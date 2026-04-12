"""
Federal Register ingestion for TariffIQ.

Fetches tariff-related documents from 5 USTR/trade agencies,
strips HTML to plain text, and MERGE-upserts into Snowflake.

Usage:
  # Test mode — 1 page (20 docs) from USTR only:
  python -m ingestion.federal_register_client --test

  # Full mode — all agencies, all pages (600+ docs):
  python -m ingestion.federal_register_client
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Iterator, List, Optional

import boto3
import requests

from ingestion.connection import get_snowflake_conn
from ingestion.html_parser import strip_html

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

BASE_URL = "https://www.federalregister.gov/api/v1/documents.json"

ALL_AGENCIES = [
    "trade-representative-office-of-united-states",  # Section 301, IEEPA, FTA, GSP
    "commerce-department",                            # Section 232 steel/aluminum
    "customs-and-border-protection",                  # HTS classification rulings
    "international-trade-commission",                 # USITC injury determinations
    "international-trade-administration",             # ADD/CVD orders
]

TEST_AGENCIES = ["trade-representative-office-of-united-states"]

FIELDS = [
    "title",
    "abstract",
    "document_number",
    "publication_date",
    "html_url",
    "full_text_xml_url",
    "agencies",
    "type",
]

PAGE_SLEEP_SECONDS = 0.5   # be polite to the FR API
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAYS = [1, 2, 4]   # exponential backoff in seconds
SNOWFLAKE_BATCH_SIZE = 50  # rows per MERGE batch
S3_BUCKET = os.environ.get("S3_BUCKET", "tariffiq-raw-docs")


# ── HTTP helpers ─────────────────────────────────────────────────────────────

def _get_with_retry(url: str, params=None) -> requests.Response:
    """GET with exponential backoff on 429/5xx."""
    for attempt, delay in enumerate(RETRY_DELAYS):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt < len(RETRY_DELAYS) - 1:
                    logger.warning(
                        "HTTP %s from %s — retrying in %ds", resp.status_code, url, delay
                    )
                    time.sleep(delay)
                    continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt == len(RETRY_DELAYS) - 1:
                raise
            logger.warning("Request error %s — retrying in %ds", exc, delay)
            time.sleep(delay)

    raise RuntimeError(f"All retries exhausted for {url}")


# ── Fetch layer ───────────────────────────────────────────────────────────────

def _fetch_page(page: int, agencies: List[str]) -> dict:
    """Fetch one page of document metadata from the FR API."""
    # FR API prefers params as dict with lists, not as param tuples
    params = {
        "per_page": 20,
        "page": page,
        "order": "newest",
        "conditions[agencies][]": agencies,
        "fields[]": FIELDS,
    }

    resp = _get_with_retry(BASE_URL, params=params)
    return resp.json()


def _fetch_full_text(xml_url: str, document_number: str = None, publication_date: str = None) -> tuple[str, Optional[str]]:
    """
    Fetches the full document text from the Federal Register XML endpoint.
    The XML URL is a direct file download — not CAPTCHA-protected.
    Also uploads raw XML to S3 for long-term storage.

    Returns:
        (plain_text, s3_key) where s3_key is None if S3 upload failed
    """
    if not xml_url:
        return "", None

    try:
        resp = _get_with_retry(xml_url)
        xml_content = resp.content

        # Upload raw XML to S3 (optional, non-blocking)
        s3_key = None
        if document_number and publication_date:
            s3_key = upload_raw_xml_to_s3(document_number, xml_content, publication_date)

        # Parse XML to plain text
        plain_text = _parse_xml_to_text(xml_content)
        return plain_text, s3_key

    except Exception as exc:
        logger.warning("fetch_full_text failed url=%s error=%s", xml_url, exc)
        return "", None


def _parse_xml_to_text(xml_bytes: bytes) -> str:
    """
    Extracts plain text from a Federal Register XML document.
    Walks all elements and concatenates their text, preserving paragraph breaks.
    """
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        logger.warning("XML parse error: %s", exc)
        return ""

    parts = []
    for elem in root.iter():
        text = (elem.text or "").strip()
        tail = (elem.tail or "").strip()
        if text:
            parts.append(text)
        if tail:
            parts.append(tail)

    return "\n\n".join(parts)


def _get_s3_client():
    """Get boto3 S3 client (lazy init)."""
    return boto3.client(
        "s3",
        region_name=os.environ.get("AWS_REGION", "us-central1"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )


def upload_raw_xml_to_s3(document_number: str, xml_content: bytes, publication_date: str) -> Optional[str]:
    """
    Upload raw XML to S3 for long-term storage before processing.

    Args:
        document_number: Federal Register document number (e.g., "2024-1234")
        xml_content: Raw XML bytes
        publication_date: Publication date (e.g., "2024-01-15")

    Returns:
        S3 key if successful, None on failure
    """
    if not xml_content or not S3_BUCKET:
        return None

    try:
        # Parse date to extract year/month
        try:
            dt = datetime.fromisoformat(publication_date)
            year = dt.year
            month = dt.month
        except (ValueError, TypeError):
            year, month = "unknown", "unknown"

        # S3 key: s3://bucket/federal-register/2024/01/2024-1234.xml
        s3_key = f"federal-register/{year}/{month:02d}/{document_number}.xml"

        s3_client = _get_s3_client()

        # Check if key already exists (idempotent)
        try:
            s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
            logger.debug("S3 key already exists: %s", s3_key)
            return s3_key
        except Exception as e:
            # Key doesn't exist (404) or other error - proceed with upload
            if "404" not in str(e) and "NoSuchKey" not in str(e):
                logger.debug("head_object check: %s", e)
            pass

        # Upload
        s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=xml_content)
        logger.info("Uploaded to S3: s3://%s/%s", S3_BUCKET, s3_key)
        return s3_key

    except Exception as exc:
        logger.warning("Failed to upload to S3 doc=%s error=%s", document_number, exc)
        return None


def _iter_pages(agencies: List[str], max_pages: Optional[int] = None) -> Iterator[List[dict]]:
    """
    Yields one page of raw API results at a time.
    Stops when next_page_url is null or max_pages is reached.
    """
    page = 1
    while True:
        logger.info("Fetching page %d agencies=%s", page, agencies)
        data = _fetch_page(page, agencies)

        results = data.get("results", [])
        if not results:
            logger.info("Empty results on page %d — stopping", page)
            break

        yield results

        if not data.get("next_page_url"):
            logger.info("No next_page_url — all pages fetched")
            break

        if max_pages and page >= max_pages:
            logger.info("Reached max_pages=%d — stopping", max_pages)
            break

        page += 1
        time.sleep(PAGE_SLEEP_SECONDS)


def fetch_federal_register_docs(test_mode: bool = False) -> List[dict]:
    """
    Fetches Federal Register documents and their full text.

    test_mode=True  → 1 page (up to 20 docs) from USTR only
    test_mode=False → all agencies, all pages (600+ docs)

    Returns List[dict] matching FEDERAL_REGISTER_NOTICES schema.
    """
    agencies = TEST_AGENCIES if test_mode else ALL_AGENCIES
    max_pages = 1 if test_mode else None

    logger.info(
        "fetch_federal_register_docs test_mode=%s agencies=%s", test_mode, agencies
    )

    docs: List[dict] = []
    skipped = 0

    # Fetch from each agency separately to avoid API errors
    for agency in agencies:
        logger.info("Fetching from agency: %s", agency)
        for page_results in _iter_pages([agency], max_pages=max_pages):
            for item in page_results:
                html_url = item.get("html_url", "")
                xml_url = item.get("full_text_xml_url", "")
                document_number = item.get("document_number", "")
                publication_date = item.get("publication_date", "")

                if not xml_url:
                    logger.debug("No xml_url for %s — skipping", document_number)
                    skipped += 1
                    continue

                full_text, s3_key = _fetch_full_text(xml_url, document_number, publication_date)
                if not full_text.strip():
                    logger.debug(
                        "Skipping empty doc document_number=%s", document_number
                    )
                    skipped += 1
                    continue

                # Normalise agency_names to a plain list of strings
                raw_agencies = item.get("agencies") or []
                agency_names = [
                    a.get("raw_name") or a.get("name") or ""
                    for a in raw_agencies
                    if isinstance(a, dict)
                ]

                docs.append(
                    {
                        "document_number": document_number,
                        "title": item.get("title", ""),
                        "publication_date": publication_date,  # "YYYY-MM-DD"
                        "html_url": html_url,
                        "body_html_url": item.get("full_text_xml_url", ""),
                        "document_type": item.get("type", ""),
                        "agency_names": agency_names,
                        "char_count": len(full_text),
                        "chunk_count": 0,       # updated later by chunker
                        "s3_key": s3_key,       # raw XML stored in S3
                        "raw_json": item,       # store full API response
                        "processing_status": "downloaded",  # Mark as downloaded
                    }
                )

    logger.info(
        "fetch done total=%d skipped=%d", len(docs), skipped
    )
    return docs


# ── Snowflake load layer ──────────────────────────────────────────────────────

def load_to_snowflake(rows: List[dict]) -> int:
    """
    MERGE-upserts documents into FEDERAL_REGISTER_NOTICES.

    On match   → UPDATE all fields (in case full_text changed)
    On no match → INSERT

    Processes in batches of SNOWFLAKE_BATCH_SIZE.
    Returns total rows processed.
    """
    if not rows:
        logger.info("load_to_snowflake: nothing to load")
        return 0

    conn = get_snowflake_conn()
    cur = conn.cursor()
    loaded = 0

    try:
        for i in range(0, len(rows), SNOWFLAKE_BATCH_SIZE):
            batch = rows[i : i + SNOWFLAKE_BATCH_SIZE]
            for doc in batch:
                cur.execute(
                    """
                    MERGE INTO FEDERAL_REGISTER_NOTICES AS t
                    USING (
                        SELECT
                            %s  AS document_number,
                            %s  AS title,
                            %s  AS publication_date,
                            %s  AS document_type,
                            PARSE_JSON(%s) AS agency_names,
                            %s  AS html_url,
                            %s  AS body_html_url,
                            %s  AS chunk_count,
                            %s  AS s3_key,
                            PARSE_JSON(%s) AS raw_json,
                            %s  AS processing_status
                    ) AS s
                    ON t.document_number = s.document_number
                    WHEN MATCHED THEN UPDATE SET
                        t.title          = s.title,
                        t.publication_date = s.publication_date,
                        t.document_type  = s.document_type,
                        t.agency_names   = s.agency_names,
                        t.html_url       = s.html_url,
                        t.body_html_url  = s.body_html_url,
                        t.chunk_count    = s.chunk_count,
                        t.s3_key         = s.s3_key,
                        t.raw_json       = s.raw_json,
                        t.processing_status = s.processing_status
                    WHEN NOT MATCHED THEN INSERT
                        (document_number, title, publication_date,
                         document_type, agency_names, html_url,
                         body_html_url, chunk_count, s3_key, raw_json, processing_status)
                    VALUES
                        (s.document_number, s.title, s.publication_date,
                         s.document_type, s.agency_names, s.html_url,
                         s.body_html_url, s.chunk_count, s.s3_key, s.raw_json, s.processing_status)
                    """,
                    (
                        doc["document_number"],
                        doc["title"],
                        doc["publication_date"],
                        doc["document_type"],
                        json.dumps(doc["agency_names"]),
                        doc["html_url"],
                        doc["body_html_url"],
                        doc["chunk_count"],
                        doc.get("s3_key"),
                        json.dumps(doc["raw_json"]),
                        doc.get("processing_status"),
                    ),
                )
                loaded += 1

            logger.info(
                "load_to_snowflake batch %d-%d done (total so far: %d)",
                i + 1, i + len(batch), loaded,
            )
    finally:
        cur.close()
        conn.close()

    logger.info("load_to_snowflake complete total_loaded=%d", loaded)
    return loaded


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Federal Register documents")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: fetch 1 page (20 docs) from USTR only",
    )
    args = parser.parse_args()

    docs = fetch_federal_register_docs(test_mode=args.test)
    print(f"\n{'='*60}")
    print(f"Fetched {len(docs)} documents")
    if docs:
        print(f"Sample: [{docs[0]['document_number']}] {docs[0]['title'][:80]}")
        print(f"        {docs[0]['char_count']:,} chars of full text")
    print(f"{'='*60}\n")

    loaded = load_to_snowflake(docs)
    print(f"\nLoaded {loaded} documents into FEDERAL_REGISTER_NOTICES")
    print("Done.")
