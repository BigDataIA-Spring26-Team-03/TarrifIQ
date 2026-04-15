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
    "trade-representative-office-of-united-states",
]

TEST_AGENCIES = ["trade-representative-office-of-united-states"]

# FR API search terms for server-side filtering
TARIFF_SEARCH_TERMS = []

ALLOWED_DOC_TYPES = {
    "Rule",
    "Notice",
    "Presidential Document",
    "Proposed Rule",
}

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
# Federal Register XML lives under raw/ in the bucket (override with S3_BUCKET in .env).
DEFAULT_S3_BUCKET = "tariff-iq-federal-registry-bucket"
S3_FR_RAW_PREFIX = "raw/federal-register"
S3_BUCKET = (os.environ.get("S3_BUCKET") or DEFAULT_S3_BUCKET).strip() or None


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

def _fetch_page(page: int, agencies: List[str], search_term: Optional[str] = None) -> dict:
    """
    Fetch one page of document metadata from the FR API.
    Optionally filter by search_term using conditions[term].
    """
    # FR API prefers params as dict with lists, not as param tuples
    params = {
        "per_page": 20,
        "page": page,
        "order": "newest",
        "conditions[agencies][]": agencies,
        "fields[]": FIELDS,
    }

    # Layer 2: Server-side filtering using FR API's conditions[term] parameter
    if search_term:
        params["conditions[term]"] = search_term

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
        try:
            dt = datetime.fromisoformat(publication_date)
            year_str = str(dt.year)
            month_str = f"{dt.month:02d}"
        except (ValueError, TypeError):
            year_str = "unknown"
            month_str = "unknown"

        # s3://<bucket>/raw/federal-register/2024/01/2024-1234.xml
        s3_key = f"{S3_FR_RAW_PREFIX}/{year_str}/{month_str}/{document_number}.xml"

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


def _iter_pages(agencies: List[str], max_pages: Optional[int] = None, cutoff_year: int = 2018, search_term: Optional[str] = None) -> Iterator[List[dict]]:
    """
    Yields one page of raw API results at a time.
    Stops when next_page_url is null, max_pages is reached, or documents are older than cutoff_year.

    If search_term is provided, uses server-side filtering via conditions[term] (Layer 2).
    Default cutoff_year: 2018 (Section 301 era onwards).
    """
    page = 1
    while True:
        logger.info("Fetching page %d agencies=%s search_term=%s", page, agencies, search_term)
        data = _fetch_page(page, agencies, search_term=search_term)

        results = data.get("results", [])
        if not results:
            logger.info("Empty results on page %d — stopping", page)
            break

        # Check if any document is older than cutoff_year
        all_older_than_cutoff = True
        filtered_results = []
        for item in results:
            pub_date = item.get("publication_date", "")
            if pub_date:
                try:
                    year = int(pub_date.split("-")[0])
                    if year >= cutoff_year:
                        filtered_results.append(item)
                        all_older_than_cutoff = False
                except (ValueError, IndexError):
                    filtered_results.append(item)
                    all_older_than_cutoff = False
            else:
                filtered_results.append(item)
                all_older_than_cutoff = False

        if filtered_results:
            yield filtered_results

        if all_older_than_cutoff and results:
            logger.info("All documents older than %d — stopping", cutoff_year)
            break

        if not data.get("next_page_url"):
            logger.info("No next_page_url — all pages fetched")
            break

        if max_pages and page >= max_pages:
            logger.info("Reached max_pages=%d — stopping", max_pages)
            break

        page += 1
        time.sleep(PAGE_SLEEP_SECONDS)


def fetch_and_load_incrementally(
    test_mode: bool = False,
    cutoff_year: int = 2018,
    batch_size: int = 50,
    max_documents: Optional[int] = None,
) -> int:
    """
    INCREMENTAL FETCH + LOAD: Fetches documents from FR API and loads to Snowflake in batches.

    Better than fetch-all-then-load-all because:
    - Snowflake populates immediately (not empty for 30+ mins)
    - Memory efficient (doesn't hold 600+ docs in RAM)
    - Progress visible in real-time
    Default cutoff_year: 2018 (Section 301 era onwards)
    - If fetch fails, at least some documents were saved

    Args:
        test_mode: True = 1 page from USTR only; False = all agencies, all pages
        cutoff_year: Only fetch documents from this year onwards (default: 2018)
        batch_size: Documents to fetch before loading to Snowflake (default: 50)
        max_documents: Stop after this many documents are collected (after S3/XML success); None = no cap

    Returns: Total documents loaded to Snowflake
    """
    agencies = TEST_AGENCIES if test_mode else ALL_AGENCIES
    max_pages = 1 if test_mode else None
    search_terms = [None]  # Always fetch without search terms — agency is the filter

    logger.info(
        "fetch_and_load_incrementally test_mode=%s agencies=%s cutoff_year=%d batch_size=%d max_documents=%s",
        test_mode, agencies, cutoff_year, batch_size, max_documents,
    )

    batch = []
    total_loaded = 0
    skipped = 0
    documents_collected = 0
    seen_document_numbers: set[str] = set()

    # Fetch from each agency and term; deduplicate by document_number across terms.
    for agency in agencies:
        logger.info("Fetching from agency: %s (documents from %d onwards)", agency, cutoff_year)
        fetched_for_agency = 0
        skipped_by_type = 0
        skipped_by_dedup = 0
        skipped_by_api_error = 0

        for search_term in search_terms:
            logger.info("Fetching agency=%s with search_term=%s", agency, search_term)
            try:
                page_iter = _iter_pages([agency], max_pages=max_pages, cutoff_year=cutoff_year, search_term=None)
                for page_results in page_iter:
                    for item in page_results:
                        document_type = (item.get("type") or "").strip()
                        if document_type not in ALLOWED_DOC_TYPES:
                            logger.debug(
                                "skip_disallowed_type doc=%s type=%s",
                                item.get("document_number", ""),
                                document_type,
                            )
                            skipped_by_type += 1
                            continue

                        document_number = item.get("document_number", "")
                        if not document_number:
                            skipped += 1
                            continue
                        if document_number in seen_document_numbers:
                            skipped_by_dedup += 1
                            continue
                        seen_document_numbers.add(document_number)

                        html_url = item.get("html_url", "")
                        xml_url = item.get("full_text_xml_url", "")
                        publication_date = item.get("publication_date", "")

                        if not xml_url:
                            logger.debug("No xml_url for %s — skipping", document_number)
                            skipped += 1
                            continue

                        full_text, s3_key = _fetch_full_text(xml_url, document_number, publication_date)
                        if not full_text.strip():
                            logger.debug("Skipping empty doc document_number=%s", document_number)
                            skipped += 1
                            continue

                        # Normalise agency_names
                        raw_agencies = item.get("agencies") or []
                        agency_names = [
                            a.get("raw_name") or a.get("name") or ""
                            for a in raw_agencies
                            if isinstance(a, dict)
                        ]

                        batch.append({
                            "document_number": document_number,
                            "title": item.get("title", ""),
                            "publication_date": publication_date,
                            "html_url": html_url,
                            "body_html_url": item.get("full_text_xml_url", ""),
                            "document_type": item.get("type", ""),
                            "agency_names": agency_names,
                            "char_count": len(full_text),
                            "chunk_count": 0,
                            "s3_key": s3_key,
                            "raw_json": item,
                            "processing_status": "downloaded",
                        })
                        documents_collected += 1
                        fetched_for_agency += 1

                        # Load batch when it reaches batch_size
                        if len(batch) >= batch_size:
                            loaded = load_to_snowflake(batch)
                            total_loaded += loaded
                            logger.info("✓ Batch loaded: %d documents (total so far: %d)", loaded, total_loaded)
                            batch = []  # Reset for next batch

                        if max_documents is not None and documents_collected >= max_documents:
                            if batch:
                                loaded = load_to_snowflake(batch)
                                total_loaded += loaded
                                logger.info("✓ Final batch loaded: %d documents", loaded)
                                batch = []
                            logger.info(
                                "fetch_and_load_incrementally: max_documents=%s reached (collected=%s, loaded=%s)",
                                max_documents,
                                documents_collected,
                                total_loaded,
                            )
                            return total_loaded
            except requests.RequestException as exc:
                skipped_by_api_error += 1
                logger.error(
                    "Skipping agency=%s due to FR API error: %s",
                    agency,
                    exc,
                )
                continue

        logger.info(
            "agency=%s fetched=%d skipped_type=%d skipped_dedup=%d skipped_api_error=%d",
            agency, fetched_for_agency, skipped_by_type, skipped_by_dedup, skipped_by_api_error
        )

    # Load remaining documents
    if batch:
        loaded = load_to_snowflake(batch)
        total_loaded += loaded
        logger.info("✓ Final batch loaded: %d documents", loaded)

    logger.info("fetch_and_load_incrementally complete: total=%d skipped=%d", total_loaded, skipped)
    return total_loaded


def fetch_federal_register_docs(test_mode: bool = False, cutoff_year: int = 2018) -> List[dict]:
    """
    Fetches Federal Register documents using FR API term filtering.

    test_mode=True  → 1 page (up to 20 docs) from USTR only, no search terms
    test_mode=False → all agencies, all search terms, all pages (documents from cutoff_year onwards, default: 2018+)
    cutoff_year     → only fetch documents published in cutoff_year or later (default: 2018)

    Returns List[dict] matching FEDERAL_REGISTER_NOTICES schema, deduplicated by document_number.
    """
    agencies = TEST_AGENCIES if test_mode else ALL_AGENCIES
    max_pages = 1 if test_mode else None
    search_terms = [None]  # Always fetch without search terms

    logger.info(
        "fetch_federal_register_docs test_mode=%s agencies=%s cutoff_year=%d search_terms=%d",
        test_mode, agencies, cutoff_year, len(search_terms)
    )

    # Fetch with each term and deduplicate by document_number.
    docs_by_number = {}  # {document_number: doc_dict} for deduplication
    skipped_by_type = 0
    skipped_by_empty = 0
    skipped_by_no_xml = 0

    if not search_terms:
        # Test mode: fetch without search terms
        search_terms = [None]

    for search_term in search_terms:
        logger.info("Fetching with search_term=%s", search_term)
        try:
            page_iter = _iter_pages(agencies, max_pages=max_pages, cutoff_year=cutoff_year, search_term=None)
            for page_results in page_iter:
                for item in page_results:
                    document_number = item.get("document_number", "")
                    if not document_number or document_number in docs_by_number:
                        # Already processed or no document number
                        continue

                    document_type = (item.get("type") or "").strip()
                    if document_type not in ALLOWED_DOC_TYPES:
                        logger.debug("Filtered by type: %s type=%s", document_number, document_type)
                        skipped_by_type += 1
                        continue

                    html_url = item.get("html_url", "")
                    xml_url = item.get("full_text_xml_url", "")
                    publication_date = item.get("publication_date", "")

                    if not xml_url:
                        logger.debug("No xml_url for %s — skipping", document_number)
                        skipped_by_no_xml += 1
                        continue

                    full_text, s3_key = _fetch_full_text(xml_url, document_number, publication_date)
                    if not full_text.strip():
                        logger.debug("Skipping empty doc %s", document_number)
                        skipped_by_empty += 1
                        continue

                    # Normalise agency_names to a plain list of strings
                    raw_agencies = item.get("agencies") or []
                    agency_names = [
                        a.get("raw_name") or a.get("name") or ""
                        for a in raw_agencies
                        if isinstance(a, dict)
                    ]

                    docs_by_number[document_number] = {
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
        except requests.RequestException as exc:
            logger.error("Skipping bulk fetch due to FR API error: %s", exc)
            continue

    docs = list(docs_by_number.values())
    logger.info(
        "fetch done total=%d skipped_by_type=%d skipped_by_empty=%d skipped_by_no_xml=%d",
        len(docs), skipped_by_type, skipped_by_empty, skipped_by_no_xml
    )
    return docs


# ── Snowflake load layer ──────────────────────────────────────────────────────

def load_to_snowflake(rows: List[dict]) -> int:
    """
    MERGE-upserts documents into FEDERAL_REGISTER_NOTICES using batch operations.

    On match   → UPDATE all fields (in case full_text changed)
    On no match → INSERT

    Uses batch MERGE with VALUES clause for much better performance.
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

            # Build VALUES clause with all rows in batch
            values_clauses = []
            params = []

            for doc in batch:
                values_clauses.append("(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)")
                params.extend([
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
                ])

            # Single batch MERGE statement
            merge_sql = f"""
            MERGE INTO FEDERAL_REGISTER_NOTICES AS t
            USING (
                SELECT
                    column1 AS document_number,
                    column2 AS title,
                    column3 AS publication_date,
                    column4 AS document_type,
                    PARSE_JSON(column5) AS agency_names,
                    column6 AS html_url,
                    column7 AS body_html_url,
                    column8 AS chunk_count,
                    column9 AS s3_key,
                    PARSE_JSON(column10) AS raw_json,
                    column11 AS processing_status
                FROM (VALUES {','.join(values_clauses)})
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
            """

            cur.execute(merge_sql, params)
            loaded += len(batch)

            logger.info(
                "load_to_snowflake batch %d-%d done (total so far: %d)",
                i + 1, i + len(batch), loaded,
            )
        conn.commit()
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
