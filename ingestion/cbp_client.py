"""
CBP Federal Register ingestion for TariffIQ.

Fetches CBP (customs-and-border-protection) agency documents from the Federal Register API,
strips HTML to plain text, and MERGE-upserts into Snowflake's CBP_FEDERAL_REGISTER_NOTICES table.

Usage:
  # Test mode — 1 page (20 docs) from CBP only:
  python -m ingestion.cbp_client --test

  # Full mode — all pages (up to 600+ CBP-specific docs):
  python -m ingestion.cbp_client
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
from ingestion.federal_register_client import (
    _get_with_retry,
    _get_s3_client,
    _parse_xml_to_text,
    PAGE_SLEEP_SECONDS,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

BASE_URL = "https://www.federalregister.gov/api/v1/documents.json"

# CBP-specific agency numeric ID (Federal Register API — more stable than slug)
CBP_AGENCY_ID = 501  # U.S. Customs and Border Protection

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

SNOWFLAKE_BATCH_SIZE = 50  # rows per MERGE batch
DEFAULT_S3_BUCKET = "tariff-iq-federal-registry-bucket"
S3_CBP_RAW_PREFIX = "raw/cbp"
S3_BUCKET = (os.environ.get("S3_BUCKET") or DEFAULT_S3_BUCKET).strip() or None

# ── Document keyword filter (title + abstract) ────────────────────────────────

# REMOVE keywords: if ANY found in (title + abstract), skip document entirely
KEYWORDS_REMOVE = frozenset({
    "laboratory accreditation", "gauger", "information collection",
    "paperwork reduction", "copyright", "trademark",
    "intellectual property", "arrival restrictions", "visa waiver",
    "canine", "ctpat", "interest rate on overdue",
    "flights to and from", "export manifest", "surety bond",
    "accreditation of", "approval of",
})

# KEEP keywords: if ANY found in title OR abstract, fetch and ingest document
KEYWORDS_KEEP = frozenset({
    "rate of duty", "ad valorem", "tariff rate", "tariff-rate quota",
    "harmonized tariff schedule", "htsus", "chapter 99", "9903.",
    "additional duties", "executive order", "section 232",
    "section 201", "ieepa", "proclamation", "usmca",
    "rules of origin", "preferential tariff", "country of origin",
    "quota", "entered for consumption", "antidumping",
    "countervailing", "de minimis", "steel", "aluminum",
    "duty-free", "tariff preference", "trade remedy",
    "modification", "subheading", "classification", "tariff", "country", "section 301"
})


def _document_passes_filter(title: str, abstract: str) -> bool:
    """
    Filter documents by title + abstract.

    Priority: REMOVE keywords always win over KEEP keywords.

    1. Check (title + abstract) against REMOVE keywords (blacklist)
       If ANY remove keyword found → skip (return False)
    2. Check title and abstract separately against KEEP keywords (whitelist)
       If ANY keep keyword found in title OR abstract → keep (return True)
    3. Otherwise → skip (return False)
    """
    # Combine title + abstract for REMOVE keyword check
    combined = f"{title} {abstract}".lower()

    # REMOVE keywords: if ANY found, reject the document
    for kw in KEYWORDS_REMOVE:
        if kw in combined:
            return False

    # KEEP keywords: check title and abstract separately
    title_lower = title.lower()
    abstract_lower = abstract.lower()

    return any(kw in title_lower or kw in abstract_lower for kw in KEYWORDS_KEEP)


# ── S3 upload layer ───────────────────────────────────────────────────────────

def _upload_cbp_xml_to_s3(document_number: str, xml_content: bytes, publication_date: str) -> Optional[str]:
    """
    Upload CBP raw XML to S3 under raw/cbp/ prefix (CBP-specific, not shared with FR).

    Returns: S3 key if successful, None on failure
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

        # s3://<bucket>/raw/cbp/2025/08/2026-16499.xml
        s3_key = f"{S3_CBP_RAW_PREFIX}/{year_str}/{month_str}/{document_number}.xml"

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
        logger.info("Uploaded CBP to S3: s3://%s/%s", S3_BUCKET, s3_key)
        return s3_key

    except Exception as exc:
        logger.warning("Failed to upload CBP to S3 doc=%s error=%s", document_number, exc)
        return None


# ── Fetch layer ───────────────────────────────────────────────────────────────

def _fetch_page(page: int) -> dict:
    """
    Fetch one page of CBP document metadata from the FR API.
    Uses numeric agency ID (more robust than slug).
    """
    params = {
        "per_page": 20,
        "page": page,
        "order": "newest",
        "conditions[agency_ids][]": [CBP_AGENCY_ID],
        "fields[]": FIELDS,
    }

    resp = _get_with_retry(BASE_URL, params=params)
    return resp.json()


def _fetch_full_text(xml_url: str, document_number: str = None, publication_date: str = None) -> tuple[str, Optional[str]]:
    """
    Fetches the full document text from the Federal Register XML endpoint.
    The XML URL is a direct file download — not CAPTCHA-protected.
    Also uploads raw XML to S3 under raw/cbp/ prefix for long-term storage.

    Returns:
        (plain_text, s3_key) where s3_key is None if S3 upload failed
    """
    if not xml_url:
        return "", None

    try:
        resp = _get_with_retry(xml_url)
        xml_content = resp.content

        # Upload raw XML to S3 under CBP prefix (optional, non-blocking)
        s3_key = None
        if document_number and publication_date:
            s3_key = _upload_cbp_xml_to_s3(document_number, xml_content, publication_date)

        # Parse XML to plain text
        plain_text = _parse_xml_to_text(xml_content)
        return plain_text, s3_key

    except Exception as exc:
        logger.warning("fetch_full_text failed url=%s error=%s", xml_url, exc)
        return "", None


def _iter_pages(max_pages: Optional[int] = None, cutoff_year: int = 2016) -> Iterator[List[dict]]:
    """
    Yields one page of raw API results at a time for CBP agency.
    Stops when next_page_url is null, max_pages is reached, or documents are older than cutoff_year.

    Default cutoff_year: 2016 (CBP tariff regime baseline)
    """
    page = 1
    while True:
        logger.info("Fetching CBP page %d", page)
        data = _fetch_page(page)

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


def fetch_and_load_cbp_incrementally(
    test_mode: bool = False,
    cutoff_year: int = 2016,
    batch_size: int = 50,
    max_documents: Optional[int] = None,
) -> int:
    """
    INCREMENTAL FETCH + LOAD: Fetches CBP documents from FR API and loads to Snowflake in batches.

    Better than fetch-all-then-load-all because:
    - Snowflake populates immediately (not empty for 30+ mins)
    - Memory efficient (doesn't hold 600+ docs in RAM)
    - Progress visible in real-time
    - If fetch fails, at least some documents were saved

    Args:
        test_mode: True = 1 page from CBP only; False = all pages
        cutoff_year: Only fetch documents from this year onwards (default: 2016)
        batch_size: Documents to fetch before loading to Snowflake (default: 50)
        max_documents: Stop after this many documents are collected (after S3/XML success); None = no cap

    Returns: Total documents loaded to Snowflake
    """
    max_pages = 1 if test_mode else None

    logger.info(
        "fetch_and_load_cbp_incrementally test_mode=%s cutoff_year=%d batch_size=%d max_documents=%s",
        test_mode, cutoff_year, batch_size, max_documents,
    )

    batch = []
    total_loaded = 0
    skipped = 0
    documents_collected = 0
    seen_document_numbers: set[str] = set()

    # Load already-ingested document numbers for idempotent skipping
    already_in_db: set[str] = set()
    try:
        _conn = get_snowflake_conn()
        _cur = _conn.cursor()
        _cur.execute("SELECT document_number FROM CBP_FEDERAL_REGISTER_NOTICES")
        already_in_db = {row[0] for row in _cur.fetchall()}
        _cur.close()
        _conn.close()
        logger.info("Loaded %d existing doc numbers from Snowflake (will skip these)", len(already_in_db))
    except Exception as _exc:
        logger.warning("Could not load existing doc numbers (idempotency disabled): %s", _exc)

    logger.info("Fetching from CBP agency (documents from %d onwards)", cutoff_year)
    fetched = 0
    skipped_by_type = 0
    skipped_by_dedup = 0
    total_filtered_out = 0

    try:
        page_iter = _iter_pages(max_pages=max_pages, cutoff_year=cutoff_year)
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

                # Skip documents already in Snowflake (idempotent re-runs)
                if document_number in already_in_db:
                    logger.debug("skip_already_ingested doc=%s", document_number)
                    skipped += 1
                    continue

                # Apply document filter (title + abstract) BEFORE downloading full text
                title = item.get("title", "")
                abstract = item.get("abstract", "")
                if not _document_passes_filter(title, abstract):
                    logger.debug("skip_document_filter doc=%s title=%s", document_number, title)
                    total_filtered_out += 1
                    continue

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

                abstract = item.get("abstract", "")

                batch.append({
                    "document_number": document_number,
                    "title": title,
                    "publication_date": publication_date,
                    "html_url": html_url,
                    "body_html_url": item.get("full_text_xml_url", ""),
                    "document_type": item.get("type", ""),
                    "agency_names": agency_names,
                    "abstract": abstract,
                    "full_text": full_text,
                    "char_count": len(full_text),
                    "chunk_count": 0,
                    "s3_key": s3_key,
                    "raw_json": item,
                    "processing_status": "downloaded",
                })
                documents_collected += 1
                fetched += 1

                # Load batch when it reaches batch_size
                if len(batch) >= batch_size:
                    loaded = load_cbp_to_snowflake(batch)
                    total_loaded += loaded
                    logger.info("✓ CBP Batch loaded: %d documents (total so far: %d)", loaded, total_loaded)
                    batch = []  # Reset for next batch

                if max_documents is not None and documents_collected >= max_documents:
                    if batch:
                        loaded = load_cbp_to_snowflake(batch)
                        total_loaded += loaded
                        logger.info("✓ CBP Final batch loaded: %d documents", loaded)
                        batch = []
                    logger.info(
                        "fetch_and_load_cbp_incrementally: max_documents=%s reached (collected=%s, loaded=%s)",
                        max_documents,
                        documents_collected,
                        total_loaded,
                    )
                    return total_loaded

    except requests.RequestException as exc:
        logger.error(
            "CBP API error: %s",
            exc,
        )

    logger.info(
        "CBP agency fetched=%d skipped_type=%d skipped_dedup=%d "
        "skipped_already_in_db=%d title_filtered_out=%d",
        fetched, skipped_by_type, skipped_by_dedup,
        len(already_in_db & seen_document_numbers),
        total_filtered_out,
    )

    # Load remaining documents
    if batch:
        loaded = load_cbp_to_snowflake(batch)
        total_loaded += loaded
        logger.info("✓ CBP Final batch loaded: %d documents", loaded)

    logger.info(
        "fetch_and_load_cbp_incrementally complete: "
        "total_ingested=%d total_filtered_out=%d total_skipped=%d",
        total_loaded, total_filtered_out, skipped,
    )
    return total_loaded


# ── Snowflake load layer ──────────────────────────────────────────────────────

def load_cbp_to_snowflake(rows: List[dict]) -> int:
    """
    MERGE-upserts documents into CBP_FEDERAL_REGISTER_NOTICES using batch operations.

    On match   → UPDATE all fields (in case full_text changed)
    On no match → INSERT

    Uses batch MERGE with VALUES clause for much better performance.
    Processes in batches of SNOWFLAKE_BATCH_SIZE.
    Returns total rows processed.

    Columns stored: document_number, title, publication_date, document_type,
                    agency_names, abstract, full_text, html_url, body_html_url,
                    char_count, chunk_count, s3_key, raw_json, processing_status
    """
    if not rows:
        logger.info("load_cbp_to_snowflake: nothing to load")
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
                # 14 columns: document_number, title, publication_date, document_type,
                #             agency_names, abstract, full_text, html_url, body_html_url,
                #             char_count, chunk_count, s3_key, raw_json, processing_status
                values_clauses.append("(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)")
                params.extend([
                    doc["document_number"],
                    doc["title"],
                    doc["publication_date"],
                    doc["document_type"],
                    json.dumps(doc["agency_names"]),
                    doc.get("abstract", ""),
                    doc.get("full_text", ""),
                    doc["html_url"],
                    doc["body_html_url"],
                    doc["char_count"],
                    doc["chunk_count"],
                    doc.get("s3_key"),
                    json.dumps(doc["raw_json"]),
                    doc.get("processing_status"),
                ])

            # Single batch MERGE statement
            merge_sql = f"""
            MERGE INTO CBP_FEDERAL_REGISTER_NOTICES AS t
            USING (
                SELECT
                    column1 AS document_number,
                    column2 AS title,
                    column3 AS publication_date,
                    column4 AS document_type,
                    PARSE_JSON(column5) AS agency_names,
                    column6 AS abstract,
                    column7 AS full_text,
                    column8 AS html_url,
                    column9 AS body_html_url,
                    column10 AS char_count,
                    column11 AS chunk_count,
                    column12 AS s3_key,
                    PARSE_JSON(column13) AS raw_json,
                    column14 AS processing_status
                FROM (VALUES {','.join(values_clauses)})
            ) AS s
            ON t.document_number = s.document_number
            WHEN MATCHED THEN UPDATE SET
                t.title          = s.title,
                t.publication_date = s.publication_date,
                t.document_type  = s.document_type,
                t.agency_names   = s.agency_names,
                t.abstract       = s.abstract,
                t.full_text      = s.full_text,
                t.html_url       = s.html_url,
                t.body_html_url  = s.body_html_url,
                t.char_count     = s.char_count,
                t.chunk_count    = s.chunk_count,
                t.s3_key         = s.s3_key,
                t.raw_json       = s.raw_json,
                t.processing_status = s.processing_status
            WHEN NOT MATCHED THEN INSERT
                (document_number, title, publication_date,
                 document_type, agency_names, abstract, full_text,
                 html_url, body_html_url, char_count, chunk_count,
                 s3_key, raw_json, processing_status)
            VALUES
                (s.document_number, s.title, s.publication_date,
                 s.document_type, s.agency_names, s.abstract, s.full_text,
                 s.html_url, s.body_html_url, s.char_count, s.chunk_count,
                 s.s3_key, s.raw_json, s.processing_status)
            """

            cur.execute(merge_sql, params)
            loaded += len(batch)

            logger.info(
                "load_cbp_to_snowflake batch %d-%d done (total so far: %d)",
                i + 1, i + len(batch), loaded,
            )
        conn.commit()
    finally:
        cur.close()
        conn.close()

    logger.info("load_cbp_to_snowflake complete total_loaded=%d", loaded)
    return loaded


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest CBP Federal Register documents")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: fetch 1 page (20 docs) from CBP only",
    )
    args = parser.parse_args()

    loaded = fetch_and_load_cbp_incrementally(test_mode=args.test)
    print(f"\n{'='*60}")
    print(f"Loaded {loaded} CBP documents into CBP_FEDERAL_REGISTER_NOTICES")
    print(f"{'='*60}\n")
