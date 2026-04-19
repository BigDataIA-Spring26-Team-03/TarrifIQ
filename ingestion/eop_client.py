"""
Executive Office of the President (EOP) Federal Register ingestion for TariffIQ.

Same flow as ITC/USTR: FR API → S3 raw XML → Snowflake EOP_DOCUMENTS.

Agency slug: executive-office-of-the-president
Document types: Presidential Document, Notice (Proclamations / Executive Orders live under Presidential Document)
Search terms: tariff-relevant keywords (Section 301/232, preference programs, HTSUS, etc.)
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from typing import Iterator, List, Optional

import requests

from ingestion.connection import get_snowflake_conn
from ingestion.federal_register_client import (
    BASE_URL,
    FIELDS,
    PAGE_SLEEP_SECONDS,
    _get_s3_client,
    _get_with_retry,
    _parse_xml_to_text,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

ALL_AGENCIES = ["executive-office-of-the-president"]
TEST_AGENCIES = ["executive-office-of-the-president"]

# Final keyword set for EOP tariff / trade coverage (each term is queried separately; results deduped).
TARIFF_SEARCH_TERMS = [
    "Section 301",
    "Section 232",
    "Surcharge",
    "Reciprocal",
    "GSP",
    "LDBDC",
    "CNL",
    "Beneficiary",
    "AGOA",
    "CBTPA",
    "CAFTA",
    "USMCA",
    "ASEAN",
    "HTSUS",
    "Ad Valorem",
    "Duty-Free",
    "Modification",
]

ALLOWED_DOC_TYPES = {"Presidential Document", "Notice"}

SNOWFLAKE_BATCH_SIZE = 50
DEFAULT_S3_BUCKET = "tariff-iq-federal-registry-bucket"
S3_EOP_RAW_PREFIX = "raw/eop"
S3_BUCKET = (os.environ.get("S3_BUCKET") or DEFAULT_S3_BUCKET).strip() or None


def upload_eop_raw_xml_to_s3(document_number: str, xml_content: bytes, publication_date: str) -> Optional[str]:
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

        s3_key = f"{S3_EOP_RAW_PREFIX}/{year_str}/{month_str}/{document_number}.xml"
        s3_client = _get_s3_client()
        try:
            s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
            return s3_key
        except Exception:
            pass
        s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=xml_content)
        return s3_key
    except Exception as exc:
        logger.warning("Failed to upload EOP XML doc=%s error=%s", document_number, exc)
        return None


def _fetch_page(page: int, agencies: List[str], search_term: Optional[str] = None) -> dict:
    params = {
        "per_page": 20,
        "page": page,
        "order": "newest",
        "conditions[agencies][]": agencies,
        "fields[]": FIELDS,
    }
    if search_term:
        params["conditions[term]"] = search_term
    resp = _get_with_retry(BASE_URL, params=params)
    return resp.json()


def _fetch_full_text(xml_url: str, document_number: str = None, publication_date: str = None) -> tuple[str, Optional[str]]:
    if not xml_url:
        return "", None
    try:
        resp = _get_with_retry(xml_url)
        xml_content = resp.content
        s3_key = None
        if document_number and publication_date:
            s3_key = upload_eop_raw_xml_to_s3(document_number, xml_content, publication_date)
        plain_text = _parse_xml_to_text(xml_content)
        return plain_text, s3_key
    except Exception as exc:
        logger.warning("fetch_full_text failed url=%s error=%s", xml_url, exc)
        return "", None


def _iter_pages(
    agencies: List[str],
    max_pages: Optional[int] = None,
    cutoff_year: int = 2021,
    search_term: Optional[str] = None,
) -> Iterator[List[dict]]:
    page = 1
    while True:
        data = _fetch_page(page, agencies, search_term=search_term)
        results = data.get("results", [])
        if not results:
            break

        all_older = True
        filtered_results = []
        for item in results:
            pub_date = item.get("publication_date", "")
            if pub_date:
                try:
                    year = int(pub_date.split("-")[0])
                    if year >= cutoff_year:
                        filtered_results.append(item)
                        all_older = False
                except (ValueError, IndexError):
                    filtered_results.append(item)
                    all_older = False
            else:
                filtered_results.append(item)
                all_older = False

        if filtered_results:
            yield filtered_results

        if all_older and results:
            break
        if not data.get("next_page_url"):
            break
        if max_pages and page >= max_pages:
            break
        page += 1
        time.sleep(PAGE_SLEEP_SECONDS)


def fetch_and_load_eop_incrementally(
    test_mode: bool = False,
    cutoff_year: int = 2021,
    batch_size: int = 50,
    max_documents: Optional[int] = None,
) -> int:
    agencies = TEST_AGENCIES if test_mode else ALL_AGENCIES
    max_pages = 1 if test_mode else None
    search_terms = TARIFF_SEARCH_TERMS or [None]

    existing_doc_numbers: set[str] = set()
    try:
        conn = get_snowflake_conn()
        cur = conn.cursor()
        cur.execute("SELECT document_number FROM EOP_DOCUMENTS")
        existing_doc_numbers = {row[0] for row in (cur.fetchall() or []) if row and row[0]}
        cur.close()
        conn.close()
    except Exception as exc:
        logger.warning("Could not load existing EOP document numbers: %s", exc)

    batch = []
    total_loaded = 0
    collected = 0
    seen_document_numbers: set[str] = set()

    for agency in agencies:
        for search_term in search_terms:
            try:
                logger.info("EOP fetch agency=%s term=%s", agency, search_term or "(none)")
                page_iter = _iter_pages(
                    [agency],
                    max_pages=max_pages,
                    cutoff_year=cutoff_year,
                    search_term=search_term,
                )
                for page_results in page_iter:
                    for item in page_results:
                        document_type = (item.get("type") or "").strip()
                        if document_type not in ALLOWED_DOC_TYPES:
                            continue

                        document_number = item.get("document_number", "")
                        if not document_number:
                            continue
                        if document_number in seen_document_numbers:
                            continue
                        seen_document_numbers.add(document_number)
                        if document_number in existing_doc_numbers:
                            continue

                        xml_url = item.get("full_text_xml_url", "")
                        publication_date = item.get("publication_date", "")
                        if not xml_url:
                            continue

                        full_text, s3_key = _fetch_full_text(xml_url, document_number, publication_date)
                        if not full_text.strip():
                            continue

                        raw_agencies = item.get("agencies") or []
                        agency_names = [
                            a.get("raw_name") or a.get("name") or ""
                            for a in raw_agencies
                            if isinstance(a, dict)
                        ]

                        batch.append(
                            {
                                "document_number": document_number,
                                "title": item.get("title", ""),
                                "publication_date": publication_date,
                                "html_url": item.get("html_url", ""),
                                "body_html_url": item.get("full_text_xml_url", ""),
                                "document_type": item.get("type", ""),
                                "agency_names": agency_names,
                                "char_count": len(full_text),
                                "chunk_count": 0,
                                "s3_key": s3_key,
                                "raw_json": item,
                                "processing_status": "downloaded",
                            }
                        )
                        collected += 1

                        if len(batch) >= batch_size:
                            total_loaded += load_eop_to_snowflake(batch)
                            batch = []

                        if max_documents is not None and collected >= max_documents:
                            if batch:
                                total_loaded += load_eop_to_snowflake(batch)
                                batch = []
                            return total_loaded
            except requests.RequestException as exc:
                logger.error("Skipping agency=%s search_term=%s due to API error: %s", agency, search_term, exc)
                continue

    if batch:
        total_loaded += load_eop_to_snowflake(batch)
    return total_loaded


def load_eop_to_snowflake(rows: List[dict]) -> int:
    if not rows:
        return 0

    conn = get_snowflake_conn()
    cur = conn.cursor()
    loaded = 0

    try:
        for i in range(0, len(rows), SNOWFLAKE_BATCH_SIZE):
            batch = rows[i : i + SNOWFLAKE_BATCH_SIZE]
            values_clauses = []
            params = []
            for doc in batch:
                values_clauses.append("(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)")
                params.extend(
                    [
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
                    ]
                )

            merge_sql = f"""
            MERGE INTO EOP_DOCUMENTS AS t
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
                t.title = s.title,
                t.publication_date = s.publication_date,
                t.document_type = s.document_type,
                t.agency_names = s.agency_names,
                t.html_url = s.html_url,
                t.body_html_url = s.body_html_url,
                t.chunk_count = s.chunk_count,
                t.s3_key = s.s3_key,
                t.raw_json = s.raw_json,
                t.processing_status = s.processing_status
            WHEN NOT MATCHED THEN INSERT
                (document_number, title, publication_date, document_type, agency_names, html_url,
                 body_html_url, chunk_count, s3_key, raw_json, processing_status)
            VALUES
                (s.document_number, s.title, s.publication_date, s.document_type, s.agency_names, s.html_url,
                 s.body_html_url, s.chunk_count, s.s3_key, s.raw_json, s.processing_status)
            """
            cur.execute(merge_sql, params)
            loaded += len(batch)
        conn.commit()
    finally:
        cur.close()
        conn.close()
    return loaded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest EOP Federal Register documents")
    parser.add_argument("--test", action="store_true", help="Test mode: fetch 1 page from EOP")
    args = parser.parse_args()
    loaded = fetch_and_load_eop_incrementally(test_mode=args.test, cutoff_year=2021)
    print(f"Loaded {loaded} documents into EOP_DOCUMENTS")
