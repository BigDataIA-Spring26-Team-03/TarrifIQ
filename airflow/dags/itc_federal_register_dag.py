"""
ITC-specific Federal Register ingestion pipeline:
1. Fetch & store raw ITC docs -> S3 + Snowflake (DOWNLOADED)
2. Parse documents -> sections + hash (PARSED)
3. Extract HTS codes -> NOTICE_HTS_CODES_ITC (HTS_EXTRACTED)
4. Chunk semantically -> ITC_CHUNKS (CHUNKED)

processing_status flow:
  downloaded -> parsed -> hts_extracted -> chunked
                    -> hts_extraction_failed (on error)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

_DEFAULT_SQL_CAP = 1000
_DEFAULT_PARALLEL_SHARDS = 4


def _itc_pipeline_max_docs() -> int | None:
    raw = os.environ.get("ITC_PIPELINE_MAX_DOCS", "").strip()
    if raw == "" or raw == "0":
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    return n if n > 0 else None


def _itc_sql_row_limit(cap: int | None) -> int:
    if cap is None:
        return _DEFAULT_SQL_CAP
    return min(_DEFAULT_SQL_CAP, cap)


def _parallel_shard_count() -> int:
    raw = os.environ.get("ITC_PIPELINE_PARALLEL_SHARDS", str(_DEFAULT_PARALLEL_SHARDS)).strip()
    try:
        n = int(raw)
    except ValueError:
        return _DEFAULT_PARALLEL_SHARDS
    if n < 1:
        return 1
    return min(n, 16)


def _task_fetch_and_store_raw() -> None:
    from ingestion.itc_client import fetch_and_load_itc_incrementally

    cap = _itc_pipeline_max_docs()
    loaded = fetch_and_load_itc_incrementally(
        test_mode=False,
        cutoff_year=2018,
        batch_size=50,
        max_documents=cap,
    )
    logger.info("ITC fetch complete loaded=%d", loaded)


def _task_parse_documents(shard_id: int = 0, total_shards: int = 1) -> None:
    from ingestion.connection import get_snowflake_conn
    from ingestion.html_parser import parse_fr_document
    from ingestion.itc_client import S3_BUCKET
    from ingestion.federal_register_client import _get_s3_client, _parse_xml_to_text

    conn = get_snowflake_conn()
    cur = conn.cursor()
    s3_client = _get_s3_client()
    try:
        row_limit = _itc_sql_row_limit(_itc_pipeline_max_docs())
        cur.execute(
            f"""
            SELECT document_number, document_type, title, agency_names, publication_date, s3_key
            FROM ITC_DOCUMENTS
            WHERE (processing_status IS NULL OR processing_status = 'pending' OR processing_status = 'downloaded')
              AND YEAR(publication_date) >= 2018
              AND MOD(ABS(HASH(document_number)), %s) = %s
            ORDER BY publication_date DESC
            LIMIT {row_limit}
            """,
            (total_shards, shard_id),
        )
        rows = cur.fetchall()
        if not rows:
            return
        if not S3_BUCKET:
            logger.error("ITC parse failed: S3_BUCKET not set")
            return

        parsed_count = 0
        for document_number, document_type, title, agency_names_raw, publication_date, s3_key in rows:
            try:
                agency_names = json.loads(agency_names_raw) if isinstance(agency_names_raw, str) else (agency_names_raw or [])
                if not s3_key:
                    continue
                s3_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
                xml_content = s3_obj["Body"].read()
                full_text = _parse_xml_to_text(xml_content)
                parsed_doc = parse_fr_document(
                    document_number=document_number,
                    document_type=document_type,
                    title=title,
                    agency_names=agency_names,
                    publication_date=publication_date,
                    full_text=full_text,
                    source_s3_key=s3_key,
                )
                cur.execute(
                    """
                    UPDATE ITC_DOCUMENTS
                    SET processing_status = 'parsed', content_hash = %s, word_count = %s
                    WHERE document_number = %s
                    """,
                    (parsed_doc.content_hash, parsed_doc.word_count, document_number),
                )
                parsed_count += 1
            except Exception as exc:
                cur.execute(
                    """
                    UPDATE ITC_DOCUMENTS
                    SET processing_status = 'failed', processing_error = %s
                    WHERE document_number = %s
                    """,
                    (str(exc)[:500], document_number),
                )
        conn.commit()
        logger.info("ITC parse complete shard=%s/%s parsed=%d", shard_id, total_shards, parsed_count)
    finally:
        cur.close()
        conn.close()


def _task_extract_hts_codes(shard_id: int = 0, total_shards: int = 1) -> None:
    from ingestion.connection import get_snowflake_conn
    from ingestion.itc_client import S3_BUCKET
    from ingestion.federal_register_client import _get_s3_client, _parse_xml_to_text
    from ingestion.itc_hts_extractor import run_extraction_pipeline

    conn = get_snowflake_conn()
    cur = conn.cursor()
    s3_client = _get_s3_client()

    try:
        cur.execute(
            """
            SELECT document_number, title, agency_names, raw_json, s3_key
            FROM ITC_DOCUMENTS
            WHERE processing_status = 'parsed'
              AND MOD(ABS(HASH(document_number)), %s) = %s
            ORDER BY publication_date DESC
            """,
            (total_shards, shard_id),
        )
        docs = cur.fetchall()
        cap = _itc_pipeline_max_docs()
        if cap is not None and len(docs) > cap:
            docs = docs[:cap]

        extracted = 0
        failed = 0
        for document_number, title, agency_names_raw, raw_json, s3_key in docs:
            try:
                if not s3_key:
                    continue
                s3_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
                xml_content = s3_obj["Body"].read()
                full_text = _parse_xml_to_text(xml_content)
                if not full_text.strip():
                    continue

                agency = ""
                if isinstance(agency_names_raw, list) and agency_names_raw:
                    agency = str(agency_names_raw[0])
                elif isinstance(agency_names_raw, str):
                    try:
                        parsed = json.loads(agency_names_raw)
                        if isinstance(parsed, list) and parsed:
                            agency = str(parsed[0])
                        else:
                            agency = agency_names_raw
                    except Exception:
                        agency = agency_names_raw

                docket_number = None
                if isinstance(raw_json, dict):
                    docket_number = raw_json.get("docket_id") or raw_json.get("docket_number") or raw_json.get("regulation_id_number")

                run_extraction_pipeline(
                    document_number,
                    full_text,
                    conn,
                    title=str(title or ""),
                    agency=agency,
                    docket_number=str(docket_number) if docket_number else None,
                )
                cur.execute(
                    """
                    UPDATE ITC_DOCUMENTS
                    SET processing_status = 'hts_extracted'
                    WHERE document_number = %s
                    """,
                    (document_number,),
                )
                conn.commit()
                extracted += 1
            except Exception as exc:
                failed += 1
                cur.execute(
                    """
                    UPDATE ITC_DOCUMENTS
                    SET processing_status = 'hts_extraction_failed', processing_error = %s
                    WHERE document_number = %s
                    """,
                    (str(exc)[:500], document_number),
                )
                conn.commit()
        logger.info("ITC extract complete shard=%s/%s extracted=%d failed=%d", shard_id, total_shards, extracted, failed)
    finally:
        cur.close()
        conn.close()


def _task_chunk_documents(shard_id: int = 0, total_shards: int = 1) -> None:
    from ingestion.connection import get_snowflake_conn
    from ingestion.html_parser import ParsedFRDocument, extract_fr_sections
    from ingestion.chunker import SemanticFRChunker
    from ingestion.embedder import Embedder
    from ingestion.itc_client import S3_BUCKET
    from ingestion.federal_register_client import _get_s3_client, _parse_xml_to_text

    embedder = Embedder()
    chunker = SemanticFRChunker(embedder)
    s3_client = _get_s3_client()

    conn = get_snowflake_conn()
    cur = conn.cursor()
    try:
        row_limit = _itc_sql_row_limit(_itc_pipeline_max_docs())
        cur.execute(
            f"""
            SELECT document_number, document_type, title, agency_names, publication_date, content_hash, s3_key
            FROM ITC_DOCUMENTS
            WHERE processing_status IN ('parsed', 'hts_extracted', 'hts_extraction_failed')
              AND YEAR(publication_date) >= 2018
              AND MOD(ABS(HASH(document_number)), %s) = %s
            ORDER BY publication_date DESC
            LIMIT {row_limit}
            """,
            (total_shards, shard_id),
        )
        rows = cur.fetchall()
        if not rows:
            return
        if not S3_BUCKET:
            logger.error("ITC chunk failed: S3_BUCKET not set")
            return

        chunked = 0
        for document_number, document_type, title, agency_names_raw, publication_date, content_hash, s3_key in rows:
            try:
                agency_names = json.loads(agency_names_raw) if isinstance(agency_names_raw, str) else (agency_names_raw or [])
                if not s3_key:
                    continue
                s3_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
                xml_content = s3_obj["Body"].read()
                full_text = _parse_xml_to_text(xml_content)
                sections = extract_fr_sections(full_text, document_type)
                parsed_doc = ParsedFRDocument(
                    document_number=document_number,
                    document_type=document_type,
                    title=title,
                    agency_names=agency_names,
                    publication_date=publication_date,
                    full_text=full_text,
                    sections=sections,
                    content_hash=content_hash,
                    word_count=len(full_text.split()),
                )
                chunks = chunker.chunk_document(parsed_doc, {})
                for chunk in chunks:
                    chunk_id = f"{document_number}_{chunk['section']}_{chunk['chunk_index']}"
                    cur.execute(
                        """
                        INSERT INTO ITC_CHUNKS (chunk_id, document_number, chunk_index, chunk_text, section, word_count)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            chunk_id,
                            chunk["document_number"],
                            chunk["chunk_index"],
                            chunk["chunk_text"],
                            chunk["section"],
                            len(chunk["chunk_text"].split()),
                        ),
                    )
                cur.execute(
                    """
                    UPDATE ITC_DOCUMENTS
                    SET processing_status = 'chunked', chunk_count = %s
                    WHERE document_number = %s
                    """,
                    (len(chunks), document_number),
                )
                chunked += 1
            except Exception as exc:
                cur.execute(
                    """
                    UPDATE ITC_DOCUMENTS
                    SET processing_status = 'failed', processing_error = %s
                    WHERE document_number = %s
                    """,
                    (str(exc)[:500], document_number),
                )
        conn.commit()
        logger.info("ITC chunk complete shard=%s/%s chunked=%d", shard_id, total_shards, chunked)
    finally:
        cur.close()
        conn.close()


DEFAULT_ARGS = {
    "owner": "tariffiq",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="itc_federal_register_ingest",
    default_args=DEFAULT_ARGS,
    description="ITC Federal Register ingestion: fetch -> parse -> hts -> chunk",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["tariffiq", "ingestion", "itc"],
) as dag:
    shard_count = _parallel_shard_count()

    task_fetch = PythonOperator(
        task_id="itc_fetch_and_store_raw",
        python_callable=_task_fetch_and_store_raw,
    )

    for shard_id in range(shard_count):
        parse_t = PythonOperator(
            task_id=f"itc_parse_documents_shard_{shard_id}",
            python_callable=_task_parse_documents,
            op_kwargs={"shard_id": shard_id, "total_shards": shard_count},
        )
        extract_t = PythonOperator(
            task_id=f"itc_extract_hts_codes_shard_{shard_id}",
            python_callable=_task_extract_hts_codes,
            op_kwargs={"shard_id": shard_id, "total_shards": shard_count},
        )
        chunk_t = PythonOperator(
            task_id=f"itc_chunk_documents_shard_{shard_id}",
            python_callable=_task_chunk_documents,
            op_kwargs={"shard_id": shard_id, "total_shards": shard_count},
        )

        task_fetch >> parse_t >> extract_t >> chunk_t
