"""
Daily Federal Register ingestion pipeline:
1. Fetch & store raw → S3 + Snowflake (DOWNLOADED)
2. Parse documents → sections + hash (PARSED)
3. Extract HTS codes → NOTICE_HTS_CODES (HTS_EXTRACTED)
4. Chunk semantically → section-aware chunks (CHUNKED)

processing_status flow:
  downloaded → parsed → hts_extracted → chunked
                    ↘ hts_extraction_failed (on error)

Env FR_PIPELINE_MAX_DOCS: when set to a positive integer (e.g. 10), caps all stages for trial runs:
  fetch (new loads), parse, chunk, HTS extraction. Omit or set 0 for no limit.
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


def _fr_pipeline_max_docs() -> int | None:
    """Positive cap from FR_PIPELINE_MAX_DOCS, or None if unlimited."""
    raw = os.environ.get("FR_PIPELINE_MAX_DOCS", "").strip()
    if raw == "" or raw == "0":
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    return n if n > 0 else None


def _fr_sql_row_limit(cap: int | None) -> int:
    """Snowflake LIMIT for parse/chunk tasks (never above _DEFAULT_SQL_CAP)."""
    if cap is None:
        return _DEFAULT_SQL_CAP
    return min(_DEFAULT_SQL_CAP, cap)


def _parallel_shard_count() -> int:
    """
    Number of parallel shard lanes for parse/chunk/extract.
    Set FR_PIPELINE_PARALLEL_SHARDS in env; defaults to 4.
    """
    raw = os.environ.get("FR_PIPELINE_PARALLEL_SHARDS", str(_DEFAULT_PARALLEL_SHARDS)).strip()
    try:
        n = int(raw)
    except ValueError:
        return _DEFAULT_PARALLEL_SHARDS
    if n < 1:
        return 1
    return min(n, 16)


def _task_fetch_and_store_raw() -> None:
    """Task 1: Fetch documents from FR API incrementally, store raw XML in S3, load to Snowflake in batches."""
    from ingestion.federal_register_client import fetch_and_load_incrementally

    try:
        cap = _fr_pipeline_max_docs()
        logger.info(
            "Task 1: Starting incremental fetch from FR API (cutoff_year=2018, FR_PIPELINE_MAX_DOCS=%s)",
            cap or "unlimited",
        )
        loaded = fetch_and_load_incrementally(
            test_mode=False,
            cutoff_year=2018,
            batch_size=50,
            max_documents=cap,
        )
        logger.info(f"Task 1 complete: fetched and loaded {loaded} documents to Snowflake")
    except Exception as exc:
        logger.error(f"Task 1 FAILED: {str(exc)}", exc_info=True)
        raise


def _task_parse_documents(shard_id: int = 0, total_shards: int = 1) -> None:
    """
    Task 2: Load raw XML from S3, parse sections, compute content hash, update status.
    """
    from ingestion.connection import get_snowflake_conn
    from ingestion.html_parser import parse_fr_document
    from ingestion.federal_register_client import (
        S3_BUCKET,
        _get_s3_client,
        _parse_xml_to_text,
    )
    import json

    conn = get_snowflake_conn()
    cur = conn.cursor()

    try:
        row_limit = _fr_sql_row_limit(_fr_pipeline_max_docs())
        logger.info(
            "Task 2: parse row limit=%s shard=%s/%s",
            row_limit, shard_id, total_shards,
        )
        cur.execute(
            f"""
            SELECT document_number, document_type, title, agency_names, publication_date,
                   s3_key
            FROM FEDERAL_REGISTER_NOTICES
            WHERE (processing_status IS NULL OR processing_status = 'pending' OR processing_status = 'downloaded')
              AND YEAR(publication_date) >= 2018
              AND MOD(ABS(HASH(document_number)), %s) = %s
            ORDER BY publication_date DESC
            LIMIT {row_limit}
            """
            ,
            (total_shards, shard_id),
        )
        rows = cur.fetchall()

        if not rows:
            logger.info("Task 2: No documents to parse for shard=%s/%s", shard_id, total_shards)
            return

        if not S3_BUCKET:
            logger.error("Task 2: S3_BUCKET is not set — cannot fetch raw XML from S3")
            return

        s3_client = _get_s3_client()
        parsed_count = 0
        for row in rows:
            document_number, document_type, title, agency_names_json, publication_date, s3_key = row

            try:
                # Parse agency_names from JSON
                agency_names = json.loads(agency_names_json) if isinstance(agency_names_json, str) else agency_names_json or []

                # Fetch full_text from S3
                if not s3_key:
                    logger.warning(f"No S3 key for {document_number} - skipping")
                    continue

                try:
                    s3_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
                    xml_content = s3_obj["Body"].read()
                    full_text = _parse_xml_to_text(xml_content)
                except Exception as e:
                    logger.error(f"Failed to fetch from S3 for {document_number}: {e}")
                    continue

                # Parse document
                parsed_doc = parse_fr_document(
                    document_number=document_number,
                    document_type=document_type,
                    title=title,
                    agency_names=agency_names,
                    publication_date=publication_date,
                    full_text=full_text,
                    source_s3_key=s3_key,
                )

                # Update Snowflake with parsed metadata
                sections_json = json.dumps(parsed_doc.sections)
                cur.execute(
                    """
                    UPDATE FEDERAL_REGISTER_NOTICES
                    SET processing_status = 'parsed',
                        content_hash = %s,
                        word_count = %s
                    WHERE document_number = %s
                    """,
                    (parsed_doc.content_hash, parsed_doc.word_count, document_number),
                )
                parsed_count += 1

            except Exception as exc:
                logger.error(f"Parse failed for doc {document_number}: {exc}")
                cur.execute(
                    """
                    UPDATE FEDERAL_REGISTER_NOTICES
                    SET processing_status = 'failed', processing_error = %s
                    WHERE document_number = %s
                    """,
                    (str(exc), document_number),
                )

        conn.commit()
        logger.info("Task 2 complete: parsed %d documents shard=%s/%s", parsed_count, shard_id, total_shards)

    finally:
        cur.close()
        conn.close()


def _task_chunk_documents(shard_id: int = 0, total_shards: int = 1) -> None:
    """
    Task 3: Load HTS-extracted docs from S3, run semantic chunking, update chunk_count.
    """
    from ingestion.connection import get_snowflake_conn
    from ingestion.html_parser import ParsedFRDocument, extract_fr_sections
    from ingestion.chunker import SemanticFRChunker
    from ingestion.embedder import Embedder
    from ingestion.federal_register_client import (
        S3_BUCKET,
        _get_s3_client,
        _parse_xml_to_text,
    )
    import json

    embedder = Embedder()
    chunker = SemanticFRChunker(embedder)
    s3_client = _get_s3_client()

    conn = get_snowflake_conn()
    cur = conn.cursor()

    try:
        row_limit = _fr_sql_row_limit(_fr_pipeline_max_docs())
        logger.info(
            "Task 3: chunk row limit=%s shard=%s/%s",
            row_limit, shard_id, total_shards,
        )
        cur.execute(
            f"""
            SELECT document_number, document_type, title, agency_names, publication_date,
                   content_hash, s3_key
            FROM FEDERAL_REGISTER_NOTICES
            WHERE processing_status IN ('parsed', 'hts_extracted', 'hts_extraction_failed')
              AND YEAR(publication_date) >= 2018
              AND MOD(ABS(HASH(document_number)), %s) = %s
            ORDER BY publication_date DESC
            LIMIT {row_limit}
            """
            ,
            (total_shards, shard_id),
        )
        rows = cur.fetchall()

        if not rows:
            logger.info("Task 3: No documents to chunk for shard=%s/%s", shard_id, total_shards)
            return

        if not S3_BUCKET:
            logger.error("Task 3: S3_BUCKET is not set — cannot fetch raw XML from S3")
            return

        chunked_count = 0
        for row in rows:
            document_number, document_type, title, agency_names_json, publication_date, content_hash, s3_key = row

            try:
                # Parse agency names
                agency_names = json.loads(agency_names_json) if isinstance(agency_names_json, str) else agency_names_json or []

                # Fetch full_text from S3
                if not s3_key:
                    logger.warning(f"No S3 key for {document_number} - skipping")
                    continue

                try:
                    s3_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
                    xml_content = s3_obj["Body"].read()
                    full_text = _parse_xml_to_text(xml_content)
                except Exception as e:
                    logger.error(f"Failed to fetch from S3 for {document_number}: {e}")
                    continue

                # Reconstruct ParsedFRDocument (sections extracted on-the-fly from full_text)
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

                # Semantic chunk (without HTS annotation for now)
                chunks = chunker.chunk_document(parsed_doc, {})

                # Insert chunks into CHUNKS table
                for chunk in chunks:
                    # Layer 4: Fixed chunk ID format to include section, preventing duplicates
                    chunk_id = f"{document_number}_{chunk['section']}_{chunk['chunk_index']}"
                    cur.execute(
                        """
                        INSERT INTO CHUNKS (chunk_id, document_number, chunk_index, chunk_text, section, word_count)
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

                # Update Snowflake: chunk_count, status
                cur.execute(
                    """
                    UPDATE FEDERAL_REGISTER_NOTICES
                    SET processing_status = 'chunked', chunk_count = %s
                    WHERE document_number = %s
                    """,
                    (len(chunks), document_number),
                )

                chunked_count += 1

            except Exception as exc:
                logger.error(f"Chunk failed for doc {document_number}: {exc}")
                cur.execute(
                    """
                    UPDATE FEDERAL_REGISTER_NOTICES
                    SET processing_status = 'failed', processing_error = %s
                    WHERE document_number = %s
                    """,
                    (str(exc), document_number),
                )

        conn.commit()
        logger.info("Task 3 complete: chunked %d documents shard=%s/%s", chunked_count, shard_id, total_shards)

    finally:
        cur.close()
        conn.close()


def _task_extract_hts_codes(shard_id: int = 0, total_shards: int = 1) -> None:
    """
    Task 4: HTS extraction for parsed documents — NOTICE_HTS_CODES.
    """
    from ingestion.connection import get_snowflake_conn
    from ingestion.federal_register_client import S3_BUCKET, _get_s3_client, _parse_xml_to_text
    from ingestion.hts_extractor import run_extraction_pipeline

    conn = get_snowflake_conn()
    cur = conn.cursor()
    s3_client = _get_s3_client()

    try:
        cur.execute(
            """
            SELECT document_number, title, agency_names, raw_json, s3_key
            FROM FEDERAL_REGISTER_NOTICES
            WHERE processing_status = 'parsed'
              AND MOD(ABS(HASH(document_number)), %s) = %s
            ORDER BY publication_date DESC
            """
            ,
            (total_shards, shard_id),
        )
        docs = cur.fetchall()
        cap = _fr_pipeline_max_docs()
        total_eligible = len(docs)
        if cap is not None and len(docs) > cap:
            logger.info(
                "hts_extraction shard=%s/%s: limiting run to %d of %d eligible documents "
                "(unset FR_PIPELINE_MAX_DOCS or set 0 to process all)",
                shard_id, total_shards,
                cap,
                total_eligible,
            )
            docs = docs[:cap]
        else:
            logger.info(
                "hts_extraction shard=%s/%s: processing %d eligible documents (no cap)",
                shard_id, total_shards,
                len(docs),
            )

        total_extracted = 0
        total_verified = 0
        failed = 0

        for doc_number, title, agency_names_raw, raw_json, s3_key in docs:
            try:
                if not s3_key:
                    logger.warning("hts_extraction skip doc=%s reason=missing_s3_key", doc_number)
                    continue

                s3_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
                xml_content = s3_obj["Body"].read()
                full_text = _parse_xml_to_text(xml_content)
                if not full_text.strip():
                    logger.warning("hts_extraction skip doc=%s reason=empty_text", doc_number)
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
                    docket_number = (
                        raw_json.get("docket_id")
                        or raw_json.get("docket_number")
                        or raw_json.get("regulation_id_number")
                    )

                summary = run_extraction_pipeline(
                    doc_number,
                    full_text,
                    conn,
                    title=str(title or ""),
                    agency=agency,
                    docket_number=str(docket_number) if docket_number else None,
                )
                total_extracted += summary.get("total_extracted", 0)
                total_verified += summary.get("verified", 0)

                cur.execute(
                    """
                    UPDATE FEDERAL_REGISTER_NOTICES
                    SET processing_status = 'hts_extracted'
                    WHERE document_number = %s
                    """,
                    (doc_number,),
                )

                conn.commit()
                logger.info(
                    "hts_extracted doc=%s extracted=%d verified=%d",
                    doc_number,
                    summary.get("total_extracted", 0),
                    summary.get("verified", 0),
                )

            except Exception as e:
                failed += 1
                logger.error("hts_extraction_failed doc=%s error=%s", doc_number, e)
                cur.execute(
                    """
                    UPDATE FEDERAL_REGISTER_NOTICES
                    SET processing_status = 'hts_extraction_failed',
                        processing_error = %s
                    WHERE document_number = %s
                    """,
                    (str(e)[:500], doc_number),
                )
                conn.commit()
                continue

        logger.info(
            "hts_extraction complete shard=%s/%s total_extracted=%d total_verified=%d failed=%d",
            shard_id, total_shards,
            total_extracted,
            total_verified,
            failed,
        )

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
    dag_id="federal_register_ingest",
    default_args=DEFAULT_ARGS,
    description="Daily Federal Register ingestion: fetch → parse → hts → chunk",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["tariffiq", "ingestion"],
) as dag:
    shard_count = _parallel_shard_count()

    task_fetch = PythonOperator(
        task_id="fetch_and_store_raw",
        python_callable=_task_fetch_and_store_raw,
        doc="Fetch documents from FR API, store raw XML in S3, insert into Snowflake",
    )

    parse_tasks = []
    chunk_tasks = []
    extract_tasks = []

    for shard_id in range(shard_count):
        parse_t = PythonOperator(
            task_id=f"parse_documents_shard_{shard_id}",
            python_callable=_task_parse_documents,
            op_kwargs={"shard_id": shard_id, "total_shards": shard_count},
            doc="Parse documents into sections, compute content hash",
        )
        chunk_t = PythonOperator(
            task_id=f"chunk_documents_shard_{shard_id}",
            python_callable=_task_chunk_documents,
            op_kwargs={"shard_id": shard_id, "total_shards": shard_count},
            doc="Semantic chunking with section awareness",
        )
        extract_t = PythonOperator(
            task_id=f"extract_hts_codes_task_shard_{shard_id}",
            python_callable=_task_extract_hts_codes,
            op_kwargs={"shard_id": shard_id, "total_shards": shard_count},
            doc="Extract HTS codes to NOTICE_HTS_CODES",
        )
        parse_tasks.append(parse_t)
        chunk_tasks.append(chunk_t)
        extract_tasks.append(extract_t)

        # Per-shard pipeline: parse → extract → chunk
        task_fetch >> parse_t >> extract_t >> chunk_t
