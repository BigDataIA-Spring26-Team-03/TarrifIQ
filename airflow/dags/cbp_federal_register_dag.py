"""
CBP-specific Federal Register ingestion pipeline:
1. Fetch & store raw CBP docs → S3 + Snowflake (DOWNLOADED)
2. Parse documents → sections + hash (PARSED)
3. Extract HTS codes at document level → CBP_NOTICE_HTS_CODES (HTS_EXTRACTED)
4. Chunk semantically + extract HTS codes per chunk with rates → CBP_CHUNKS (CHUNKED)

processing_status flow:
  downloaded → parsed → hts_extracted → chunked
                              ↘ failed (on error)

Env CBP_PIPELINE_MAX_DOCS: when set to a positive integer (e.g. 10), caps all stages for trial runs.
Omit or set 0 for no limit.
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


def _cbp_pipeline_max_docs() -> int | None:
    """Positive cap from CBP_PIPELINE_MAX_DOCS, or None if unlimited."""
    raw = os.environ.get("CBP_PIPELINE_MAX_DOCS", "").strip()
    if raw == "" or raw == "0":
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    return n if n > 0 else None


def _cbp_sql_row_limit(cap: int | None) -> int:
    """Snowflake LIMIT for parse/chunk tasks (never above _DEFAULT_SQL_CAP)."""
    if cap is None:
        return _DEFAULT_SQL_CAP
    return min(_DEFAULT_SQL_CAP, cap)


def _parallel_shard_count() -> int:
    """
    Number of parallel shard lanes for parse/extract/chunk.
    Set CBP_PIPELINE_PARALLEL_SHARDS in env; defaults to 4.
    """
    raw = os.environ.get("CBP_PIPELINE_PARALLEL_SHARDS", str(_DEFAULT_PARALLEL_SHARDS)).strip()
    try:
        n = int(raw)
    except ValueError:
        return _DEFAULT_PARALLEL_SHARDS
    if n < 1:
        return 1
    return min(n, 16)


def _task_fetch_and_store_raw() -> None:
    """Task 1: Fetch CBP documents from FR API incrementally, store raw XML in S3, load to Snowflake in batches."""
    from ingestion.cbp_client import fetch_and_load_cbp_incrementally

    try:
        cap = _cbp_pipeline_max_docs()
        logger.info(
            "Task 1 (CBP): Starting incremental fetch from FR API (cutoff_year=2018, CBP_PIPELINE_MAX_DOCS=%s)",
            cap or "unlimited",
        )
        loaded = fetch_and_load_cbp_incrementally(
            test_mode=False,
            cutoff_year=2018,
            batch_size=50,
            max_documents=cap,
        )
        logger.info(f"Task 1 (CBP) complete: fetched and loaded {loaded} documents to Snowflake")
    except Exception as exc:
        logger.error(f"Task 1 (CBP) FAILED: {str(exc)}", exc_info=True)
        raise


def _task_parse_documents(shard_id: int = 0, total_shards: int = 1) -> None:
    """
    Task 2 (CBP): Parse documents using stored full_text, compute content hash, update status.
    """
    from ingestion.connection import get_snowflake_conn
    from ingestion.html_parser import parse_fr_document
    import json

    conn = get_snowflake_conn()
    cur = conn.cursor()

    try:
        row_limit = _cbp_sql_row_limit(_cbp_pipeline_max_docs())
        logger.info(
            "Task 2 (CBP): parse row limit=%s shard=%s/%s",
            row_limit, shard_id, total_shards,
        )
        cur.execute(
            f"""
            SELECT document_number, document_type, title, agency_names, publication_date,
                   s3_key, full_text
            FROM CBP_FEDERAL_REGISTER_NOTICES
            WHERE (processing_status IS NULL OR processing_status = 'pending' OR processing_status = 'downloaded')
              AND MOD(ABS(HASH(document_number)), %s) = %s
            ORDER BY publication_date DESC
            LIMIT {row_limit}
            """
            ,
            (total_shards, shard_id),
        )
        rows = cur.fetchall()

        if not rows:
            logger.info("Task 2 (CBP): No documents to parse for shard=%s/%s", shard_id, total_shards)
            return

        parsed_count = 0
        for row in rows:
            document_number, document_type, title, agency_names_json, publication_date, s3_key, full_text = row

            try:
                # Parse agency_names from JSON
                agency_names = json.loads(agency_names_json) if isinstance(agency_names_json, str) else agency_names_json or []

                # Validate full_text is stored
                if not full_text or not full_text.strip():
                    logger.warning(f"No full_text stored for {document_number} - skipping")
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
                cur.execute(
                    """
                    UPDATE CBP_FEDERAL_REGISTER_NOTICES
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
                    UPDATE CBP_FEDERAL_REGISTER_NOTICES
                    SET processing_status = 'failed', processing_error = %s
                    WHERE document_number = %s
                    """,
                    (str(exc), document_number),
                )

        conn.commit()
        logger.info("Task 2 (CBP) complete: parsed %d documents shard=%s/%s", parsed_count, shard_id, total_shards)

    finally:
        cur.close()
        conn.close()


def _task_extract_document_hts(shard_id: int = 0, total_shards: int = 1) -> None:
    """
    Task 3 (CBP): Extract HTS codes at document level from parsed documents.
    - Extracts HTS codes from stored full_text
    - Looks up general_rate from HTS_CODES table
    - MERGEs into CBP_NOTICE_HTS_CODES
    - Updates document status to HTS_EXTRACTED
    """
    from ingestion.connection import get_snowflake_conn
    from ingestion.hts_extractor import extract_hts_entities, _to_code_records

    conn = get_snowflake_conn()
    cur = conn.cursor()

    try:
        row_limit = _cbp_sql_row_limit(_cbp_pipeline_max_docs())
        logger.info(
            "Task 3 (CBP): extract HTS row limit=%s shard=%s/%s",
            row_limit, shard_id, total_shards,
        )
        cur.execute(
            f"""
            SELECT document_number, full_text
            FROM CBP_FEDERAL_REGISTER_NOTICES
            WHERE processing_status = 'parsed'
              AND MOD(ABS(HASH(document_number)), %s) = %s
            ORDER BY publication_date DESC
            LIMIT {row_limit}
            """
            ,
            (total_shards, shard_id),
        )
        rows = cur.fetchall()

        if not rows:
            logger.info("Task 3 (CBP): No documents to extract HTS from for shard=%s/%s", shard_id, total_shards)
            return

        extracted_count = 0
        failed = 0

        for row in rows:
            document_number, full_text = row

            try:
                # Validate full_text is stored
                if not full_text or not full_text.strip():
                    logger.warning(f"No full_text stored for {document_number} - skipping")
                    continue

                # Extract HTS codes with context snippets
                entities = extract_hts_entities(full_text)
                code_records = _to_code_records(document_number, full_text, entities)

                # MERGE into CBP_NOTICE_HTS_CODES (document-level index)
                for rec in code_records:
                    hts_code = rec["hts_code"]
                    hts_chapter = rec["hts_chapter"]
                    context_snippet = rec["context_snippet"]

                    # Lookup rate from HTS_CODES
                    cur.execute(
                        """
                        SELECT general_rate FROM HTS_CODES WHERE hts_code = %s LIMIT 1
                        """,
                        (hts_code,),
                    )
                    rate_row = cur.fetchone()
                    general_rate = rate_row[0] if rate_row else None

                    # MERGE with context_snippet
                    cur.execute(
                        """
                        MERGE INTO CBP_NOTICE_HTS_CODES AS t
                        USING (
                            SELECT %s AS document_number, %s AS hts_code, %s AS hts_chapter,
                                   %s AS context_snippet, %s AS general_rate
                        ) AS s
                        ON t.document_number = s.document_number AND t.hts_code = s.hts_code
                        WHEN MATCHED THEN UPDATE SET
                            t.hts_chapter = s.hts_chapter,
                            t.context_snippet = s.context_snippet,
                            t.general_rate = s.general_rate,
                            t.match_status = 'VERIFIED'
                        WHEN NOT MATCHED THEN INSERT
                            (document_number, hts_code, hts_chapter, context_snippet, general_rate, match_status)
                        VALUES
                            (s.document_number, s.hts_code, s.hts_chapter, s.context_snippet, s.general_rate, 'VERIFIED')
                        """,
                        (document_number, hts_code, hts_chapter, context_snippet, general_rate),
                    )

                # Update document status
                cur.execute(
                    """
                    UPDATE CBP_FEDERAL_REGISTER_NOTICES
                    SET processing_status = 'hts_extracted'
                    WHERE document_number = %s
                    """,
                    (document_number,),
                )
                conn.commit()
                extracted_count += 1
                logger.info("Task 3 (CBP) extracted doc=%s hts_codes=%d", document_number, len(code_records))

            except Exception as e:
                failed += 1
                logger.error("Task 3 (CBP) extraction_failed doc=%s error=%s", document_number, e)
                cur.execute(
                    """
                    UPDATE CBP_FEDERAL_REGISTER_NOTICES
                    SET processing_status = 'failed',
                        processing_error = %s
                    WHERE document_number = %s
                    """,
                    (str(e)[:500], document_number),
                )
                conn.commit()
                continue

        logger.info(
            "Task 3 (CBP) complete shard=%s/%s extracted=%d failed=%d",
            shard_id, total_shards,
            extracted_count,
            failed,
        )

    finally:
        cur.close()
        conn.close()


def _task_chunk_and_extract_rates(shard_id: int = 0, total_shards: int = 1) -> None:
    """
    Task 4 (CBP): Chunk documents semantically + extract HTS codes per chunk with rate lookup.
    - Chunks document text from stored full_text
    - Extracts HTS codes from each chunk
    - Looks up general_rate from HTS_CODES
    - Inserts into CBP_CHUNKS with hts_code, hts_chapter, general_rate
    - Updates document status to CHUNKED
    """
    from ingestion.connection import get_snowflake_conn
    from ingestion.html_parser import ParsedFRDocument, extract_fr_sections
    from ingestion.chunker import SemanticFRChunker
    from ingestion.embedder import Embedder
    from ingestion.hts_extractor import extract_hts_entities
    import json

    embedder = Embedder()
    chunker = SemanticFRChunker(embedder)

    conn = get_snowflake_conn()
    cur = conn.cursor()

    try:
        row_limit = _cbp_sql_row_limit(_cbp_pipeline_max_docs())
        logger.info(
            "Task 4 (CBP): chunk row limit=%s shard=%s/%s",
            row_limit, shard_id, total_shards,
        )
        cur.execute(
            f"""
            SELECT document_number, document_type, title, agency_names, publication_date,
                   content_hash, full_text
            FROM CBP_FEDERAL_REGISTER_NOTICES
            WHERE processing_status = 'hts_extracted'
              AND MOD(ABS(HASH(document_number)), %s) = %s
            ORDER BY publication_date DESC
            LIMIT {row_limit}
            """
            ,
            (total_shards, shard_id),
        )
        rows = cur.fetchall()

        if not rows:
            logger.info("Task 4 (CBP): No documents to chunk for shard=%s/%s", shard_id, total_shards)
            return

        chunked_count = 0
        for row in rows:
            document_number, document_type, title, agency_names_json, publication_date, content_hash, full_text = row

            try:
                # Parse agency names
                agency_names = json.loads(agency_names_json) if isinstance(agency_names_json, str) else agency_names_json or []

                # Validate full_text is stored
                if not full_text or not full_text.strip():
                    logger.warning(f"No full_text stored for {document_number} - skipping")
                    continue

                # Reconstruct ParsedFRDocument
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

                # Semantic chunk
                chunks = chunker.chunk_document(parsed_doc, {})

                # Insert chunks into CBP_CHUNKS table with HTS code and rate enrichment
                for chunk in chunks:
                    chunk_id = f"{document_number}_{chunk['section']}_{chunk['chunk_index']}"

                    # Extract HTS codes from this chunk
                    chunk_entities = extract_hts_entities(chunk["chunk_text"])
                    chunk_hts_codes = [e["entity_text"] for e in chunk_entities if e["label"] == "HTS_CODE"]

                    # Take first HTS code (or None if no codes in chunk)
                    chunk_hts_code = chunk_hts_codes[0] if chunk_hts_codes else None
                    chunk_hts_chapter = chunk_hts_code[:2] if chunk_hts_code and len(chunk_hts_code) >= 2 else None

                    # Lookup rate for this chunk
                    chunk_general_rate = None
                    if chunk_hts_code:
                        cur.execute(
                            """
                            SELECT general_rate FROM HTS_CODES WHERE hts_code = %s LIMIT 1
                            """,
                            (chunk_hts_code,),
                        )
                        chunk_rate_row = cur.fetchone()
                        chunk_general_rate = chunk_rate_row[0] if chunk_rate_row else None

                    # Insert chunk with HTS annotation
                    cur.execute(
                        """
                        INSERT INTO CBP_CHUNKS (chunk_id, document_number, chunk_index, chunk_text, section,
                                               hts_code, hts_chapter, general_rate, word_count)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            chunk_id,
                            document_number,
                            chunk["chunk_index"],
                            chunk["chunk_text"],
                            chunk["section"],
                            chunk_hts_code,
                            chunk_hts_chapter,
                            chunk_general_rate,
                            len(chunk["chunk_text"].split()),
                        ),
                    )

                # Update document status
                cur.execute(
                    """
                    UPDATE CBP_FEDERAL_REGISTER_NOTICES
                    SET processing_status = 'chunked', chunk_count = %s
                    WHERE document_number = %s
                    """,
                    (len(chunks), document_number),
                )
                conn.commit()

                chunked_count += 1
                logger.info("Task 4 (CBP) chunked doc=%s chunks=%d", document_number, len(chunks))

            except Exception as exc:
                logger.error(f"Chunk failed for doc {document_number}: {exc}")
                cur.execute(
                    """
                    UPDATE CBP_FEDERAL_REGISTER_NOTICES
                    SET processing_status = 'failed', processing_error = %s
                    WHERE document_number = %s
                    """,
                    (str(exc), document_number),
                )
                conn.commit()

        logger.info("Task 4 (CBP) complete: chunked %d documents shard=%s/%s", chunked_count, shard_id, total_shards)

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
    dag_id="cbp_federal_register_ingest",
    default_args=DEFAULT_ARGS,
    description="CBP-specific FR ingestion: fetch → parse → extract HTS (doc-level) → chunk+extract HTS (chunk-level with rates)",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["tariffiq", "ingestion", "cbp"],
) as dag:
    shard_count = _parallel_shard_count()

    task_fetch = PythonOperator(
        task_id="cbp_fetch_and_store_raw",
        python_callable=_task_fetch_and_store_raw,
        doc="Fetch CBP documents from FR API, store raw XML in S3, insert into Snowflake",
    )

    parse_tasks = []
    extract_hts_tasks = []
    chunk_tasks = []

    for shard_id in range(shard_count):
        parse_t = PythonOperator(
            task_id=f"cbp_parse_documents_shard_{shard_id}",
            python_callable=_task_parse_documents,
            op_kwargs={"shard_id": shard_id, "total_shards": shard_count},
            doc="Parse CBP documents into sections, compute content hash",
        )
        extract_hts_t = PythonOperator(
            task_id=f"cbp_extract_document_hts_shard_{shard_id}",
            python_callable=_task_extract_document_hts,
            op_kwargs={"shard_id": shard_id, "total_shards": shard_count},
            doc="Extract document-level HTS codes, populate CBP_NOTICE_HTS_CODES",
        )
        chunk_t = PythonOperator(
            task_id=f"cbp_chunk_with_rates_shard_{shard_id}",
            python_callable=_task_chunk_and_extract_rates,
            op_kwargs={"shard_id": shard_id, "total_shards": shard_count},
            doc="Chunk documents, extract chunk-level HTS codes with rates, populate CBP_CHUNKS",
        )
        parse_tasks.append(parse_t)
        extract_hts_tasks.append(extract_hts_t)
        chunk_tasks.append(chunk_t)

        # Per-shard pipeline: parse → extract HTS (doc-level) → chunk+extract HTS (chunk-level)
        task_fetch >> parse_t >> extract_hts_t >> chunk_t
