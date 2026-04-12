"""
Daily Federal Register ingestion pipeline:
1. Fetch & store raw → S3 + Snowflake (DOWNLOADED)
2. Parse documents → sections + hash (PARSED)
3. Chunk semantically → section-aware chunks (CHUNKED)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)


def _task_fetch_and_store_raw() -> None:
    """Task 1: Fetch documents from FR API, store raw XML in S3, insert into Snowflake."""
    from ingestion.federal_register_client import fetch_federal_register_docs, load_to_snowflake

    docs = fetch_federal_register_docs(test_mode=False)
    if docs:
        loaded = load_to_snowflake(docs)
        logger.info(f"Task 1 complete: fetched {len(docs)} docs, loaded {loaded} to Snowflake")
    else:
        logger.info("Task 1 complete: no new documents")


def _task_parse_documents() -> None:
    """
    Task 2: Load raw XML from S3, parse sections, compute content hash, update status.
    """
    from ingestion.connection import get_snowflake_conn
    from ingestion.html_parser import parse_fr_document
    from ingestion.federal_register_client import _parse_xml_to_text, _get_s3_client
    import json

    conn = get_snowflake_conn()
    cur = conn.cursor()

    try:
        # Fetch documents that are ready for parsing (PENDING or DOWNLOADED status)
        cur.execute(
            """
            SELECT document_number, document_type, title, agency_names, publication_date,
                   s3_key
            FROM FEDERAL_REGISTER_NOTICES
            WHERE processing_status IS NULL OR processing_status = 'pending' OR processing_status = 'downloaded'
            ORDER BY publication_date DESC
            LIMIT 1000
            """
        )
        rows = cur.fetchall()

        if not rows:
            logger.info("Task 2: No documents to parse")
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
                    s3_obj = s3_client.get_object(Bucket="tariffiq-raw-docs", Key=s3_key)
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
        logger.info(f"Task 2 complete: parsed {parsed_count} documents")

    finally:
        cur.close()
        conn.close()


def _task_chunk_documents() -> None:
    """
    Task 3: Load parsed docs from S3, run semantic chunking, update chunk_count.
    """
    from ingestion.connection import get_snowflake_conn
    from ingestion.html_parser import ParsedFRDocument, extract_fr_sections
    from ingestion.chunker import SemanticFRChunker
    from ingestion.embedder import Embedder
    from ingestion.federal_register_client import _parse_xml_to_text, _get_s3_client
    import json

    embedder = Embedder()
    chunker = SemanticFRChunker(embedder)
    s3_client = _get_s3_client()

    conn = get_snowflake_conn()
    cur = conn.cursor()

    try:
        # Fetch documents ready for chunking (PARSED status)
        cur.execute(
            """
            SELECT document_number, document_type, title, agency_names, publication_date,
                   content_hash, s3_key
            FROM FEDERAL_REGISTER_NOTICES
            WHERE processing_status = 'parsed'
            ORDER BY publication_date DESC
            LIMIT 1000
            """
        )
        rows = cur.fetchall()

        if not rows:
            logger.info("Task 3: No documents to chunk")
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
                    s3_obj = s3_client.get_object(Bucket="tariffiq-raw-docs", Key=s3_key)
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
                    chunk_id = f"{document_number}_{chunk['chunk_index']}"
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
        logger.info(f"Task 3 complete: chunked {chunked_count} documents")

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
    description="Daily Federal Register ingestion: fetch → parse → chunk",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["tariffiq", "ingestion"],
) as dag:

    task_fetch = PythonOperator(
        task_id="fetch_and_store_raw",
        python_callable=_task_fetch_and_store_raw,
        doc="Fetch documents from FR API, store raw XML in S3, insert into Snowflake",
    )

    task_parse = PythonOperator(
        task_id="parse_documents",
        python_callable=_task_parse_documents,
        doc="Parse documents into sections, compute content hash",
    )

    task_chunk = PythonOperator(
        task_id="chunk_documents",
        python_callable=_task_chunk_documents,
        doc="Semantic chunking with section awareness",
    )

    # Pipeline: fetch → parse → chunk
    task_fetch >> task_parse >> task_chunk
