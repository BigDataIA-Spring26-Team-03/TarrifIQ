"""
Complete Federal Register pipeline test (Tasks 1-3 without Airflow).

Task 1: Fetch & store raw documents (already done above)
Task 2: Parse documents
Task 3: Chunk documents
"""

import json
import logging

from ingestion.connection import get_snowflake_conn
from ingestion.html_parser import ParsedFRDocument, extract_fr_sections, parse_fr_document
from ingestion.chunker import SemanticFRChunker
from ingestion.embedder import Embedder
from ingestion.federal_register_client import _parse_xml_to_text, _get_s3_client

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)


def task_2_parse_documents():
    """Task 2: Load raw docs, parse sections, compute content hash."""
    conn = get_snowflake_conn()
    cur = conn.cursor()

    try:
        # Fetch documents ready for parsing
        cur.execute("""
            SELECT document_number, document_type, title, agency_names, publication_date, s3_key
            FROM FEDERAL_REGISTER_NOTICES
            WHERE processing_status IS NULL OR processing_status = 'downloaded'
            ORDER BY publication_date DESC
            LIMIT 20
        """)
        rows = cur.fetchall()

        if not rows:
            logger.info("Task 2: No documents to parse")
            return 0

        s3_client = _get_s3_client()
        parsed_count = 0
        for row in rows:
            document_number, document_type, title, agency_names_json, publication_date, s3_key = row

            try:
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

                parsed_doc = parse_fr_document(
                    document_number=document_number,
                    document_type=document_type,
                    title=title,
                    agency_names=agency_names,
                    publication_date=publication_date,
                    full_text=full_text,
                    source_s3_key=s3_key,
                )

                # Update Snowflake
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
                logger.info(f"  {document_number}: parsed ({parsed_doc.word_count} words)")

            except Exception as exc:
                logger.error(f"  {document_number}: PARSE FAILED - {exc}")
                cur.execute(
                    "UPDATE FEDERAL_REGISTER_NOTICES SET processing_status = 'failed', processing_error = %s WHERE document_number = %s",
                    (str(exc), document_number),
                )

        conn.commit()
        logger.info(f"Task 2 complete: parsed {parsed_count} documents")
        return parsed_count

    finally:
        cur.close()
        conn.close()


def task_3_chunk_documents():
    """Task 3: Load parsed docs from S3, run semantic chunking, extract HTS entities."""
    conn = get_snowflake_conn()
    cur = conn.cursor()

    embedder = Embedder()
    chunker = SemanticFRChunker(embedder)
    s3_client = _get_s3_client()

    try:
        # Fetch documents ready for chunking
        cur.execute("""
            SELECT document_number, document_type, title, agency_names, publication_date, content_hash, s3_key
            FROM FEDERAL_REGISTER_NOTICES
            WHERE processing_status = 'parsed'
            ORDER BY publication_date DESC
            LIMIT 20
        """)
        rows = cur.fetchall()

        if not rows:
            logger.info("Task 3: No documents to chunk")
            return 0

        chunked_count = 0
        for row in rows:
            document_number, document_type, title, agency_names_json, publication_date, content_hash, s3_key = row

            try:
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

                # Update Snowflake
                cur.execute(
                    "UPDATE FEDERAL_REGISTER_NOTICES SET processing_status = 'chunked', chunk_count = %s WHERE document_number = %s",
                    (len(chunks), document_number),
                )
                chunked_count += 1
                logger.info(f"  {document_number}: chunked ({len(chunks)} chunks)")

            except Exception as exc:
                logger.error(f"  {document_number}: CHUNK FAILED - {exc}")
                cur.execute(
                    "UPDATE FEDERAL_REGISTER_NOTICES SET processing_status = 'failed', processing_error = %s WHERE document_number = %s",
                    (str(exc), document_number),
                )

        conn.commit()
        logger.info(f"Task 3 complete: chunked {chunked_count} documents")
        return chunked_count

    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Starting Full Pipeline Test (Tasks 2-3)")
    logger.info("=" * 60)

    # Reset any existing processing status for fresh test
    conn = get_snowflake_conn()
    cur = conn.cursor()
    cur.execute("UPDATE FEDERAL_REGISTER_NOTICES SET processing_status = NULL WHERE processing_status = 'parsed' OR processing_status = 'chunked'")
    conn.commit()
    cur.close()
    conn.close()
    logger.info("Reset processing status for fresh test")

    try:
        # Task 2: Parse
        parsed = task_2_parse_documents()
        logger.info(f"Task 2 result: {parsed} documents parsed")

        # Task 3: Chunk
        chunked = task_3_chunk_documents()
        logger.info(f"Task 3 result: {chunked} documents chunked")

        logger.info("=" * 60)
        if parsed > 0 and chunked > 0:
            logger.info("SUCCESS: All tasks completed successfully!")
            logger.info(f"  Parsed: {parsed} docs")
            logger.info(f"  Chunked: {chunked} docs")
            logger.info("Pipeline is ready for Airflow DAG deployment")
        else:
            logger.error(f"INCOMPLETE: Some tasks had no documents to process")
            logger.error(f"  Parsed: {parsed}, Chunked: {chunked}")

    except Exception as e:
        logger.error(f"Pipeline test FAILED: {e}", exc_info=True)
