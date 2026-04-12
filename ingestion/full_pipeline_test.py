"""
Complete Federal Register pipeline test (all 4 DAG tasks without Airflow).

Task 1: Fetch & store raw documents (already done above)
Task 2: Parse documents
Task 3: Chunk documents
Task 4: Embed and index to ChromaDB
"""

import json
import logging
from datetime import datetime

from storage.chromadb_client import get_chromadb_client
from ingestion.connection import get_snowflake_conn
from ingestion.html_parser import ParsedFRDocument, extract_fr_sections, parse_fr_document
from ingestion.chunker import SemanticFRChunker
from ingestion.embedder import Embedder
from ingestion.hts_extractor import extract_hts_entities

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
            SELECT document_number, document_type, title, agency_names, publication_date, full_text, s3_key
            FROM FEDERAL_REGISTER_NOTICES
            WHERE processing_status IS NULL OR processing_status = 'downloaded'
            ORDER BY publication_date DESC
            LIMIT 20
        """)
        rows = cur.fetchall()

        if not rows:
            logger.info("Task 2: No documents to parse")
            return 0

        parsed_count = 0
        for row in rows:
            document_number, document_type, title, agency_names_json, publication_date, full_text, s3_key = row

            try:
                agency_names = json.loads(agency_names_json) if isinstance(agency_names_json, str) else agency_names_json or []

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
    """Task 3: Load parsed docs, run semantic chunking, extract HTS entities."""
    conn = get_snowflake_conn()
    cur = conn.cursor()

    embedder = Embedder()
    chunker = SemanticFRChunker(embedder)

    try:
        # Fetch documents ready for chunking
        cur.execute("""
            SELECT document_number, document_type, title, agency_names, publication_date, full_text, content_hash
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
            document_number, document_type, title, agency_names_json, publication_date, full_text, content_hash = row

            try:
                agency_names = json.loads(agency_names_json) if isinstance(agency_names_json, str) else agency_names_json or []
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

                # Extract HTS entities
                hts_entities = extract_hts_entities(full_text)
                hts_annotations = {}
                for entity in hts_entities:
                    entity_text = full_text[entity["start_char"] : entity["end_char"]]
                    hts_annotations[entity_text] = {
                        "hts_code": entity["entity_text"] if entity["label"] == "HTS_CODE" else None,
                        "hts_chapter": entity["entity_text"][:2] if entity["label"] == "HTS_CODE" else (
                            entity["entity_text"].split()[-1] if entity["label"] == "HTS_CHAPTER" else None
                        ),
                    }

                # Semantic chunk
                chunks = chunker.chunk_document(parsed_doc, hts_annotations)

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


def task_4_embed_and_index():
    """Task 4: Load chunked docs, embed chunks, upsert to ChromaDB."""
    conn = get_snowflake_conn()
    cur = conn.cursor()

    embedder = Embedder()
    chroma_client = get_chromadb_client()
    policy_rag = chroma_client.get_or_create_collection(name="policy_rag")

    try:
        # Fetch documents ready for embedding
        cur.execute("""
            SELECT document_number, full_text, document_type
            FROM FEDERAL_REGISTER_NOTICES
            WHERE processing_status = 'chunked'
            ORDER BY publication_date DESC
            LIMIT 20
        """)
        rows = cur.fetchall()

        if not rows:
            logger.info("Task 4: No documents to embed")
            return 0

        chunker = SemanticFRChunker(embedder)
        indexed_count = 0

        for row in rows:
            document_number, full_text, document_type = row

            try:
                # Re-chunk to get chunk objects
                sections = extract_fr_sections(full_text, document_type)
                parsed_doc = ParsedFRDocument(
                    document_number=document_number,
                    document_type=document_type,
                    title="",
                    agency_names=[],
                    publication_date="",
                    full_text=full_text,
                    sections=sections,
                    content_hash="",
                    word_count=len(full_text.split()),
                )

                chunks = chunker.chunk_document(parsed_doc, {})

                if chunks:
                    # Embed chunk texts
                    chunk_texts = [c["chunk_text"] for c in chunks]
                    embeddings = embedder.embed_batch(chunk_texts)

                    # Prepare for ChromaDB upsert
                    ids = [f"{document_number}_{i}" for i in range(len(chunks))]
                    metadatas = [
                        {
                            "document_number": c["document_number"],
                            "section": c["section"],
                            "chunk_index": str(c["chunk_index"]),
                            "hts_code": c.get("hts_code") or "",
                            "hts_chapter": c.get("hts_chapter") or "",
                        }
                        for c in chunks
                    ]

                    # Upsert to ChromaDB
                    policy_rag.upsert(
                        ids=ids,
                        embeddings=embeddings,
                        documents=chunk_texts,
                        metadatas=metadatas,
                    )

                    # Update Snowflake
                    cur.execute(
                        "UPDATE FEDERAL_REGISTER_NOTICES SET processing_status = 'indexed' WHERE document_number = %s",
                        (document_number,),
                    )
                    indexed_count += 1
                    logger.info(f"  {document_number}: indexed ({len(chunks)} chunks)")

            except Exception as exc:
                logger.error(f"  {document_number}: EMBED FAILED - {exc}")
                cur.execute(
                    "UPDATE FEDERAL_REGISTER_NOTICES SET processing_status = 'failed', processing_error = %s WHERE document_number = %s",
                    (str(exc), document_number),
                )

        conn.commit()
        logger.info(f"Task 4 complete: indexed {indexed_count} documents")

        # Verify ChromaDB has data
        collection_count = policy_rag.count()
        logger.info(f"ChromaDB policy_rag collection now has {collection_count} chunks")

        return indexed_count

    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Starting Full Pipeline Test (Tasks 2-4)")
    logger.info("=" * 60)

    # Reset any existing processing status for fresh test
    conn = get_snowflake_conn()
    cur = conn.cursor()
    cur.execute("UPDATE FEDERAL_REGISTER_NOTICES SET processing_status = NULL WHERE processing_status != 'indexed'")
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

        # Task 4: Embed & Index
        indexed = task_4_embed_and_index()
        logger.info(f"Task 4 result: {indexed} documents indexed")

        logger.info("=" * 60)
        if parsed > 0 and chunked > 0 and indexed > 0:
            logger.info("SUCCESS: All tasks completed successfully!")
            logger.info(f"  Parsed: {parsed} docs")
            logger.info(f"  Chunked: {chunked} docs")
            logger.info(f"  Indexed: {indexed} docs")
            logger.info("Pipeline is ready for Airflow DAG deployment")
        else:
            logger.error(f"INCOMPLETE: Some tasks had no documents to process")
            logger.error(f"  Parsed: {parsed}, Chunked: {chunked}, Indexed: {indexed}")

    except Exception as e:
        logger.error(f"Pipeline test FAILED: {e}", exc_info=True)
