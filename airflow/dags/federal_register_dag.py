"""
Daily Federal Register ingestion pipeline:
1. Fetch & store raw → S3 + Snowflake (DOWNLOADED)
2. Parse documents → sections + hash (PARSED)
3. Chunk semantically → section-aware chunks (CHUNKED)
4. Embed & index → ChromaDB policy_rag (INDEXED)
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
    Task 2: Load raw docs from Snowflake, parse sections, compute content hash, update status.
    """
    from ingestion.connection import get_snowflake_conn
    from ingestion.html_parser import parse_fr_document
    import json

    conn = get_snowflake_conn()
    cur = conn.cursor()

    try:
        # Fetch documents that are ready for parsing (PENDING or DOWNLOADED status)
        cur.execute(
            """
            SELECT document_number, document_type, title, agency_names, publication_date,
                   full_text, s3_key
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

        parsed_count = 0
        for row in rows:
            document_number, document_type, title, agency_names_json, publication_date, full_text, s3_key = row

            try:
                # Parse agency_names from JSON
                agency_names = json.loads(agency_names_json) if isinstance(agency_names_json, str) else agency_names_json or []

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
    Task 3: Load parsed docs, run semantic chunking, extract HTS entities, update chunk_count.
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
        # Fetch documents ready for chunking (PARSED status)
        cur.execute(
            """
            SELECT document_number, document_type, title, agency_names, publication_date,
                   full_text, content_hash
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
            document_number, document_type, title, agency_names_json, publication_date, full_text, content_hash = row

            try:
                # Parse agency names
                agency_names = json.loads(agency_names_json) if isinstance(agency_names_json, str) else agency_names_json or []

                # Reconstruct ParsedFRDocument (sections already extracted, stored in the DB)
                # For now, extract sections on-the-fly from full_text
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

                # Extract HTS entities from full text
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


def _task_embed_and_index() -> None:
    """
    Task 4: Load chunked docs, embed chunks, upsert into ChromaDB policy_rag.
    """
    from ingestion.connection import get_snowflake_conn
    from ingestion.embedder import Embedder
    from storage.chromadb_client import get_chromadb_client
    import json

    embedder = Embedder()
    chroma_client = get_chromadb_client()
    policy_rag = chroma_client.get_or_create_collection(name="policy_rag")

    conn = get_snowflake_conn()
    cur = conn.cursor()

    try:
        # Fetch documents ready for embedding (CHUNKED status)
        cur.execute(
            """
            SELECT document_number, full_text
            FROM FEDERAL_REGISTER_NOTICES
            WHERE processing_status = 'chunked'
            ORDER BY publication_date DESC
            LIMIT 1000
            """
        )
        rows = cur.fetchall()

        if not rows:
            logger.info("Task 4: No documents to embed")
            return

        from ingestion.html_parser import extract_fr_sections
        from ingestion.chunker import SemanticFRChunker

        chunker = SemanticFRChunker(embedder)

        indexed_count = 0
        for row in rows:
            document_number, full_text = row

            try:
                # Re-chunk to get chunk objects (in production, store chunks in DB for efficiency)
                sections = extract_fr_sections(full_text, "Notice")  # assume Notice for fallback
                parsed_doc_mini = type('ParsedFRDocument', (), {
                    'document_number': document_number,
                    'sections': sections,
                    'content_hash': '',
                    'word_count': len(full_text.split()),
                })()

                chunks = chunker.chunk_document(parsed_doc_mini, {})

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

                    # Update Snowflake: status = INDEXED
                    cur.execute(
                        """
                        UPDATE FEDERAL_REGISTER_NOTICES
                        SET processing_status = 'indexed'
                        WHERE document_number = %s
                        """,
                        (document_number,),
                    )
                    indexed_count += 1

            except Exception as exc:
                logger.error(f"Embed failed for doc {document_number}: {exc}")
                cur.execute(
                    """
                    UPDATE FEDERAL_REGISTER_NOTICES
                    SET processing_status = 'failed', processing_error = %s
                    WHERE document_number = %s
                    """,
                    (str(exc), document_number),
                )

        conn.commit()
        logger.info(f"Task 4 complete: indexed {indexed_count} documents")

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
    description="Daily Federal Register ingestion: fetch → parse → chunk → embed → index",
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

    task_embed = PythonOperator(
        task_id="embed_and_index",
        python_callable=_task_embed_and_index,
        doc="Embed chunks and upsert to ChromaDB policy_rag",
    )

    # Pipeline: fetch → parse → chunk → embed
    task_fetch >> task_parse >> task_chunk >> task_embed
