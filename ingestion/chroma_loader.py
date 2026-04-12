"""
ChromaDB loader for TariffIQ.

Pulls pre-chunked Federal Register chunks from Snowflake CHUNKS table,
embeds via all-MiniLM-L6-v2, and loads into ChromaDB collection 'federal_register'.

Called by rebuild_on_startup() in api/main.py on FastAPI boot.
"""

import logging
import os
from typing import Optional

import chromadb

from ingestion.connection import get_snowflake_conn
from ingestion.embedder import Embedder

logger = logging.getLogger(__name__)

COLLECTION_NAME = "federal_register"
EMBED_BATCH_SIZE = 64
UPSERT_BATCH_SIZE = 100


def get_chroma_client() -> chromadb.HttpClient:
    host = os.environ.get("CHROMADB_HOST", "chromadb")
    port = int(os.environ.get("CHROMADB_PORT", 8000))
    return chromadb.HttpClient(host=host, port=port)


def load_federal_register_to_chroma(limit: Optional[int] = None) -> int:
    """
    Pull chunks from TARIFFIQ.RAW.CHUNKS, embed, and upsert into ChromaDB.

    Args:
        limit: Optional row limit for testing (None = all rows)

    Returns:
        Total number of chunks loaded into ChromaDB
    """
    conn = get_snowflake_conn()
    cur = conn.cursor()

    try:
        query = """
            SELECT
                c.document_number,
                c.chunk_index,
                MAX(c.chunk_text) AS chunk_text,
                MAX(c.section) AS section,
                COALESCE(MAX(f.title), '') AS title,
                COALESCE(MAX(f.publication_date::VARCHAR), '') AS publication_date
            FROM TARIFFIQ.RAW.CHUNKS c
            LEFT JOIN TARIFFIQ.RAW.FEDERAL_REGISTER_NOTICES f
                ON c.document_number = f.document_number
            WHERE c.chunk_text IS NOT NULL
              AND LENGTH(c.chunk_text) > 20
            GROUP BY c.document_number, c.chunk_index
        """
        if limit:
            query += f" LIMIT {limit}"

        cur.execute(query)
        rows = cur.fetchall()

    finally:
        cur.close()
        conn.close()

    if not rows:
        logger.warning("chroma_loader: no chunks found in CHUNKS table")
        return 0

    logger.info("chroma_loader: fetched %d chunks from Snowflake", len(rows))

    embedder = Embedder()
    chroma = get_chroma_client()

    collection = chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total_loaded = 0

    # Process in batches
    for i in range(0, len(rows), UPSERT_BATCH_SIZE):
        batch = rows[i: i + UPSERT_BATCH_SIZE]

        ids = [f"{r[0]}__{r[1]}" for r in batch]
        texts = [r[2] for r in batch]
        metadatas = [
            {
                "document_number": r[0],
                "chunk_index": r[1],
                "section": r[3] or "",
                "title": r[4],
                "publication_date": r[5],
            }
            for r in batch
        ]

        embeddings = embedder.embed_batch(texts, batch_size=EMBED_BATCH_SIZE)

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        total_loaded += len(batch)
        logger.info(
            "chroma_loader: upserted batch %d-%d total=%d",
            i + 1, i + len(batch), total_loaded,
        )

    logger.info("chroma_loader: complete total_chunks=%d", total_loaded)
    return total_loaded