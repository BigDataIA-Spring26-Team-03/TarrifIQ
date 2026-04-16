"""
ChromaDB initialization script for TariffIQ.

Builds two collections:
1. policy_notices - from CHUNKS, CBP_CHUNKS, ITC_CHUNKS Snowflake tables
2. hts_descriptions - from HTS_CODES table

Skips if collections already exist and are populated (>1000 docs).
Persists to ./chroma_data directory.
"""

import os
import time
from typing import Optional, List, Dict
import logging

import snowflake.connector
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global client instance
_chroma_client = None
_embedder = None


def get_snowflake_conn():
    """Create a Snowflake connection using environment variables."""
    return snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
    )


def get_chroma_client():
    """Returns persistent ChromaDB client (singleton)."""
    global _chroma_client
    if _chroma_client is None:
        persist_dir = os.path.join(os.path.dirname(__file__), "..", "chroma_data")
        os.makedirs(persist_dir, exist_ok=True)
        _chroma_client = PersistentClient(path=persist_dir)
    return _chroma_client


def get_embedder():
    """Returns sentence transformer embedder (singleton)."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def delete_collection_if_exists(chroma: PersistentClient, collection_name: str) -> None:
    """Delete collection if it exists."""
    try:
        chroma.delete_collection(name=collection_name)
        logger.info(f"Deleted existing {collection_name} collection")
    except Exception:
        pass  # Collection doesn't exist, which is fine


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts using sentence-transformers."""
    embedder = get_embedder()
    return embedder.encode(texts).tolist()


def build_policy_notices_collection(chroma: PersistentClient):
    """Build policy_notices collection from CHUNKS, CBP_CHUNKS, ITC_CHUNKS."""
    logger.info("Building policy_notices collection...")

    conn = get_snowflake_conn()
    cur = conn.cursor()

    try:
        collection = chroma.get_or_create_collection(
            name="policy_notices",
            metadata={"hnsw:space": "cosine"}
        )

        documents = []
        metadatas = []
        ids = []

        # Load USTR chunks
        logger.info("Loading USTR chunks from CHUNKS table...")
        cur.execute(
            """
            SELECT chunk_id, document_number, chunk_index, chunk_text,
                   section, hts_code, hts_chapter
            FROM CHUNKS
            WHERE chunk_text IS NOT NULL
            ORDER BY chunk_id
            """
        )
        ustr_rows = cur.fetchall()

        for i, row in enumerate(ustr_rows):
            chunk_id, doc_num, idx, text, section, hts_code, hts_chapter = row
            documents.append(text)
            ids.append(f"USTR_{i}")  # Enumerate to guarantee uniqueness
            metadatas.append({
                "chunk_id": chunk_id,
                "document_number": doc_num or "",
                "hts_chapter": hts_chapter or "",
                "hts_code": hts_code or "",
                "source": "USTR",
                "section": section or ""
            })

        logger.info(f"Loaded {len(ustr_rows)} USTR chunks")

        # Load CBP chunks
        logger.info("Loading CBP chunks from CBP_CHUNKS table...")
        cur.execute(
            """
            SELECT chunk_id, document_number, chunk_index, chunk_text,
                   section, hts_code, hts_chapter, general_rate
            FROM CBP_CHUNKS
            WHERE chunk_text IS NOT NULL
            ORDER BY chunk_id
            """
        )
        cbp_rows = cur.fetchall()

        for i, row in enumerate(cbp_rows):
            chunk_id, doc_num, idx, text, section, hts_code, hts_chapter, gen_rate = row
            documents.append(text)
            ids.append(f"CBP_{i}")  # Enumerate to guarantee uniqueness
            metadatas.append({
                "chunk_id": chunk_id,
                "document_number": doc_num or "",
                "hts_chapter": hts_chapter or "",
                "hts_code": hts_code or "",
                "source": "CBP",
                "section": section or "",
                "general_rate": gen_rate or ""
            })

        logger.info(f"Loaded {len(cbp_rows)} CBP chunks")

        # Load ITC chunks
        logger.info("Loading ITC chunks from ITC_CHUNKS table...")
        cur.execute(
            """
            SELECT chunk_id, document_number, chunk_index, chunk_text,
                   section, hts_code, hts_chapter
            FROM ITC_CHUNKS
            WHERE chunk_text IS NOT NULL
            ORDER BY chunk_id
            """
        )
        itc_rows = cur.fetchall()

        for i, row in enumerate(itc_rows):
            chunk_id, doc_num, idx, text, section, hts_code, hts_chapter = row
            documents.append(text)
            ids.append(f"USITC_{i}")  # Enumerate to guarantee uniqueness
            metadatas.append({
                "chunk_id": chunk_id,
                "document_number": doc_num or "",
                "hts_chapter": hts_chapter or "",
                "hts_code": hts_code or "",
                "source": "USITC",
                "section": section or ""
            })

        logger.info(f"Loaded {len(itc_rows)} ITC chunks")

        # Embed and upsert in batches
        batch_size = 64
        total = len(documents)

        for i in range(0, total, batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            batch_metas = metadatas[i : i + batch_size]

            try:
                embeddings = embed_batch(batch_docs)
                collection.upsert(
                    documents=batch_docs,
                    ids=batch_ids,
                    metadatas=batch_metas,
                    embeddings=embeddings
                )
            except Exception as e:
                logger.warning(f"Error embedding batch {i}-{i+batch_size}: {e}")
                # Try individual documents
                for j, (doc, doc_id, meta) in enumerate(zip(batch_docs, batch_ids, batch_metas)):
                    try:
                        emb = embed_batch([doc])
                        collection.upsert(
                            documents=[doc],
                            ids=[doc_id],
                            metadatas=[meta],
                            embeddings=emb
                        )
                    except Exception as e2:
                        logger.warning(f"Failed to embed document {doc_id}: {e2}")

            if (i + batch_size) % 500 == 0 or i + batch_size >= total:
                logger.info(f"Embedding policy_notices: {min(i + batch_size, total)}/{total} chunks...")

        logger.info(f"policy_notices collection complete: {total} documents indexed")
        return total

    finally:
        cur.close()
        conn.close()


def build_hts_descriptions_collection(chroma: PersistentClient) -> int:
    """Build hts_descriptions collection from HTS_CODES table."""
    logger.info("Building hts_descriptions collection...")

    conn = get_snowflake_conn()
    cur = conn.cursor()

    try:
        collection = chroma.get_or_create_collection(
            name="hts_descriptions",
            metadata={"hnsw:space": "cosine"}
        )

        documents = []
        metadatas = []
        ids = []

        # Load HTS codes with valid descriptions
        logger.info("Loading HTS codes from HTS_CODES table...")
        cur.execute(
            """
            SELECT hts_code, description, general_rate, special_rate, chapter, is_chapter99
            FROM HTS_CODES
            WHERE is_header_row = FALSE
            AND description IS NOT NULL
            AND LENGTH(description) > 10
            ORDER BY hts_code
            """
        )
        hts_rows = cur.fetchall()

        for row in hts_rows:
            hts_code, desc, gen_rate, spec_rate, chapter, is_ch99 = row
            documents.append(desc)
            ids.append(hts_code)
            metadatas.append({
                "hts_code": hts_code,
                "general_rate": gen_rate or "",
                "special_rate": spec_rate or "",
                "chapter": chapter or "",
                "is_chapter99": str(is_ch99) if is_ch99 is not None else "False"
            })

        logger.info(f"Loaded {len(hts_rows)} HTS codes")

        # Embed and upsert in batches
        batch_size = 64
        total = len(documents)

        for i in range(0, total, batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            batch_metas = metadatas[i : i + batch_size]

            try:
                embeddings = embed_batch(batch_docs)
                collection.upsert(
                    documents=batch_docs,
                    ids=batch_ids,
                    metadatas=batch_metas,
                    embeddings=embeddings
                )
            except Exception as e:
                logger.warning(f"Error embedding batch {i}-{i+batch_size}: {e}")
                # Try individual documents
                for j, (doc, doc_id, meta) in enumerate(zip(batch_docs, batch_ids, batch_metas)):
                    try:
                        emb = embed_batch([doc])
                        collection.upsert(
                            documents=[doc],
                            ids=[doc_id],
                            metadatas=[meta],
                            embeddings=emb
                        )
                    except Exception as e2:
                        logger.warning(f"Failed to embed HTS code {doc_id}: {e2}")

            if (i + batch_size) % 500 == 0 or i + batch_size >= total:
                logger.info(f"Embedding hts_descriptions: {min(i + batch_size, total)}/{total} codes...")

        logger.info(f"hts_descriptions collection complete: {total} documents indexed")
        return total

    finally:
        cur.close()
        conn.close()


def search_policy(
    query: str,
    hts_chapter: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 5
) -> List[Dict]:
    """
    Search policy_notices collection.

    Args:
        query: Search query string
        hts_chapter: Optional filter by chapter (e.g., "85")
        source: Optional filter by source ("USTR", "CBP", "USITC")
        limit: Maximum results to return

    Returns:
        List of matching documents with metadata and distance
    """
    chroma = get_chroma_client()

    try:
        collection = chroma.get_collection(name="policy_notices")
    except Exception:
        raise RuntimeError("ChromaDB not initialized. Run chromadb_init.py first.")

    # Build where filter
    where_filters = None
    if source or hts_chapter:
        where_filters = {}
        if source:
            where_filters["source"] = {"$eq": source}
        if hts_chapter:
            if "source" in where_filters:
                where_filters = {"$and": [
                    where_filters,
                    {"hts_chapter": {"$eq": hts_chapter}}
                ]}
            else:
                where_filters["hts_chapter"] = {"$eq": hts_chapter}

    # Search
    results = collection.query(
        query_texts=[query],
        n_results=limit,
        where=where_filters,
        include=["documents", "metadatas", "distances"]
    )

    # Format results
    formatted = []
    if results["documents"] and len(results["documents"]) > 0:
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]

            formatted.append({
                "chunk_id": meta.get("chunk_id", ""),
                "document_number": meta.get("document_number", ""),
                "chunk_text": doc,
                "source": meta.get("source", ""),
                "hts_chapter": meta.get("hts_chapter", ""),
                "hts_code": meta.get("hts_code", ""),
                "distance": distance
            })

    return formatted


def search_hts(
    query: str,
    chapter: Optional[str] = None,
    limit: int = 5
) -> List[Dict]:
    """
    Search hts_descriptions collection.

    Args:
        query: Search query string
        chapter: Optional filter by chapter (e.g., "84")
        limit: Maximum results to return

    Returns:
        List of matching HTS codes with metadata and distance
    """
    chroma = get_chroma_client()

    try:
        collection = chroma.get_collection(name="hts_descriptions")
    except Exception:
        raise RuntimeError("ChromaDB not initialized. Run chromadb_init.py first.")

    # Build where filter
    where_filters = None
    if chapter:
        where_filters = {"chapter": {"$eq": chapter}}

    # Search
    results = collection.query(
        query_texts=[query],
        n_results=limit,
        where=where_filters,
        include=["documents", "metadatas", "distances"]
    )

    # Format results
    formatted = []
    if results["documents"] and len(results["documents"]) > 0:
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]

            formatted.append({
                "hts_code": meta.get("hts_code", ""),
                "description": doc,
                "general_rate": meta.get("general_rate", ""),
                "chapter": meta.get("chapter", ""),
                "distance": distance
            })

    return formatted


def initialize_chromadb():
    """
    Initialize ChromaDB collections.

    Always rebuilds from scratch to ensure consistency with Snowflake data.
    Rebuilds take ~30-60 seconds for 11K+ documents.
    """
    logger.info("Starting ChromaDB initialization (always rebuilding)...")
    start_time = time.time()

    chroma = get_chroma_client()

    # Delete existing collections to rebuild from scratch
    delete_collection_if_exists(chroma, "policy_notices")
    delete_collection_if_exists(chroma, "hts_descriptions")

    # Build fresh collections
    policy_count = build_policy_notices_collection(chroma)
    hts_count = build_hts_descriptions_collection(chroma)

    elapsed = time.time() - start_time

    logger.info(f"ChromaDB initialization complete")
    logger.info(f"  policy_notices: {policy_count} documents indexed")
    logger.info(f"  hts_descriptions: {hts_count} documents indexed")
    logger.info(f"  Total time: {elapsed:.1f} seconds")


if __name__ == "__main__":
    initialize_chromadb()
