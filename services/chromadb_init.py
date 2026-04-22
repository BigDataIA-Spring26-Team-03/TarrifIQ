"""
ChromaDB initialization for TariffIQ.

Builds two collections using HttpClient (ChromaDB container):
  1. policy_notices  — CHUNKS (USTR) + CBP_CHUNKS + ITC_CHUNKS
  2. hts_descriptions — HTS_CODES descriptions

Skip-if-populated: if collection already has >10000 docs, skip rebuild.
Set CHROMADB_FORCE_REBUILD=true to force full rebuild.

NEVER deletes collections on every startup — that would wipe data for 60-90s.
"""

import os
import time
from typing import Optional, List, Dict
import logging

import snowflake.connector
import chromadb
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_embedder = None
SKIP_THRESHOLD = 10000


def get_snowflake_conn():
    return snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
    )


def get_chroma_client() -> chromadb.HttpClient:
    """HttpClient pointing to ChromaDB container — NOT PersistentClient."""
    host = os.environ.get("CHROMADB_HOST", os.environ.get("CHROMA_HOST", "chromadb"))
    port = int(os.environ.get("CHROMADB_PORT", os.environ.get("CHROMA_PORT", 8000)))
    return chromadb.HttpClient(host=host, port=port)


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def embed_batch(texts: List[str]) -> List[List[float]]:
    return get_embedder().encode(texts).tolist()


def _needs_build(chroma: chromadb.HttpClient, name: str) -> bool:
    force = os.environ.get("CHROMADB_FORCE_REBUILD", "false").lower() == "true"
    if force:
        logger.info("chroma_init: force rebuild %s", name)
        try:
            chroma.delete_collection(name)
        except Exception:
            pass
        return True
    try:
        count = chroma.get_collection(name).count()
        if count > SKIP_THRESHOLD:
            logger.info("chroma_init: %s has %d docs — skipping", name, count)
            return False
        logger.info("chroma_init: %s has %d docs — rebuilding", name, count)
        return True
    except Exception:
        logger.info("chroma_init: %s does not exist — building", name)
        return True


def build_policy_notices_collection(chroma: chromadb.HttpClient) -> int:
    """Build policy_notices from USTR + CBP + ITC + ITA + EOP chunks."""
    conn = get_snowflake_conn()
    cur = conn.cursor()
    try:
        col = chroma.get_or_create_collection("policy_notices", metadata={"hnsw:space": "cosine"})
        documents, metadatas, ids = [], [], []

        # USTR chunks
        cur.execute("""
            SELECT c.chunk_id, c.document_number, c.chunk_index, c.chunk_text,
                   c.section, c.hts_code, c.hts_chapter,
                   f.publication_date::VARCHAR
            FROM CHUNKS c
            LEFT JOIN FEDERAL_REGISTER_NOTICES f ON c.document_number = f.document_number
            WHERE c.chunk_text IS NOT NULL ORDER BY c.chunk_id
        """)
        for i, row in enumerate(cur.fetchall()):
            chunk_id, doc_num, idx, text, section, hts_code, hts_chapter, pub_date = row
            dated_text = f"[{pub_date}] {text}" if pub_date else text
            documents.append(dated_text)
            ids.append(f"USTR_{chunk_id}" if chunk_id else f"USTR_{i}")
            metadatas.append({
                "chunk_id": chunk_id or "",
                "document_number": doc_num or "",
                "hts_chapter": hts_chapter or "",
                "hts_code": hts_code or "",
                "source": "USTR",
                "section": section or "",
                "publication_date": pub_date or "",
            })
        logger.info("chroma_init: loaded %d USTR chunks", len(documents))

        # CBP chunks
        cbp_start = len(documents)
        try:
            cur.execute("""
                SELECT c.chunk_id, c.document_number, c.chunk_index, c.chunk_text,
                       c.section, c.hts_code, c.hts_chapter, c.general_rate,
                       f.publication_date::VARCHAR
                FROM CBP_CHUNKS c
                LEFT JOIN CBP_FEDERAL_REGISTER_NOTICES f ON c.document_number = f.document_number
                WHERE c.chunk_text IS NOT NULL ORDER BY c.chunk_id
            """)
            for i, row in enumerate(cur.fetchall()):
                chunk_id, doc_num, idx, text, section, hts_code, hts_chapter, gen_rate, pub_date = row
                dated_text = f"[{pub_date}] {text}" if pub_date else text
                documents.append(dated_text)
                ids.append(f"CBP_{chunk_id}" if chunk_id else f"CBP_{i}")
                metadatas.append({
                    "chunk_id": chunk_id or "",
                    "document_number": doc_num or "",
                    "hts_chapter": hts_chapter or "",
                    "hts_code": hts_code or "",
                    "source": "CBP",
                    "section": section or "",
                    "general_rate": gen_rate or "",
                    "publication_date": pub_date or "",
                })
            logger.info("chroma_init: loaded %d CBP chunks", len(documents) - cbp_start)
        except Exception as e:
            logger.warning("chroma_init: CBP_CHUNKS unavailable: %s", e)

        # ITC chunks
        itc_start = len(documents)
        try:
            cur.execute("""
                SELECT c.chunk_id, c.document_number, c.chunk_index, c.chunk_text,
                       c.section, c.hts_code, c.hts_chapter,
                       f.publication_date::VARCHAR
                FROM ITC_CHUNKS c
                LEFT JOIN ITC_DOCUMENTS f ON c.document_number = f.document_number
                WHERE c.chunk_text IS NOT NULL ORDER BY c.chunk_id
            """)
            for i, row in enumerate(cur.fetchall()):
                chunk_id, doc_num, idx, text, section, hts_code, hts_chapter, pub_date = row
                dated_text = f"[{pub_date}] {text}" if pub_date else text
                documents.append(dated_text)
                ids.append(f"USITC_{chunk_id}" if chunk_id else f"USITC_{i}")
                metadatas.append({
                    "chunk_id": chunk_id or "",
                    "document_number": doc_num or "",
                    "hts_chapter": hts_chapter or "",
                    "hts_code": hts_code or "",
                    "source": "USITC",
                    "section": section or "",
                    "publication_date": pub_date or "",
                })
            logger.info("chroma_init: loaded %d ITC chunks", len(documents) - itc_start)
        except Exception as e:
            logger.warning("chroma_init: ITC_CHUNKS unavailable: %s", e)

        # ITA chunks
        ita_start = len(documents)
        try:
            cur.execute("""
                SELECT c.chunk_id, c.document_number, c.chunk_index, c.chunk_text,
                       c.section, c.hts_code, c.hts_chapter,
                       f.publication_date::VARCHAR
                FROM ITA_CHUNKS c
                LEFT JOIN ITA_FEDERAL_REGISTER_NOTICES f ON c.document_number = f.document_number
                WHERE c.chunk_text IS NOT NULL ORDER BY c.chunk_id
            """)
            for i, row in enumerate(cur.fetchall()):
                chunk_id, doc_num, idx, text, section, hts_code, hts_chapter, pub_date = row
                dated_text = f"[{pub_date}] {text}" if pub_date else text
                documents.append(dated_text)
                ids.append(f"ITA_{chunk_id}" if chunk_id else f"ITA_{i}")
                metadatas.append({
                    "chunk_id": chunk_id or "",
                    "document_number": doc_num or "",
                    "hts_chapter": hts_chapter or "",
                    "hts_code": hts_code or "",
                    "source": "ITA",
                    "section": section or "",
                    "publication_date": pub_date or "",
                })
            logger.info("chroma_init: loaded %d ITA chunks", len(documents) - ita_start)
        except Exception as e:
            logger.warning("chroma_init: ITA_CHUNKS unavailable: %s", e)

        # EOP chunks
        eop_start = len(documents)
        try:
            cur.execute("""
                SELECT c.chunk_id, c.document_number, c.chunk_index, c.chunk_text,
                       c.section, c.hts_code, c.hts_chapter,
                       f.publication_date::VARCHAR
                FROM EOP_CHUNKS c
                LEFT JOIN EOP_DOCUMENTS f ON c.document_number = f.document_number
                WHERE c.chunk_text IS NOT NULL ORDER BY c.chunk_id
            """)
            for i, row in enumerate(cur.fetchall()):
                chunk_id, doc_num, idx, text, section, hts_code, hts_chapter, pub_date = row
                dated_text = f"[{pub_date}] {text}" if pub_date else text
                documents.append(dated_text)
                ids.append(f"EOP_{chunk_id}" if chunk_id else f"EOP_{i}")
                metadatas.append({
                    "chunk_id": chunk_id or "",
                    "document_number": doc_num or "",
                    "hts_chapter": hts_chapter or "",
                    "hts_code": hts_code or "",
                    "source": "EOP",
                    "section": section or "",
                    "publication_date": pub_date or "",
                })
            logger.info("chroma_init: loaded %d EOP chunks", len(documents) - eop_start)
        except Exception as e:
            logger.warning("chroma_init: EOP_CHUNKS unavailable: %s", e)

        # Deduplicate by ID before upserting — prevents batch failure on duplicate chunk_ids
        seen_ids = set()
        deduped_documents, deduped_ids, deduped_metadatas = [], [], []
        for doc, id_, meta in zip(documents, ids, metadatas):
            if id_ not in seen_ids:
                seen_ids.add(id_)
                deduped_documents.append(doc)
                deduped_ids.append(id_)
                deduped_metadatas.append(meta)

        documents, ids, metadatas = deduped_documents, deduped_ids, deduped_metadatas
        logger.info("chroma_init: after dedup total=%d", len(documents))

        # Embed and upsert
        batch_size = 32
        total = len(documents)
        for i in range(0, total, batch_size):
            try:
                col.upsert(
                    documents=documents[i:i+batch_size],
                    ids=ids[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size],
                    embeddings=embed_batch(documents[i:i+batch_size]),
                )
                logger.info("chroma_init: policy batch %d-%d upserted", i, min(i+batch_size, total))
            except Exception as e:
                logger.error("chroma_init: policy batch %d failed: %s", i, e)

        logger.info("chroma_init: policy_notices complete total=%d", total)
        return total
    finally:
        cur.close(); conn.close()


def build_hts_descriptions_collection(chroma: chromadb.HttpClient) -> int:
    """Build hts_descriptions from HTS_CODES."""
    conn = get_snowflake_conn()
    cur = conn.cursor()
    try:
        col = chroma.get_or_create_collection("hts_descriptions", metadata={"hnsw:space": "cosine"})
        cur.execute(
            "SELECT hts_code, description, general_rate, special_rate, chapter, is_chapter99 FROM HTS_CODES WHERE is_header_row = FALSE AND description IS NOT NULL AND LENGTH(description) > 10 ORDER BY hts_code"
        )
        rows = cur.fetchall()
        documents = [r[1] for r in rows]
        ids = [r[0] for r in rows]
        metadatas = [{"hts_code": r[0], "general_rate": r[2] or "", "special_rate": r[3] or "",
                      "chapter": r[4] or "", "is_chapter99": str(r[5]) if r[5] is not None else "False"}
                     for r in rows]

        batch_size = 32
        for i in range(0, len(documents), batch_size):
            try:
                col.upsert(
                    documents=documents[i:i+batch_size],
                    ids=ids[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size],
                    embeddings=embed_batch(documents[i:i+batch_size]),
                )
                logger.info("chroma_init: hts batch %d-%d upserted", i, min(i+batch_size, len(documents)))
            except Exception as e:
                logger.error("chroma_init: hts batch %d failed: %s", i, e)

        logger.info("chroma_init: hts_descriptions complete total=%d", len(documents))
        return len(documents)
    finally:
        cur.close(); conn.close()


def search_policy(query: str, hts_chapter: Optional[str] = None, source: Optional[str] = None, limit: int = 5) -> List[Dict]:
    """Search policy_notices collection. Called by search_policy_vector.py."""
    chroma = get_chroma_client()
    try:
        col = chroma.get_collection("policy_notices")
    except Exception:
        raise RuntimeError("policy_notices not initialized.")

    if hts_chapter:
        query = f"HTS chapter {hts_chapter} tariff {query}"
    where = {"source": {"$eq": source}} if source else None

    results = col.query(query_texts=[query], n_results=limit, where=where,
                        include=["documents", "metadatas", "distances"])

    out = []
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            out.append({"chunk_id": meta.get("chunk_id", ""), "document_number": meta.get("document_number", ""),
                        "chunk_text": doc, "source": meta.get("source", ""), "hts_chapter": meta.get("hts_chapter", ""),
                        "hts_code": meta.get("hts_code", ""), "distance": results["distances"][0][i]})
    return out


def search_hts(query: str, chapter: Optional[str] = None, limit: int = 5) -> List[Dict]:
    """Search hts_descriptions collection. Called by search_hts_vector.py."""
    chroma = get_chroma_client()
    try:
        col = chroma.get_collection("hts_descriptions")
    except Exception:
        raise RuntimeError("hts_descriptions not initialized.")

    if chapter:
        query = f"HTS chapter {chapter} {query}"
    where = None
    results = col.query(query_texts=[query], n_results=limit, where=where,
                        include=["documents", "metadatas", "distances"])

    out = []
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            out.append({"hts_code": meta.get("hts_code", ""), "description": doc,
                        "general_rate": meta.get("general_rate", ""),
                        "chapter": meta.get("chapter", ""), "distance": results["distances"][0][i]})
    return out


def initialize_chromadb():
    """
    Initialize ChromaDB on startup.
    Skips if collections already populated (>10000 docs).
    Set CHROMADB_FORCE_REBUILD=true to force full rebuild.
    Set CHROMADB_SKIP_INIT=true to skip init entirely.
    """
    if os.environ.get("CHROMADB_SKIP_INIT", "false").lower() == "true":
        logger.info("chroma_init: skipped (CHROMADB_SKIP_INIT=true)")
        return
    logger.info("chroma_init: starting")
    start = time.time()
    chroma = get_chroma_client()

    policy_count = build_policy_notices_collection(chroma) if _needs_build(chroma, "policy_notices") else 0
    hts_count = build_hts_descriptions_collection(chroma) if _needs_build(chroma, "hts_descriptions") else 0

    logger.info("chroma_init: complete policy_notices=%d hts_descriptions=%d time=%.1fs",
                policy_count, hts_count, time.time() - start)


if __name__ == "__main__":
    initialize_chromadb()