"""
Policy Agent — TariffIQ Pipeline Step 4

Retrieves relevant Federal Register policy context using:
  1. HyDE: Generate hypothetical FR sentence, embed as query vector
  2. Dense retrieval: ChromaDB vector search filtered to HTS chapter
  3. BM25: Keyword retrieval over same chunks
  4. RRF: Reciprocal Rank Fusion to merge results
  5. LiteLLM: Synthesize policy summary with FR document citations
"""

import logging
import os
from typing import Dict, Any, List, Optional

import litellm
from rank_bm25 import BM25Okapi

import chromadb

from ingestion.connection import get_snowflake_conn
from ingestion.embedder import Embedder
from agents.state import TariffState

logger = logging.getLogger(__name__)

TOP_K = 5
RRF_K = 60

HYDE_PROMPT = """Write one sentence that would appear in a US Federal Register tariff notice
about the following product and HTS chapter. Be specific about tariff rates and policy context.
Product: {product}
HTS Chapter: {hts_chapter}"""

POLICY_SYSTEM_PROMPT = """You are a US trade policy analyst.
Answer the user's question using only the Federal Register excerpts provided.
Cite the exact document number in parentheses for every factual claim.
If the context is insufficient, say so explicitly.
Do not use knowledge of tariff rates or policy outside the provided documents."""


def _get_chroma_client() -> chromadb.HttpClient:
    host = os.environ.get("CHROMADB_HOST", "chromadb")
    port = int(os.environ.get("CHROMADB_PORT", 8000))
    return chromadb.HttpClient(host=host, port=port)


def _fetch_chunks_for_chapter(hts_chapter: str) -> List[Dict[str, Any]]:
    """
    Fetch all chunks for a given HTS chapter from Snowflake CHUNKS table.
    Used as the BM25 corpus.
    """
    conn = get_snowflake_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT c.chunk_id, c.chunk_text, c.document_number, c.section,
                   f.title, f.publication_date::VARCHAR
            FROM TARIFFIQ.RAW.CHUNKS c
            LEFT JOIN TARIFFIQ.RAW.FEDERAL_REGISTER_NOTICES f
                ON c.document_number = f.document_number
            LEFT JOIN TARIFFIQ.RAW.NOTICE_HTS_CODES n
                ON c.document_number = n.document_number
            WHERE n.hts_chapter = %s
              AND c.chunk_text IS NOT NULL
            LIMIT 200
            """,
            (hts_chapter,),
        )
        rows = cur.fetchall()
        return [
            {
                "chunk_id": r[0],
                "chunk_text": r[1],
                "document_number": r[2],
                "section": r[3],
                "title": r[4] or "",
                "publication_date": r[5] or "",
            }
            for r in rows
        ]
    finally:
        cur.close()
        conn.close()


def _hyde_query(product: str, hts_chapter: str) -> str:
    """Generate hypothetical FR sentence for better query embedding."""
    try:
        response = litellm.completion(
            model=os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514"),
            messages=[
                {
                    "role": "user",
                    "content": HYDE_PROMPT.format(
                        product=product, hts_chapter=hts_chapter
                    ),
                }
            ],
            temperature=0.3,
            max_tokens=80,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("hyde_generation_failed error=%s", e)
        return f"{product} tariff chapter {hts_chapter} Federal Register notice"


def _dense_retrieval(
    query_text: str, hts_chapter: str, embedder: Embedder
) -> List[Dict[str, Any]]:
    """Dense vector retrieval from ChromaDB filtered to HTS chapter."""
    try:
        query_vec = embedder.embed_batch([query_text])[0]
        chroma = _get_chroma_client()
        collection = chroma.get_collection("federal_register")

        # Try chapter-filtered retrieval first
        where_filter = {"hts_chapter": {"$eq": hts_chapter}} if hts_chapter else None

        results = collection.query(
            query_embeddings=[query_vec],
            n_results=TOP_K,
            where=where_filter if where_filter else None,
            include=["documents", "metadatas", "distances"],
        )

        # If no results with chapter filter, broaden
        if not results["ids"][0] and where_filter:
            results = collection.query(
                query_embeddings=[query_vec],
                n_results=TOP_K,
                include=["documents", "metadatas", "distances"],
            )

        chunks = []
        for i, doc_id in enumerate(results["ids"][0]):
            chunks.append(
                {
                    "chunk_id": doc_id,
                    "chunk_text": results["documents"][0][i],
                    "document_number": results["metadatas"][0][i].get("document_number", ""),
                    "title": results["metadatas"][0][i].get("title", ""),
                    "publication_date": results["metadatas"][0][i].get("publication_date", ""),
                    "distance": results["distances"][0][i],
                }
            )
        return chunks

    except Exception as e:
        logger.warning("dense_retrieval_failed error=%s", e)
        return []


def _bm25_retrieval(query_text: str, corpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """BM25 keyword retrieval over corpus chunks."""
    if not corpus:
        return []

    tokenized_corpus = [c["chunk_text"].lower().split() for c in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query_text.lower().split())

    ranked = sorted(
        zip(scores, corpus), key=lambda x: x[0], reverse=True
    )
    return [c for _, c in ranked[:TOP_K]]


def _rrf_fusion(
    dense_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion of dense and BM25 results.
    RRF score = sum(1 / (k + rank)) across result lists.
    """
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict[str, Any]] = {}

    for rank, chunk in enumerate(dense_results):
        cid = chunk.get("chunk_id") or chunk.get("document_number", "") + str(rank)
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (RRF_K + rank + 1)
        chunk_map[cid] = chunk

    for rank, chunk in enumerate(bm25_results):
        cid = chunk.get("chunk_id") or chunk.get("document_number", "") + str(rank)
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (RRF_K + rank + 1)
        chunk_map[cid] = chunk

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    return [chunk_map[cid] for cid in sorted_ids[:TOP_K]]


def run_policy_agent(state: TariffState) -> Dict[str, Any]:
    """
    Retrieve Federal Register policy context and synthesize with LLM.

    Args:
        state: TariffState with product, hts_code, query populated

    Returns:
        Dict with policy_chunks and policy_summary
    """
    product = state.get("product") or state.get("query", "")
    hts_code = state.get("hts_code") or ""
    hts_chapter = hts_code[:2] if hts_code else ""
    query = state.get("query", "")

    logger.info("policy_agent_start product=%s chapter=%s", product, hts_chapter)

    embedder = Embedder()

    # Step 1 — HyDE
    hyde_query = _hyde_query(product, hts_chapter)
    logger.info("policy_agent_hyde query=%s", hyde_query)

    # Step 2 — Dense retrieval
    dense_results = _dense_retrieval(hyde_query, hts_chapter, embedder)

    # Step 3 — BM25 retrieval
    corpus = _fetch_chunks_for_chapter(hts_chapter)
    if not corpus:
        corpus = dense_results  # fallback to dense results as corpus
    bm25_results = _bm25_retrieval(f"{product} {hts_chapter} tariff", corpus)

    # Step 4 — RRF fusion
    fused = _rrf_fusion(dense_results, bm25_results)

    if not fused:
        logger.warning("policy_agent_no_chunks product=%s", product)
        return {
            "policy_chunks": [],
            "policy_summary": "No Federal Register policy context found for this product.",
        }

    # Step 5 — LLM synthesis
    context_block = "\n\n".join(
        [
            f"[{c.get('document_number', 'UNKNOWN')}] {c['chunk_text']}"
            for c in fused
        ]
    )

    try:
        response = litellm.completion(
            model=os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514"),
            messages=[
                {"role": "system", "content": POLICY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nFederal Register excerpts:\n{context_block}",
                },
            ],
            temperature=0.1,
            max_tokens=500,
        )
        summary = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("policy_llm_failed error=%s", e)
        summary = "Policy synthesis failed. See raw chunks for context."

    logger.info("policy_agent_done chunks=%d", len(fused))

    return {
        "policy_chunks": fused,
        "policy_summary": summary,
    }