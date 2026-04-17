"""
Policy Agent — TariffIQ Pipeline Step 4

Retrieves relevant policy context from USTR + CBP + ITC documents using:
  1. HyDE: Generate hypothetical FR sentence, embed as query vector
  2. Dense retrieval: ChromaDB vector search on policy_notices collection
     filtered to confirmed HTS code (falls back to chapter → unfiltered)
  3. BM25: Keyword retrieval over same chunks from Snowflake
  4. RRF: Reciprocal Rank Fusion to merge results
  5. LiteLLM: Synthesize policy summary with document citations

Dense retrieval filters ChromaDB to chunks from notices that mention the
EXACT confirmed HTS code (via NOTICE_HTS_CODES join), instead of just the
2-digit chapter. This dramatically improves precision.

Fallback chain: exact hts_code filter → chapter filter → unfiltered

Collection: policy_notices (USTR + CBP + ITC chunks via Ayush's chromadb_init.py)
"""

import logging
import os
from typing import Dict, Any, List, Optional, Set

import litellm
from rank_bm25 import BM25Okapi
import chromadb

from ingestion.connection import get_snowflake_conn
from ingestion.embedder import Embedder
from agents.state import TariffState
from services.retrieval.hybrid import HybridRetriever
from services.llm.router import get_router, TaskType

logger = logging.getLogger(__name__)

TOP_K = 5
RRF_K = 60

HYDE_PROMPT = """Write one sentence that would appear in a US Federal Register tariff notice
about the following product and HTS code. Be specific about tariff rates and policy context.
Product: {product}
HTS Code: {hts_code}
HTS Chapter: {hts_chapter}"""

POLICY_SYSTEM_PROMPT = """You are a US trade policy analyst specializing in import tariffs and customs law.
Answer the user's question using only the document excerpts provided below.
These excerpts come from:
  - USTR Federal Register notices: Section 301, IEEPA, safeguard tariff actions
  - CBP rulings: Customs classification rulings, country-of-origin determinations
  - USITC notices: ITC tariff investigations and determinations

Rules:
1. Cite the exact document number in parentheses for every factual claim
2. If a CBP ruling addresses country-of-origin, explicitly note it
3. If context is insufficient, say so explicitly
4. Do not use knowledge outside the provided documents
5. Be concise — 3-5 sentences maximum"""


def _get_chroma_client() -> chromadb.HttpClient:
    # Support both CHROMADB_HOST (our convention) and CHROMA_HOST (Ayush's convention)
    host = os.environ.get("CHROMADB_HOST", os.environ.get("CHROMA_HOST", "chromadb"))
    port = int(os.environ.get("CHROMADB_PORT", os.environ.get("CHROMA_PORT", 8000)))
    return chromadb.HttpClient(host=host, port=port)


def _fetch_doc_numbers_for_hts_code(hts_code: str) -> Set[str]:
    """
    Fetch all FR document numbers that reference the exact HTS code.
    Used to pre-filter ChromaDB to only relevant notices.

    Filters by exact HTS code (e.g. '8541.43.00' → only solar cell notices)
    instead of chapter (e.g. '85' → all Chapter 85 products).
    """
    if not hts_code:
        return set()
    conn = get_snowflake_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT DISTINCT document_number
            FROM TARIFFIQ.RAW.NOTICE_HTS_CODES
            WHERE hts_code = %s
            """,
            (hts_code,),
        )
        doc_numbers = {r[0] for r in cur.fetchall() if r[0]}
        logger.info(
            "policy_agent_hts_doc_filter hts=%s matched_notices=%d",
            hts_code, len(doc_numbers),
        )
        return doc_numbers
    except Exception as e:
        logger.warning("policy_agent_hts_doc_filter_error hts=%s error=%s", hts_code, e)
        return set()
    finally:
        cur.close()
        conn.close()


def _fetch_chunks_for_hts_code(hts_code: str, hts_chapter: str) -> List[Dict[str, Any]]:
    """
    Fetch chunks for BM25 corpus from Snowflake CHUNKS table.
    Tries exact HTS code first, falls back to chapter-level if < 10 chunks found.
    """
    conn = get_snowflake_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT c.chunk_id, c.chunk_text, c.document_number, c.section,
                   f.title, f.publication_date::VARCHAR
            FROM TARIFFIQ.RAW.CHUNKS c
            INNER JOIN TARIFFIQ.RAW.NOTICE_HTS_CODES n
                ON c.document_number = n.document_number
            LEFT JOIN TARIFFIQ.RAW.FEDERAL_REGISTER_NOTICES f
                ON c.document_number = f.document_number
            WHERE n.hts_code = %s
              AND c.chunk_text IS NOT NULL
            LIMIT 200
            """,
            (hts_code,),
        )
        rows = cur.fetchall()

        if len(rows) >= 10:
            logger.info("policy_agent_bm25_corpus hts_code=%s chunks=%d source=exact_code", hts_code, len(rows))
            return [{"chunk_id": r[0], "chunk_text": r[1], "document_number": r[2],
                     "section": r[3], "title": r[4] or "", "publication_date": r[5] or "",
                     "source": "USTR"} for r in rows]

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
        logger.info("policy_agent_bm25_corpus hts_chapter=%s chunks=%d source=chapter_fallback", hts_chapter, len(rows))
        return [{"chunk_id": r[0], "chunk_text": r[1], "document_number": r[2],
                 "section": r[3], "title": r[4] or "", "publication_date": r[5] or "",
                 "source": "USTR"} for r in rows]
    finally:
        cur.close()
        conn.close()


def _hyde_query(product: str, hts_code: str, hts_chapter: str) -> str:
    try:
        response = litellm.completion(
            model=os.environ["LLM_MODEL"],  # no default — must be set in .env
            messages=[
                {
                    "role": "user",
                    "content": HYDE_PROMPT.format(
                        product=product,
                        hts_code=hts_code or f"chapter {hts_chapter}",
                        hts_chapter=hts_chapter,
                    ),
                }
            ],
            temperature=0.3,
            max_tokens=80,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("hyde_generation_failed error=%s", e)
        return f"{product} HTS {hts_code} tariff Federal Register notice"


def _dense_retrieval(
    query_text: str,
    hts_code: str,
    hts_chapter: str,
    doc_numbers_for_hts: Set[str],
    embedder: Embedder,
) -> List[Dict[str, Any]]:
    """
    Dense vector retrieval from ChromaDB policy_notices collection.

    Filter priority:
      1. Filter by exact document_number set (notices that cite this HTS code)
      2. Fall back to chapter-level filter
      3. Fall back to unfiltered
    """
    try:
        query_vec = embedder.embed_batch([query_text])[0]
        chroma = _get_chroma_client()
        collection = chroma.get_collection("policy_notices")

        # Attempt 1: filter by exact HTS-linked document numbers
        if doc_numbers_for_hts:
            results = collection.query(
                query_embeddings=[query_vec],
                n_results=TOP_K,
                where={"document_number": {"$in": list(doc_numbers_for_hts)}},
                include=["documents", "metadatas", "distances"],
            )
            if results["ids"][0]:
                logger.info("dense_retrieval_source=hts_code_filter hts=%s chunks=%d", hts_code, len(results["ids"][0]))
                return _format_chroma_results(results)

        # Attempt 2: chapter filter
        if hts_chapter:
            results = collection.query(
                query_embeddings=[query_vec],
                n_results=TOP_K,
                where={"hts_chapter": {"$eq": hts_chapter}},
                include=["documents", "metadatas", "distances"],
            )
            if results["ids"][0]:
                logger.info("dense_retrieval_source=chapter_filter chapter=%s chunks=%d", hts_chapter, len(results["ids"][0]))
                return _format_chroma_results(results)

        # Attempt 3: unfiltered
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"],
        )
        logger.info("dense_retrieval_source=unfiltered chunks=%d", len(results["ids"][0]))
        return _format_chroma_results(results)

    except Exception as e:
        logger.warning("dense_retrieval_failed error=%s", e)
        return []


def _format_chroma_results(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    chunks = []
    for i, doc_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        chunks.append({
            "chunk_id": doc_id,
            "chunk_text": results["documents"][0][i],
            "document_number": meta.get("document_number", ""),
            "title": meta.get("title", ""),
            "publication_date": meta.get("publication_date", ""),
            "source": meta.get("source", "USTR"),
            "distance": results["distances"][0][i],
        })
    return chunks


def _bm25_retrieval(query_text: str, corpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not corpus:
        return []
    tokenized = [c["chunk_text"].lower().split() for c in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query_text.lower().split())
    ranked = sorted(zip(scores, corpus), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:TOP_K]]


def _rrf_fusion(
    dense_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
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


async def run_policy_agent(state: TariffState) -> Dict[str, Any]:
    """
    Retrieve policy context from USTR/CBP/ITC using HyDE + HybridRetriever.

    Flow:
      1. HyDE: Generate hypothetical FR excerpt from product/country/HTS
      2. HybridRetriever.search_policy(): Dense + Sparse + RRF on enhanced query
      3. ModelRouter.complete(): Synthesize summary with citations
    """
    product = state.get("product") or state.get("query", "")
    hts_code = state.get("hts_code") or ""
    hts_chapter = hts_code[:2] if hts_code else ""
    country = state.get("country")
    query = state.get("query", "")

    logger.info(
        "policy_agent_start",
        product=product,
        hts_code=hts_code,
        chapter=hts_chapter,
        country=country
    )

    # Step 1: HyDE query enhancement
    try:
        from services.retrieval.hyde import get_enhancer
        enhancer = get_enhancer()
        enhanced_query = await enhancer.enhance(
            query=query,
            product=product,
            country=country or "unknown",
            hts_chapter=hts_chapter
        )
        logger.info(
            "policy_agent_hyde",
            original_len=len(query),
            enhanced_len=len(enhanced_query)
        )
    except Exception as e:
        logger.warning("policy_agent_hyde_failed error=%s using original query", e)
        enhanced_query = query

    # Step 2: HybridRetriever search with optional HTS chapter filter
    try:
        retriever = HybridRetriever()
        results = retriever.search_policy(
            query=enhanced_query,
            hts_chapter=hts_chapter if hts_chapter else None,
            source=None,
            top_k=5
        )

        if not results:
            logger.warning("policy_agent_no_results product=%s hts=%s", product, hts_code)
            return {
                "policy_chunks": [],
                "policy_summary": "No policy context found for this product.",
            }

        logger.info(
            "policy_agent_retrieval",
            result_count=len(results),
            retrieval_method=results[0].get("retrieval_method", "unknown")
        )

    except Exception as e:
        logger.error("policy_agent_retrieval_failed error=%s", e)
        return {
            "policy_chunks": [],
            "policy_summary": "Policy retrieval failed.",
        }

    # Step 3: Format context and generate summary with ModelRouter
    context_block = "\n\n".join([
        f"[{r.get('source', 'USTR').upper()} | {r.get('document_number', 'UNKNOWN')}] {r['chunk_text']}"
        for r in results
    ])

    try:
        router = get_router()
        response = await router.complete(
            task=TaskType.POLICY_ANALYSIS,
            messages=[
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nDocument excerpts:\n{context_block}",
                },
            ],
        )
        summary = response.choices[0].message.content.strip()
        logger.info("policy_agent_synthesis_success", summary_len=len(summary))

    except Exception as e:
        logger.error("policy_agent_synthesis_failed error=%s", e)
        summary = "Policy synthesis failed. See raw chunks for context."

    logger.info("policy_agent_done", chunk_count=len(results))

    return {
        "policy_chunks": results,
        "policy_summary": summary,
    }