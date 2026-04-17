"""
Policy Agent — Pipeline Step 4

Retrieves Federal Register policy context using Ayush's services:
  1. HyDEQueryEnhancer.enhance_sync()  — generates hypothetical FR sentence
  2. HybridRetriever.search_policy()   — dense + BM25 + RRF in one call
  3. ModelRouter(POLICY_ANALYSIS)      — LLM synthesis with numbered citations

Both HyDE and HybridRetriever use singletons — instantiated once at startup,
never per-request (BM25 index build takes seconds).

Citation enforcement:
  LLM receives [1]…[N] indexed list. Post-generation, _resolve_citations()
  replaces indices with actual doc numbers and strips any hallucinated ones.

Redis cache: 6-hour TTL keyed on (hts_code + query_hash).
"""

import asyncio
import hashlib
import json
import logging
import os
import re
from typing import Dict, Any, List, Optional, Set, Tuple

from agents.state import TariffState
from agents import tools

logger = logging.getLogger(__name__)

TOP_K = 5
CACHE_TTL = 21_600  # 6 h

POLICY_PROMPT_SUFFIX = """
CITATION RULES (strictly enforced):
1. Cite using the bracketed index: [1], [2] — exactly as shown in the excerpts
2. Every factual claim needs a citation — no unsourced statements
3. Do NOT cite an index not present in the list
4. If a CBP ruling covers country-of-origin, say so explicitly
5. If context is insufficient: "Insufficient policy context for this query."
6. Use NO knowledge outside the provided documents
7. Maximum 3-5 sentences"""


# ── Redis ─────────────────────────────────────────────────────────────────────

def _redis():
    try:
        import redis
        c = redis.Redis(
            host=os.environ.get("REDIS_HOST", "redis"),
            port=int(os.environ.get("REDIS_PORT", "6379")),
            socket_connect_timeout=2, socket_timeout=2,
        )
        c.ping()
        return c
    except Exception:
        return None


def _cache_key(hts_code: str, query: str) -> str:
    qhash = hashlib.md5(query.lower().strip().encode()).hexdigest()[:8]
    return f"tariffiq:policy:{hts_code}:{qhash}"


def _cache_get(hts_code: str, query: str) -> Optional[Dict]:
    r = _redis()
    if not r:
        return None
    try:
        raw = r.get(_cache_key(hts_code, query))
        if raw:
            logger.info("policy_agent_cache_hit hts=%s", hts_code)
            return json.loads(raw)
    except Exception:
        pass
    return None


def _cache_set(hts_code: str, query: str, result: Dict) -> None:
    r = _redis()
    if not r or not result.get("policy_chunks"):
        return
    try:
        r.setex(_cache_key(hts_code, query), CACHE_TTL, json.dumps(result))
    except Exception:
        pass


# ── Citation building + resolution ────────────────────────────────────────────

def _build_numbered_context(chunks: List[Dict]) -> Tuple[str, Dict[int, str]]:
    """Build [1]…[N] numbered context block. Returns (text, index→doc_number map)."""
    index_to_doc: Dict[int, str] = {}
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        doc = chunk.get("document_number", "UNKNOWN")
        src = chunk.get("source", "USTR").upper()
        title = chunk.get("title", "") or ""
        pub = chunk.get("publication_date", "") or ""
        header = f"[{i}] {src} | {doc}"
        if title:
            header += f' | "{title[:80]}"'
        if pub:
            header += f" | {pub}"
        text = chunk.get("chunk_text", "")
        lines.append(f"{header}\n{text[:400]}")
        index_to_doc[i] = doc
    return "\n\n".join(lines), index_to_doc


def _resolve_citations(summary: str, index_to_doc: Dict[int, str]) -> Tuple[str, List[str]]:
    """Replace [N] with (doc_number). Return (resolved_text, invalid_index_list)."""
    cited = set(int(m) for m in re.findall(r"\[(\d+)\]", summary))
    invalid = [str(i) for i in cited if i not in index_to_doc]

    def _replace(m: re.Match) -> str:
        idx = int(m.group(1))
        doc = index_to_doc.get(idx)
        return f"({doc})" if doc else f"[INVALID-{idx}]"

    resolved = re.sub(r"\[(\d+)\]", _replace, summary)
    return resolved, invalid


# ── Main agent ────────────────────────────────────────────────────────────────

def run_policy_agent(state: TariffState) -> Dict[str, Any]:
    product = state.get("product") or state.get("query", "")
    hts_code = state.get("hts_code") or ""
    hts_chapter = hts_code[:2] if hts_code else ""
    query = state.get("query", "")
    country = state.get("country") or "unspecified"

    logger.info("policy_agent_start product=%s hts=%s chapter=%s", product, hts_code, hts_chapter)

    # Cache check
    cached = _cache_get(hts_code, query)
    if cached:
        return cached

    # ── Step 1: HyDE — enhance query with hypothetical FR sentence ────────────
    from services.retrieval.hyde import get_enhancer
    hyde_query = get_enhancer().enhance_sync(
        query=query,
        product=product,
        country=country,
        hts_chapter=hts_chapter or None,
    )
    logger.info("policy_agent_hyde=%s", hyde_query[:120])

    # ── Step 2: Hybrid retrieval (dense + BM25 + RRF) ─────────────────────────
    # Use HTS chapter filter for precision. HybridRetriever is a singleton —
    # BM25 index is built once at startup, not per-request.
    from services.retrieval.hybrid import get_retriever
    retriever = get_retriever()

    # Pass source filter based on country:
    # China queries → prioritise USTR (Section 301 notices)
    # All others → no source filter (CBP rulings may be relevant for any country)
    country_lower = (state.get("country") or "").lower().strip()
    source_filter = "USTR" if country_lower in ("china", "prc") else None

    chunks = retriever.search_policy(
        query=hyde_query,
        hts_chapter=hts_chapter or None,
        source=source_filter,
        top_k=TOP_K,
    )

    # If filtered returns nothing, retry unfiltered
    if not chunks:
        logger.info("policy_agent_filter_empty hts=%s source=%s — retrying unfiltered",
                    hts_code, source_filter)
        chunks = retriever.search_policy(query=hyde_query, top_k=TOP_K)

    if not chunks:
        logger.warning("policy_agent_no_chunks product=%s hts=%s", product, hts_code)
        return {"policy_chunks": [], "policy_summary": "No policy context found."}

    # ── Step 3: LLM synthesis with numbered citations ─────────────────────────
    from services.llm.router import get_router, TaskType
    router = get_router()

    context_block, index_to_doc = _build_numbered_context(chunks)
    user_content = (
        f"Question: {query}\n\n"
        f"Numbered excerpts:\n{context_block}"
        f"\n\n{POLICY_PROMPT_SUFFIX}"
    )

    try:
        loop = asyncio.new_event_loop()
        try:
            resp = loop.run_until_complete(
                router.complete(
                    task=TaskType.POLICY_ANALYSIS,
                    messages=[{"role": "user", "content": user_content}],
                )
            )
        finally:
            loop.close()
        raw_summary = resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error("policy_llm_failed error=%s", e)
        raw_summary = "Policy synthesis failed."
        index_to_doc = {}

    # ── Step 4: Resolve + validate citations ──────────────────────────────────
    resolved_summary, invalid = _resolve_citations(raw_summary, index_to_doc)
    if invalid:
        logger.warning("policy_invalid_citations invalid=%s", invalid)
        resolved_summary = re.sub(r"\[INVALID-\d+\]", "", resolved_summary).strip()

    logger.info("policy_agent_done chunks=%d invalid_citations=%d", len(chunks), len(invalid))

    result = {"policy_chunks": chunks, "policy_summary": resolved_summary}
    _cache_set(hts_code, query, result)
    return result