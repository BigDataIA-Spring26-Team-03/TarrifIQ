"""
Policy Agent — Pipeline Step 4

Retrieves policy context from Chroma (USTR, CBP, USITC, ITA, EOP chunks) plus
Snowflake notice–HTS fallbacks for those same sources where tables exist.

  1. HyDEQueryEnhancer.enhance_sync()  — hypothetical FR-style query text
  2. HybridRetriever.search_policy()   — main hybrid search + per-source legs
  3. ModelRouter(POLICY_ANALYSIS)      — LLM synthesis with numbered citations

HyDE and HybridRetriever use singletons — instantiated once at startup,
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

TOP_K = 12
SOURCE_LEG_K = 6  # per-agency hybrid leg so CBP/USITC/ITA/EOP are not drowned out by USTR hits
POLICY_MERGE_MAX = 28
# Full HTS-linked corpus can be huge; only the policy-analysis LLM is budgeted (env override).
POLICY_CONTEXT_MAX_CHARS = int(os.environ.get("POLICY_CONTEXT_MAX_CHARS", "600000"))
# Per-chunk slice in the numbered context sent to the policy LLM (full chunk kept in merge until budget).
POLICY_CHUNK_TEXT_MAX_CHARS = max(120, int(os.environ.get("POLICY_CHUNK_TEXT_MAX_CHARS", "520")))
CACHE_TTL = 21_600  # 6 h

POLICY_PROMPT_SUFFIX = """
CITATION RULES (strictly enforced):
1. Cite using the bracketed index: [1], [2] — exactly as shown in the excerpts
2. Every factual claim needs a citation — no unsourced statements
3. Do NOT cite an index not present in the list
4. If a CBP ruling covers country-of-origin, say so explicitly
5. If context is insufficient: "Insufficient policy context for this query."
6. Use NO knowledge outside the provided documents
7. Maximum 8–12 sentences; mention each distinct notice (agency + date) when relevant."""


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

def _chunk_key(c: Dict[str, Any]) -> str:
    cid = c.get("chunk_id")
    if cid is not None and cid != "":
        return f"id:{cid}"
    snippet = (c.get("chunk_text") or "")[:120]
    return f"doc:{c.get('document_number', '')}|sec:{c.get('section', '')}|h:{hash(snippet)}"


def _merge_policy_chunks_round_robin(
    *lists: List[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Interleave lists so merged results span USTR/CBP/chapter scan without flooding duplicates."""
    seen: Set[str] = set()
    out: List[Dict[str, Any]] = []
    lists_n = [x for x in lists if x]
    if not lists_n:
        return []
    max_len = max(len(x) for x in lists_n)
    for i in range(max_len):
        for lst in lists_n:
            if i >= len(lst):
                continue
            c = lst[i]
            k = _chunk_key(c)
            if k in seen:
                continue
            seen.add(k)
            out.append(c)
            if len(out) >= POLICY_MERGE_MAX:
                return out
    return out


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
        raw = chunk.get("chunk_text", "") or ""
        text = raw[:POLICY_CHUNK_TEXT_MAX_CHARS]
        if len(raw) > POLICY_CHUNK_TEXT_MAX_CHARS:
            text += " …"
        lines.append(f"{header}\n{text}")
        index_to_doc[i] = doc
    return "\n\n".join(lines), index_to_doc


def _approx_numbered_chunk_chars(c: Dict[str, Any]) -> int:
    """Rough size of one chunk block in `_build_numbered_context` (header + sliced body)."""
    raw = c.get("chunk_text") or ""
    body = min(len(raw), POLICY_CHUNK_TEXT_MAX_CHARS)
    if len(raw) > POLICY_CHUNK_TEXT_MAX_CHARS:
        body += 2  # ellipsis
    # [n] agency doc | "title[:80]" | pub — buffer for variable header fields
    return body + 280


def _trim_chunks_for_policy_llm(
    chunks: List[Dict[str, Any]], max_chars: int
) -> Tuple[List[Dict[str, Any]], str]:
    """Include chunks in order until character budget is reached (LLM context limit)."""
    total = 0
    out: List[Dict[str, Any]] = []
    for c in chunks:
        t = _approx_numbered_chunk_chars(c)
        if total + t > max_chars and out:
            break
        out.append(c)
        total += t
    omitted = len(chunks) - len(out)
    if not omitted:
        return out, ""
    return out, (
        f"\n[Truncated policy context: sent {len(out)} of {len(chunks)} chunks "
        f"within POLICY_CONTEXT_MAX_CHARS={max_chars}. Increase the env var to include more.]\n"
    )


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
    from services.retrieval.hybrid import get_retriever
    retriever = get_retriever()

    # All sources in Chroma (USTR, CBP, USITC, ITA, EOP) — do not filter to USTR-only
    # (older behavior for China drowned out CBP rulings and USITC notices).
    main_hybrid = retriever.search_policy(
        query=hyde_query,
        hts_chapter=hts_chapter or None,
        source=None,
        top_k=TOP_K,
    )
    cbp_hybrid = retriever.search_policy(
        query=hyde_query,
        hts_chapter=hts_chapter or None,
        source="CBP",
        top_k=SOURCE_LEG_K,
    )
    usitc_hybrid = retriever.search_policy(
        query=hyde_query,
        hts_chapter=hts_chapter or None,
        source="USITC",
        top_k=SOURCE_LEG_K,
    )
    ita_hybrid = retriever.search_policy(
        query=hyde_query,
        hts_chapter=hts_chapter or None,
        source="ITA",
        top_k=SOURCE_LEG_K,
    )
    eop_hybrid = retriever.search_policy(
        query=hyde_query,
        hts_chapter=hts_chapter or None,
        source="EOP",
        top_k=SOURCE_LEG_K,
    )

    chapter_scan: List[Dict[str, Any]] = []
    if hts_chapter:
        chapter_scan = retriever.search_policy(
            query=(
                f"import tariff duty classification Federal Register "
                f"HTS chapter {hts_chapter} {product}"
            ),
            hts_chapter=hts_chapter,
            source=None,
            top_k=8,
        )

    chunks = _merge_policy_chunks_round_robin(
        main_hybrid,
        cbp_hybrid,
        usitc_hybrid,
        ita_hybrid,
        eop_hybrid,
        chapter_scan,
    )

    if not chunks:
        logger.info("policy_agent_filter_empty hts=%s — retrying unfiltered", hts_code)
        chunks = retriever.search_policy(query=hyde_query, top_k=TOP_K)

    # ── Step 2b: Exhaustive Snowflake load — every chunk for every doc linked to this HTS ──
    exhaustive: List[Dict[str, Any]] = []
    if hts_code:
        exhaustive = tools.fetch_all_hts_linked_policy_chunks(hts_code)
    if exhaustive:
        ex_ids = {c.get("chunk_id") for c in exhaustive if c.get("chunk_id")}
        vector_extra = [c for c in chunks if c.get("chunk_id") not in ex_ids]
        chunks = exhaustive + vector_extra
        logger.info(
            "policy_agent_hts_exhaustive exhaustive=%d vector_supplement=%d total=%d",
            len(exhaustive),
            len(vector_extra),
            len(chunks),
        )

    if not chunks:
        logger.warning("policy_agent_no_chunks product=%s hts=%s", product, hts_code)
        return {"policy_chunks": [], "policy_summary": "No policy context found."}

    # ── Step 3: LLM synthesis with numbered citations ─────────────────────────
    from services.llm.router import get_router, TaskType
    router = get_router()

    chunks_for_llm, trim_note = _trim_chunks_for_policy_llm(chunks, POLICY_CONTEXT_MAX_CHARS)
    context_block, index_to_doc = _build_numbered_context(chunks_for_llm)
    user_content = (
        f"Question: {query}{trim_note}\n\n"
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