"""
Policy Agent — Pipeline Step 4

Retrieves policy context from Chroma (USTR, CBP, USITC, ITA, EOP chunks) plus
Snowflake notice–HTS fallbacks for those same sources where tables exist.

  1. HyDEQueryEnhancer.enhance_sync()  — hypothetical FR-style query text
  2. HybridRetriever.search_policy()   — main hybrid search + per-source legs
  3. fetch_all_hts_linked_policy_chunks — exhaustive Snowflake HTS-linked load
  4. _bm25_rerank_exhaustive — re-rank exhaustive chunks by BM25 relevance (best 5, not first 5)
  5. _resolve_xrefs — cross-reference resolution: scan chunk text for FR doc refs, fetch those docs
  6. ModelRouter(POLICY_ANALYSIS) — LLM synthesis with numbered citations

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
SOURCE_LEG_K = 6
POLICY_MERGE_MAX = 28
POLICY_CONTEXT_MAX_CHARS = int(os.environ.get("POLICY_CONTEXT_MAX_CHARS", "600000"))
POLICY_CHUNK_TEXT_MAX_CHARS = max(120, int(os.environ.get("POLICY_CHUNK_TEXT_MAX_CHARS", "520")))
CACHE_TTL = 21_600  # 6 h

# Cross-reference patterns to detect FR document references in chunk text
XREF_PATTERNS = [
    r"Federal Register[,\s]+Vol\.?\s*\d+[,\s]+No\.?\s*\d+",
    r"FR Doc\.?\s*([\w\-]+)",
    r"(\d{4}-\d{5,6})",  # document number pattern like 2024-21217
    r"(\d{2} FR \d+)",
    r"document number\s+([\w\-]+)",
    r"Docket\s+No\.?\s*([\w\-]+)",
]

POLICY_PROMPT_SUFFIX = """
CITATION RULES (strictly enforced):
1. Cite using the bracketed index: [1], [2] — exactly as shown in the excerpts
2. Every factual claim needs a citation — no unsourced statements
3. Do NOT cite an index not present in the list
4. If a CBP ruling covers country-of-origin, say so explicitly
5. If context is insufficient: "Insufficient policy context for this query."
6. Use NO knowledge outside the provided documents
7. Maximum 8-12 sentences; mention each distinct notice (agency + date) when relevant.
8. The CONFIRMED RATE section above is authoritative — always reference those confirmed figures when stating duty rates."""


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


# ── Chunk utilities ────────────────────────────────────────────────────────────

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


def _bm25_rerank_exhaustive(
    chunks: List[Dict[str, Any]],
    query: str,
    top_n: int = 10,
    min_score_ratio: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Re-rank exhaustive HTS-linked chunks by BM25 relevance against the query.
    Returns top_n most relevant chunks above a minimum relevance threshold.

    min_score_ratio: chunks scoring below (max_score * min_score_ratio) are dropped.
    This removes low-relevance chunks from large documents like USMCA rules
    that happen to mention HTS codes but are not tariff policy relevant.

    Falls back to date-sorted order if BM25 fails.
    """
    if not chunks or not query:
        # Fallback: sort by publication date descending (most recent first)
        return sorted(
            chunks,
            key=lambda x: x.get("publication_date") or "",
            reverse=True,
        )[:top_n]
    try:
        from rank_bm25 import BM25Okapi
        tokenized = [
            (c.get("chunk_text") or "").lower().split()
            for c in chunks
        ]
        valid = [(i, t) for i, t in enumerate(tokenized) if t]
        if not valid:
            return chunks[:top_n]
        indices, valid_tokens = zip(*valid)
        bm25 = BM25Okapi(list(valid_tokens))
        scores = bm25.get_scores(query.lower().split())

        # Build (original_index, score) pairs for all valid chunks
        scored = [(indices[j], scores[j]) for j in range(len(indices))]

        # Drop chunks below relevance threshold
        max_score = max(s for _, s in scored) if scored else 0
        threshold = max_score * min_score_ratio if max_score > 0 else 0
        scored = [(i, s) for i, s in scored if s >= threshold]

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        top_indices = [i for i, _ in scored[:top_n]]
        result = [chunks[i] for i in top_indices]

        # Log score distribution for debugging
        scores_used = [s for _, s in scored[:top_n]]
        logger.info(
            "bm25_rerank exhaustive=%d above_threshold=%d top_n=%d "
            "max_score=%.1f min_used=%.1f",
            len(chunks), len(scored), len(result),
            max_score, min(scores_used) if scores_used else 0,
        )
        return result
    except Exception as e:
        logger.warning("bm25_rerank_failed error=%s — using date-sort fallback", e)
        return sorted(
            chunks,
            key=lambda x: x.get("publication_date") or "",
            reverse=True,
        )[:top_n]


def _extract_xref_doc_numbers(chunks: List[Dict[str, Any]]) -> Set[str]:
    """
    Scan chunk text for cross-references to other FR documents.
    Returns set of document numbers found.
    Filters to realistic FR doc number format: YYYY-NNNNN
    """
    found: Set[str] = set()
    doc_num_pattern = re.compile(r"\b(\d{4}-\d{4,6})\b")
    for chunk in chunks:
        text = chunk.get("chunk_text") or ""
        for m in doc_num_pattern.finditer(text):
            candidate = m.group(1)
            # Basic sanity: year between 2000-2030
            year = int(candidate.split("-")[0])
            if 2000 <= year <= 2030:
                found.add(candidate)
    return found


def _resolve_xrefs(
    chunks: List[Dict[str, Any]],
    existing_doc_numbers: Set[str],
    max_total: int = 15,
) -> List[Dict[str, Any]]:
    """
    Cross-reference resolution step.

    1. Scan chunk text for FR document number patterns
    2. For each referenced doc not already in corpus, check if it exists in Snowflake
    3. If found, fetch its chunks from Snowflake and append
    4. Deduplicate by chunk_id. Cap total at max_total chunks added.

    Logs each resolved/missing cross-reference.
    Never fails — errors are caught and logged.
    """
    if not chunks:
        return chunks

    xref_docs = _extract_xref_doc_numbers(chunks)
    new_docs = xref_docs - existing_doc_numbers
    if not new_docs:
        return chunks

    logger.info("xref_resolution found=%d new_docs=%d", len(xref_docs), len(new_docs))

    existing_chunk_ids = {_chunk_key(c) for c in chunks}
    added_count = 0
    result = list(chunks)

    try:
        conn = tools._sf()
        cur = conn.cursor()

        for doc_num in sorted(new_docs):
            if added_count >= max_total:
                break

            # Check which table this doc lives in
            source = None
            for table, src_label in [
                ("TARIFFIQ.RAW.FEDERAL_REGISTER_NOTICES", "USTR"),
                ("TARIFFIQ.RAW.CBP_FEDERAL_REGISTER_NOTICES", "CBP"),
                ("TARIFFIQ.RAW.ITC_DOCUMENTS", "USITC"),
                ("TARIFFIQ.RAW.EOP_DOCUMENTS", "EOP"),
                ("TARIFFIQ.RAW.ITA_FEDERAL_REGISTER_NOTICES", "ITA"),
            ]:
                try:
                    cur.execute(
                        f"SELECT 1 FROM {table} WHERE document_number = %s LIMIT 1",
                        (doc_num,),
                    )
                    if cur.fetchone():
                        source = src_label
                        break
                except Exception:
                    continue

            if not source:
                logger.debug("xref_missing doc=%s", doc_num)
                continue

            # Fetch chunks for this doc
            chunk_table_map = {
                "USTR": ("TARIFFIQ.RAW.CHUNKS", "TARIFFIQ.RAW.FEDERAL_REGISTER_NOTICES"),
                "CBP": ("TARIFFIQ.RAW.CBP_CHUNKS", "TARIFFIQ.RAW.CBP_FEDERAL_REGISTER_NOTICES"),
                "USITC": ("TARIFFIQ.RAW.ITC_CHUNKS", "TARIFFIQ.RAW.ITC_DOCUMENTS"),
                "EOP": ("TARIFFIQ.RAW.EOP_CHUNKS", "TARIFFIQ.RAW.EOP_DOCUMENTS"),
            }
            if source not in chunk_table_map:
                continue

            chunk_tbl, notice_tbl = chunk_table_map[source]
            try:
                cur.execute(
                    f"""
                    SELECT c.chunk_id, c.chunk_text, c.document_number, c.section,
                           f.title, f.publication_date::VARCHAR
                    FROM {chunk_tbl} c
                    LEFT JOIN {notice_tbl} f ON c.document_number = f.document_number
                    WHERE c.document_number = %s AND c.chunk_text IS NOT NULL
                    ORDER BY c.chunk_index ASC NULLS LAST
                    LIMIT 5
                    """,
                    (doc_num,),
                )
                new_chunks = cur.fetchall()
                xref_added = 0
                for row in new_chunks:
                    if added_count >= max_total:
                        break
                    chunk_id, chunk_text, doc_number, section, title, pub_date = row
                    if not chunk_id or not chunk_text:
                        continue
                    c = {
                        "chunk_id": str(chunk_id),
                        "chunk_text": str(chunk_text),
                        "document_number": str(doc_number or ""),
                        "section": str(section or ""),
                        "title": str(title or ""),
                        "publication_date": str(pub_date or ""),
                        "source": source,
                        "retrieval_method": "xref_resolved",
                    }
                    k = _chunk_key(c)
                    if k not in existing_chunk_ids:
                        existing_chunk_ids.add(k)
                        result.append(c)
                        added_count += 1
                        xref_added += 1
                if xref_added:
                    logger.info("xref_resolved doc=%s source=%s chunks_added=%d", doc_num, source, xref_added)
                else:
                    logger.debug("xref_resolved_no_new_chunks doc=%s", doc_num)
            except Exception as e:
                logger.debug("xref_fetch_error doc=%s error=%s", doc_num, e)

        cur.close()
        conn.close()
    except Exception as e:
        logger.warning("xref_resolution_error error=%s", e)

    return result


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
        raw = chunk.get("chunk_text", "") or ""
        text = raw[:POLICY_CHUNK_TEXT_MAX_CHARS]
        if len(raw) > POLICY_CHUNK_TEXT_MAX_CHARS:
            text += " …"
        lines.append(f"{header}\n{text}")
        index_to_doc[i] = doc
    return "\n\n".join(lines), index_to_doc


def _approx_numbered_chunk_chars(c: Dict[str, Any]) -> int:
    raw = c.get("chunk_text") or ""
    body = min(len(raw), POLICY_CHUNK_TEXT_MAX_CHARS)
    if len(raw) > POLICY_CHUNK_TEXT_MAX_CHARS:
        body += 2
    return body + 280


def _trim_chunks_for_policy_llm(
    chunks: List[Dict[str, Any]], max_chars: int
) -> Tuple[List[Dict[str, Any]], str]:
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
        f"\n[Truncated: sent {len(out)} of {len(chunks)} chunks "
        f"within POLICY_CONTEXT_MAX_CHARS={max_chars}.]\n"
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

    # ── Step 1: HyDE ──────────────────────────────────────────────────────────
    from services.retrieval.hyde import get_enhancer
    hyde_query = get_enhancer().enhance_sync(
        query=query,
        product=product,
        country=country,
        hts_chapter=hts_chapter or None,
    )
    logger.info("policy_agent_hyde=%s", hyde_query[:120])

    # ── Step 2: Hybrid retrieval — multi-agency round-robin ───────────────────
    from services.retrieval.hybrid import get_retriever
    retriever = get_retriever()

    main_hybrid = retriever.search_policy(
        query=hyde_query, hts_chapter=hts_chapter or None, source=None, top_k=TOP_K,
    )
    cbp_hybrid = retriever.search_policy(
        query=hyde_query, hts_chapter=hts_chapter or None, source="CBP", top_k=SOURCE_LEG_K,
    )
    usitc_hybrid = retriever.search_policy(
        query=hyde_query, hts_chapter=hts_chapter or None, source="USITC", top_k=SOURCE_LEG_K,
    )
    ita_hybrid = retriever.search_policy(
        query=hyde_query, hts_chapter=hts_chapter or None, source="ITA", top_k=SOURCE_LEG_K,
    )
    eop_hybrid = retriever.search_policy(
        query=hyde_query, hts_chapter=hts_chapter or None, source="EOP", top_k=SOURCE_LEG_K,
    )
    chapter_scan: List[Dict[str, Any]] = []
    if hts_chapter:
        chapter_scan = retriever.search_policy(
            query=f"import tariff duty classification Federal Register HTS chapter {hts_chapter} {product}",
            hts_chapter=hts_chapter, source=None, top_k=8,
        )

    chunks = _merge_policy_chunks_round_robin(
        main_hybrid, cbp_hybrid, usitc_hybrid, ita_hybrid, eop_hybrid, chapter_scan,
    )

    if not chunks:
        logger.info("policy_agent_filter_empty hts=%s — retrying unfiltered", hts_code)
        chunks = retriever.search_policy(query=hyde_query, top_k=TOP_K)

    # ── Step 2b: Exhaustive Snowflake HTS-linked load — re-ranked by BM25 ────
    # Seed with notice_doc from Step 5 (adder_rate_agent) if available
    # This prioritizes the FR document that sourced the adder rate in retrieval
    notice_doc = state.get("notice_doc")
    exhaustive: List[Dict[str, Any]] = []
    if hts_code:
        raw_exhaustive = tools.fetch_all_hts_linked_policy_chunks(hts_code)
        if raw_exhaustive:
            bm25_query = f"{product} {hts_code} tariff duty {country}"
            exhaustive = _bm25_rerank_exhaustive(raw_exhaustive, bm25_query, top_n=10)
            if notice_doc and exhaustive:
                priority = [c for c in exhaustive if c.get("document_number") == notice_doc]
                rest = [c for c in exhaustive if c.get("document_number") != notice_doc]
                exhaustive = priority + rest
            logger.info(
                "policy_agent_hts_exhaustive raw=%d reranked=%d",
                len(raw_exhaustive), len(exhaustive),
            )

    if exhaustive:
        ex_ids = {c.get("chunk_id") for c in exhaustive if c.get("chunk_id")}
        vector_extra = [c for c in chunks if c.get("chunk_id") not in ex_ids]
        chunks = exhaustive + vector_extra
        logger.info(
            "policy_agent_merged exhaustive=%d vector_supplement=%d total=%d",
            len(exhaustive), len(vector_extra), len(chunks),
        )

    if not chunks:
        logger.warning("policy_agent_no_chunks product=%s hts=%s", product, hts_code)
        return {"policy_chunks": [], "policy_summary": "No policy context found."}

    # ── Step 2b2: Drop chunks from wrong HTS chapter ─────────────────────────────
    if hts_chapter:
        before = len(chunks)
        country_lower = (country or "").lower().strip()
        is_china = country_lower in ("china", "prc", "people's republic of china")
        is_india = "india" in country_lower

        def _keep(c):
            c_hts = (c.get("hts_code") or "").replace(".","")[:2]
            c_chap = (c.get("hts_chapter") or "").strip().lstrip("0")
            title = (c.get("title") or "").lower()

            # Drop China-specific docs for non-China queries
            if not is_china and any(kw in title for kw in ["china", "chinese", "people's republic"]):
                return False
            # Drop India-specific docs for non-India queries
            if not is_india and "india" in title and "executive order" in title:
                return False

            # No HTS info on chunk — keep it (generic policy doc)
            if not c_hts and not c_chap:
                return True
            # Has HTS info — must match query chapter
            if c_hts and c_hts != hts_chapter:
                return False
            if c_chap and c_chap != hts_chapter.lstrip("0") and c_chap != str(int(hts_chapter)):
                return False
            return True
        chunks = [c for c in chunks if _keep(c)]
        logger.info("policy_agent_chapter_filter before=%d after=%d chapter=%s", before, len(chunks), hts_chapter)

    # ── Step 2c: Cross-reference resolution ───────────────────────────────────
    # Scan chunk text for FR doc references, fetch those docs if in corpus
    existing_doc_numbers = {c.get("document_number", "") for c in chunks}
    chunks = _resolve_xrefs(chunks, existing_doc_numbers, max_total=15)

    # ── Step 2d: Final relevance filter across ALL chunks ─────────────────────
    # Drop low-relevance chunks regardless of retrieval method.
    # Prevents large documents like 2020-13865 (USMCA) from flooding context
    # when retrieved via hybrid search but irrelevant to the tariff query.
    if len(chunks) > 8:
        bm25_query = f"{product} {hts_code} tariff duty section 301 {country}"
        chunks = _bm25_rerank_exhaustive(chunks, bm25_query, top_n=12, min_score_ratio=0.35)
        logger.info("policy_agent_final_filter chunks_after=%d", len(chunks))

    # ── Step 3: LLM synthesis with numbered citations ─────────────────────────
    from services.llm.router import get_router, TaskType
    router = get_router()

    chunks_for_llm, trim_note = _trim_chunks_for_policy_llm(chunks, POLICY_CONTEXT_MAX_CHARS)
    # Inject confirmed adder rate as authoritative context for LLM
    adder_rate = state.get("adder_rate")
    total_duty = state.get("total_duty")
    adder_doc = state.get("adder_doc")
    base_rate = state.get("base_rate") or 0.0

    confirmed_rate_note = ""
    if adder_rate is not None and total_duty is not None:
        confirmed_rate_note = (
            f"\n\nCONFIRMED RATE (from HTS schedule, authoritative):\n"
            f"  Base MFN rate: {base_rate}%\n"
            f"  Additional duty (Section 301/232/IEEPA): {adder_rate}%\n"
            f"  Total effective rate: {total_duty}%\n"
            f"  Source document: {adder_doc or 'HTS Schedule'}\n"
            f"  When summarizing the duty rate, use these confirmed figures.\n"
        )
    context_block, index_to_doc = _build_numbered_context(chunks_for_llm)
    user_content = (
        f"Question: {query}{confirmed_rate_note}{trim_note}\n\n"
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