"""
Synthesis Agent — Pipeline Step 7

Combines upstream outputs into a cited Markdown answer (fixed ## section layout).
LLM call via ModelRouter(TaskType.ANSWER_SYNTHESIS). Validation via tools.py.

Context includes: duty stack, up to 20 FR excerpts (chronological), policy summary,
Snowflake FR notice history for the HTS, Census trade line, and alternative-origin
base rates. Ambiguity is handled upstream (query_agent / Streamlit chips).

Citation validation:
  1. Every FR doc number in the answer must appear in policy_chunks / adder / history
  2. tools.verify_docs_batch() against Snowflake notice tables

Rate record: tools.hts_verify(rate_record_id). Pipeline confidence: HIGH/MEDIUM/LOW.
"""

import asyncio
import logging
import os
import re
import time
from typing import Dict, Any, List, Optional, Set
from urllib.parse import quote

from agents.state import TariffState
from agents import tools

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
RETRY_DELAY = 1.0

SYNTHESIS_CONTEXT_TEMPLATE = """
USER QUERY: {query}
PRODUCT: {product}
COUNTRY OF ORIGIN: {country}
PIPELINE CONFIDENCE: {pipeline_confidence}

HTS CLASSIFICATION: {hts_code} — {hts_description} (confidence: {confidence})

VERIFIED DUTY RATE [HTS code {record_id} — verified in Snowflake HTS_CODES]:
  {rate_line}
  Section 301/IEEPA:     {adder_rate:.2f}% {adder_source}
  Total effective duty:  {total_duty:.2f}%
{footnote_line}
POLICY CHUNKS (full text per chunk until SYNTHESIS_* env budgets; cite ONLY these FR doc numbers as (FR: doc_number), e.g. (FR: 2025-07325): {valid_docs}):
{policy_excerpts}

POLICY ANALYSIS (intermediate summary from retrieval): {policy_summary}

{rate_history_block}

TRADE VOLUME [Census Bureau {period}]: {trade_line}

{top_importers_block}

{comparison_section}

FINAL ANSWER FORMAT (Markdown — use exactly these section headings, in this order):

## 1. Product, origin, and HTS classification
State product, country of origin, HTS code and description, and classification confidence. If the match is broad, say so.

## 2. All US import charges affecting this product
Cover: preferential (FTA) duty vs MFN, base duty, Chapter 99 / Section 301 / 232 / IEEPA or other adders (with FR document numbers from the excerpt list only), and total estimated duty. Explicitly mention import surcharges or trade-remedy charges by name when they appear in the excerpts.

## 3. Policy notices in chronological order
For each materially relevant Federal Register (or CBP/USITC) notice that affects this product or HTS chapter, oldest first: publication or effective context, agency, one or two sentences on what changed, and a link line:
`https://www.federalregister.gov/documents/{{YYYY-XXXXX}}`
Use only document numbers that appear in VALID FR DOCUMENTS / excerpts above. Skip notices that are not substantively about this product or chapter.

## 4. Alternative sourcing countries (indicative baseline rates)
Use the ALTERNATIVE SOURCING block when present: compare MFN/FTA baseline rates for other origins. State clearly that Section 301/232 adders are US-measures and are not recomputed per country here.

## 5. Historical tariff and notice trail
Synthesize the RATE / NOTICE HISTORY block and excerpt dates into a short dated timeline. Link each FR action you mention.

## 6. Census trade snapshot (US imports)
Summarize the latest Census period, import value, and YoY trend from the trade line (for the user’s country of origin when applicable).

## 7. Top US import partners by country (Census)
Using the TOP IMPORT PARTNERS block: report which partner countries account for the largest US import values for this HS over the summed monthly series; for each major partner list cumulative import dollars and the indicative MFN or FTA baseline duty rate for this HTS from the block. Clarify that Section 301 / 232 / IEEPA-style adders are applied at the US border under separate rules and are not recalculated per partner in that table.

If a section has no supporting data in this context, write one honest sentence (do not invent notices, rates, or links).

CONFIDENCE TONE:
- HIGH: direct wording; still cite FR numbers for each policy claim.
- MEDIUM: one short caveat on the weakest data element.
- LOW: flag uncertainty on classification, adder source, or sparse policy excerpts.

Write the final user-visible answer using the section headings above (## 1 through ## 7).
"""


def _dedupe_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Keep ALL chunks for LLM context — different chunks from same doc
    # have different text. Deduplication only happens in _build_citations().
    return chunks if chunks else []


def _sort_chunks_chronologically(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _key(c: Dict[str, Any]) -> tuple:
        d = c.get("publication_date") or ""
        return (d, c.get("document_number") or "")

    return sorted(chunks, key=_key)


def _budgeted_policy_excerpts_for_synthesis(chunks: List[Dict[str, Any]]) -> str:
    """Include full chunk texts up to SYNTHESIS_* env budgets (default: very large)."""
    if not chunks:
        return "No relevant policy documents found."

    max_chars = int(os.environ.get("SYNTHESIS_POLICY_EXCERPT_MAX_CHARS", "600000"))
    max_n = int(os.environ.get("SYNTHESIS_POLICY_MAX_CHUNKS", "600"))

    ordered = _sort_chunks_chronologically(chunks)
    lines: List[str] = []
    total = 0
    n = 0
    for c in ordered:
        if n >= max_n:
            lines.append(
                f"[SYNTHESIS chunk cap reached at SYNTHESIS_POLICY_MAX_CHUNKS={max_n}; "
                "increase env var to include more chunks]"
            )
            break
        txt = (c.get("chunk_text") or "")
        line = "[{doc}] {pub} | {src} | {txt}".format(
            doc=c.get("document_number", "UNKNOWN"),
            pub=c.get("publication_date", "") or "?",
            src=(c.get("source", "") or "").upper(),
            txt=txt,
        )
        if total + len(line) > max_chars and lines:
            remaining = len(ordered) - n
            lines.append(
                f"[SYNTHESIS excerpt budget reached; omitted {remaining} remaining chunks — "
                f"SYNTHESIS_POLICY_EXCERPT_MAX_CHARS={max_chars}]"
            )
            break
        lines.append(line)
        total += len(line)
        n += 1

    return "\n".join(lines)


def _verify_docs(candidates: Set[str]) -> Set[str]:
    if not candidates:
        return set()
    return tools.verify_docs_batch(candidates)


def _preferred_http_url(*vals: Optional[str]) -> str:
    for v in vals:
        s = (v or "").strip()
        if s.startswith(("http://", "https://")):
            return s
    return ""


def _fetch_doc_metadata(doc_numbers: Set[str]) -> Dict[str, Dict]:
    """
    Fetch title, publication_date, FR html links from Snowflake notice tables.
    Returns dict keyed by document_number.
    """
    if not doc_numbers:
        return {}
    try:
        conn = tools._sf()
        cur = conn.cursor()
        meta: Dict[str, Dict[str, Any]] = {}
        docs_list = list(doc_numbers)

        def _load_batch(
            table_sql: str,
            source_tag: str,
            snowflake_table: str,
            ids: List[str],
        ) -> None:
            if not ids:
                return
            ph = ",".join(["%s"] * len(ids))
            cur.execute(
                f"SELECT document_number, title, publication_date, html_url, body_html_url "
                f"FROM {table_sql} "
                f"WHERE document_number IN ({ph})",
                ids,
            )
            for doc_num, title, pub_date, html_url, body_html_url in cur.fetchall():
                url = _preferred_http_url(
                    str(html_url) if html_url is not None else "",
                    str(body_html_url) if body_html_url is not None else "",
                )
                meta[doc_num] = {
                    "title": title or "",
                    "publication_date": str(pub_date) if pub_date else "",
                    "source": source_tag,
                    "snowflake_table": snowflake_table,
                    "html_url": url,
                    "body_html_url": (body_html_url or "").strip()
                    if body_html_url
                    else "",
                }

        _load_batch(
            "TARIFFIQ.RAW.FEDERAL_REGISTER_NOTICES",
            "USTR",
            "TARIFFIQ.RAW.FEDERAL_REGISTER_NOTICES",
            docs_list,
        )

        remaining = [d for d in docs_list if d not in meta]
        _load_batch(
            "TARIFFIQ.RAW.CBP_FEDERAL_REGISTER_NOTICES",
            "CBP",
            "TARIFFIQ.RAW.CBP_FEDERAL_REGISTER_NOTICES",
            remaining,
        )

        remaining = [d for d in docs_list if d not in meta]
        _load_batch(
            "TARIFFIQ.RAW.ITC_DOCUMENTS",
            "USITC",
            "TARIFFIQ.RAW.ITC_DOCUMENTS",
            remaining,
        )

        remaining = [d for d in docs_list if d not in meta]
        _load_batch(
            "TARIFFIQ.RAW.EOP_DOCUMENTS",
            "EOP",
            "TARIFFIQ.RAW.EOP_DOCUMENTS",
            remaining,
        )

        remaining = [d for d in docs_list if d not in meta]
        try:
            _load_batch(
                "TARIFFIQ.RAW.ITA_FEDERAL_REGISTER_NOTICES",
                "ITA",
                "TARIFFIQ.RAW.ITA_FEDERAL_REGISTER_NOTICES",
                remaining,
            )
        except Exception as e:
            logger.debug("fetch_doc_metadata_ita_skip error=%s", e)

        cur.close()
        conn.close()
        return meta
    except Exception as e:
        logger.warning("fetch_doc_metadata_error error=%s", e)
        return {}


def _validate_citations(text: str, valid_docs: Set[str]) -> tuple[bool, Set[str]]:
    if not valid_docs:
        return True, set()
    # Extract doc numbers from both formats:
    #   bare: 2025-07325
    #   prefixed: (FR: 2025-07325)  ← format we now use
    cited: Set[str] = set()
    # Prefixed format first — strip the FR: prefix
    for m in re.finditer(r"\(FR:\s*([\w\-]+)\)", text):
        cited.add(m.group(1).strip())
    # Also catch bare YYYY-NNNNN format (e.g. in section headings or history block)
    for m in re.finditer(r"\b(\d{4}-\d{4,6})\b", text):
        cited.add(m.group(1))
    if not cited:
        return True, set()
    hallucinated = cited - valid_docs
    if hallucinated:
        logger.warning("synthesis_hallucination cited=%s hallucinated=%s", cited, hallucinated)
        return False, hallucinated
    return True, set()


def _compute_confidence(
    classification_confidence: float,
    rate_verified: bool,
    adder_doc_verified: bool,
    fr_docs_verified: bool,
    hitl_was_triggered: bool,
) -> str:
    failures = 0
    if classification_confidence < 0.60:
        failures += 2
    elif classification_confidence < 0.80:
        failures += 1
    if not rate_verified:
        failures += 1
    if not adder_doc_verified:
        failures += 1
    if not fr_docs_verified:
        failures += 1
    if hitl_was_triggered:
        failures += 1
    if failures == 0:
        return "HIGH"
    elif failures <= 2:
        return "MEDIUM"
    return "LOW"


def _format_rate_history_block(history: Optional[List[Dict[str, Any]]]) -> str:
    if not history:
        return (
            "RATE / NOTICE HISTORY (Snowflake NOTICE_HTS_CODES → FR metadata, chronological):\n"
            "  (No indexed rows returned for this HTS in the current query window.)\n"
        )
    lines = [
        "RATE / NOTICE HISTORY (Snowflake — FR notices linked to this HTS; oldest first for narrative):"
    ]
    for row in sorted(
        history,
        key=lambda x: (x.get("publication_date") or "", x.get("document_number") or ""),
    ):
        doc = row.get("document_number") or ""
        src = row.get("source") or ""
        title = (row.get("title") or "").replace("\n", " ")[:200]
        pub = row.get("publication_date") or ""
        url = f"https://www.federalregister.gov/documents/{doc}" if doc else ""
        lines.append(f"  - {pub} | {src} | ({doc}) {title}")
        if url:
            lines.append(f"    {url}")
    return "\n".join(lines) + "\n"


def _format_top_importers_block(rows: Optional[List[Dict[str, Any]]]) -> str:
    if not rows:
        return (
            "TOP IMPORT PARTNERS (Census HS imports — summed monthly GEN_VAL_MO over trailing months):\n"
            "  (No partner-country rows returned; Census may have no detail for this HS level or period.)\n"
        )
    months = int(rows[0].get("months_in_sample") or 24)
    lines = [
        f"TOP IMPORT PARTNERS (Census: summed monthly US import value GEN_VAL_MO over ~{months} months; "
        "ranked by partner country). Baseline = HTS MFN or FTA rate from Snowflake for that origin — excludes Section 301/232/IEEPA add-on layers:"
    ]
    for i, r in enumerate(rows, 1):
        name = r.get("census_country_name") or r.get("lookup_country") or "unknown"
        usd = float(r.get("imports_usd_trailing") or 0)
        br = r.get("base_rate")
        mfn = r.get("mfn_rate")
        fta = r.get("fta_program")
        applied = r.get("fta_applied")
        cc = r.get("cty_code") or ""
        if br is None:
            rate_txt = "baseline rate lookup failed"
        else:
            try:
                mfn_f = float(mfn) if mfn is not None else float(br)
                extra = f" — MFN {mfn_f:.2f}%"
                if applied and fta:
                    extra += f", FTA applied: {fta}"
                rate_txt = f"baseline {float(br):.2f}%{extra}"
            except (TypeError, ValueError):
                rate_txt = "see HTS"
        lines.append(
            f"  {i}. {name} [Census CTY_CODE {cc}] — imports ~USD {usd:,.0f} — {rate_txt}"
        )
    return "\n".join(lines) + "\n"


def _build_context(
    state: TariffState,
    deduped: List[Dict],
    valid_docs: Set[str],
    rate_change_history: Optional[List[Dict[str, Any]]] = None,
    *,
    country_comparison: Optional[List[Dict[str, Any]]] = None,
    top_importers_block: str = "",
) -> str:
    hts_code = state.get("hts_code") or "Unknown"
    record_id = state.get("rate_record_id") or hts_code
    rate_source = state.get("rate_source") or "TARIFFIQ.RAW.HTS_CODES"
    base_rate = state.get("base_rate") or 0.0
    adder_rate = state.get("adder_rate") or 0.0
    total_duty = state.get("total_duty") or 0.0
    adder_doc = state.get("adder_doc")

    adder_source = f"(sourced from {adder_doc})" if adder_doc else "(source: not identified)"

    fta_applied = state.get("fta_applied", False)
    fta_program = state.get("fta_program")
    mfn_rate = state.get("mfn_rate") or base_rate
    hts_footnotes = state.get("hts_footnotes") or []

    if fta_applied and fta_program:
        rate_line = f"{fta_program} preferential rate {base_rate:.2f}% (MFN would be {mfn_rate:.2f}%)"
    else:
        rate_line = f"Base MFN {base_rate:.2f}%"

    footnote_line = ("HTS FOOTNOTES: " + "; ".join(hts_footnotes[:3]) + "\n") if hts_footnotes else ""

    excerpts = (
        _budgeted_policy_excerpts_for_synthesis(deduped)
        if deduped
        else "No relevant policy documents found."
    )

    rate_history_block = _format_rate_history_block(rate_change_history)

    trade_suppressed = state.get("trade_suppressed")
    import_value = state.get("import_value_usd")
    period = state.get("trade_period", "")
    trend_label = state.get("trade_trend_label")

    if trade_suppressed is False and import_value is not None:
        trade_line = f"${import_value:,.0f} USD"
        if trend_label:
            trade_line += f" | {trend_label}"
    elif trade_suppressed is False:
        trade_line = "Value suppressed by Census Bureau"
    else:
        trade_line = "Not available from Census Bureau"

    # Compute pipeline confidence for context
    conf_val = state.get("pipeline_confidence") or "UNKNOWN"

    # Build country comparison section if available (passed in — not on state until after synthesis)
    country_comp = country_comparison if country_comparison is not None else (
        state.get("country_comparison") or []
    )
    if country_comp:
        comp_lines = []
        for c in country_comp[:5]:
            fta = f" ({c['fta_program']})" if c.get("fta_program") else ""
            comp_lines.append(f"  {c['country']}: {c['base_rate']:.1f}% base rate{fta}")
        comparison_section = "ALTERNATIVE SOURCING (base MFN rates only):\n" + "\n".join(comp_lines)
    else:
        comparison_section = ""

    return SYNTHESIS_CONTEXT_TEMPLATE.format(
        query=state.get("query", ""),
        product=state.get("product", "Unknown"),
        country=state.get("country", "Not specified"),
        pipeline_confidence=conf_val,
        comparison_section=comparison_section,
        hts_code=hts_code,
        hts_description=state.get("hts_description", ""),
        confidence=f"{(state.get('classification_confidence') or 0):.0%}",
        record_id=record_id,
        rate_line=rate_line,
        base_rate=base_rate,
        adder_rate=adder_rate,
        adder_source=adder_source,
        total_duty=total_duty,
        footnote_line=footnote_line,
        valid_docs=", ".join(sorted(valid_docs)) if valid_docs else "none",
        policy_excerpts=excerpts,
        policy_summary=state.get("policy_summary", ""),
        rate_history_block=rate_history_block,
        period=period,
        trade_line=trade_line,
        top_importers_block=top_importers_block,
    )


# Agency metadata for citation enrichment
_SOURCE_AGENCY_MAP = {
    "USTR":  {"agency": "Office of the U.S. Trade Representative", "agency_short": "USTR"},
    "CBP":   {"agency": "U.S. Customs and Border Protection",       "agency_short": "CBP"},
    "USITC": {"agency": "U.S. International Trade Commission",      "agency_short": "USITC"},
    "ITC":   {"agency": "U.S. International Trade Commission",      "agency_short": "USITC"},
    "EOP":   {"agency": "Executive Office of the President",        "agency_short": "EOP"},
    "ITA":   {"agency": "International Trade Administration",        "agency_short": "ITA"},
}


def _fr_url(doc_number: str) -> str:
    """
    Build a Federal Register URL for a document number.
    Uses the /search endpoint with the doc number as query — this reliably
    surfaces the correct document as the top result without needing the slug.
    """
    dn = (doc_number or "").strip()
    if not dn:
        return ""
    # FR search with exact doc number surfaces the correct doc as top result
    return f"https://www.federalregister.gov/search?query={quote(dn, safe='')}"


def _usitc_hts_lookup_url(hts_code: Optional[str]) -> Optional[str]:
    """HTS_CODES has no public URL column; link to USITC HTS Online search."""
    hc = (hts_code or "").strip()
    if not hc:
        return None
    return f"https://hts.usitc.gov/?query={quote(hc, safe='')}"


def _build_citations(
    state: TariffState,
    deduped: List[Dict],
    valid_docs: Set[str],
    doc_metadata: Optional[Dict[str, Dict]] = None,
    rate_change_history: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    citations = []
    seen: Set[str] = set()
    meta = doc_metadata or {}

    record_id = state.get("rate_record_id")
    if record_id and record_id not in seen:
        seen.add(record_id)
        fta_applied = state.get("fta_applied", False)
        fta_program = state.get("fta_program")
        rate_text = (
            f"HTS {state.get('hts_code')} — "
            f"{fta_program + ' rate' if fta_applied and fta_program else 'base MFN'} "
            f"{state.get('base_rate', 0):.2f}% + "
            f"adder {state.get('adder_rate', 0):.2f}% = "
            f"total {state.get('total_duty', 0):.2f}%"
        )
        citations.append({
            "type": "snowflake_hts",
            "id": record_id,
            "agency": "USITC",
            "agency_short": "USITC",
            "title": f"HTS {state.get('hts_code')} — {state.get('hts_description', '')}",
            "text": rate_text,
            "source": state.get("rate_source") or "TARIFFIQ.RAW.HTS_CODES",
            "url": _usitc_hts_lookup_url(state.get("hts_code")),
            "effective_date": None,
        })

    adder_doc = state.get("adder_doc")
    if adder_doc and adder_doc not in seen:
        seen.add(adder_doc)
        adder_meta = meta.get(adder_doc, {})
        # Plain English explanation for Chapter 99 codes
        if adder_doc.startswith("9903"):
            ch99_title = f"Chapter 99 surcharge code {adder_doc}"
            ch99_text = f"Section 301/IEEPA adder rate — {adder_meta.get('title', adder_doc)}"
            # Chapter 99 codes are HTS codes — link to USITC HTS lookup, not FR
            adder_url = _usitc_hts_lookup_url(adder_doc) or ""
        else:
            ch99_title = adder_meta.get("title", f"FR {adder_doc}")
            ch99_text = f"Section 301/IEEPA adder rate — method: {state.get('adder_method', 'unknown')}"
            # Use Snowflake html_url if available, else build FR search URL
            adder_url = (adder_meta.get("html_url") or "").strip()
            if not adder_url:
                adder_url = _fr_url(adder_doc)
        citations.append({
            "type": "adder_source",
            "id": adder_doc,
            "agency": "Office of the U.S. Trade Representative",
            "agency_short": "USTR",
            "title": ch99_title,
            "text": ch99_text,
            "source": adder_meta.get("snowflake_table") or "federalregister.gov",
            "url": adder_url or None,
            "effective_date": adder_meta.get("publication_date"),
        })

    query_hts_code = (state.get("hts_code") or "").strip()
    query_hts_chapter = query_hts_code[:2]
    is_china_query = (state.get("country") or "").lower().strip() in (
        "china", "prc", "people's republic of china"
    )

    # Ground truth: docs explicitly linked to this HTS in Snowflake
    hts_linked_docs: Set[str] = set()
    if query_hts_code:
        try:
            hts_linked_docs = tools.fetch_doc_numbers_for_hts(query_hts_code)
            parts = query_hts_code.split(".")
            while len(parts) > 1:
                parts = parts[:-1]
                hts_linked_docs |= tools.fetch_doc_numbers_for_hts(".".join(parts))
        except Exception:
            pass

    # Docs from rate_change_history are explicitly fetched for this HTS — always keep
    history_docs: Set[str] = set()
    if rate_change_history:
        history_docs = {r.get("document_number") for r in rate_change_history if r.get("document_number")}

    def _is_relevant_citation(doc: str, chunk: dict, doc_meta: dict) -> bool:
        # Always keep adder doc
        if doc == adder_doc:
            return True

        title = (doc_meta.get("title") or chunk.get("title") or "").lower()

        # Drop China 301 docs for non-China queries
        if not is_china_query:
            if any(s in title for s in [
                "china's acts", "china's policies",
                "people's republic of china",
                "opioid supply chain in the people's republic",
            ]):
                return False

        # Always drop antidumping/CVD, trade remedy investigations, and
        # border enforcement docs — never relevant to import duty rates
        if any(s in title for s in [
            "less than fair value", "less-than-fair-value",
            "antidumping", "countervailing",
            "dumping margin", "sales at less than",
            "opioid supply chain", "synthetic opioid",
            "flow of illicit drugs", "illicit drugs across",
            "northern border", "southern border",
            "initiation of", "covered merchandise referral",
            "affirmative determination", "negative determination",
            "preliminary determination", "final determination",
        ]):
            return False

        # Drop USMCA implementing regulations (rules of origin, not tariff rates)
        if "implementing regulations" in title and any(
            s in title for s in ["textile", "apparel", "automotive", "usmca"]
        ):
            return False

        # Keep if Snowflake HTS linkage confirms this doc is about this product
        if hts_linked_docs and doc in hts_linked_docs:
            return True

        # Keep if explicitly fetched from rate_change_history for this HTS
        if doc in history_docs:
            return True

        # For China queries: keep Section 301/IEEPA policy docs even without
        # HTS linkage in Snowflake — Section 301 lists are ingested at heading
        # level and often not linked to individual 10-digit subheadings
        if is_china_query and any(s in title for s in [
            "section 301", "china's acts", "china's policies",
            "people's republic of china", "technology transfer",
            "intellectual property", "reciprocal tariff",
        ]):
            return True

        # If chunk has explicit HTS chapter metadata, use it
        chunk_hts = (chunk.get("hts_code") or "").replace(".", "")[:2]
        chunk_chap = (chunk.get("hts_chapter") or "").strip().lstrip("0")
        if chunk_hts and query_hts_chapter:
            return chunk_hts == query_hts_chapter
        if chunk_chap and query_hts_chapter:
            return chunk_chap == query_hts_chapter.lstrip("0") or \
                   chunk_chap == str(int(query_hts_chapter))

        # No linkage data — drop it
        # The policy_agent should have filtered these out but if it didn't,
        # we'd rather show fewer citations than irrelevant ones
        return False

    # Dedup by doc number here (not in _dedupe_chunks)
    for chunk in deduped:
        doc = chunk.get("document_number")
        if doc and doc not in seen and doc in valid_docs:
            doc_meta = meta.get(doc, {})

            if not _is_relevant_citation(doc, chunk, doc_meta):
                logger.debug("citations_skip_irrelevant doc=%s", doc)
                continue

            seen.add(doc)
            src = chunk.get("source", "USTR").upper()
            agency_meta = _SOURCE_AGENCY_MAP.get(src, {"agency": src, "agency_short": src})
            chunk_text = chunk.get("chunk_text", "") or ""
            text = chunk_text[:120].strip() if chunk_text else ""
            # URL priority: Snowflake html_url → FR search URL (always populated)
            doc_url = (doc_meta.get("html_url") or "").strip()
            if not doc_url:
                doc_url = _fr_url(doc)
            sf_tbl = doc_meta.get("snowflake_table")
            if not sf_tbl:
                sf_tbl = (
                    "TARIFFIQ.RAW.CBP_FEDERAL_REGISTER_NOTICES"
                    if src == "CBP"
                    else "TARIFFIQ.RAW.FEDERAL_REGISTER_NOTICES"
                )
            citations.append({
                "type": "federal_register",
                "id": doc,
                "agency": agency_meta["agency"],
                "agency_short": agency_meta["agency_short"],
                "title": doc_meta.get("title", f"Federal Register Notice {doc}"),
                "text": text,
                "source": sf_tbl,
                "url": doc_url,
                "effective_date": doc_meta.get("publication_date", ""),
            })

    if not state.get("trade_suppressed") and state.get("import_value_usd") is not None:
        period = state.get("trade_period", "")
        cid = f"census_{period}"
        if cid not in seen:
            seen.add(cid)
            trend = state.get("trade_trend_label")
            text = f"US imports {period} — ${state.get('import_value_usd', 0):,.0f} USD"
            if trend:
                text += f" ({trend})"
            citations.append({
                "type": "census_bureau",
                "id": cid,
                "agency": "U.S. Census Bureau",
                "agency_short": "Census",
                "text": text,
                "source": "api.census.gov",
                "url": "https://www.census.gov/foreign-trade/",
            })

    return citations


def _build_country_comparison(
    hts_code: Optional[str],
    current_country: Optional[str],
) -> List[Dict[str, Any]]:
    """Country comparison disabled — hardcoded adder tiers removed.
    Will be re-enabled once IEEPA EO data is ingested into NOTICE_HTS_CODES."""
    return []


def run_synthesis_agent(state: TariffState) -> Dict[str, Any]:
    logger.info("synthesis_agent_start query=%s", state.get("query", "")[:80])

    raw_chunks = state.get("policy_chunks") or []
    deduped = _sort_chunks_chronologically(_dedupe_chunks(raw_chunks))

    hts_for_history = state.get("hts_code") or state.get("rate_record_id")
    rate_change_history: Optional[List[Dict[str, Any]]] = None
    if hts_for_history:
        rate_change_history = tools.fetch_rate_change_history(
            hts_for_history, country=state.get("country")
        )

    # Candidate docs: policy chunks + adder_doc + history rows
    candidate_docs: Set[str] = {
        c.get("document_number", "") for c in deduped if c.get("document_number")
    }
    adder_doc = state.get("adder_doc")
    if adder_doc:
        candidate_docs.add(adder_doc)
    if rate_change_history:
        for row in rate_change_history:
            dn = row.get("document_number")
            if dn:
                candidate_docs.add(dn)

    # Verify all docs against Snowflake
    valid_docs = _verify_docs(candidate_docs)

    # Fetch metadata for citation enrichment
    doc_metadata = _fetch_doc_metadata(valid_docs)
    if candidate_docs - valid_docs:
        logger.warning("synthesis_unverified_docs=%s", candidate_docs - valid_docs)

    # Verify rate record
    record_id = state.get("rate_record_id")
    rate_verified = tools.hts_verify(record_id) if record_id else False
    adder_doc_verified = adder_doc in valid_docs if adder_doc else True

    hts_comp = state.get("hts_code") or state.get("rate_record_id")
    country_comparison_pre = (
        _build_country_comparison(
            hts_code=hts_comp,
            current_country=state.get("country"),
        )
        if hts_comp
        else []
    )
    top_importers_pre: List[Dict[str, Any]] = []
    if hts_comp:
        top_importers_pre = tools.fetch_top_importer_countries(
            hts_comp, months=24, top_n=8
        )
    top_importers_block = _format_top_importers_block(top_importers_pre)

    context = _build_context(
        state,
        deduped,
        valid_docs,
        rate_change_history,
        country_comparison=country_comparison_pre,
        top_importers_block=top_importers_block,
    )

    # Prepend intent hint for rate-change queries so LLM leads with history
    query_intent = state.get("query_intent")
    if query_intent == "rate_change":
        context = (
            "QUERY INTENT: rate_change — the user is asking about tariff history or changes. "
            "Lead with ## 5 (historical timeline) and make it the most detailed section. "
            "## 2 should still cover current rates but be brief.\n\n"
        ) + context
    elif query_intent == "country_compare":
        context = (
            "QUERY INTENT: country_compare — the user is comparing sourcing countries. "
            "Lead with ## 4 (alternative sourcing) and make it the most detailed section. "
            "Explicitly rank the countries by total effective duty in ## 4. "
            "## 2 should cover the primary country's charges briefly.\n\n"
        ) + context

    # LLM call via ModelRouter — ANSWER_SYNTHESIS (claude-haiku)
    from services.llm.router import get_router, TaskType
    router = get_router()

    final_response = None
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            loop = asyncio.new_event_loop()
            try:
                resp = loop.run_until_complete(
                    router.complete(
                        task=TaskType.ANSWER_SYNTHESIS,
                        messages=[{"role": "user", "content": context}],
                    )
                )
            finally:
                loop.close()
            final_response = resp.choices[0].message.content.strip()
            break
        except RuntimeError as e:
            last_error = str(e)
            logger.error("synthesis_router_failed error=%s", e)
            break
        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

    if not final_response:
        return {
            "final_response": None,
            "citations": [],
            "pipeline_confidence": "LOW",
            "hitl_required": True,
            "hitl_reason": "citation_failure",
            "error": f"Synthesis LLM failed: {last_error}",
        }

    citations_valid, hallucinated = _validate_citations(final_response, valid_docs)
    if not citations_valid:
        logger.warning("synthesis_unverified_citations hallucinated=%s — continuing", hallucinated)

    conf = _compute_confidence(
        classification_confidence=float(state.get("classification_confidence") or 0.0),
        rate_verified=rate_verified,
        adder_doc_verified=adder_doc_verified,
        fr_docs_verified=bool(valid_docs),
        hitl_was_triggered=bool(state.get("hitl_required", False)),
    )
    citations = _build_citations(state, deduped, valid_docs, doc_metadata, rate_change_history)

    logger.info("synthesis_agent_done citations=%d confidence=%s", len(citations), conf)

    return {
        "final_response": final_response,
        "citations": citations,
        "pipeline_confidence": conf,
        "country_comparison": country_comparison_pre,
        "top_importers": top_importers_pre,
        "rate_change_history": rate_change_history,
        "hitl_required": bool(state.get("hitl_required", False)),
        "hitl_reason": state.get("hitl_reason"),
    }