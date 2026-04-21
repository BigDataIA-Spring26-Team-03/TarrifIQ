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

VERIFIED DUTY RATE [HTS {record_id}]:
  {rate_line}
  Section 301/IEEPA:     {adder_rate:.2f}% {adder_source}
  Total effective duty:  {total_duty:.2f}%
{footnote_line}
POLICY CHUNKS (full text per chunk until SYNTHESIS_* env budgets; cite ONLY these FR doc numbers: {valid_docs}):
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
    cited = set(re.findall(r"\b(\d{4}-\d{5,6})\b", text))
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
    """Fallback when Snowflake has no html_url — FR keyword search by document number."""
    dn = (doc_number or "").strip()
    if not dn:
        return ""
    return (
        "https://www.federalregister.gov/documents/search"
        f"?conditions%5Bterm%5D={quote(dn, safe='')}"
    )


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
            "source": "TARIFFIQ.RAW.HTS_CODES",
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
        else:
            ch99_title = adder_meta.get("title", f"FR {adder_doc}")
            ch99_text = f"Section 301/IEEPA adder rate — method: {state.get('adder_method', 'unknown')}"
        adder_url = (adder_meta.get("html_url") or "").strip()
        if not adder_url and not adder_doc.startswith("9903"):
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

    # Dedup by doc number here (not in _dedupe_chunks)
    for chunk in deduped:
        doc = chunk.get("document_number")
        if doc and doc not in seen and doc in valid_docs:
            seen.add(doc)
            src = chunk.get("source", "USTR").upper()
            agency_meta = _SOURCE_AGENCY_MAP.get(src, {"agency": src, "agency_short": src})
            doc_meta = meta.get(doc, {})
            chunk_text = chunk.get("chunk_text", "") or ""
            text = chunk_text[:120].strip() if chunk_text else ""
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
                "title": doc_meta.get("title", f"Federal Register {doc}"),
                "text": text,
                "source": sf_tbl,
                "url": doc_url or None,
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


# Key alternative sourcing countries to compare
COMPARISON_COUNTRIES = [
    "Vietnam", "Mexico", "India", "South Korea",
    "Taiwan", "Germany", "Japan", "Canada",
]


def _build_country_comparison(
    hts_code: Optional[str],
    current_country: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Build country comparison with estimated total effective duty.
    Uses base rate from HTS_CODES + known country-specific adder tiers.
    Sorted by estimated total ascending so cheapest sourcing is first.
    """
    if not hts_code:
        return []

    # Known IEEPA / Section 301 adder tiers by country (as of 2025-2026).
    # These are country-level surcharges applied on top of MFN base rate.
    # Source: Executive Orders 14257, 14259, 14266, 14329 and Section 301 actions.
    # Not exhaustive — only major trading partners with active adders.
    COUNTRY_ADDER_TIERS = {
        "china":       {"adder": 145.0, "program": "Section 301 + IEEPA"},
        "india":       {"adder": 26.0,  "program": "IEEPA (EO 14329)"},
        "vietnam":     {"adder": 46.0,  "program": "IEEPA reciprocal"},
        "taiwan":      {"adder": 32.0,  "program": "IEEPA reciprocal"},
        "thailand":    {"adder": 36.0,  "program": "IEEPA reciprocal"},
        "indonesia":   {"adder": 32.0,  "program": "IEEPA reciprocal"},
        "bangladesh":  {"adder": 37.0,  "program": "IEEPA reciprocal"},
        "cambodia":    {"adder": 49.0,  "program": "IEEPA reciprocal"},
        "japan":       {"adder": 24.0,  "program": "IEEPA reciprocal"},
        "south korea": {"adder": 25.0,  "program": "IEEPA reciprocal"},
        # USMCA partners — no IEEPA adder on USMCA-qualifying goods
        "canada":      {"adder": 0.0,   "program": None},
        "mexico":      {"adder": 0.0,   "program": None},
        # EU countries — framework agreement, reduced
        "germany":     {"adder": 15.0,  "program": "US-EU Framework"},
        "france":      {"adder": 15.0,  "program": "US-EU Framework"},
        "italy":       {"adder": 15.0,  "program": "US-EU Framework"},
    }

    results = []
    current_lower = (current_country or "").lower().strip()

    for country in COMPARISON_COUNTRIES:
        if country.lower() == current_lower:
            continue
        try:
            rate_result = tools.hts_base_rate_lookup(hts_code, country=country)
            if rate_result is None:
                continue

            base = rate_result.get("base_rate", 0.0)
            mfn = rate_result.get("mfn_rate", 0.0)
            fta_program = rate_result.get("fta_program")
            fta_applied = rate_result.get("fta_applied", False)

            # Get known adder for this country
            tier = COUNTRY_ADDER_TIERS.get(country.lower(), {})
            adder = tier.get("adder", 0.0)
            adder_program = tier.get("program")

            # If FTA applied, base is already 0 — adder still applies on top
            estimated_total = round(base + adder, 2)

            # Sourcing note
            if fta_applied and fta_program and adder == 0:
                note = f"{fta_program} — {estimated_total:.1f}% total"
            elif adder > 0 and adder_program:
                note = f"{adder_program} +{adder:.0f}% adder — {estimated_total:.1f}% total"
            elif estimated_total == 0:
                note = "0% — cheapest source"
            else:
                note = f"{estimated_total:.1f}% total"

            results.append({
                "country": country,
                "base_rate": round(base, 2),
                "adder_rate": adder,
                "adder_program": adder_program or ("FTA" if fta_applied else "None"),
                "fta_program": fta_program,
                "estimated_total": estimated_total,
                "note": note,
            })
        except Exception as e:
            logger.debug("country_comparison_error country=%s error=%s", country, e)
            continue

    # Sort by estimated total ascending — cheapest first
    results.sort(key=lambda x: x["estimated_total"])
    return results[:6]


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
        logger.warning("synthesis_citation_failure hallucinated=%s", hallucinated)
        return {
            "final_response": final_response,
            "citations": _build_citations(state, deduped, valid_docs),
            "pipeline_confidence": "LOW",
            "hitl_required": True,
            "hitl_reason": "citation_failure",
        }

    conf = _compute_confidence(
        classification_confidence=float(state.get("classification_confidence") or 0.0),
        rate_verified=rate_verified,
        adder_doc_verified=adder_doc_verified,
        fr_docs_verified=bool(valid_docs),
        hitl_was_triggered=bool(state.get("hitl_required", False)),
    )
    citations = _build_citations(state, deduped, valid_docs, doc_metadata)

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