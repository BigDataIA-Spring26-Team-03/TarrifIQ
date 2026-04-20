"""
Synthesis Agent — Pipeline Step 7

Combines all upstream agent outputs into a cited natural language answer.
LLM call via ModelRouter(TaskType.ANSWER_SYNTHESIS).
All validation via tools.py.

Citation validation (two layers):
  1. Every FR doc number cited must be in policy_chunks + adder_doc set
  2. Every doc verified against Snowflake via tools.verify_fr_doc()

Rate record validation: tools.hts_verify() confirms rate_record_id exists.

Trade trend: trade_trend_label included in context + census citation.

Adder provenance: adder_doc cited as dedicated "adder_source" entry.

FTA awareness: if base_rate_agent applied an FTA rate (USMCA, KORUS, etc.),
the context block says so explicitly so the LLM can report the correct rate.

Pipeline confidence: HIGH / MEDIUM / LOW based on component verification.
"""

import asyncio
import logging
import os
import re
import time
from typing import Dict, Any, List, Optional, Set

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
{specific_duty_line}
  Total effective duty:  {total_duty:.2f}%
{footnote_line}
FEDERAL REGISTER POLICY EXCERPTS
(cite ONLY these document numbers: {valid_docs}):
{policy_excerpts}

POLICY ANALYSIS: {policy_summary}

TRADE VOLUME [Census Bureau {period}]: {trade_line}

{comparison_section}

RESPONSE INSTRUCTIONS BASED ON CONFIDENCE:
- HIGH confidence: Give a direct, confident answer. No hedging. No "verify with CBP".
- MEDIUM confidence: Give the answer, add ONE brief note about what is uncertain.
- LOW confidence: Give the answer with explicit uncertainty flags on specific data points.
Never use markdown headers (no #). Write in plain prose paragraphs only.
"""


def _dedupe_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Keep ALL chunks for LLM context — different chunks from same doc
    # have different text. Deduplication only happens in _build_citations().
    return chunks if chunks else []


def _verify_docs(candidates: Set[str]) -> Set[str]:
    """
    Verify doc numbers against all three source tables:
    FEDERAL_REGISTER_NOTICES (USTR), CBP_FEDERAL_REGISTER_NOTICES, ITC_DOCUMENTS.
    Uses tools.verify_fr_doc() which checks USTR + CBP.
    Also checks ITC_DOCUMENTS for USITC notices.
    """
    if not candidates:
        return set()
    verified = set()
    for doc in candidates:
        if tools.verify_fr_doc(doc):
            verified.add(doc)
        elif tools.verify_itc_doc(doc):
            verified.add(doc)
    return verified


def _fetch_doc_metadata(doc_numbers: Set[str]) -> Dict[str, Dict]:
    """
    Fetch title, publication_date, and abstract for FR documents.
    Returns dict keyed by document_number.
    """
    if not doc_numbers:
        return {}
    try:
        conn = tools._sf()
        cur = conn.cursor()
        meta = {}
        docs_list = list(doc_numbers)

        # Fetch from USTR notices
        placeholders = ",".join(["%s"] * len(docs_list))
        cur.execute(
            f"SELECT document_number, title, publication_date, html_url "
            f"FROM TARIFFIQ.RAW.FEDERAL_REGISTER_NOTICES "
            f"WHERE document_number IN ({placeholders})",
            docs_list,
        )
        for doc_num, title, pub_date, html_url in cur.fetchall():
            meta[doc_num] = {
                "title": title or "",
                "publication_date": str(pub_date) if pub_date else "",
                "url": html_url or "",
                "source": "USTR",
            }

        # Fetch from CBP notices (for ones not found above)
        remaining = [d for d in docs_list if d not in meta]
        if remaining:
            placeholders2 = ",".join(["%s"] * len(remaining))
            cur.execute(
                f"SELECT document_number, title, publication_date, html_url "
                f"FROM TARIFFIQ.RAW.CBP_FEDERAL_REGISTER_NOTICES "
                f"WHERE document_number IN ({placeholders2})",
                remaining,
            )
            for doc_num, title, pub_date, html_url in cur.fetchall():
                meta[doc_num] = {
                    "title": title or "",
                    "publication_date": str(pub_date) if pub_date else "",
                    "url": html_url or "",
                    "source": "CBP",
                }

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


def _build_context(state: TariffState, deduped: List[Dict], valid_docs: Set[str]) -> str:
    hts_code = state.get("hts_code") or "Unknown"
    record_id = state.get("rate_record_id") or hts_code
    base_rate = state.get("base_rate") or 0.0
    adder_rate = state.get("adder_rate") or 0.0
    total_duty = state.get("total_duty") or 0.0
    adder_doc = state.get("adder_doc")
    adder_specific_duty = (state.get("adder_specific_duty") or "").strip()

    adder_source = f"(sourced from {adder_doc})" if adder_doc else "(source: not identified)"
    specific_duty_line = ""
    if adder_specific_duty:
        if adder_doc:
            specific_duty_line = f"  Chapter 99 specific duty: {adder_specific_duty} [{adder_doc}]"
        else:
            specific_duty_line = f"  Chapter 99 specific duty: {adder_specific_duty}"

    fta_applied = state.get("fta_applied", False)
    fta_program = state.get("fta_program")
    mfn_rate = state.get("mfn_rate") or base_rate
    hts_footnotes = state.get("hts_footnotes") or []

    if fta_applied and fta_program:
        rate_line = f"{fta_program} preferential rate {base_rate:.2f}% (MFN would be {mfn_rate:.2f}%)"
    else:
        rate_line = f"Base MFN {base_rate:.2f}%"

    footnote_line = ("HTS FOOTNOTES: " + "; ".join(hts_footnotes[:3]) + "\n") if hts_footnotes else ""

    if deduped:
        excerpts = "\n".join(
            f"[{c.get('document_number', 'UNKNOWN')}] {c.get('chunk_text', '')[:300]}"
            for c in deduped[:5]
        )
    else:
        excerpts = "No relevant policy documents found."

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

    # Build country comparison section if available
    country_comp = state.get("country_comparison") or []
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
        specific_duty_line=specific_duty_line,
        total_duty=total_duty,
        footnote_line=footnote_line,
        valid_docs=", ".join(sorted(valid_docs)) if valid_docs else "none",
        policy_excerpts=excerpts,
        policy_summary=state.get("policy_summary", ""),
        period=period,
        trade_line=trade_line,
    )


# Agency metadata for citation enrichment
_SOURCE_AGENCY_MAP = {
    "USTR":  {"agency": "Office of the U.S. Trade Representative", "agency_short": "USTR"},
    "CBP":   {"agency": "U.S. Customs and Border Protection",       "agency_short": "CBP"},
    "USITC": {"agency": "U.S. International Trade Commission",      "agency_short": "USITC"},
    "ITC":   {"agency": "U.S. International Trade Commission",      "agency_short": "USITC"},
}


def _fr_url(doc_number: str) -> str:
    return f"https://www.federalregister.gov/documents/{doc_number}"


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
            "url": None,
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
        citations.append({
            "type": "adder_source",
            "id": adder_doc,
            "agency": "Office of the U.S. Trade Representative",
            "agency_short": "USTR",
            "title": ch99_title,
            "text": ch99_text,
            "source": "federalregister.gov",
            "url": _fr_url(adder_doc) if not adder_doc.startswith("9903") else None,
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
            citations.append({
                "type": "federal_register",
                "id": doc,
                "agency": agency_meta["agency"],
                "agency_short": agency_meta["agency_short"],
                "title": doc_meta.get("title", f"Federal Register {doc}"),
                "text": text,
                "source": f"TARIFFIQ.RAW.{'CBP_' if src == 'CBP' else ''}FEDERAL_REGISTER_NOTICES",
                "url": doc_meta.get("url") or _fr_url(doc),
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
    Run base rate lookup for key alternative sourcing countries.
    Returns a list of {country, base_rate, fta_program, fta_applied, total_note}
    sorted by effective rate ascending.
    Skips the current country and countries that fail lookup.
    """
    if not hts_code:
        return []

    results = []
    current_lower = (current_country or "").lower().strip()

    for country in COMPARISON_COUNTRIES:
        if country.lower() == current_lower:
            continue
        try:
            rate_result = tools.hts_base_rate_lookup(hts_code, country=country)
            if rate_result is None:
                continue
            results.append({
                "country": country,
                "base_rate": rate_result.get("base_rate", 0.0),
                "mfn_rate": rate_result.get("mfn_rate", 0.0),
                "fta_program": rate_result.get("fta_program"),
                "fta_applied": rate_result.get("fta_applied", False),
                # Note: adder rates not computed here (would require full pipeline per country)
                "note": rate_result.get("fta_program") or "MFN rate",
            })
        except Exception as e:
            logger.debug("country_comparison_error country=%s error=%s", country, e)
            continue

    # Sort by effective base rate ascending
    results.sort(key=lambda x: x["base_rate"])
    return results[:5]  # top 5 cheapest alternatives


def run_synthesis_agent(state: TariffState) -> Dict[str, Any]:
    logger.info("synthesis_agent_start query=%s", state.get("query", "")[:80])

    raw_chunks = state.get("policy_chunks") or []
    deduped = _dedupe_chunks(raw_chunks)

    # Candidate docs: policy chunks + adder_doc
    candidate_docs: Set[str] = {
        c.get("document_number", "") for c in deduped if c.get("document_number")
    }
    adder_doc = state.get("adder_doc")
    if adder_doc:
        candidate_docs.add(adder_doc)

    # Verify all docs against Snowflake
    valid_docs = _verify_docs(candidate_docs)

    # Fetch metadata for citation enrichment
    doc_metadata = _fetch_doc_metadata(valid_docs)
    unverified_docs = candidate_docs - valid_docs
    # Suppress unverified doc warning for chapter99_lookup sourced adder docs
    adder_doc = state.get("adder_doc") or ""
    adder_method = state.get("adder_method") or ""
    if adder_method == "chapter99_lookup" and adder_doc in unverified_docs:
        unverified_docs.discard(adder_doc)
        logger.info("synthesis_chap99_doc_suppressed doc=%s", adder_doc)
    if unverified_docs:
        logger.warning("synthesis_unverified_docs=%s", unverified_docs)

    # Verify rate record
    record_id = state.get("rate_record_id")
    rate_verified = tools.hts_verify(record_id) if record_id else False
    adder_doc_verified = adder_doc in valid_docs if adder_doc else True

    context = _build_context(state, deduped, valid_docs)

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

    # Country comparison — run base rate lookup for key alternative origins
    country_comparison = _build_country_comparison(
        hts_code=state.get("hts_code") or state.get("rate_record_id"),
        current_country=state.get("country"),
    )

    # Rate change history — fetch if intent is rate_change
    rate_change_history = None
    if state.get("query_intent") == "rate_change":
        hts_for_history = state.get("hts_code") or state.get("rate_record_id")
        if hts_for_history:
            rate_change_history = tools.fetch_rate_change_history(
                hts_for_history, country=state.get("country")
            )

    logger.info("synthesis_agent_done citations=%d confidence=%s", len(citations), conf)

    return {
        "final_response": final_response,
        "citations": citations,
        "pipeline_confidence": conf,
        "country_comparison": country_comparison,
        "rate_change_history": rate_change_history,
        "hitl_required": bool(state.get("hitl_required", False)),
        "hitl_reason": state.get("hitl_reason"),
    }