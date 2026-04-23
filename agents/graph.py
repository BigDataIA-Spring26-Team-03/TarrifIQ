"""
TariffIQ — LangGraph pipeline (9 steps)

Sequential pipeline:
  1. query → 2. classify → 3. base_rate → 4. adder_rate → 5. policy → 6. trade → 7. synthesis

HITL exits:
  - after classify:   confidence < 0.80  → hitl_step → END
  - after synthesis:  citation failure   → hitl_step → END

Notes:
  - adder_rate_step (Steps 4-7) now runs BEFORE policy_step
    Reason: Steps 4-5 only need hts_code + hts_footnotes from base_rate_step
  - policy_chunks are supplementary context; adder_rate_agent handles policy_chunks=None gracefully
  - hitl_node writes to TARIFFIQ.RAW.HITL_RECORDS via tools.write_hitl_record()
  - Alias write-back (self-improvement) fires in base_rate_node after rate confirmed
  - Pipeline latency logged in ms on every run
"""

import logging
import time
import uuid
from typing import Dict, Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agents.state import TariffState
from agents.query_agent import run_query_agent
from agents.classification_agent import run_classification_agent
from agents.base_rate_agent import run_base_rate_agent
from agents.policy_agent import run_policy_agent
from agents.adder_rate_agent import run_adder_rate_agent
from agents.trade_agent import run_trade_agent
from agents.synthesis_agent import run_synthesis_agent
from agents import tools

logger = logging.getLogger(__name__)


def _fetch_subcategory_suggestions(hts_code: str, product: str, country: str) -> list:
    if not hts_code:
        return []
    try:
        heading = hts_code.split(".")[0] if "." in hts_code else hts_code[:4]
        country_suffix = f" from {country}" if country else ""
        # Expand product query with synonyms for better HTS matching
        SYNONYMS = {
            'pipes': 'tubes', 'pipe': 'tube',
            'steel pipes': 'steel tubes', 'iron pipes': 'iron tubes',
        }
        search_product = product.lower().strip()
        for term, synonym in SYNONYMS.items():
            if term in search_product:
                search_product = search_product.replace(term, synonym)
                break
        rows = tools.hts_keyword_search(query=search_product, limit=12, heading_filter=heading)
        # If too few results, broaden to chapter level
        if len(rows) < 3:
            chapter = heading[:2]
            broader = tools.hts_keyword_search(query=product, limit=12, chapter_filter=chapter)
            # Merge, dedupe by hts_code
            seen_codes = {r['hts_code'] for r in rows}
            for r in broader:
                if r['hts_code'] not in seen_codes:
                    rows.append(r)
                    seen_codes.add(r['hts_code'])
        # Still too few — search globally without any filter
        if len(rows) < 3:
            broader = tools.hts_keyword_search(query=product, limit=12)
            seen_codes = {r["hts_code"] for r in rows}
            for r in broader:
                if r["hts_code"] not in seen_codes:
                    rows.append(r)
                    seen_codes.add(r["hts_code"])
        if not rows:
            return []
        suggestions = []
        seen: set = set()
        for r in rows:
            desc = (r.get("description") or "").strip()
            if not desc or len(desc) < 4:
                continue
            if desc.lower().startswith(("of ", "other", "not ", "nesoi", ":")):
                continue
            dl = desc.lower()
            if dl in seen:
                continue
            seen.add(dl)
            label = desc if len(desc) <= 70 else desc[:67] + "..."
            suggestions.append({"label": label, "query": f"{dl}{country_suffix}"})
            if len(suggestions) >= 5:
                break
        return suggestions
    except Exception as e:
        logger.debug("fetch_subcategory_suggestions_error hts=%s error=%s", hts_code, e)
        return []


def query_node(state: TariffState) -> Dict[str, Any]:
    result = run_query_agent(state)
    if result.get("clarification_needed"):
        return {
            "clarification_needed": True,
            "clarification_message": result.get("message"),
            "clarification_suggestions": result.get("suggestions", []),
            "product": result.get("product"),
            "country": result.get("country"),
        }
    return result


def classification_node(state: TariffState) -> Dict[str, Any]:
    result = run_classification_agent(state)
    if result.get("hitl_required") and result.get("hitl_reason") in ("low_confidence", "semantic_mismatch"):
        hts_code = result.get("hts_code")
        product = state.get("product") or ""
        country = state.get("country") or ""
        suggestions = _fetch_subcategory_suggestions(hts_code, product, country)
        if suggestions:
            result["clarification_needed"] = True
            result["clarification_message"] = (
                f'"{product}" matches multiple HTS subcategories with different rates. '
                f"Which type are you importing{' from ' + country if country else ''}?"
            )
            result["clarification_suggestions"] = suggestions
            logger.info("classification_node_suggestions product=%s count=%d", product, len(suggestions))
    return result


def base_rate_node(state: TariffState) -> Dict[str, Any]:
    result = run_base_rate_agent(state)
    # Auto alias write-back disabled — was writing wrong codes automatically
    return result


def policy_node(state: TariffState) -> Dict[str, Any]:
    return run_policy_agent(state)


def adder_rate_node(state: TariffState) -> Dict[str, Any]:
    return run_adder_rate_agent(state)


def trade_node(state: TariffState) -> Dict[str, Any]:
    return run_trade_agent(state)


def synthesis_node(state: TariffState) -> Dict[str, Any]:
    return run_synthesis_agent(state)


def hitl_node(state: TariffState) -> Dict[str, Any]:
    reason = state.get("hitl_reason", "unknown")
    query_text = state.get("query", "")
    hts = state.get("hts_code")
    conf = state.get("classification_confidence")
    logger.warning("hitl_escalation query=%s reason=%s hts=%s conf=%s",
                   query_text[:80], reason, hts, conf)
    hitl_id = tools.write_hitl_record(
        query_text=query_text,
        trigger_reason=reason,
        classifier_hts=hts,
        classifier_conf=conf,
    )
    if hitl_id:
        logger.info("hitl_record_written id=%s", hitl_id)
    return {"hitl_required": True}


def after_query(state: TariffState) -> str:
    if state.get("clarification_needed"):
        return "end"
    return "classify"


def after_classification(state: TariffState) -> str:
    if state.get("clarification_needed"):
        return "hitl"
    if state.get("hitl_required") and state.get("hitl_reason") in ("low_confidence", "semantic_mismatch"):
        return "hitl"
    return "base_rate"


def after_synthesis(state: TariffState) -> str:
    if state.get("hitl_required") and state.get("hitl_reason") == "citation_failure":
        return "hitl"
    return "end"


def build_graph() -> StateGraph:
    wf = StateGraph(TariffState)
    wf.add_node("query_step",      query_node)
    wf.add_node("classify_step",   classification_node)
    wf.add_node("base_rate_step",  base_rate_node)
    wf.add_node("policy_step",     policy_node)
    wf.add_node("adder_rate_step", adder_rate_node)
    wf.add_node("trade_step",      trade_node)
    wf.add_node("synthesis_step",  synthesis_node)
    wf.add_node("hitl_step",       hitl_node)
    wf.set_entry_point("query_step")
    wf.add_conditional_edges(
        "query_step", after_query,
        {"classify": "classify_step", "end": END},
    )
    wf.add_conditional_edges(
        "classify_step", after_classification,
        {"base_rate": "base_rate_step", "hitl": "hitl_step"},
    )
    wf.add_edge("base_rate_step",  "adder_rate_step")
    wf.add_edge("adder_rate_step", "policy_step")
    wf.add_edge("policy_step",     "trade_step")
    wf.add_edge("trade_step",      "synthesis_step")
    wf.add_conditional_edges("synthesis_step", after_synthesis, {"end": END, "hitl": "hitl_step"})
    wf.add_edge("hitl_step", END)
    return wf.compile(checkpointer=MemorySaver())


tariff_graph = build_graph()


def run_pipeline(query: str) -> Dict[str, Any]:
    t0 = time.monotonic()
    logger.info("pipeline_start query=%s", query[:100])
    initial_state: TariffState = {
        "query": query,
        "product": None, "country": None,
        "clarification_needed": None, "clarification_message": None, "clarification_suggestions": None,
        "hts_code": None, "hts_description": None, "classification_confidence": None,
        "base_rate": None, "mfn_rate": None, "fta_rate": None, "fta_program": None, "fta_applied": None, "rate_record_id": None, "hts_footnotes": None,
        "chapter99_adder": None, "chapter99_doc": None,
        "notice_adder": None, "notice_doc": None, "notice_basis": None,
        "policy_chunks": None, "policy_summary": None,
        "adder_rate": None, "adder_doc": None, "adder_basis": None, "adder_method": None, "total_duty": None,
        "import_value_usd": None, "import_quantity": None,
        "trade_period": None, "trade_country_code": None,
        "trade_suppressed": None, "trade_trend_pct": None, "trade_trend_label": None,
        "final_response": None, "citations": None, "pipeline_confidence": None,
        "country_comparison": None, "top_importers": None, "rate_change_history": None,
        "query_intent": None, "hitl_required": None, "hitl_reason": None,
        "_product_for_feedback": None, "error": None,
    }
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = tariff_graph.invoke(initial_state, config=config)
    elapsed_ms = round((time.monotonic() - t0) * 1000)
    logger.info("pipeline_done query=%s hitl=%s latency_ms=%d",
                query[:60], result.get("hitl_required"), elapsed_ms)
    return result


def run_comparison_pipeline(query: str, countries: list) -> Dict[str, Any]:
    """
    Run the pipeline for multiple countries and return side-by-side comparison.
    Used for queries like "is it cheaper to import steel from China or Germany?"

    Returns a dict with:
    - product, hts_code, hts_description
    - comparison: list of {country, base_rate, adder_rate, total_duty, adder_doc, fta_applied}
    - cheapest_country: country with lowest total_duty
    """
    from agents.query_agent import run_query_agent

    import re as _re
    # Strip "vs/or COUNTRY" but keep the product + first country intact
    clean_query = _re.sub(r'\s+(vs\.?|or)\s+\S+.*$', '', query, flags=_re.IGNORECASE).strip()
    # Also strip trailing "?" 
    clean_query = clean_query.rstrip('?').strip()
    initial = {"query": clean_query}
    parsed = run_query_agent(initial)
    product = parsed.get("product")

    if not product:
        return {"error": "Could not parse product from query"}

    logger.info("comparison_pipeline_start product=%s countries=%s", product, countries)

    comparison = []
    hts_code = None
    hts_description = None

    for country in countries:
        country_query = f"{product} from {country}"
        result = run_pipeline(country_query)

        if not hts_code:
            hts_code = result.get("hts_code")
            hts_description = result.get("hts_description")

        comparison.append({
            "country": country,
            "base_rate": result.get("base_rate") or 0.0,
            "mfn_rate": result.get("mfn_rate") or 0.0,
            "adder_rate": result.get("adder_rate") or 0.0,
            "section122_adder": result.get("section122_adder") or 0.0,
            "total_duty": result.get("total_duty") or 0.0,
            "adder_doc": result.get("adder_doc"),
            "adder_method": result.get("adder_method"),
            "fta_applied": result.get("fta_applied", False),
            "fta_program": result.get("fta_program"),
            "policy_summary": result.get("policy_summary"),
            "hitl_required": result.get("hitl_required", False),
        })

    # Sort by total_duty ascending
    comparison.sort(key=lambda x: x["total_duty"])
    cheapest = comparison[0]["country"] if comparison else None

    logger.info(
        "comparison_pipeline_done product=%s countries=%d cheapest=%s",
        product, len(countries), cheapest
    )

    return {
        "product": product,
        "hts_code": hts_code,
        "hts_description": hts_description,
        "comparison": comparison,
        "cheapest_country": cheapest,
        "query": query,
    }


def run_pipeline_auto(query: str) -> Dict[str, Any]:
    """
    Auto-detects comparison queries and routes appropriately.
    Falls back to standard run_pipeline for non-comparison queries.
    """
    import re

    # Pattern 1: "from X vs/or Y"
    m = re.search(
        r'\bfrom\s+([A-Za-z][a-z]+(?:\s[A-Za-z][a-z]+)?)\s+(?:or|vs\.?)\s+([A-Za-z][a-z]+(?:\s[A-Za-z][a-z]+)?)',
        query, re.IGNORECASE
    )
    if m:
        return run_comparison_pipeline(query, [m.group(1), m.group(2)])

    # Pattern 2: any "X or Y" / "X vs Y" where both are capitalized country-like words
    m2 = re.search(
        r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s+(?:or|vs\.?)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)',
        query
    )
    if m2:
        c1, c2 = m2.group(1), m2.group(2)
        # Skip common non-country words
        skip = {"the", "and", "for", "not", "but", "with", "from", "this", "that"}
        if c1.lower() not in skip and c2.lower() not in skip and len(c1) > 2 and len(c2) > 2:
            logger.info("auto_routing comparison c1=%s c2=%s", c1, c2)
            return run_comparison_pipeline(query, [c1, c2])

    return run_pipeline(query)