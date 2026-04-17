"""
TariffIQ — LangGraph pipeline

7-step sequential pipeline:
  query → classify → base_rate → policy → adder_rate → trade → synthesis

HITL exits:
  - after classify:   confidence < 0.80  → hitl_step → END
  - after synthesis:  citation failure   → hitl_step → END

hitl_node writes to TARIFFIQ.RAW.HITL_RECORDS via tools.write_hitl_record().
Alias write-back (self-improvement) fires in base_rate_node after rate confirmed.
Pipeline latency logged in ms on every run.
"""

import logging
import time
from typing import Dict, Any

from langgraph.graph import StateGraph, END

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


# ── Node wrappers ──────────────────────────────────────────────────────────────

def query_node(state: TariffState) -> Dict[str, Any]:
    return run_query_agent(state)


def classification_node(state: TariffState) -> Dict[str, Any]:
    return run_classification_agent(state)


def base_rate_node(state: TariffState) -> Dict[str, Any]:
    result = run_base_rate_agent(state)
    # Self-improvement write-back: only when classification passed and rate found
    if not state.get("hitl_required") and result.get("rate_record_id"):
        product = state.get("_product_for_feedback") or state.get("product")
        hts_code = state.get("hts_code")
        confidence = float(state.get("classification_confidence") or 0.0)
        if product and hts_code:
            tools.alias_write(product, hts_code, confidence)
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


# ── Conditional edges ──────────────────────────────────────────────────────────

def after_classification(state: TariffState) -> str:
    if state.get("hitl_required") and state.get("hitl_reason") == "low_confidence":
        return "hitl"
    return "base_rate"


def after_synthesis(state: TariffState) -> str:
    if state.get("hitl_required") and state.get("hitl_reason") == "citation_failure":
        return "hitl"
    return "end"


# ── Graph construction ─────────────────────────────────────────────────────────

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
    wf.add_edge("query_step",      "classify_step")
    wf.add_conditional_edges(
        "classify_step", after_classification,
        {"base_rate": "base_rate_step", "hitl": "hitl_step"},
    )
    wf.add_edge("base_rate_step",  "policy_step")
    wf.add_edge("policy_step",     "adder_rate_step")
    wf.add_edge("adder_rate_step", "trade_step")
    wf.add_edge("trade_step",      "synthesis_step")
    wf.add_conditional_edges(
        "synthesis_step", after_synthesis,
        {"end": END, "hitl": "hitl_step"},
    )
    wf.add_edge("hitl_step", END)

    return wf.compile()


tariff_graph = build_graph()


# ── Public entry point ─────────────────────────────────────────────────────────

def run_pipeline(query: str) -> Dict[str, Any]:
    t0 = time.monotonic()
    logger.info("pipeline_start query=%s", query[:100])

    initial_state: TariffState = {
        "query": query,
        "product": None, "country": None,
        "hts_code": None, "hts_description": None, "classification_confidence": None,
        "base_rate": None, "mfn_rate": None, "fta_rate": None, "fta_program": None, "fta_applied": None, "rate_record_id": None, "hts_footnotes": None,
        "policy_chunks": None, "policy_summary": None,
        "adder_rate": None, "adder_doc": None, "adder_method": None, "total_duty": None,
        "import_value_usd": None, "import_quantity": None,
        "trade_period": None, "trade_country_code": None,
        "trade_suppressed": None,
        "trade_trend_pct": None, "trade_trend_label": None,
        "final_response": None, "citations": None, "pipeline_confidence": None,
        "hitl_required": None, "hitl_reason": None,
        "_product_for_feedback": None,
        "error": None,
    }

    result = tariff_graph.invoke(initial_state)
    elapsed_ms = round((time.monotonic() - t0) * 1000)
    logger.info("pipeline_done query=%s hitl=%s latency_ms=%d",
                query[:60], result.get("hitl_required"), elapsed_ms)
    return result