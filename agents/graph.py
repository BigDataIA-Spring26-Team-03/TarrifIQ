"""
LangGraph pipeline for TariffIQ.
Sequential: query -> classification -> rate -> policy -> trade -> synthesis
HITL exits after classification (low confidence) or synthesis (citation failure).
"""

import logging
from typing import Dict, Any

from langgraph.graph import StateGraph, END

from agents.state import TariffState
from agents.query_agent import run_query_agent
from agents.classification_agent import run_classification_agent, maybe_write_alias_feedback
from agents.rate_agent import run_rate_agent
from agents.policy_agent import run_policy_agent
from agents.trade_agent import run_trade_agent
from agents.synthesis_agent import run_synthesis_agent

logger = logging.getLogger(__name__)


def query_node(state: TariffState) -> Dict[str, Any]:
    return run_query_agent(state)


def classification_node(state: TariffState) -> Dict[str, Any]:
    return run_classification_agent(state)


def rate_node(state: TariffState) -> Dict[str, Any]:
    result = run_rate_agent(state)
    # Alias write-back fires only when both classification and rate are verified
    rate_found = result.get("rate_record_id") is not None
    maybe_write_alias_feedback(state, rate_found)
    return result


def policy_node(state: TariffState) -> Dict[str, Any]:
    return run_policy_agent(state)


def trade_node(state: TariffState) -> Dict[str, Any]:
    return run_trade_agent(state)


def synthesis_node(state: TariffState) -> Dict[str, Any]:
    return run_synthesis_agent(state)


def hitl_node(state: TariffState) -> Dict[str, Any]:
    logger.warning("hitl_escalation query=%s reason=%s", state.get("query"), state.get("hitl_reason"))
    return {"hitl_required": True}


def after_classification(state: TariffState) -> str:
    if state.get("hitl_required") and state.get("hitl_reason") == "low_confidence":
        return "hitl"
    return "rate"


def after_synthesis(state: TariffState) -> str:
    if state.get("hitl_required") and state.get("hitl_reason") == "citation_failure":
        return "hitl"
    return "end"


def build_graph() -> StateGraph:
    workflow = StateGraph(TariffState)
    workflow.add_node("query_step", query_node)
    workflow.add_node("classify_step", classification_node)
    workflow.add_node("rate_step", rate_node)
    workflow.add_node("policy_step", policy_node)
    workflow.add_node("trade_step", trade_node)
    workflow.add_node("synthesis_step", synthesis_node)
    workflow.add_node("hitl_step", hitl_node)
    workflow.set_entry_point("query_step")
    workflow.add_edge("query_step", "classify_step")
    workflow.add_conditional_edges("classify_step", after_classification, {"rate": "rate_step", "hitl": "hitl_step"})
    workflow.add_edge("rate_step", "policy_step")
    workflow.add_edge("policy_step", "trade_step")
    workflow.add_edge("trade_step", "synthesis_step")
    workflow.add_conditional_edges("synthesis_step", after_synthesis, {"end": END, "hitl": "hitl_step"})
    workflow.add_edge("hitl_step", END)
    return workflow.compile()


tariff_graph = build_graph()


def run_pipeline(query: str) -> Dict[str, Any]:
    logger.info("pipeline_start query=%s", query)
    initial_state: TariffState = {
        "query": query,
        "product": None, "country": None,
        "hts_code": None, "hts_description": None, "classification_confidence": None,
        "base_rate": None, "adder_rate": None, "total_duty": None, "rate_record_id": None,
        "policy_chunks": None, "policy_summary": None,
        "import_value_usd": None, "import_quantity": None, "trade_period": None,
        "trade_country_code": None, "trade_suppressed": None,
        "final_response": None, "citations": None, "pipeline_confidence": None,
        "hitl_required": None, "hitl_reason": None, "error": None,
    }
    result = tariff_graph.invoke(initial_state)
    logger.info("pipeline_done query=%s hitl=%s", query, result.get("hitl_required"))
    return result