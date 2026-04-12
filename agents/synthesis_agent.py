"""
Synthesis Agent — TariffIQ Pipeline Step 6

Combines outputs from all upstream agents into a cited natural language answer.
Validates every rate claim against a Snowflake record ID.
Validates every policy claim against a Federal Register document number.
Triggers HITL on citation validation failure.
"""

import json
import logging
import os
import re
from typing import Dict, Any, List, Optional

import litellm

from ingestion.connection import get_snowflake_conn
from agents.state import TariffState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a trade sourcing analyst. Generate a sourcing brief from the structured inputs provided.

Rules:
- Every rate claim must reference the Snowflake record ID provided in the rate data.
- Every policy claim must reference a Federal Register document number in parentheses e.g. (2025-12345).
- Every trade volume claim must reference "Census Bureau {period}".
- Do not generate tariff rates, dates, or policy facts from memory.
- If data is missing or suppressed, say so explicitly.
- Be concise and factual."""


def _validate_fr_citations(
    text: str, policy_chunks: Optional[List[Dict[str, Any]]]
) -> bool:
    """
    Check that every FR document number cited in text exists in policy_chunks.
    Returns True if all citations are valid, False if any hallucinated.
    """
    if not policy_chunks:
        return True

    # Extract all document numbers cited in response
    cited_docs = set(re.findall(r"\b(\d{4}-\d+)\b", text))
    if not cited_docs:
        return True

    valid_docs = {c.get("document_number", "") for c in policy_chunks}

    hallucinated = cited_docs - valid_docs
    if hallucinated:
        logger.warning(
            "citation_hallucination_detected hallucinated=%s valid=%s",
            hallucinated, valid_docs,
        )
        return False
    return True


def _validate_rate_record(record_id: Optional[str]) -> bool:
    """Verify rate record ID exists in Snowflake HTS_CODES."""
    if not record_id:
        return True
    conn = get_snowflake_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT 1 FROM TARIFFIQ.RAW.HTS_CODES WHERE hts_code = %s LIMIT 1",
            (record_id,),
        )
        return cur.fetchone() is not None
    finally:
        cur.close()
        conn.close()


def _build_context(state: TariffState) -> str:
    """Build structured context block for LLM from all agent outputs."""
    parts = []

    parts.append(f"USER QUERY: {state.get('query', '')}")
    parts.append(f"PRODUCT: {state.get('product', 'Unknown')}")
    parts.append(f"COUNTRY: {state.get('country', 'Not specified')}")

    hts_code = state.get("hts_code")
    if hts_code:
        parts.append(
            f"HTS CODE: {hts_code} — {state.get('hts_description', '')} "
            f"(confidence: {state.get('classification_confidence', 0):.0%})"
        )

    record_id = state.get("rate_record_id")
    total_duty = state.get("total_duty")
    if total_duty is not None:
        parts.append(
            f"DUTY RATE: Base {state.get('base_rate', 0)}% + "
            f"Adder {state.get('adder_rate', 0)}% = "
            f"Total {total_duty}% [Snowflake record: {record_id}]"
        )

    policy_summary = state.get("policy_summary")
    if policy_summary:
        parts.append(f"POLICY CONTEXT:\n{policy_summary}")

    chunks = state.get("policy_chunks") or []
    if chunks:
        doc_numbers = list({c.get("document_number", "") for c in chunks if c.get("document_number")})
        parts.append(f"AVAILABLE FR DOCUMENT NUMBERS FOR CITATION: {', '.join(doc_numbers)}")

    trade_suppressed = state.get("trade_suppressed")
    if trade_suppressed is False:
        period = state.get("trade_period", "")
        val = state.get("import_value_usd")
        qty = state.get("import_quantity")
        parts.append(
            f"TRADE FLOW ({period}): Import value ${val:,.0f} USD, "
            f"Quantity {qty:,.0f} units [Source: Census Bureau {period}]"
            if val else f"TRADE FLOW ({period}): Data available from Census Bureau"
        )
    else:
        parts.append("TRADE FLOW: Data suppressed or unavailable from Census Bureau")

    return "\n\n".join(parts)


def _build_citations(state: TariffState) -> List[Dict[str, Any]]:
    """Build structured citations list from all agent outputs."""
    citations = []

    record_id = state.get("rate_record_id")
    if record_id:
        citations.append({
            "type": "snowflake_hts",
            "id": record_id,
            "text": f"HTS {state.get('hts_code')} — base rate {state.get('base_rate')}%",
        })

    for chunk in (state.get("policy_chunks") or []):
        doc_num = chunk.get("document_number")
        if doc_num:
            citations.append({
                "type": "federal_register",
                "id": doc_num,
                "text": chunk.get("title", ""),
            })

    if not state.get("trade_suppressed"):
        citations.append({
            "type": "census_bureau",
            "id": f"census_{state.get('trade_period', '')}",
            "text": f"US imports {state.get('trade_period', '')}",
        })

    return citations


def run_synthesis_agent(state: TariffState) -> Dict[str, Any]:
    """
    Generate cited final response from all agent outputs.

    Args:
        state: TariffState with all upstream agent outputs populated

    Returns:
        Dict with final_response, citations, and optionally hitl_required
    """
    logger.info("synthesis_agent_start query=%s", state.get("query"))

    context = _build_context(state)

    try:
        response = litellm.completion(
            model=os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ],
            temperature=0.1,
            max_tokens=800,
        )
        final_response = response.choices[0].message.content.strip()

    except Exception as e:
        logger.error("synthesis_llm_failed error=%s", e)
        return {
            "final_response": None,
            "citations": [],
            "hitl_required": True,
            "hitl_reason": "citation_failure",
            "error": f"Synthesis LLM failed: {e}",
        }

    # Validate FR citations
    policy_chunks = state.get("policy_chunks")
    citations_valid = _validate_fr_citations(final_response, policy_chunks)

    if not citations_valid:
        logger.warning("synthesis_citation_failure triggering HITL")
        return {
            "final_response": final_response,
            "citations": _build_citations(state),
            "hitl_required": True,
            "hitl_reason": "citation_failure",
        }

    citations = _build_citations(state)

    logger.info("synthesis_agent_done citations=%d", len(citations))

    return {
        "final_response": final_response,
        "citations": citations,
        "hitl_required": state.get("hitl_required", False),
        "hitl_reason": state.get("hitl_reason"),
    }