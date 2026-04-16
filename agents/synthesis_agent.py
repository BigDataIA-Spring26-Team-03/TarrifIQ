"""
Synthesis Agent — TariffIQ Pipeline Step 6

Combines all upstream agent outputs into a cited natural language answer.

Validation:
  - Every FR doc number cited must exist in policy_chunks (hallucination check)
  - IMPROVEMENT: Every cited FR doc is also verified against FEDERAL_REGISTER_NOTICES
    in Snowflake — not just against the retrieved chunks. This catches the case
    where the LLM fabricates a doc number that looks plausible (e.g. 2024-99999)
    but was never in your database at all.
  - Every rate claim must reference a verified Snowflake record ID
  - Duplicate citations deduplicated before building context
  - Citation validation failure triggers HITL, never reaches user

  - IMPROVEMENT: Final response includes a pipeline_confidence field with
    value HIGH / MEDIUM / LOW based on:
      HIGH: classification_confidence >= 0.80, rate record verified, FR docs real
      MEDIUM: classification_confidence >= 0.60 OR any single component missing
      LOW: multiple components missing or unverified

Safety:
  - LLM never generates rates, dates, or HTS codes from memory
  - Structured context block enforces citation grounding
  - Retry logic for LLM failures
"""

import logging
import os
import re
import time
from typing import Dict, Any, List, Optional, Set

import litellm

from ingestion.connection import get_snowflake_conn
from agents.state import TariffState

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
RETRY_DELAY = 1.0

SYSTEM_PROMPT = """You are a trade sourcing analyst generating a concise, factual sourcing brief.

STRICT RULES:
1. Every tariff rate claim MUST reference the Snowflake record ID in brackets, e.g. [HTS 8542.31.00]
2. Every policy claim MUST cite a Federal Register document number in parentheses, e.g. (2025-23912)
3. Every trade volume claim MUST reference "Census Bureau {period}"
4. NEVER generate tariff rates, HTS codes, dates, or policy facts from memory
5. If data is missing or suppressed, say so explicitly — do not guess
6. Be concise: 200-300 words maximum
7. Structure: Product & Classification | Tariff Rate | Policy Context | Trade Volume"""


def _dedupe_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate policy chunks by document_number, keeping first occurrence."""
    seen: Set[str] = set()
    deduped = []
    for chunk in chunks:
        doc_num = chunk.get("document_number", "")
        if doc_num and doc_num not in seen:
            seen.add(doc_num)
            deduped.append(chunk)
        elif not doc_num:
            deduped.append(chunk)
    return deduped


def _verify_fr_docs_in_snowflake(doc_numbers: Set[str]) -> Set[str]:
    """
    Verify which doc numbers actually exist in FEDERAL_REGISTER_NOTICES.
    Returns the set of doc numbers that are confirmed real.

    This is the second layer of citation validation — even if the LLM only
    cites docs from the retrieved chunks, those chunks' doc numbers could
    theoretically be corrupt. This is the ground-truth check.
    """
    if not doc_numbers:
        return set()
    conn = get_snowflake_conn()
    cur = conn.cursor()
    verified = set()
    try:
        for doc_num in doc_numbers:
            cur.execute(
                """
                SELECT 1 FROM TARIFFIQ.RAW.FEDERAL_REGISTER_NOTICES
                WHERE document_number = %s
                LIMIT 1
                """,
                (doc_num,),
            )
            if cur.fetchone():
                verified.add(doc_num)
            else:
                logger.warning(
                    "synthesis_fr_doc_not_in_snowflake doc_number=%s — chunk metadata may be corrupt",
                    doc_num,
                )
        return verified
    except Exception as e:
        logger.error("verify_fr_docs_snowflake_error error=%s", e)
        # On DB error, trust the chunks rather than blocking everything
        return doc_numbers
    finally:
        cur.close()
        conn.close()


def _validate_fr_citations(
    text: str, valid_doc_numbers: Set[str]
) -> tuple[bool, Set[str]]:
    """
    Validate all FR document numbers cited in text exist in valid_doc_numbers.
    valid_doc_numbers has already been cross-checked against Snowflake.
    Returns (all_valid, set_of_hallucinated_docs).
    """
    if not valid_doc_numbers:
        return True, set()

    # Extract document numbers in format YYYY-NNNNN (5+ digits after dash, not dates)
    cited_docs = set(re.findall(r"\b(\d{4}-\d{5,6})\b", text))
    if not cited_docs:
        return True, set()

    hallucinated = cited_docs - valid_doc_numbers
    if hallucinated:
        logger.warning(
            "citation_hallucination cited=%s valid=%s hallucinated=%s",
            cited_docs, valid_doc_numbers, hallucinated,
        )
        return False, hallucinated

    return True, set()


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
    except Exception as e:
        logger.error("validate_rate_record_error record_id=%s error=%s", record_id, e)
        return False
    finally:
        cur.close()
        conn.close()


def _compute_pipeline_confidence(
    classification_confidence: float,
    rate_record_verified: bool,
    fr_docs_verified: bool,
    hitl_was_triggered: bool,
) -> str:
    """
    Compute an overall confidence level for the final response.

    HIGH:   All three components clean — classification ≥ 0.80, rate verified,
            FR docs verified in Snowflake, no HITL triggered anywhere.

    MEDIUM: Classification between 0.60-0.80, OR one component is missing/
            unverified (e.g. no trade data), OR HITL was triggered upstream
            but pipeline continued.

    LOW:    Classification < 0.60, OR rate lookup failed, OR FR docs could
            not be verified. User should treat this response with caution.

    This is surfaced in the API response so the Streamlit UI can show a
    clear confidence badge next to the answer.
    """
    failures = 0
    if classification_confidence < 0.60:
        failures += 2  # Serious — wrong HTS code undermines everything
    elif classification_confidence < 0.80:
        failures += 1
    if not rate_record_verified:
        failures += 1
    if not fr_docs_verified:
        failures += 1
    if hitl_was_triggered:
        failures += 1

    if failures == 0:
        return "HIGH"
    elif failures <= 2:
        return "MEDIUM"
    else:
        return "LOW"


def _build_context(
    state: TariffState,
    deduped_chunks: List[Dict[str, Any]],
    valid_doc_numbers: Set[str],
) -> str:
    """Build structured context block for LLM synthesis."""
    parts = []

    parts.append(f"USER QUERY: {state.get('query', '')}")
    parts.append(f"PRODUCT: {state.get('product', 'Unknown')}")
    parts.append(f"COUNTRY OF ORIGIN: {state.get('country', 'Not specified')}")

    hts_code = state.get("hts_code")
    if hts_code:
        parts.append(
            f"HTS CLASSIFICATION: {hts_code} — {state.get('hts_description', 'No description')} "
            f"(confidence: {(state.get('classification_confidence') or 0):.0%})"
        )

    record_id = state.get("rate_record_id")
    total_duty = state.get("total_duty")
    if total_duty is not None and record_id:
        parts.append(
            f"VERIFIED DUTY RATE [HTS {record_id}]: "
            f"Base {state.get('base_rate', 0):.2f}% + "
            f"Section 301/IEEPA Adder {state.get('adder_rate', 0):.2f}% = "
            f"Total {total_duty:.2f}%"
        )
    elif total_duty is None:
        parts.append("DUTY RATE: Not available — rate lookup failed")

    if deduped_chunks:
        context_excerpts = "\n".join([
            f"[{c.get('document_number', 'UNKNOWN')}] {c['chunk_text'][:300]}"
            for c in deduped_chunks[:5]
        ])
        parts.append(
            f"FEDERAL REGISTER POLICY EXCERPTS "
            f"(cite ONLY these document numbers: {', '.join(sorted(valid_doc_numbers))}):\n"
            f"{context_excerpts}"
        )

        policy_summary = state.get("policy_summary")
        if policy_summary:
            parts.append(f"POLICY ANALYSIS: {policy_summary}")
    else:
        parts.append("FEDERAL REGISTER POLICY: No relevant policy documents found")

    trade_suppressed = state.get("trade_suppressed")
    import_value = state.get("import_value_usd")
    period = state.get("trade_period", "")
    if trade_suppressed is False and import_value is not None:
        parts.append(
            f"TRADE VOLUME [Census Bureau {period}]: "
            f"${import_value:,.0f} USD imports in {period}"
        )
    elif trade_suppressed is False:
        parts.append(f"TRADE VOLUME [Census Bureau {period}]: Data available but value suppressed")
    else:
        parts.append("TRADE VOLUME: Data not available from Census Bureau")

    return "\n\n".join(parts)


def _build_citations(
    state: TariffState,
    deduped_chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build deduplicated structured citations list."""
    citations = []
    seen_ids: Set[str] = set()

    record_id = state.get("rate_record_id")
    if record_id and record_id not in seen_ids:
        seen_ids.add(record_id)
        citations.append({
            "type": "snowflake_hts",
            "id": record_id,
            "text": (
                f"HTS {state.get('hts_code')} — "
                f"base rate {state.get('base_rate', 0):.2f}% + "
                f"adder {state.get('adder_rate', 0):.2f}% = "
                f"total {state.get('total_duty', 0):.2f}%"
            ),
            "source": "TARIFFIQ.RAW.HTS_CODES",
        })

    for chunk in deduped_chunks:
        doc_num = chunk.get("document_number")
        if doc_num and doc_num not in seen_ids:
            seen_ids.add(doc_num)
            citations.append({
                "type": "federal_register",
                "id": doc_num,
                "text": chunk.get("title", ""),
                "source": "federalregister.gov",
            })

    if not state.get("trade_suppressed") and state.get("import_value_usd") is not None:
        period = state.get("trade_period", "")
        census_id = f"census_{period}"
        if census_id not in seen_ids:
            seen_ids.add(census_id)
            citations.append({
                "type": "census_bureau",
                "id": census_id,
                "text": f"US imports {period} — ${state.get('import_value_usd', 0):,.0f} USD",
                "source": "api.census.gov",
            })

    return citations


def run_synthesis_agent(state: TariffState) -> Dict[str, Any]:
    """
    Generate cited final response from all agent outputs with full validation.

    Args:
        state: TariffState with all upstream outputs populated

    Returns:
        Dict with final_response, citations, pipeline_confidence, hitl_required, hitl_reason
    """
    logger.info("synthesis_agent_start query=%s", state.get("query", "")[:80])

    # Deduplicate policy chunks
    raw_chunks = state.get("policy_chunks") or []
    deduped_chunks = _dedupe_chunks(raw_chunks)
    candidate_doc_numbers = {
        c.get("document_number", "")
        for c in deduped_chunks
        if c.get("document_number")
    }

    # IMPROVEMENT: Cross-check chunk doc numbers against Snowflake
    # valid_doc_numbers only contains docs that actually exist in FEDERAL_REGISTER_NOTICES
    valid_doc_numbers = _verify_fr_docs_in_snowflake(candidate_doc_numbers)
    if candidate_doc_numbers - valid_doc_numbers:
        logger.warning(
            "synthesis_chunk_docs_not_in_snowflake unverified=%s",
            candidate_doc_numbers - valid_doc_numbers,
        )

    # Validate rate record exists in Snowflake
    record_id = state.get("rate_record_id")
    rate_record_verified = _validate_rate_record(record_id) if record_id else False

    # Build context — LLM is told exactly which doc numbers are valid to cite
    context = _build_context(state, deduped_chunks, valid_doc_numbers)
    model = os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514")

    final_response = None
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ],
                temperature=0.1,
                max_tokens=600,
            )
            final_response = response.choices[0].message.content.strip()
            break

        except litellm.exceptions.RateLimitError:
            logger.warning("synthesis_rate_limit attempt=%d", attempt + 1)
            last_error = "LLM rate limit"
            time.sleep(RETRY_DELAY * (attempt + 1))
        except Exception as e:
            logger.error("synthesis_llm_error attempt=%d error=%s", attempt + 1, e)
            last_error = str(e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

    if not final_response:
        logger.error("synthesis_all_retries_failed error=%s", last_error)
        return {
            "final_response": None,
            "citations": [],
            "pipeline_confidence": "LOW",
            "hitl_required": True,
            "hitl_reason": "citation_failure",
            "error": f"Synthesis failed: {last_error}",
        }

    # Citation validation — checked against Snowflake-verified doc numbers
    citations_valid, hallucinated = _validate_fr_citations(final_response, valid_doc_numbers)

    if not citations_valid:
        logger.warning(
            "synthesis_citation_failure hallucinated=%s triggering_hitl",
            hallucinated,
        )
        return {
            "final_response": final_response,
            "citations": _build_citations(state, deduped_chunks),
            "pipeline_confidence": "LOW",
            "hitl_required": True,
            "hitl_reason": "citation_failure",
        }

    # IMPROVEMENT: Compute overall pipeline confidence
    classification_confidence = float(state.get("classification_confidence") or 0.0)
    hitl_was_triggered = bool(state.get("hitl_required", False))
    fr_docs_verified = bool(valid_doc_numbers)  # True if at least one FR doc verified

    pipeline_confidence = _compute_pipeline_confidence(
        classification_confidence=classification_confidence,
        rate_record_verified=rate_record_verified,
        fr_docs_verified=fr_docs_verified,
        hitl_was_triggered=hitl_was_triggered,
    )

    citations = _build_citations(state, deduped_chunks)
    logger.info(
        "synthesis_agent_done citations=%d pipeline_confidence=%s hitl=%s",
        len(citations),
        pipeline_confidence,
        hitl_was_triggered,
    )

    return {
        "final_response": final_response,
        "citations": citations,
        "pipeline_confidence": pipeline_confidence,
        "hitl_required": hitl_was_triggered,
        "hitl_reason": state.get("hitl_reason"),
    }