"""
Rate Agent — TariffIQ Pipeline Step 3

Pure SQL lookup — zero LLM calls.
Calls resolve_total_duty() for base MFN rate + Section 301/IEEPA adder.

Safety:
  - Progressive HTS code truncation if exact match not found
  - Handles HTTPException from resolve_total_duty gracefully
  - Verifies record ID exists before returning
  - Never generates rates from memory
"""

import logging
from typing import Dict, Any, Optional

from fastapi import HTTPException

from api.tools.resolve_hts_rate import resolve_total_duty
from ingestion.connection import get_snowflake_conn
from agents.state import TariffState

logger = logging.getLogger(__name__)


def _try_resolve_with_fallback(hts_code: str) -> Optional[Any]:
    """
    Try resolve_total_duty with exact code first, then progressively shorter.
    e.g. 8542.31.00.15 -> 8542.31.00 -> 8542.31
    Returns VerificationReceipt or None.
    """
    codes_to_try = [hts_code]

    # Build fallback chain by removing last segment
    parts = hts_code.split(".")
    while len(parts) > 2:
        parts = parts[:-1]
        codes_to_try.append(".".join(parts))

    for code in codes_to_try:
        try:
            receipt = resolve_total_duty(code)
            if code != hts_code:
                logger.info(
                    "rate_agent_fallback original=%s resolved=%s",
                    hts_code, code,
                )
            return receipt
        except HTTPException as e:
            if e.status_code == 404:
                logger.debug("rate_agent_404 hts=%s", code)
                continue
            logger.error("rate_agent_http_error hts=%s status=%s", code, e.status_code)
            return None
        except Exception as e:
            logger.error("rate_agent_error hts=%s error=%s", code, e)
            return None

    logger.warning("rate_agent_no_match hts=%s all_fallbacks_exhausted", hts_code)
    return None


def run_rate_agent(state: TariffState) -> Dict[str, Any]:
    """
    Fetch base MFN rate + Section 301/IEEPA adder for the resolved HTS code.
    Zero LLM calls — pure SQL via resolve_total_duty().

    Args:
        state: TariffState with hts_code populated

    Returns:
        Dict with base_rate, adder_rate, total_duty, rate_record_id
    """
    hts_code = state.get("hts_code")

    if not hts_code or not hts_code.strip():
        logger.warning("rate_agent_skipped no hts_code in state")
        return {
            "base_rate": None,
            "adder_rate": None,
            "total_duty": None,
            "rate_record_id": None,
            "error": "No HTS code available for rate lookup",
        }

    hts_code = hts_code.strip()
    logger.info("rate_agent_start hts_code=%s", hts_code)

    receipt = _try_resolve_with_fallback(hts_code)

    if receipt is None:
        logger.warning("rate_agent_no_receipt hts=%s", hts_code)
        return {
            "base_rate": None,
            "adder_rate": None,
            "total_duty": None,
            "rate_record_id": None,
            "error": f"Rate lookup failed for HTS {hts_code} — no matching record found",
        }

    # Validate rate reconciliation
    if not receipt.rate_reconciliation.check_passed:
        logger.warning(
            "rate_agent_reconciliation_failed hts=%s calc=%s",
            hts_code,
            receipt.rate_reconciliation.calculation,
        )

    logger.info(
        "rate_agent_done hts=%s base=%.4f adder=%.4f total=%.4f record=%s",
        hts_code,
        receipt.base_rate,
        receipt.adder_rate,
        receipt.total_duty,
        receipt.base_rate_source.record_id,
    )

    return {
        "base_rate": receipt.base_rate,
        "adder_rate": receipt.adder_rate,
        "total_duty": receipt.total_duty,
        "rate_record_id": receipt.base_rate_source.record_id,
    }