"""
Rate Agent — TariffIQ Pipeline Step 3

Pure SQL lookup — zero LLM calls.
Calls existing resolve_total_duty() to get base MFN rate + Section 301/IEEPA adder.
Returns rate components with Snowflake record IDs for citation.
"""

import logging
from typing import Dict, Any

from api.tools.resolve_hts_rate import resolve_total_duty
from agents.state import TariffState

logger = logging.getLogger(__name__)


def run_rate_agent(state: TariffState) -> Dict[str, Any]:
    """
    Fetch base rate + adder for the resolved HTS code.

    Args:
        state: TariffState with hts_code populated

    Returns:
        Dict with base_rate, adder_rate, total_duty, rate_record_id
    """
    hts_code = state.get("hts_code")
    if not hts_code:
        logger.warning("rate_agent_skipped no hts_code in state")
        return {
            "base_rate": None,
            "adder_rate": None,
            "total_duty": None,
            "rate_record_id": None,
            "error": "No HTS code available for rate lookup",
        }

    logger.info("rate_agent_start hts_code=%s", hts_code)

    try:
        receipt = resolve_total_duty(hts_code)

        logger.info(
            "rate_agent_done hts=%s base=%.2f adder=%.2f total=%.2f",
            hts_code,
            receipt.base_rate,
            receipt.adder_rate,
            receipt.total_duty,
        )

        return {
            "base_rate": receipt.base_rate,
            "adder_rate": receipt.adder_rate,
            "total_duty": receipt.total_duty,
            "rate_record_id": receipt.base_rate_source.record_id,
        }

    except Exception as e:
        logger.error("rate_agent_failed hts=%s error=%s", hts_code, e)
        return {
            "base_rate": None,
            "adder_rate": None,
            "total_duty": None,
            "rate_record_id": None,
            "error": f"Rate lookup failed: {e}",
        }