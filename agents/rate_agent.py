"""
Rate Agent — TariffIQ Pipeline Step 3

Pure SQL lookup — zero LLM calls.
Calls resolve_total_duty() for base MFN rate + Section 301/IEEPA adder.
Country-aware: passes country so China-specific adders don't apply to Taiwan/Korea/Vietnam.
Progressive HTS fallback: 8542.31.00.15 → 8542.31.00 → 8542.31
"""

import logging
from typing import Dict, Any, Optional

from fastapi import HTTPException

from api.tools.resolve_hts_rate import resolve_total_duty
from agents.state import TariffState

logger = logging.getLogger(__name__)


def _try_resolve(hts_code: str, country: Optional[str] = None) -> Optional[Any]:
    codes = [hts_code]
    parts = hts_code.split(".")
    while len(parts) > 2:
        parts = parts[:-1]
        codes.append(".".join(parts))

    for code in codes:
        try:
            receipt = resolve_total_duty(code, country)
            if code != hts_code:
                logger.info("rate_agent_fallback original=%s resolved=%s", hts_code, code)
            return receipt
        except HTTPException as e:
            if e.status_code == 404:
                continue
            logger.error("rate_agent_http_error hts=%s status=%s", code, e.status_code)
            return None
        except Exception as e:
            logger.error("rate_agent_error hts=%s error=%s", code, e)
            return None

    logger.warning("rate_agent_no_match hts=%s", hts_code)
    return None


def run_rate_agent(state: TariffState) -> Dict[str, Any]:
    hts_code = (state.get("hts_code") or "").strip()
    country = state.get("country")

    if not hts_code:
        return {"base_rate": None, "adder_rate": None, "total_duty": None,
                "rate_record_id": None, "error": "No HTS code"}

    logger.info("rate_agent_start hts=%s country=%s", hts_code, country)
    receipt = _try_resolve(hts_code, country)

    if receipt is None:
        return {"base_rate": None, "adder_rate": None, "total_duty": None,
                "rate_record_id": None, "error": f"Rate lookup failed for HTS {hts_code}"}

    logger.info("rate_agent_done hts=%s country=%s base=%.4f adder=%.4f total=%.4f record=%s",
                hts_code, country, receipt.base_rate, receipt.adder_rate,
                receipt.total_duty, receipt.base_rate_source.record_id)

    return {
        "base_rate": receipt.base_rate,
        "adder_rate": receipt.adder_rate,
        "total_duty": receipt.total_duty,
        "rate_record_id": receipt.base_rate_source.record_id,
    }