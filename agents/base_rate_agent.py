"""
Base Rate Agent — Pipeline Step 3

Pure SQL lookup of MFN general_rate from HTS_CODES.
Zero LLM calls. Zero NOTICE table reads. Zero regex on ambiguous snippets.

Answers one question only: "What is the base MFN duty rate for this HTS code?"
The Section 301/IEEPA adder is computed in adder_rate_agent (Step 5)
AFTER policy_agent has retrieved the actual Federal Register chunks.

Redis cache: 1-hour TTL (rate records are stable within a day).
"""

import json
import logging
import os
from typing import Dict, Any, Optional

from agents.state import TariffState
from agents import tools

logger = logging.getLogger(__name__)

CACHE_TTL = 3_600  # 1 h


def _redis():
    try:
        import redis
        c = redis.Redis(
            host=os.environ.get("REDIS_HOST", "redis"),
            port=int(os.environ.get("REDIS_PORT", "6379")),
            socket_connect_timeout=2, socket_timeout=2,
        )
        c.ping()
        return c
    except Exception:
        return None


def _cache_get(hts_code: str) -> Optional[Dict]:
    r = _redis()
    if not r:
        return None
    try:
        raw = r.get(f"tariffiq:base_rate:{hts_code}")
        if raw:
            logger.info("base_rate_cache_hit hts=%s", hts_code)
            return json.loads(raw)
    except Exception:
        pass
    return None


def _cache_set(hts_code: str, result: Dict) -> None:
    r = _redis()
    if not r:
        return
    try:
        if result.get("rate_record_id"):
            r.setex(f"tariffiq:base_rate:{hts_code}", CACHE_TTL, json.dumps(result))
    except Exception:
        pass


def run_base_rate_agent(state: TariffState) -> Dict[str, Any]:
    hts_code = (state.get("hts_code") or "").strip()
    country = state.get("country")

    if not hts_code:
        return {
            "base_rate": None, "mfn_rate": None,
            "fta_rate": None, "fta_program": None, "fta_applied": False,
            "rate_record_id": None, "hts_footnotes": [],
            "error": "No HTS code for base rate lookup",
        }

    logger.info("base_rate_agent_start hts=%s country=%s", hts_code, country)

    cached = _cache_get(hts_code)
    if cached:
        return cached

    result = tools.hts_base_rate_lookup(hts_code, country=country)

    if result is None:
        msg = f"Base rate lookup failed for HTS {hts_code}"
        logger.warning(msg)
        return {
            "base_rate": None, "mfn_rate": None,
            "fta_rate": None, "fta_program": None, "fta_applied": False,
            "rate_record_id": None, "hts_footnotes": [],
            "error": msg,
        }

    if result.get("fta_applied"):
        logger.info("base_rate_agent_fta hts=%s country=%s mfn=%.4f fta=%.4f program=%s",
                    hts_code, country, result["mfn_rate"], result["base_rate"], result["fta_program"])
    else:
        logger.info("base_rate_agent_done hts=%s resolved=%s base=%.4f",
                    hts_code, result["hts_code"], result["base_rate"])

    out = {
        "base_rate": result["base_rate"],
        "mfn_rate": result["mfn_rate"],
        "fta_rate": result["fta_rate"],
        "fta_program": result["fta_program"],
        "fta_applied": result["fta_applied"],
        "rate_record_id": result["hts_code"],
        "hts_footnotes": result.get("footnotes", []),
    }
    _cache_set(hts_code, out)
    return out