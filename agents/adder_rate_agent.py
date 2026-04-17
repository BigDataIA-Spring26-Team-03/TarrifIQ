"""
Adder Rate Agent — Pipeline Step 5

Determines the Section 301 / IEEPA / Section 232 adder rate by having
the LLM read the actual policy chunks retrieved by policy_agent.

Uses ModelRouter(TaskType.POLICY_ANALYSIS) — the same model (claude-haiku)
and system prompt that policy_agent uses, which is correct since this is
also a policy document reading task.

WHY THIS AGENT EXISTS
──────────────────────
The old rate_agent computed the adder by running a regex on a short
context_snippet stored in NOTICE_HTS_CODES. That snippet often contains
multiple percentages (2018 rate, 2019 escalation, 2025 IEEPA rate) and
the regex blindly returned the first one.

This agent reads the full policy chunks with an LLM and asks specifically:
"what is the CURRENT effective adder for HTS X from country Y?"
Falls back to regex on chunk text if the LLM fails or returns null.

Redis cache: 1-hour TTL keyed on (hts_code + country).
"""

import asyncio
import json
import logging
import os
import re
from typing import Dict, Any, List, Optional, Set

from agents.state import TariffState
from agents import tools

logger = logging.getLogger(__name__)

CACHE_TTL = 3_600  # 1 h

ADDER_PROMPT = """Given the Federal Register excerpts below, determine the CURRENT effective
Section 301, Section 232, or IEEPA additional duty rate for:

Product HTS code: {hts_code}
Country of origin: {country}

Return ONLY valid JSON:
{{"adder_rate": <number or null>, "document_number": "<FR doc or null>", "basis": "<one sentence>"}}

Rules:
- adder_rate is a percentage NUMBER (e.g. 25.0 for 25%) — not the base MFN rate
- If multiple rates exist, return the MOST RECENT currently effective one
- If this country is not subject to additional duties, return adder_rate: 0
- If context is insufficient, return adder_rate: null
- document_number must come from the excerpts — never fabricate one

Federal Register excerpts:
{context}"""


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


def _cache_key(hts_code: str, country: Optional[str]) -> str:
    return f"tariffiq:adder_rate:{hts_code}:{(country or 'ALL').lower().replace(' ','_')}"


def _cache_get(hts_code: str, country: Optional[str]) -> Optional[Dict]:
    r = _redis()
    if not r:
        return None
    try:
        raw = r.get(_cache_key(hts_code, country))
        if raw:
            logger.info("adder_rate_cache_hit hts=%s country=%s", hts_code, country)
            return json.loads(raw)
    except Exception:
        pass
    return None


def _cache_set(hts_code: str, country: Optional[str], result: Dict) -> None:
    r = _redis()
    if not r:
        return
    try:
        if result.get("adder_method") in ("llm_policy", "none"):
            r.setex(_cache_key(hts_code, country), CACHE_TTL, json.dumps(result))
    except Exception:
        pass


def _build_context(chunks: List[Dict[str, Any]]) -> str:
    lines = []
    for i, chunk in enumerate(chunks[:5], start=1):
        doc = chunk.get("document_number", "UNKNOWN")
        src = chunk.get("source", "USTR").upper()
        pub = chunk.get("publication_date", "") or ""
        text = chunk.get("chunk_text", "")
        lines.append(f"[{i}] {src} | {doc} | {pub}\n{text[:500]}")
    return "\n\n".join(lines)


def _regex_fallback(chunks: List[Dict[str, Any]], hts_code: str) -> float:
    """Last-resort regex on chunk text. Returns 0.0 if nothing plausible found."""
    for chunk in chunks:
        text = chunk.get("chunk_text", "")
        escaped = re.escape(hts_code.strip())
        m = re.search(escaped + r".{0,200}?(\d{1,3}(?:\.\d+)?)\s*%", text, re.DOTALL)
        if m:
            rate = float(m.group(1))
            if 0 < rate <= 200:
                logger.info("adder_rate_regex_hts_match hts=%s rate=%.1f", hts_code, rate)
                return rate
        m2 = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", text)
        if m2:
            rate = float(m2.group(1))
            if 0 < rate <= 200:
                logger.info("adder_rate_regex_broad hts=%s rate=%.1f", hts_code, rate)
                return rate
    return 0.0


def run_adder_rate_agent(state: TariffState) -> Dict[str, Any]:
    hts_code = (state.get("hts_code") or "").strip()
    country = state.get("country")
    base_rate = state.get("base_rate") or 0.0
    policy_chunks = state.get("policy_chunks") or []

    if not hts_code:
        return {
            "adder_rate": 0.0, "adder_doc": None,
            "adder_method": "none",
            "total_duty": base_rate,
        }

    logger.info("adder_rate_agent_start hts=%s country=%s chunks=%d",
                hts_code, country, len(policy_chunks))

    # Cache check (recompute total_duty with current base_rate)
    cached = _cache_get(hts_code, country)
    if cached:
        adder = cached.get("adder_rate") or 0.0
        cached["total_duty"] = round(base_rate + adder, 4)
        return cached

    if not policy_chunks:
        result = {"adder_rate": 0.0, "adder_doc": None, "adder_method": "none",
                  "total_duty": round(base_rate, 4)}
        _cache_set(hts_code, country, result)
        return result

    valid_docs: Set[str] = {
        c.get("document_number", "") for c in policy_chunks
        if c.get("document_number")
    }
    context = _build_context(policy_chunks)

    # LLM call via ModelRouter — POLICY_ANALYSIS task (claude-haiku)
    from services.llm.router import get_router, TaskType
    router = get_router()

    adder_rate: Optional[float] = None
    adder_doc: Optional[str] = None
    method = "none"

    try:
        loop = asyncio.new_event_loop()
        try:
            resp = loop.run_until_complete(
                router.complete(
                    task=TaskType.POLICY_ANALYSIS,
                    messages=[{
                        "role": "user",
                        "content": ADDER_PROMPT.format(
                            hts_code=hts_code,
                            country=country or "unspecified",
                            context=context,
                        ),
                    }],
                )
            )
        finally:
            loop.close()

        raw = re.sub(r"```(?:json)?", "", resp.choices[0].message.content.strip()).strip()
        # Find the first { and last } to extract the JSON object
        # Handles trailing text and nested structures from the LLM
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"No JSON object found in response: {raw[:100]}")
        parsed = json.loads(raw[start:end+1])

        raw_rate = parsed.get("adder_rate")
        raw_doc = parsed.get("document_number")

        if raw_rate is not None:
            try:
                rate_val = float(raw_rate)
                if 0 <= rate_val <= 200:
                    adder_rate = rate_val
                    method = "llm_policy"
            except (ValueError, TypeError):
                pass

        # Validate doc number against retrieved chunks only
        if raw_doc and raw_doc in valid_docs:
            adder_doc = raw_doc
        elif raw_doc:
            logger.warning("adder_rate_hallucinated_doc doc=%s — rejecting", raw_doc)

        logger.info("adder_rate_llm hts=%s country=%s rate=%s doc=%s basis=%s",
                    hts_code, country, adder_rate, adder_doc,
                    str(parsed.get("basis", ""))[:80])

    except Exception as e:
        logger.warning("adder_rate_llm_failed hts=%s error=%s", hts_code, e)

    # Regex fallback if LLM failed or returned null
    if adder_rate is None:
        adder_rate = _regex_fallback(policy_chunks, hts_code)
        method = "regex_fallback" if adder_rate > 0 else "none"

    total_duty = round(base_rate + (adder_rate or 0.0), 4)

    logger.info("adder_rate_agent_done hts=%s country=%s base=%.4f adder=%.4f total=%.4f method=%s",
                hts_code, country, base_rate, adder_rate or 0.0, total_duty, method)

    result = {
        "adder_rate": adder_rate or 0.0,
        "adder_doc": adder_doc,
        "adder_method": method,
        "total_duty": total_duty,
    }
    _cache_set(hts_code, country, result)
    return result