"""
Query Agent — TariffIQ Pipeline Step 1

Parses raw user query into structured product + country using LiteLLM/Claude.
- Redis caching: identical queries return instantly (24h TTL)
- Product alias map: "EV" → "electric vehicles", "chips" → "semiconductors"
- Country alias map: "PRC" → "China", "ROK" → "South Korea"
- Prompt injection detection
- Retry with exponential backoff
- Robust JSON extraction
"""

import json
import logging
import os
import re
import time
from typing import Dict, Any, Optional

import litellm

from agents.state import TariffState

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAYS = [0.5, 1.0, 2.0]

INJECTION_PATTERNS = [
    r"ignore (previous|above|all) instructions",
    r"you are now",
    r"act as",
    r"forget (everything|all)",
    r"system prompt",
    r"jailbreak",
]

TARIFF_SIGNALS = [
    "tariff", "duty", "import", "customs", "hts", "rate", "tax",
    "from", "sourcing", "buy", "purchase", "product", "goods",
    "section 301", "section 232", "ieepa",
]

PRODUCT_ALIASES: Dict[str, str] = {
    "ev": "electric vehicles", "evs": "electric vehicles",
    "electric car": "electric vehicles", "electric cars": "electric vehicles",
    "bev": "electric vehicles", "bevs": "electric vehicles",
    "chips": "semiconductors", "microchips": "semiconductors",
    "integrated circuits": "semiconductors", "ics": "semiconductors",
    "solar cells": "solar panels", "pv modules": "solar panels",
    "photovoltaics": "solar panels",
    "notebook computers": "laptops", "notebooks": "laptops",
    "phones": "smartphones", "mobile phones": "smartphones",
    "cell phones": "smartphones", "iphones": "smartphones",
    "android phones": "smartphones",
    "lcd": "flat panel displays", "oled displays": "flat panel displays",
    "flat panel": "flat panel displays",
    "aluminium": "aluminum", "aluminum products": "aluminum",
    "steel products": "steel", "flat rolled steel": "steel",
    "crude": "crude oil", "petroleum": "crude oil",
}

COUNTRY_ALIASES: Dict[str, str] = {
    "prc": "China", "people's republic of china": "China",
    "mainland china": "China", "chinese": "China",
    "rok": "South Korea", "republic of korea": "South Korea",
    "korean": "South Korea", "korea": "South Korea",
    "uk": "United Kingdom", "great britain": "United Kingdom",
    "britain": "United Kingdom", "british": "United Kingdom",
    "uae": "United Arab Emirates",
    "taiwanese": "Taiwan", "republic of china": "Taiwan",
    "vietnamese": "Vietnam", "japanese": "Japan",
    "german": "Germany", "french": "France", "italian": "Italy",
    "brazilian": "Brazil", "indian": "India", "mexican": "Mexico",
    "thai": "Thailand", "canadian": "Canada",
}

SYSTEM_PROMPT = """You are a query parser for a US import tariff intelligence platform.

Extract the imported PRODUCT and source COUNTRY from the user query.
Normalize product to a concise trade noun phrase. Normalize country to standard English name.

Return ONLY valid JSON:
{"product": "<product name>", "country": "<country name or null>"}

Rules:
- product must never be null
- country is null if not mentioned
- No explanation, markdown, or extra text

Examples:
- "tariff on EVs from PRC?" → {"product": "electric vehicles", "country": "China"}
- "import chips from Taiwan" → {"product": "semiconductors", "country": "Taiwan"}
- "steel tariff ROK" → {"product": "steel", "country": "South Korea"}
- "aluminum duties Mexico" → {"product": "aluminum", "country": "Mexico"}"""


def _get_redis():
    try:
        import redis
        c = redis.Redis(
            host=os.environ.get("REDIS_HOST", "redis"),
            port=int(os.environ.get("REDIS_PORT", "6379")),
            socket_connect_timeout=3, socket_timeout=3,
        )
        c.ping()
        return c
    except Exception:
        return None


def _cache_get(query: str) -> Optional[Dict]:
    r = _get_redis()
    if not r:
        return None
    try:
        v = r.get(f"tariffiq:query_agent:{query.lower().strip()}")
        if v:
            logger.info("query_agent_cache_hit query=%s", query[:50])
            return json.loads(v)
    except Exception:
        pass
    return None


def _cache_set(query: str, result: Dict) -> None:
    r = _get_redis()
    if not r:
        return
    try:
        r.setex(f"tariffiq:query_agent:{query.lower().strip()}", 86400, json.dumps(result))
    except Exception:
        pass


def _validate(query: str) -> tuple[bool, Optional[str]]:
    if not query or not query.strip():
        return False, "Empty query"
    if len(query.strip()) < 5:
        return False, "Query too short"
    if len(query) > 500:
        return False, "Query too long"
    q = query.lower()
    for p in INJECTION_PATTERNS:
        if re.search(p, q):
            return False, "Invalid query format"
    if not any(s in q for s in TARIFF_SIGNALS):
        logger.warning("query_agent_low_tariff_signal query=%s", query[:50])
    return True, None


def _extract_json(raw: str) -> Optional[Dict]:
    if not raw:
        return None
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{[^{}]*"product"[^{}]*\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def _norm_product(p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    n = " ".join(p.lower().strip().split())
    return PRODUCT_ALIASES.get(n, n) if n else None


def _norm_country(c: Optional[str]) -> Optional[str]:
    if not c:
        return None
    cleaned = re.sub(r"^(made in|imported from|from|in)\s+", "", c.strip(), flags=re.IGNORECASE).strip()
    if not cleaned:
        return None
    alias = COUNTRY_ALIASES.get(cleaned.lower())
    if alias:
        return alias
    return " ".join(cleaned.split()).title()


def run_query_agent(state: TariffState) -> Dict[str, Any]:
    query = state.get("query", "").strip()
    logger.info("query_agent_start query=%s", query[:100])

    ok, reason = _validate(query)
    if not ok:
        return {"product": None, "country": None, "error": f"Query rejected: {reason}"}

    cached = _cache_get(query)
    if cached:
        return cached

    model = os.environ["LLM_MODEL"]
    last_error = None

    for attempt, delay in enumerate(RETRY_DELAYS[:MAX_RETRIES]):
        try:
            resp = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.0, max_tokens=150,
            )
            raw = resp.choices[0].message.content
            parsed = _extract_json(raw)

            if not parsed:
                last_error = f"Failed to parse: {raw[:100]}"
                if attempt < MAX_RETRIES - 1:
                    time.sleep(delay)
                continue

            product = _norm_product(parsed.get("product"))
            country = _norm_country(parsed.get("country"))

            if not product:
                last_error = "No product extracted"
                if attempt < MAX_RETRIES - 1:
                    time.sleep(delay)
                continue

            logger.info("query_agent_done product=%s country=%s", product, country)
            result = {"product": product, "country": country}
            _cache_set(query, result)
            return result

        except litellm.exceptions.RateLimitError:
            last_error = "LLM rate limit"
            time.sleep(delay * 2)
        except litellm.exceptions.APIConnectionError:
            last_error = "LLM connection error"
            time.sleep(delay)
        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(delay)

    logger.error("query_agent_all_retries_failed error=%s", last_error)
    return {"product": None, "country": None, "error": f"Query agent failed: {last_error}"}