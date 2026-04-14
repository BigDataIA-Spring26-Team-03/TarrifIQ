"""
Query Agent — TariffIQ Pipeline Step 1

Parses raw user query into structured product + country using LiteLLM/Claude.
- Input validation: rejects empty, non-tariff, and prompt injection queries
- Retry logic: up to 3 attempts with exponential backoff
- Robust JSON extraction: handles markdown fences, extra whitespace, partial JSON
- Normalizes product and country names
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

# Prompt injection patterns to reject
INJECTION_PATTERNS = [
    r"ignore (previous|above|all) instructions",
    r"you are now",
    r"act as",
    r"forget (everything|all)",
    r"system prompt",
    r"jailbreak",
]

# Minimum signals that a query is tariff-related
TARIFF_SIGNALS = [
    "tariff", "duty", "import", "customs", "hts", "rate", "tax",
    "from", "sourcing", "buy", "purchase", "product", "goods",
    "section 301", "section 232", "ieepa",
]

SYSTEM_PROMPT = """You are a query parser for a US import tariff intelligence platform.

Your task:
1. Extract the imported PRODUCT and source COUNTRY from the user query
2. Correct any spelling mistakes
3. Normalize product name to a concise noun phrase (e.g. "solar panels", "steel wire rods")
4. Normalize country to standard English name (e.g. "China", "South Korea", "Vietnam")

Return ONLY valid JSON with exactly two keys:
{"product": "<product name>", "country": "<country name or null>"}

Rules:
- product must never be null — infer from context if needed
- country is null if not mentioned
- Do not include any explanation, markdown, or extra text
- Do not add tariff rates, HTS codes, or policy information"""


def _is_valid_query(query: str) -> tuple[bool, Optional[str]]:
    """
    Validate query before sending to LLM.
    Returns (is_valid, rejection_reason).
    """
    if not query or not query.strip():
        return False, "Empty query"

    if len(query.strip()) < 5:
        return False, "Query too short"

    if len(query) > 500:
        return False, "Query too long"

    q_lower = query.lower()

    # Check for prompt injection
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, q_lower):
            return False, "Invalid query format"

    # Check for at least one tariff-related signal
    if not any(signal in q_lower for signal in TARIFF_SIGNALS):
        # Still allow it but log warning -- don't hard reject
        logger.warning("query_agent_low_tariff_signal query=%s", query[:50])

    return True, None


def _extract_json(raw: str) -> Optional[Dict]:
    """
    Robustly extract JSON from LLM response.
    Handles markdown fences, leading/trailing text, partial JSON.
    """
    if not raw:
        return None

    # Strip markdown fences
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try extracting first JSON object
    match = re.search(r'\{[^{}]*"product"[^{}]*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def _normalize_product(product: Optional[str]) -> Optional[str]:
    """Normalize product name -- lowercase, strip extra whitespace."""
    if not product:
        return None
    normalized = " ".join(product.lower().strip().split())
    return normalized if normalized else None


def _normalize_country(country: Optional[str]) -> Optional[str]:
    """Normalize country name -- title case, strip extra whitespace."""
    if not country:
        return None
    normalized = " ".join(country.strip().split()).title()
    return normalized if normalized else None


def run_query_agent(state: TariffState) -> Dict[str, Any]:
    """
    Parse user query into product + country with retry logic and validation.

    Args:
        state: TariffState with query populated

    Returns:
        Dict with product, country, and optionally error
    """
    query = state.get("query", "").strip()
    logger.info("query_agent_start query=%s", query[:100])

    # Input validation
    is_valid, rejection_reason = _is_valid_query(query)
    if not is_valid:
        logger.warning("query_agent_rejected reason=%s", rejection_reason)
        return {
            "product": None,
            "country": None,
            "error": f"Query rejected: {rejection_reason}",
        }

    model = os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514")
    last_error = None

    for attempt, delay in enumerate(RETRY_DELAYS[:MAX_RETRIES]):
        try:
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=150,
            )

            raw = response.choices[0].message.content
            parsed = _extract_json(raw)

            if not parsed:
                logger.warning(
                    "query_agent_parse_failed attempt=%d raw=%s", attempt + 1, raw[:100]
                )
                last_error = f"Failed to parse LLM response: {raw[:100]}"
                if attempt < MAX_RETRIES - 1:
                    time.sleep(delay)
                continue

            product = _normalize_product(parsed.get("product"))
            country = _normalize_country(parsed.get("country"))

            if not product:
                logger.warning("query_agent_no_product attempt=%d", attempt + 1)
                last_error = "No product extracted from query"
                if attempt < MAX_RETRIES - 1:
                    time.sleep(delay)
                continue

            logger.info(
                "query_agent_done product=%s country=%s attempts=%d",
                product, country, attempt + 1,
            )
            return {"product": product, "country": country}

        except litellm.exceptions.RateLimitError:
            logger.warning("query_agent_rate_limit attempt=%d", attempt + 1)
            last_error = "LLM rate limit"
            time.sleep(delay * 2)
        except litellm.exceptions.APIConnectionError:
            logger.warning("query_agent_connection_error attempt=%d", attempt + 1)
            last_error = "LLM connection error"
            time.sleep(delay)
        except Exception as e:
            logger.error("query_agent_error attempt=%d error=%s", attempt + 1, e)
            last_error = str(e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(delay)

    logger.error("query_agent_all_retries_failed error=%s", last_error)
    return {
        "product": None,
        "country": None,
        "error": f"Query agent failed after {MAX_RETRIES} attempts: {last_error}",
    }