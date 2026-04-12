"""
Query Agent — TariffIQ Pipeline Step 1

Parses raw user query into structured product + country using LiteLLM/Claude.
Returns updated TariffState with product and country populated.
"""

import json
import logging
import os
from typing import Dict, Any

import litellm

from agents.state import TariffState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a query parser for a US import tariff platform.
Extract the product and country from the user query and correct any spelling.
Return ONLY valid JSON with exactly two keys: {"product": "...", "country": "..."}.
If no country is mentioned, set country to null.
Do not include any explanation or extra text."""


def run_query_agent(state: TariffState) -> Dict[str, Any]:
    """
    Parse user query into product + country.

    Args:
        state: TariffState with query populated

    Returns:
        Dict with product and country keys
    """
    query = state["query"]
    logger.info("query_agent_start query=%s", query)

    try:
        response = litellm.completion(
            model=os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=100,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)
        product = parsed.get("product") or None
        country = parsed.get("country") or None

        logger.info("query_agent_done product=%s country=%s", product, country)

        return {"product": product, "country": country}

    except Exception as e:
        logger.error("query_agent_failed error=%s", e)
        return {"product": None, "country": None, "error": f"Query agent failed: {e}"}