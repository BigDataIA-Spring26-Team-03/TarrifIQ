"""
Classification Agent — Pipeline Step 2

4-layer HTS code resolution. All data access via tools.py.
LLM calls via ModelRouter. Vector search via HybridRetriever.

  Layer 0: ModelRouter(HTS_CLASSIFICATION) — product → HTS terms + heading
  Layer 1: tools.alias_lookup()            — PRODUCT_ALIASES exact match
  Layer 2: tools.hts_keyword_search()      — SQL LIKE on HTS descriptions
  Layer 3: HybridRetriever.search_hts()    — ChromaDB semantic search

Triggers HITL if final confidence < 0.80.
"""

import asyncio
import json
import logging
import os
import re
from typing import Dict, Any, List, Optional, Tuple

from agents.state import TariffState
from agents import tools

logger = logging.getLogger(__name__)

HITL_THRESHOLD = 0.80

TRANSLATE_PROMPT = """Given this common product name, return the HTS classification details.

Common name: {product}

Return ONLY valid JSON:
{{"hts_heading": "XXXXXX", "technical_terms": ["term1", "term2", "term3"], "hts_chapter": "XX"}}

Examples:
- "automobiles"       → {{"hts_heading": "8703",   "technical_terms": ["motor vehicles", "cylinder capacity"], "hts_chapter": "87"}}
- "solar panels"      → {{"hts_heading": "854143", "technical_terms": ["photovoltaic", "modules", "panels"], "hts_chapter": "85"}}
- "laptops"           → {{"hts_heading": "847130", "technical_terms": ["portable", "automatic data processing", "weighing"], "hts_chapter": "84"}}
- "steel"             → {{"hts_heading": "7208",   "technical_terms": ["flat-rolled", "iron", "steel"], "hts_chapter": "72"}}
- "semiconductors"    → {{"hts_heading": "854231", "technical_terms": ["integrated circuits", "processors"], "hts_chapter": "85"}}
- "electric vehicles" → {{"hts_heading": "870380", "technical_terms": ["electric motor", "propulsion", "vehicles"], "hts_chapter": "87"}}

Rules:
- hts_heading: 4 or 6 digits, never starting 98 or 99
- hts_chapter: 01–97 only
- technical_terms: 2–4 words appearing in HTS legal descriptions"""


def _layer0_translate(product: str) -> Tuple[List[str], Optional[str], Optional[str]]:
    """LLM translates product name → HTS technical terms + heading hint."""
    from services.llm.router import get_router, TaskType
    router = get_router()
    try:
        loop = asyncio.new_event_loop()
        try:
            resp = loop.run_until_complete(
                router.complete(
                    task=TaskType.HTS_CLASSIFICATION,
                    messages=[{"role": "user", "content": TRANSLATE_PROMPT.format(product=product)}],
                )
            )
        finally:
            loop.close()

        raw = re.sub(r"```(?:json)?", "", resp.choices[0].message.content.strip()).strip()
        parsed = json.loads(raw)
        terms = parsed.get("technical_terms", [])
        chapter = str(parsed.get("hts_chapter", "")).zfill(2)
        heading = str(parsed.get("hts_heading", "")).strip()

        try:
            chapter = chapter if 1 <= int(chapter) <= 97 else None
        except ValueError:
            chapter = None
        if not re.match(r"^\d{4,6}$", heading) or heading[:2] in ("98", "99"):
            heading = None

        logger.info("classify_layer0 product=%s terms=%s chapter=%s heading=%s",
                    product, terms, chapter, heading)
        return terms or [], chapter, heading

    except Exception as e:
        logger.warning("classify_layer0_failed product=%s error=%s", product, e)
        return [], None, None


def _layer1_alias(product: str) -> Optional[Tuple[str, float]]:
    result = tools.alias_lookup(product)
    if result:
        logger.info("classify_layer1_hit product=%s hts=%s conf=%.2f", product, result[0], result[1])
    return result


def _layer2_keyword(
    product: str,
    technical_terms: List[str],
    hts_chapter: Optional[str],
    hts_heading: Optional[str],
) -> Optional[Tuple[str, str, float]]:
    # Heading-scoped first
    if hts_heading:
        rows = tools.hts_keyword_search(
            query=" ".join(technical_terms) if technical_terms else product,
            limit=1, heading_filter=hts_heading,
        )
        if rows:
            r = rows[0]
            logger.info("classify_layer2_heading product=%s hts=%s", product, r["hts_code"])
            return r["hts_code"], r["description"], 0.85

    # Term-by-term
    search_terms = technical_terms if technical_terms else [
        w for w in re.sub(r"[^a-zA-Z0-9 ]", " ", product.lower()).split() if len(w) > 3
    ]
    for term in search_terms:
        rows = tools.hts_keyword_search(query=term, limit=3, chapter_filter=hts_chapter)
        if rows:
            best = rows[0]
            conf = 0.80 if hts_chapter else 0.75
            logger.info("classify_layer2_keyword product=%s term=%s hts=%s conf=%.2f",
                        product, term, best["hts_code"], conf)
            return best["hts_code"], best["description"], conf
    return None


def _layer3_vector(
    product: str,
    technical_terms: List[str],
    hts_chapter: Optional[str],
) -> Optional[Tuple[str, str, float]]:
    """Uses HybridRetriever.search_hts() — the singleton from services/retrieval/hybrid.py."""
    from services.retrieval.hybrid import get_retriever
    retriever = get_retriever()

    query = " ".join(technical_terms) if technical_terms else product
    rows = retriever.search_hts(query=query, chapter=hts_chapter, top_k=5)
    if not rows:
        rows = retriever.search_hts(query=query, top_k=5)

    # Filter out chapters 98/99
    rows = [r for r in rows if not str(r.get("hts_code", "")).startswith(("98", "99"))]

    if rows:
        best = rows[0]
        distance = best.get("score", 0.5)  # HybridRetriever uses "score" key
        conf = max(0.50, min(0.70, 1.0 - distance))
        logger.info("classify_layer3_vector product=%s hts=%s dist=%.3f conf=%.2f",
                    product, best["hts_code"], distance, conf)
        return best["hts_code"], best["description"], conf
    return None


def _verify_and_shorten(hts_code: str) -> Optional[str]:
    if tools.hts_verify(hts_code):
        return hts_code
    parts = hts_code.split(".")
    while len(parts) > 2:
        parts = parts[:-1]
        shorter = ".".join(parts)
        if tools.hts_verify(shorter):
            logger.info("classify_hts_shortened original=%s verified=%s", hts_code, shorter)
            return shorter
    return None


def run_classification_agent(state: TariffState) -> Dict[str, Any]:
    product = (state.get("product") or "").strip()
    if not product:
        return {
            "hts_code": None, "hts_description": None,
            "classification_confidence": 0.0,
            "hitl_required": True, "hitl_reason": "low_confidence",
            "error": "No product extracted from query",
        }

    logger.info("classification_agent_start product=%s", product)

    technical_terms, hts_chapter, hts_heading = _layer0_translate(product)

    hts_code: Optional[str] = None
    confidence: float = 0.0
    description: Optional[str] = None

    result = _layer1_alias(product)
    if result:
        hts_code, confidence = result
        description = tools.hts_description(hts_code)

    if not hts_code:
        r2 = _layer2_keyword(product, technical_terms, hts_chapter, hts_heading)
        if r2:
            hts_code, description, confidence = r2

    if not hts_code:
        r3 = _layer3_vector(product, technical_terms, hts_chapter)
        if r3:
            hts_code, description, confidence = r3
            sf_desc = tools.hts_description(hts_code)
            if sf_desc:
                description = sf_desc

    if not hts_code:
        logger.warning("classification_agent_no_result product=%s", product)
        return {
            "hts_code": None, "hts_description": None,
            "classification_confidence": 0.0,
            "hitl_required": True, "hitl_reason": "low_confidence",
        }

    verified = _verify_and_shorten(hts_code)
    if not verified:
        return {
            "hts_code": None, "hts_description": None,
            "classification_confidence": 0.0,
            "hitl_required": True, "hitl_reason": "low_confidence",
            "error": f"HTS {hts_code} not found in Snowflake",
        }

    if verified != hts_code:
        description = tools.hts_description(verified) or description
        confidence = max(0.0, confidence - 0.05)
        hts_code = verified

    hitl = confidence < HITL_THRESHOLD
    logger.info("classification_agent_done hts=%s conf=%.2f hitl=%s", hts_code, confidence, hitl)

    return {
        "hts_code": hts_code,
        "hts_description": description,
        "classification_confidence": confidence,
        "hitl_required": hitl,
        "hitl_reason": "low_confidence" if hitl else None,
        "_product_for_feedback": product,
    }