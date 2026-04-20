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

TRANSLATE_PROMPT = """Given this common product name, return HTS classification details AND search synonyms.

Common name: {product}

Return ONLY valid JSON:
{{"hts_heading": "XXXXXX", "technical_terms": ["term1", "term2", "term3"], "hts_chapter": "XX", "synonyms": ["alt1", "alt2", "alt3"]}}

Examples:
- "chili peppers"    → {{"hts_heading": "0904",   "technical_terms": ["capsicum", "dried pepper", "crushed"], "hts_chapter": "09", "synonyms": ["chili", "peppers", "capsicum fruits", "pimenta"]}}
- "shrimp"          → {{"hts_heading": "0306",   "technical_terms": ["crustaceans", "frozen shrimp", "prawns"], "hts_chapter": "03", "synonyms": ["prawns", "crustaceans", "frozen shrimp"]}}
- "automobiles"     → {{"hts_heading": "8703",   "technical_terms": ["motor vehicles", "cylinder capacity"], "hts_chapter": "87", "synonyms": ["cars", "passenger vehicles", "motor cars"]}}
- "solar panels"    → {{"hts_heading": "854143", "technical_terms": ["photovoltaic", "modules", "panels"], "hts_chapter": "85", "synonyms": ["photovoltaic cells", "PV modules", "solar cells"]}}
- "laptops"         → {{"hts_heading": "847130", "technical_terms": ["portable", "automatic data processing", "weighing"], "hts_chapter": "84", "synonyms": ["notebook computers", "portable computers", "ADP machines"]}}
- "steel"           → {{"hts_heading": "7208",   "technical_terms": ["flat-rolled", "iron", "steel"], "hts_chapter": "72", "synonyms": ["flat rolled steel", "iron products", "steel sheets"]}}
- "semiconductors"  → {{"hts_heading": "854231", "technical_terms": ["integrated circuits", "processors"], "hts_chapter": "85", "synonyms": ["integrated circuits", "microprocessors", "chips"]}}

Rules:
- hts_heading: 4 or 6 digits, never starting 98 or 99
- hts_chapter: 01–97 only
- technical_terms: 2–4 words that APPEAR in HTS legal descriptions
- synonyms: 2–5 alternate common names, related terms, or HTS description words that could match this product in a keyword search. Include both formal HTS terminology and common trade names."""


def _layer0_translate(product: str) -> Tuple[List[str], Optional[str], Optional[str], List[str]]:
    """LLM translates product name → HTS technical terms + heading hint + synonyms."""
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
        synonyms = parsed.get("synonyms", [])
        chapter = str(parsed.get("hts_chapter", "")).zfill(2)
        heading = str(parsed.get("hts_heading", "")).strip()

        try:
            chapter = chapter if 1 <= int(chapter) <= 97 else None
        except ValueError:
            chapter = None
        if not re.match(r"^\d{4,6}$", heading) or heading[:2] in ("98", "99"):
            heading = None

        logger.info("classify_layer0 product=%s terms=%s synonyms=%s chapter=%s heading=%s",
                    product, terms, synonyms, chapter, heading)
        return terms or [], chapter, heading, synonyms or []

    except Exception as e:
        logger.warning("classify_layer0_failed product=%s error=%s", product, e)
        return [], None, None, []


def _layer1_alias(product: str) -> Optional[Tuple[str, float]]:
    """
    Only use alias if confidence is 0.95 (manually curated entries).
    Skip aliases with confidence < 0.95 — let LLM layers handle those.
    This prevents the alias table from short-circuiting classification
    for products that need proper LLM reasoning.
    """
    result = tools.alias_lookup(product)
    if result and result[1] >= 0.95:
        logger.info("classify_layer1_hit product=%s hts=%s conf=%.2f", product, result[0], result[1])
        return result
    if result:
        logger.info("classify_layer1_skip product=%s conf=%.2f below threshold", product, result[1])
    return None


def _layer2_keyword(
    product: str,
    technical_terms: List[str],
    hts_chapter: Optional[str],
    hts_heading: Optional[str],
    synonyms: Optional[List[str]] = None,
) -> Optional[Tuple[str, str, float]]:
    # All search terms: product + technical terms + synonyms
    all_terms = [product] + (technical_terms or []) + (synonyms or [])
    # Deduplicate preserving order
    seen_terms = set()
    search_pool = []
    for t in all_terms:
        tl = t.lower().strip()
        if tl and tl not in seen_terms:
            seen_terms.add(tl)
            search_pool.append(t)

    # Heading-scoped first — try every term against the heading hint
    if hts_heading:
        for q in search_pool:
            rows = tools.hts_keyword_search(query=q, limit=1, heading_filter=hts_heading)
            if rows:
                r = rows[0]
                logger.info("classify_layer2_heading product=%s hts=%s query=%s", product, r["hts_code"], q)
                return r["hts_code"], r["description"], 0.85

    # Chapter-scoped: try every term against the chapter
    for term in search_pool:
        rows = tools.hts_keyword_search(query=term, limit=3, chapter_filter=hts_chapter)
        if rows:
            best = rows[0]
            conf = 0.80 if hts_chapter else 0.75
            logger.info("classify_layer2_keyword product=%s term=%s hts=%s conf=%.2f",
                        product, term, best["hts_code"], conf)
            return best["hts_code"], best["description"], conf

    # Last resort: unscoped search with all terms
    for term in search_pool:
        rows = tools.hts_keyword_search(query=term, limit=3)
        if rows:
            best = rows[0]
            logger.info("classify_layer2_unscoped product=%s term=%s hts=%s",
                        product, term, best["hts_code"])
            return best["hts_code"], best["description"], 0.70

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


VALIDATE_PROMPT = """Does this HTS classification make sense for this product?

Product: {product}
HTS Code: {hts_code}
HTS Description: {description}

Answer with ONLY one word: YES or NO

Rules:
- YES if the HTS description plausibly covers the product (even if not perfect)
- NO if the HTS description is clearly wrong for this product
- Examples of NO: "chili peppers" classified as "leguminous vegetables", "laptops" classified as "telephones"
- Examples of YES: "chili peppers" as "dried chili peppers", "laptops" as "portable automatic data processing machines"
"""


def _semantic_validate(product: str, hts_code: str, description: str) -> bool:
    """
    Ask LLM to validate HTS classification makes sense for the product.
    Returns True if valid, False if clearly wrong.
    Falls back to True if LLM call fails (fail open).
    """
    if not product or not hts_code or not description:
        return True
    try:
        from services.llm.router import get_router, TaskType
        router = get_router()
        loop = asyncio.new_event_loop()
        try:
            resp = loop.run_until_complete(
                router.complete(
                    task=TaskType.HTS_CLASSIFICATION,
                    messages=[{
                        "role": "user",
                        "content": VALIDATE_PROMPT.format(
                            product=product,
                            hts_code=hts_code,
                            description=description,
                        ),
                    }],
                )
            )
        finally:
            loop.close()
        answer = resp.choices[0].message.content.strip().upper()
        valid = answer.startswith("YES")
        if not valid:
            logger.warning("semantic_validation_failed product=%s hts=%s desc=%s answer=%s",
                           product, hts_code, description[:60], answer)
        return valid
    except Exception as e:
        logger.debug("semantic_validate_error product=%s error=%s", product, e)
        return True  # fail open


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

    technical_terms, hts_chapter, hts_heading, synonyms = _layer0_translate(product)

    hts_code: Optional[str] = None
    confidence: float = 0.0
    description: Optional[str] = None

    result = _layer1_alias(product)
    if result:
        hts_code, confidence = result
        description = tools.hts_description(hts_code)

    if not hts_code:
        r2 = _layer2_keyword(product, technical_terms, hts_chapter, hts_heading, synonyms)
        if r2:
            hts_code, description, confidence = r2

    if not hts_code:
        # Combine technical terms + synonyms for richer vector query
        all_search_terms = technical_terms + synonyms
        r3 = _layer3_vector(product, all_search_terms, hts_chapter)
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

    # ── Semantic validation — does HTS description match the product? ────────
    # Catches cases where classification found a wrong code with high confidence
    # e.g. "chili peppers" → "leguminous vegetables" at 80% confidence.
    # Only run below 0.90 to avoid extra LLM calls on clear alias hits.
    if confidence < 0.90 and description:
        is_valid = _semantic_validate(product, hts_code, description)
        if not is_valid:
            logger.warning("classification_semantic_mismatch product=%s hts=%s — HITL",
                           product, hts_code)
            return {
                "hts_code": hts_code,
                "hts_description": description,
                "classification_confidence": confidence * 0.5,
                "hitl_required": True,
                "hitl_reason": "semantic_mismatch",
            }

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