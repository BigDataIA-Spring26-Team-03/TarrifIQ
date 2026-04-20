"""
Query Agent — Pipeline Step 1

Parses raw user query → {product, country} using the ModelRouter.

Ambiguity detection is fully dynamic — no hardcoded product lists.
Uses SQL lookup against HTS_CODES to determine if a product term
maps to multiple distinct HTS headings. If so, returns clarification
with real HTS subcategories from the database.
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Dict, Any, List, Optional

from agents.state import TariffState
from agents import tools

logger = logging.getLogger(__name__)

# ── Query Intent Detection ─────────────────────────────────────────────────────
# Detect special query intents before running the standard pipeline.
# Returns intent type and extracted entities, or None for standard pipeline.

CHANGE_PATTERNS = [
    r"(has.*changed|have.*changed|did.*change|when did.*change)",
    r"(since when|how long|history of tariff|historical rate|tariff history)",
    r"(what happened to|what will happen|upcoming tariff|future tariff|expect.*tariff)",
    r"(recent.*tariff|latest.*tariff|new.*tariff|tariff.*update|tariff.*modified)",
]

COMPARE_PATTERNS = [
    r"(cheaper to import from|which is cheaper|is it cheaper)",
    r"(which country.*import|where should.*import|best.*source.*import)",
    r"(compare.*tariff|tariff.*comparison|vs\.?\s+\w+.*tariff|tariff.*vs\.?\s+\w+)",
    r"(china.*or.*vietnam|vietnam.*or.*china|china.*vs.*mexico|mexico.*vs.*china)",
]

EXEMPT_PATTERNS = [
    r"(exempt from|excluded from|not subject to|exclusion from)",
    r"(which products.*exempt|what products.*excluded|products.*not.*tariff)",
]


def _detect_intent(query: str) -> Optional[Dict[str, Any]]:
    """
    Detect special query intents. Returns intent dict or None for standard pipeline.
    
    Intents:
      - "rate_change": "has the tariff on X changed?" / "when did tariffs on X change?"
      - "country_compare": "cheaper from China or Vietnam?"
      - "exemption_check": "what products are exempt from Section 301?"
      - None: standard "what is the tariff on X from Y?" pipeline
    """
    q = query.lower().strip()
    
    # Check for rate change intent
    if any(re.search(p, q) for p in CHANGE_PATTERNS):
        return {"intent": "rate_change"}
    
    # Check for country comparison intent  
    if any(re.search(p, q) for p in COMPARE_PATTERNS):
        return {"intent": "country_compare"}
    
    # Check for exemption intent
    if any(re.search(p, q) for p in EXEMPT_PATTERNS):
        return {"intent": "exemption_check"}
    
    return None


MAX_RETRIES = 3
RETRY_DELAYS = [0.5, 1.0, 2.0]
SEMANTIC_THRESHOLD = 0.92
CACHE_TTL = 86_400  # 24 h

INJECTION_PATTERNS = [
    r"ignore (previous|above|all) instructions",
    r"you are now", r"act as", r"forget (everything|all)",
    r"system prompt", r"jailbreak",
    r"disregard (all|previous)", r"override (instructions|system)",
]

TARIFF_SIGNALS = [
    "tariff", "duty", "import", "customs", "hts", "rate", "tax",
    "from", "sourcing", "buy", "purchase", "product", "goods",
    "section 301", "section 232", "ieepa", "trade", "supply chain",
]

# Abbreviations and technical shortforms only.
# The LLM handles full product names correctly — we only fix
# things the LLM consistently gets wrong (acronyms, shortforms).
PRODUCT_ALIASES: Dict[str, str] = {
    "ev": "electric vehicles", "evs": "electric vehicles",
    "bev": "electric vehicles", "phev": "electric vehicles",
    "ics": "semiconductors",
    "cpu": "semiconductors", "gpu": "semiconductors",
    "pv": "solar panels",
    "lcd": "flat panel displays",
    "hrc": "hot-rolled steel", "crc": "cold-rolled steel",
    "lng": "liquefied natural gas",
    "li-ion": "lithium-ion batteries",
}

COUNTRY_ALIASES: Dict[str, str] = {
    "prc": "China", "people's republic of china": "China",
    "mainland china": "China", "chinese": "China",
    "rok": "South Korea", "republic of korea": "South Korea",
    "korean": "South Korea", "korea": "South Korea",
    "uk": "United Kingdom", "great britain": "United Kingdom",
    "britain": "United Kingdom", "british": "United Kingdom",
    "uae": "United Arab Emirates",
    "taiwanese": "Taiwan", "republic of china": "Taiwan", "roc": "Taiwan",
    "vietnamese": "Vietnam", "viet nam": "Vietnam",
    "japanese": "Japan", "german": "Germany",
    "french": "France", "italian": "Italy",
    "brazilian": "Brazil", "indian": "India",
    "mexican": "Mexico", "thai": "Thailand", "canadian": "Canada",
    "eu": "European Union", "europe": "European Union",
    "asean": None,
}


# ── Redis ─────────────────────────────────────────────────────────────────────

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


def _exact_get(query: str) -> Optional[Dict]:
    r = _redis()
    if not r:
        return None
    try:
        raw = r.get(f"tariffiq:qagent:exact:{query.lower().strip()}")
        if raw:
            logger.info("query_agent_exact_hit query=%s", query[:60])
            return json.loads(raw)
    except Exception:
        pass
    return None


def _exact_set(query: str, result: Dict) -> None:
    r = _redis()
    if not r:
        return
    try:
        r.setex(f"tariffiq:qagent:exact:{query.lower().strip()}", CACHE_TTL, json.dumps(result))
    except Exception:
        pass


def _semantic_get(query: str) -> Optional[Dict]:
    r = _redis()
    if not r:
        return None
    try:
        import numpy as np
        from services.chromadb_init import get_embedder
        keys_raw = r.get("tariffiq:qagent:semkeys")
        if not keys_raw:
            return None
        keys = json.loads(keys_raw)
        if not keys:
            return None
        q_vec = np.array(get_embedder().encode([query])[0], dtype=float)
        q_norm = float(np.linalg.norm(q_vec))
        if q_norm == 0:
            return None
        best_sim, best_key = 0.0, None
        for key in keys:
            vec_raw = r.get(f"tariffiq:qagent:semvec:{key}")
            if not vec_raw:
                continue
            vec = np.array(json.loads(vec_raw), dtype=float)
            norm = float(np.linalg.norm(vec))
            if norm == 0:
                continue
            sim = float(np.dot(q_vec, vec) / (q_norm * norm))
            if sim > best_sim:
                best_sim, best_key = sim, key
        if best_sim >= SEMANTIC_THRESHOLD and best_key:
            val_raw = r.get(f"tariffiq:qagent:semval:{best_key}")
            if val_raw:
                logger.info("query_agent_semantic_hit sim=%.3f query=%s", best_sim, query[:60])
                return json.loads(val_raw)
    except Exception as e:
        logger.debug("semantic_get_failed error=%s", e)
    return None


def _semantic_set(query: str, result: Dict) -> None:
    r = _redis()
    if not r:
        return
    try:
        import numpy as np
        from services.chromadb_init import get_embedder
        safe_key = re.sub(r"[^a-z0-9]", "_", query.lower().strip())[:80]
        vec = get_embedder().encode([query])[0].tolist()
        r.setex(f"tariffiq:qagent:semvec:{safe_key}", CACHE_TTL, json.dumps(vec))
        r.setex(f"tariffiq:qagent:semval:{safe_key}", CACHE_TTL, json.dumps(result))
        keys_raw = r.get("tariffiq:qagent:semkeys")
        keys = json.loads(keys_raw) if keys_raw else []
        if safe_key not in keys:
            keys.append(safe_key)
        if len(keys) > 500:
            old = keys.pop(0)
            r.delete(f"tariffiq:qagent:semvec:{old}", f"tariffiq:qagent:semval:{old}")
        r.setex("tariffiq:qagent:semkeys", CACHE_TTL, json.dumps(keys))
    except Exception as e:
        logger.debug("semantic_set_failed error=%s", e)


# ── Ambiguity detection — dynamic SQL-based, no hardcoding ───────────────────

def _check_ambiguity(product: str, country: Optional[str]) -> Optional[Dict]:
    """
    Dynamically checks if a product term is too broad by querying HTS_CODES.

    Logic:
    - Search HTS_CODES for the product term (limit 30 results)
    - Count distinct 2-digit chapters in results
    - If 3+ chapters → show chapter-level suggestions (very broad term)
    - If 1-2 chapters but 3+ distinct 4-digit headings → show heading-level suggestions
    - Suggestions use parent heading descriptions for context, not sub-descriptions
    - Multi-word products (2+ words) skip this check — already specific enough
    """
    if not product:
        return None

    product_lower = product.lower().strip()

    # Multi-word products are already specific enough
    if len(product_lower.split()) >= 2:
        return None

    try:
        rows = tools.hts_keyword_search(product_lower, limit=30)
    except Exception as e:
        logger.debug("ambiguity_check_failed product=%s error=%s", product, e)
        return None

    if not rows:
        return None

    # Group by chapter and heading
    chapter_map: Dict[str, list] = {}
    heading_map: Dict[str, Dict] = {}

    for r in rows:
        code = r.get("hts_code", "")
        if not code:
            continue
        chapter = code[:2]
        parts = code.split(".")
        heading = parts[0]  # 4-digit heading

        if chapter not in chapter_map:
            chapter_map[chapter] = []
        chapter_map[chapter].append(r)

        if heading not in heading_map:
            heading_map[heading] = r

    # Not ambiguous if 1-2 headings
    if len(heading_map) <= 2:
        return None

    country_suffix = f" from {country}" if country else ""
    suggestions = []

    # Build suggestions — use the most specific meaningful description per heading
    # Strategy: for each heading, find the row with the most descriptive text
    # that doesn't start with "Of" or "Other" (those are sub-descriptions needing context)
    seen_headings: Dict[str, Dict] = {}
    for r in rows:
        code = r.get("hts_code", "")
        if not code:
            continue
        parts = code.split(".")
        heading = parts[0]
        desc = r.get("description", "").strip()
        if not desc:
            continue
        # Prefer descriptions that don't start with relative terms
        is_relative = desc.lower().startswith(("of ", "other", "not ", "nesoi", "except"))
        existing = seen_headings.get(heading)
        if not existing:
            seen_headings[heading] = {**r, "_is_relative": is_relative}
        elif is_relative is False and existing.get("_is_relative") is True:
            # Replace with a more absolute description
            seen_headings[heading] = {**r, "_is_relative": False}

    for heading, row in list(seen_headings.items())[:6]:
        # Try parent heading description first for context
        parent_desc = tools.hts_description(heading)
        desc = parent_desc if (parent_desc and not parent_desc.lower().startswith(
            ("of ", "other", "not ", "nesoi"))) else row.get("description", "")
        desc = desc.strip()
        # Take first meaningful clause
        desc_short = re.split(r"[;:]", desc)[0].strip()
        desc_short = re.sub(r"\s+", " ", desc_short)
        # Truncate if too long
        if len(desc_short) > 60:
            desc_short = desc_short[:57] + "..."
        if len(desc_short) < 5:
            continue
        suggestions.append({
            "label": f"{desc_short} (HTS {heading})",
            "query": f"{desc_short.lower()}{country_suffix}",
        })

    if not suggestions:
        return None

    logger.info("query_agent_ambiguous product=%s chapters=%d headings=%d",
                product, len(chapter_map), len(heading_map))

    return {
        "clarification_needed": True,
        "product": product,
        "country": country,
        "message": (
            f'"{product}" matches {len(heading_map)} different HTS categories with different tariff rates. '
            f"Which type are you asking about{country_suffix}?"
        ),
        "suggestions": suggestions,
    }


# ── Validation + normalisation ─────────────────────────────────────────────────

def _validate(query: str):
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
        logger.warning("query_agent_low_signal query=%s", query[:60])
    return True, None


def _extract_json(raw: str) -> Optional[Dict]:
    if not raw:
        return None
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
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
    if n in PRODUCT_ALIASES:
        return PRODUCT_ALIASES[n]
    for alias, canonical in PRODUCT_ALIASES.items():
        if re.search(r'\b' + re.escape(alias) + r'\b', n) and canonical:
            return canonical
    return n or None


def _norm_country(c: Optional[str]) -> Optional[str]:
    """Normalise country. Router may return 'all' — treat as None."""
    if not c or c.lower() in ("all", "null", "none", ""):
        return None
    cleaned = re.sub(
        r"^(made in|imported from|sourced from|from|in|manufactured in)\s+",
        "", c.strip(), flags=re.IGNORECASE,
    ).strip()
    if not cleaned or cleaned.lower() in ("all", "null", "none"):
        return None
    alias = COUNTRY_ALIASES.get(cleaned.lower())
    if alias is None and cleaned.lower() in COUNTRY_ALIASES:
        return None
    if alias:
        return alias
    return " ".join(cleaned.split()).title()


# ── Main agent ─────────────────────────────────────────────────────────────────

def run_query_agent(state: TariffState) -> Dict[str, Any]:
    query = state.get("query", "").strip()
    logger.info("query_agent_start query=%s", query[:100])

    ok, reason = _validate(query)
    if not ok:
        return {"product": None, "country": None, "error": f"Query rejected: {reason}"}

    # Layer 1: exact cache
    cached = _exact_get(query)
    if cached:
        return cached

    # Layer 2: semantic cache
    sem = _semantic_get(query)
    if sem:
        _exact_set(query, sem)
        return sem

    # Layer 3: ModelRouter
    from services.llm.router import get_router, TaskType
    router = get_router()
    last_error = None

    for attempt, delay in enumerate(RETRY_DELAYS[:MAX_RETRIES]):
        try:
            loop = asyncio.new_event_loop()
            try:
                resp = loop.run_until_complete(
                    router.complete(
                        task=TaskType.QUERY_PARSING,
                        messages=[{"role": "user", "content": query}],
                    )
                )
            finally:
                loop.close()

            raw = resp.choices[0].message.content
            parsed = _extract_json(raw)

            if not parsed:
                last_error = f"JSON parse failed: {raw[:80]}"
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

            # Dynamic ambiguity check — SQL-based, no hardcoding
            clarification = _check_ambiguity(product, country)
            if clarification:
                logger.info("query_agent_ambiguous product=%s — requesting clarification", product)
                return clarification

            # Intent detection — check for special query types
            intent_info = _detect_intent(query)
            if intent_info:
                intent = intent_info["intent"]
                logger.info("query_agent_intent product=%s intent=%s", product, intent)
                result = {"product": product, "country": country, "query_intent": intent}
            else:
                result = {"product": product, "country": country}

            _exact_set(query, result)
            _semantic_set(query, result)
            return result

        except RuntimeError as e:
            last_error = str(e)
            logger.error("query_agent_router_failed error=%s", e)
            break
        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(delay)

    logger.error("query_agent_failed error=%s", last_error)
    return {"product": None, "country": None, "error": f"Query agent failed: {last_error}"}