"""
Classification Agent — TariffIQ Pipeline Step 2

Four-layer HTS code resolution:
  Layer 0: LLM translates common product name to HTS technical terminology + heading hint
  Layer 1: Exact alias lookup in PRODUCT_ALIASES (Snowflake) — confidence 0.95
  Layer 2: Keyword search using technical terms against HTS_CODES — confidence 0.75-0.85
  Layer 3: Semantic search via ChromaDB federal_register collection — confidence 0.50-0.70

Safety:
  - Parameterized queries only (no SQL injection)
  - Chapters 98/99 excluded (special provisions, not product codes)
  - HTS code verified in Snowflake before returning
  - All layers fail gracefully, never crash pipeline
  - No hardcoded product mappings anywhere

Triggers HITL if final confidence < 0.80.

IMPROVEMENT: After a successful pipeline run (hitl_required=False, rate found),
_write_alias_feedback() writes the confirmed product → hts_code mapping to
PRODUCT_ALIASES. This makes the system self-improving — repeated queries for
the same product bypass all 4 layers and resolve instantly via Layer 1.
"""

import json
import logging
import os
import re
import time
from typing import Dict, Any, Optional, Tuple, List

import chromadb
import litellm

from ingestion.connection import get_snowflake_conn
from ingestion.embedder import Embedder
from agents.state import TariffState

logger = logging.getLogger(__name__)

HITL_CONFIDENCE_THRESHOLD = 0.80
HTS_CODE_PATTERN = re.compile(r"^\d{2,4}(\.\d{2}){0,4}$")

TRANSLATE_PROMPT = """You are a US Harmonized Tariff Schedule (HTS) classification expert.

Given a common product name, return:
1. The most specific HTS heading you can identify (4 or 6 digits)
2. HTS technical search terms (legal language used in HTS descriptions)
3. The 2-digit chapter number (01-97 only)

Common name: {product}

Return ONLY valid JSON:
{{"hts_heading": "XXXXXX", "technical_terms": ["term1", "term2", "term3"], "hts_chapter": "XX"}}

Examples:
- "automobiles" -> {{"hts_heading": "8703", "technical_terms": ["motor vehicles", "cylinder capacity"], "hts_chapter": "87"}}
- "solar panels" -> {{"hts_heading": "854143", "technical_terms": ["photovoltaic", "modules", "panels"], "hts_chapter": "85"}}
- "laptops" -> {{"hts_heading": "847130", "technical_terms": ["portable", "automatic data processing", "weighing"], "hts_chapter": "84"}}
- "coffee" -> {{"hts_heading": "0901", "technical_terms": ["coffee", "roasted", "beans"], "hts_chapter": "09"}}
- "steel" -> {{"hts_heading": "7208", "technical_terms": ["flat-rolled", "iron", "steel"], "hts_chapter": "72"}}
- "aluminum" -> {{"hts_heading": "7606", "technical_terms": ["aluminum", "plates", "sheets"], "hts_chapter": "76"}}
- "semiconductors" -> {{"hts_heading": "854231", "technical_terms": ["integrated circuits", "processors", "controllers"], "hts_chapter": "85"}}
- "electric vehicles" -> {{"hts_heading": "870380", "technical_terms": ["electric motor", "propulsion", "vehicles"], "hts_chapter": "87"}}

Rules:
- hts_heading can be 4 OR 6 digits — use 6 digits when you know the specific subheading
- hts_heading must never start with 98 or 99
- hts_chapter must be between 01 and 97, never 98 or 99
- technical_terms must be words that appear in HTS legal descriptions
- Return exactly 2-4 technical terms"""


def _get_chroma_client() -> chromadb.HttpClient:
    host = os.environ.get("CHROMADB_HOST", "chromadb")
    port = int(os.environ.get("CHROMADB_PORT", 8000))
    return chromadb.HttpClient(host=host, port=port)


def _is_valid_hts_code(code: str) -> bool:
    if not code:
        return False
    return bool(HTS_CODE_PATTERN.match(code.strip()))


def _normalize_hts_code(code: str) -> str:
    return code.strip()


def _layer0_llm_translate(product: str) -> Tuple[List[str], Optional[str], Optional[str]]:
    """
    Layer 0: Use LLM to translate common product name to HTS technical terms + heading hint.
    Returns (technical_terms, hts_chapter, hts_heading) or ([], None, None) on failure.
    """
    try:
        response = litellm.completion(
            model=os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514"),
            messages=[
                {
                    "role": "user",
                    "content": TRANSLATE_PROMPT.format(product=product),
                }
            ],
            temperature=0.0,
            max_tokens=150,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip()

        parsed = json.loads(raw)
        terms = parsed.get("technical_terms", [])
        chapter = str(parsed.get("hts_chapter", "")).zfill(2)
        heading = str(parsed.get("hts_heading", "")).strip()

        # Validate chapter
        try:
            chapter_int = int(chapter)
            if not (1 <= chapter_int <= 97):
                chapter = None
        except ValueError:
            chapter = None

        # Validate heading (4 or 6 digits, not starting with 98/99)
        if not re.match(r"^\d{4,6}$", heading):
            heading = None
        elif heading[:2] in ("98", "99"):
            heading = None

        if not terms:
            return [], None, None

        logger.info(
            "classification_layer0_translate product=%s terms=%s chapter=%s heading=%s",
            product, terms, chapter, heading,
        )
        return terms, chapter, heading

    except Exception as e:
        logger.warning("classification_layer0_failed product=%s error=%s", product, e)
        return [], None, None


def _layer1_alias_lookup(product: str) -> Optional[Tuple[str, float]]:
    """
    Layer 1: Exact alias lookup in PRODUCT_ALIASES.
    Only populated via HITL feedback — never hardcoded.
    """
    conn = get_snowflake_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT hts_code, confidence
            FROM TARIFFIQ.RAW.PRODUCT_ALIASES
            WHERE LOWER(alias) = LOWER(%s)
            LIMIT 1
            """,
            (product.strip(),),
        )
        row = cur.fetchone()
        if row and row[0]:
            hts_code = _normalize_hts_code(row[0])
            confidence = float(row[1]) if row[1] else 0.95
            logger.info(
                "classification_layer1_hit product=%s hts=%s conf=%.2f",
                product, hts_code, confidence,
            )
            return hts_code, confidence
        return None
    except Exception as e:
        logger.error("classification_layer1_error product=%s error=%s", product, e)
        return None
    finally:
        cur.close()
        conn.close()


def _layer2_keyword_search(
    product: str,
    technical_terms: List[str],
    hts_chapter: Optional[str],
    hts_heading: Optional[str] = None,
) -> Optional[Tuple[str, str, float]]:
    """
    Layer 2: When heading hint available, find first code within that heading.
    Otherwise fall back to keyword search with chapter filter.
    """
    conn = get_snowflake_conn()
    cur = conn.cursor()
    try:
        # When we have a specific heading, use it directly
        if hts_heading:
            n = len(hts_heading)
            if n == 6:
                # 6-digit: format as XXXX.XX for Snowflake
                formatted = f"{hts_heading[:4]}.{hts_heading[4:]}"
                left_filter = "LEFT(hts_code, 7) = %s"
                filter_val = formatted
            else:
                # 4-digit heading
                left_filter = "LEFT(hts_code, 4) = %s"
                filter_val = hts_heading

            cur.execute(
                f"""
                SELECT hts_code, description
                FROM TARIFFIQ.RAW.HTS_CODES
                WHERE {left_filter}
                  AND is_header_row = FALSE
                  AND general_rate IS NOT NULL
                  AND chapter NOT IN ('98', '99')
                ORDER BY hts_code ASC
                LIMIT 1
                """,
                (filter_val,),
            )
            row = cur.fetchone()
            if row:
                hts_code = _normalize_hts_code(row[0])
                description = row[1]
                logger.info(
                    "classification_layer2_heading_hit product=%s hts=%s desc=%s",
                    product, hts_code, description[:60],
                )
                return hts_code, description, 0.85

        # Fallback: keyword search
        search_terms = technical_terms if technical_terms else [
            w for w in re.sub(r"[^a-zA-Z0-9 ]", " ", product.lower()).split()
            if len(w) > 3
        ]
        if not search_terms:
            return None

        like_conditions = " OR ".join(
            ["LOWER(description) LIKE %s"] * min(len(search_terms), 4)
        )
        params = tuple(f"%{t.lower()}%" for t in search_terms[:4])

        chapter_filter = ""
        chapter_params = ()
        if hts_chapter:
            chapter_filter = "AND chapter = %s"
            chapter_params = (hts_chapter,)

        match_score_parts = " + ".join(
            [f"CASE WHEN LOWER(description) LIKE %s THEN 1 ELSE 0 END"
             for _ in search_terms[:4]]
        )
        starts_with_check = "CASE WHEN LOWER(description) LIKE %s THEN 2 ELSE 0 END"

        cur.execute(
            f"""
            SELECT hts_code, description,
                   ({starts_with_check} + {match_score_parts}) AS match_score
            FROM TARIFFIQ.RAW.HTS_CODES
            WHERE ({like_conditions})
              {chapter_filter}
              AND is_header_row = FALSE
              AND general_rate IS NOT NULL
              AND chapter NOT IN ('98', '99')
              AND LEFT(hts_code, 2) NOT IN ('98', '99')
            ORDER BY match_score DESC,
                     SUBSTRING(hts_code, 6, 2) ASC,
                     LENGTH(hts_code) ASC
            LIMIT 5
            """,
            (f"{search_terms[0].lower()}%",) + params + params + chapter_params,
        )
        rows = cur.fetchall()

        if not rows and hts_chapter:
            cur.execute(
                f"""
                SELECT hts_code, description,
                       ({starts_with_check} + {match_score_parts}) AS match_score
                FROM TARIFFIQ.RAW.HTS_CODES
                WHERE ({like_conditions})
                  AND is_header_row = FALSE
                  AND general_rate IS NOT NULL
                  AND chapter NOT IN ('98', '99')
                  AND LEFT(hts_code, 2) NOT IN ('98', '99')
                ORDER BY match_score DESC,
                         SUBSTRING(hts_code, 6, 2) ASC,
                         LENGTH(hts_code) ASC
                LIMIT 5
                """,
                (f"{search_terms[0].lower()}%",) + params + params,
            )
            rows = cur.fetchall()

        if not rows:
            return None

        best = rows[0]
        hts_code = _normalize_hts_code(best[0])
        description = best[1]
        match_score = best[2]
        confidence = 0.85 if match_score >= 2 else 0.75

        logger.info(
            "classification_layer2_hit product=%s hts=%s desc=%s conf=%.2f",
            product, hts_code, description[:60], confidence,
        )
        return hts_code, description, confidence

    except Exception as e:
        logger.error("classification_layer2_error product=%s error=%s", product, e)
        return None
    finally:
        cur.close()
        conn.close()


def _layer3_semantic_search(product: str, technical_terms: List[str]) -> Optional[Tuple[str, str, float]]:
    """
    Layer 3: Semantic search via ChromaDB using enriched query.
    """
    try:
        query_text = product
        if technical_terms:
            query_text = f"{product} {' '.join(technical_terms[:3])}"

        embedder = Embedder()
        query_vec = embedder.embed_batch([query_text])[0]

        chroma = _get_chroma_client()
        collection = chroma.get_collection("federal_register")

        results = collection.query(
            query_embeddings=[query_vec],
            n_results=5,
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"][0]:
            return None

        for i, meta in enumerate(results["metadatas"][0]):
            hts_code = (meta.get("hts_code") or "").strip()
            if not hts_code or not _is_valid_hts_code(hts_code):
                continue

            distance = results["distances"][0][i]
            similarity = max(0.0, 1.0 - distance)
            confidence = round(similarity * 0.70, 2)

            if confidence < 0.40:
                continue

            hts_code = _normalize_hts_code(hts_code)
            title = meta.get("title", "")

            logger.info(
                "classification_layer3_hit product=%s hts=%s conf=%.2f",
                product, hts_code, confidence,
            )
            return hts_code, title, confidence

        return None

    except Exception as e:
        logger.warning("classification_layer3_failed error=%s", e)
        return None


def _get_hts_description(hts_code: str) -> Optional[str]:
    """Fetch HTS description, trying progressively shorter codes."""
    conn = get_snowflake_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT description FROM TARIFFIQ.RAW.HTS_CODES WHERE hts_code = %s LIMIT 1",
            (hts_code,),
        )
        row = cur.fetchone()
        if row:
            return row[0]

        parts = hts_code.split(".")
        while len(parts) > 2:
            parts = parts[:-1]
            shorter = ".".join(parts)
            cur.execute(
                "SELECT description FROM TARIFFIQ.RAW.HTS_CODES WHERE hts_code = %s LIMIT 1",
                (shorter,),
            )
            row = cur.fetchone()
            if row:
                return row[0]
        return None
    except Exception as e:
        logger.error("get_hts_description_error hts=%s error=%s", hts_code, e)
        return None
    finally:
        cur.close()
        conn.close()


def _verify_hts_exists(hts_code: str) -> bool:
    """Verify HTS code exists in Snowflake."""
    conn = get_snowflake_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT 1 FROM TARIFFIQ.RAW.HTS_CODES WHERE hts_code = %s LIMIT 1",
            (hts_code,),
        )
        return cur.fetchone() is not None
    except Exception:
        return False
    finally:
        cur.close()
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# IMPROVEMENT: Self-improving alias write-back
# ─────────────────────────────────────────────────────────────────────────────

def _write_alias_feedback(
    product: str,
    hts_code: str,
    confidence: float,
    rate_was_found: bool,
) -> None:
    """
    Write confirmed product → hts_code mapping to PRODUCT_ALIASES so future
    queries for the same product hit Layer 1 instantly instead of going through
    all four layers again.

    Only called when:
      - hitl_required is False (classification confidence >= 0.80)
      - rate_was_found is True (Rate Agent found a real Snowflake record)

    Both conditions together mean the full pipeline verified this mapping end-
    to-end — the HTS code is correct AND it has a real rate. Safe to cache.

    Confidence stored is the classification confidence, capped at 0.95 (we
    never claim alias certainty higher than a direct alias lookup would give).
    """
    if not rate_was_found:
        logger.debug(
            "alias_feedback_skipped product=%s hts=%s reason=no_rate_found",
            product, hts_code,
        )
        return

    stored_confidence = min(confidence, 0.95)
    conn = get_snowflake_conn()
    cur = conn.cursor()
    try:
        # Check if alias already exists to avoid duplicate writes
        cur.execute(
            """
            SELECT hts_code, confidence
            FROM TARIFFIQ.RAW.PRODUCT_ALIASES
            WHERE LOWER(alias) = LOWER(%s)
            LIMIT 1
            """,
            (product.strip(),),
        )
        existing = cur.fetchone()

        if existing:
            existing_hts, existing_conf = existing
            if existing_hts == hts_code:
                # Same mapping — update confidence if ours is higher
                if stored_confidence > float(existing_conf or 0):
                    cur.execute(
                        """
                        UPDATE TARIFFIQ.RAW.PRODUCT_ALIASES
                        SET confidence = %s, updated_at = CURRENT_TIMESTAMP()
                        WHERE LOWER(alias) = LOWER(%s)
                        """,
                        (stored_confidence, product.strip()),
                    )
                    logger.info(
                        "alias_feedback_updated product=%s hts=%s conf=%.2f",
                        product, hts_code, stored_confidence,
                    )
            else:
                # Different HTS code for same alias — this is a conflict.
                # Do NOT overwrite; let HITL resolve. Log for visibility.
                logger.warning(
                    "alias_feedback_conflict product=%s existing_hts=%s new_hts=%s — skipping write",
                    product, existing_hts, hts_code,
                )
            return

        # New alias — insert it
        cur.execute(
            """
            INSERT INTO TARIFFIQ.RAW.PRODUCT_ALIASES (alias, hts_code, confidence, created_at, updated_at)
            VALUES (%s, %s, %s, CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP())
            """,
            (product.strip(), hts_code, stored_confidence),
        )
        logger.info(
            "alias_feedback_written product=%s hts=%s conf=%.2f",
            product, hts_code, stored_confidence,
        )

    except Exception as e:
        # Write-back failure must never crash the pipeline
        logger.error(
            "alias_feedback_error product=%s hts=%s error=%s",
            product, hts_code, e,
        )
    finally:
        cur.close()
        conn.close()


def run_classification_agent(state: TariffState) -> Dict[str, Any]:
    """
    Resolve HTS code using 4-layer lookup with LLM-assisted translation.
    No hardcoded product mappings.

    After a fully verified pipeline run, writes the confirmed alias to
    PRODUCT_ALIASES for self-improvement. The caller (pipeline orchestrator)
    is responsible for calling _write_alias_feedback() once the Rate Agent
    confirms a rate was found — see run_classification_agent_with_feedback().
    """
    product = state.get("product")
    if not product or not product.strip():
        return {
            "hts_code": None,
            "hts_description": None,
            "classification_confidence": 0.0,
            "hitl_required": True,
            "hitl_reason": "low_confidence",
            "error": "No product extracted from query",
        }

    product = product.strip()
    logger.info("classification_agent_start product=%s", product)

    technical_terms, hts_chapter, hts_heading = _layer0_llm_translate(product)

    hts_code = None
    confidence = 0.0
    description = None

    # Layer 1 — alias lookup
    result = _layer1_alias_lookup(product)
    if result:
        hts_code, confidence = result
        description = _get_hts_description(hts_code)

    # Layer 2 — heading-first or keyword search
    if not hts_code:
        result = _layer2_keyword_search(product, technical_terms, hts_chapter, hts_heading)
        if result:
            hts_code, description, confidence = result

    # Layer 3 — semantic search
    if not hts_code:
        result = _layer3_semantic_search(product, technical_terms)
        if result:
            hts_code, description, confidence = result
            snowflake_desc = _get_hts_description(hts_code)
            if snowflake_desc:
                description = snowflake_desc

    if not hts_code:
        logger.warning("classification_agent_no_result product=%s", product)
        return {
            "hts_code": None,
            "hts_description": None,
            "classification_confidence": 0.0,
            "hitl_required": True,
            "hitl_reason": "low_confidence",
        }

    # Final validation
    if not _verify_hts_exists(hts_code):
        parts = hts_code.split(".")
        found = False
        while len(parts) > 2:
            parts = parts[:-1]
            shorter = ".".join(parts)
            if _verify_hts_exists(shorter):
                hts_code = shorter
                description = _get_hts_description(shorter) or description
                confidence = max(0.0, confidence - 0.05)
                found = True
                break
        if not found:
            return {
                "hts_code": None,
                "hts_description": None,
                "classification_confidence": 0.0,
                "hitl_required": True,
                "hitl_reason": "low_confidence",
                "error": f"HTS code {hts_code} not found in Snowflake",
            }

    hitl_required = confidence < HITL_CONFIDENCE_THRESHOLD

    logger.info(
        "classification_agent_done hts=%s desc=%s confidence=%.2f hitl=%s",
        hts_code, (description or "")[:50], confidence, hitl_required,
    )

    return {
        "hts_code": hts_code,
        "hts_description": description,
        "classification_confidence": confidence,
        "hitl_required": hitl_required,
        "hitl_reason": "low_confidence" if hitl_required else None,
        # Carry product forward so the pipeline orchestrator can call
        # _write_alias_feedback() once the Rate Agent result is known.
        "_product_for_feedback": product,
    }


def maybe_write_alias_feedback(classification_result: Dict[str, Any], rate_found: bool) -> None:
    """
    Called by the pipeline orchestrator AFTER the Rate Agent completes.
    Writes alias to PRODUCT_ALIASES only when both conditions are met:
      1. Classification resolved without HITL (confidence >= 0.80)
      2. Rate Agent found a verified Snowflake record (rate_found=True)

    Usage in pipeline orchestrator (e.g. graph.py or pipeline.py):
        classification_out = run_classification_agent(state)
        state.update(classification_out)
        rate_out = run_rate_agent(state)
        state.update(rate_out)
        rate_found = rate_out.get("rate_record_id") is not None
        maybe_write_alias_feedback(classification_out, rate_found)
    """
    if classification_result.get("hitl_required"):
        return
    product = classification_result.get("_product_for_feedback")
    hts_code = classification_result.get("hts_code")
    confidence = classification_result.get("classification_confidence", 0.0)
    if product and hts_code:
        _write_alias_feedback(product, hts_code, confidence, rate_found)