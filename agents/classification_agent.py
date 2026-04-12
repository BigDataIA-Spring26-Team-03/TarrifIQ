"""
Classification Agent — TariffIQ Pipeline Step 2

Three-layer HTS code resolution:
  Layer 1: Exact alias lookup in PRODUCT_ALIASES (Snowflake)
  Layer 2: HTS description keyword search in HTS_CODES (Snowflake)
  Layer 3: Semantic search via ChromaDB

Triggers HITL if confidence < 0.80.
"""

import logging
import os
from typing import Dict, Any, Optional, Tuple

import chromadb

from ingestion.connection import get_snowflake_conn
from ingestion.embedder import Embedder
from agents.state import TariffState

logger = logging.getLogger(__name__)

HITL_CONFIDENCE_THRESHOLD = 0.80


def _get_chroma_client() -> chromadb.HttpClient:
    host = os.environ.get("CHROMADB_HOST", "chromadb")
    port = int(os.environ.get("CHROMADB_PORT", 8000))
    return chromadb.HttpClient(host=host, port=port)


def _layer1_alias_lookup(product: str) -> Optional[Tuple[str, float]]:
    """
    Layer 1: Exact alias lookup in PRODUCT_ALIASES table.
    Returns (hts_code, confidence) or None.
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
            (product,),
        )
        row = cur.fetchone()
        if row:
            logger.info("classification_layer1_hit product=%s hts=%s", product, row[0])
            return row[0], float(row[1]) if row[1] else 0.95
        return None
    finally:
        cur.close()
        conn.close()


def _layer2_keyword_search(product: str) -> Optional[Tuple[str, str, float]]:
    """
    Layer 2: Keyword search against HTS_CODES descriptions in Snowflake.
    Returns (hts_code, description, confidence) or None.
    """
    conn = get_snowflake_conn()
    cur = conn.cursor()
    try:
        words = [w for w in product.lower().split() if len(w) > 3]
        if not words:
            return None

        like_clauses = " OR ".join(
            [f"LOWER(description) LIKE '%{w}%'" for w in words[:3]]
        )

        cur.execute(
            f"""
            SELECT hts_code, description
            FROM TARIFFIQ.RAW.HTS_CODES
            WHERE ({like_clauses})
              AND is_header_row = FALSE
              AND general_rate IS NOT NULL
              AND chapter != '99'
              AND LEFT(hts_code, 2) != '99'
            ORDER BY
                CASE WHEN LOWER(description) LIKE '{words[0]}%' THEN 0 ELSE 1 END,
                LENGTH(hts_code) DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
        if row:
            hts_code, description = row[0], row[1]
            confidence = 0.85 if description.lower().startswith(words[0]) else 0.75
            logger.info(
                "classification_layer2_hit product=%s hts=%s conf=%.2f",
                product, hts_code, confidence,
            )
            return hts_code, description, confidence
        return None
    finally:
        cur.close()
        conn.close()


def _layer3_semantic_search(product: str) -> Optional[Tuple[str, str, float]]:
    """
    Layer 3: Semantic search via ChromaDB federal_register collection.
    Returns (hts_code, description, confidence) or None.
    """
    try:
        embedder = Embedder()
        query_vec = embedder.embed_batch([product])[0]

        chroma = _get_chroma_client()
        collection = chroma.get_collection("federal_register")

        results = collection.query(
            query_embeddings=[query_vec],
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"][0]:
            return None

        best_meta = results["metadatas"][0][0]
        best_distance = results["distances"][0][0]

        similarity = 1.0 - best_distance
        confidence = round(similarity * 0.70, 2)

        hts_code = best_meta.get("hts_code") or ""
        if not hts_code:
            return None

        logger.info(
            "classification_layer3_hit product=%s hts=%s confidence=%.2f",
            product, hts_code, confidence,
        )
        return hts_code, best_meta.get("title", ""), confidence

    except Exception as e:
        logger.warning("classification_layer3_failed error=%s", e)
        return None


def _get_hts_description(hts_code: str) -> Optional[str]:
    """Fetch HTS description from Snowflake for a resolved code."""
    conn = get_snowflake_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT description FROM TARIFFIQ.RAW.HTS_CODES WHERE hts_code = %s LIMIT 1",
            (hts_code,),
        )
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        cur.close()
        conn.close()


def run_classification_agent(state: TariffState) -> Dict[str, Any]:
    """
    Resolve HTS code for the product in state using 3-layer lookup.

    Args:
        state: TariffState with product populated

    Returns:
        Dict with hts_code, hts_description, classification_confidence,
        and optionally hitl_required + hitl_reason
    """
    product = state.get("product")
    if not product:
        return {
            "hts_code": None,
            "classification_confidence": 0.0,
            "hitl_required": True,
            "hitl_reason": "low_confidence",
            "error": "No product extracted from query",
        }

    logger.info("classification_agent_start product=%s", product)

    hts_code = None
    confidence = 0.0
    description = None

    # Layer 1 — alias lookup
    result = _layer1_alias_lookup(product)
    if result:
        hts_code, confidence = result
        description = _get_hts_description(hts_code)

    # Layer 2 — keyword search
    if not hts_code:
        result = _layer2_keyword_search(product)
        if result:
            hts_code, description, confidence = result

    # Layer 3 — semantic search
    if not hts_code:
        result = _layer3_semantic_search(product)
        if result:
            hts_code, description, confidence = result

    if not hts_code:
        logger.warning("classification_agent_no_result product=%s", product)
        return {
            "hts_code": None,
            "hts_description": None,
            "classification_confidence": 0.0,
            "hitl_required": True,
            "hitl_reason": "low_confidence",
        }

    hitl_required = confidence < HITL_CONFIDENCE_THRESHOLD

    logger.info(
        "classification_agent_done hts=%s confidence=%.2f hitl=%s",
        hts_code, confidence, hitl_required,
    )

    return {
        "hts_code": hts_code,
        "hts_description": description,
        "classification_confidence": confidence,
        "hitl_required": hitl_required,
        "hitl_reason": "low_confidence" if hitl_required else None,
    }