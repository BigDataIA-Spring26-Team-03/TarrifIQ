import os
from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import snowflake.connector
import structlog

logger = structlog.get_logger()
router = APIRouter()


class HTSSearchResult(BaseModel):
    hts_code: str
    description: str
    general_rate: str
    chapter: str
    confidence: float  # 1.0 = exact, 0.7 = partial


def get_snowflake_conn():
    return snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
    )


@router.get("/hts/search", response_model=List[HTSSearchResult])
async def search_hts_by_product(query: str, limit: int = 5):
    """
    GET /tools/hts/search?query=smartphones&limit=5

    Search HTS codes by product description.
    Returns matches ranked by relevance.
    """
    logger.info("search_hts_by_product_called", query=query, limit=limit)

    if not query or len(query.strip()) < 2:
        raise HTTPException(
            status_code=400,
            detail="Query must be at least 2 characters"
        )

    conn = get_snowflake_conn()
    try:
        cur = conn.cursor()

        # Search in description field, prioritize exact word matches
        search_pattern = f"%{query}%"
        search_lower = query.lower()

        cur.execute(
            """
            SELECT hts_code, description, general_rate, is_chapter99
            FROM HTS_CODES
            WHERE LOWER(description) LIKE LOWER(%s)
            AND is_header_row = FALSE
            ORDER BY
                CASE WHEN LOWER(description) = %s THEN 0 ELSE 1 END,
                CASE WHEN LOWER(description) LIKE %s THEN 0 ELSE 1 END,
                LENGTH(description) ASC
            LIMIT %s
            """,
            (search_pattern, search_lower, f"{search_lower}%", limit),
        )
        rows = cur.fetchall()

        if not rows:
            logger.info("search_hts_no_results", query=query)
            return []

        results = []
        for hts_code, description, general_rate, is_chapter99 in rows:
            chapter = hts_code[:4]

            # Confidence scoring
            desc_lower = description.lower()
            if desc_lower == search_lower:
                confidence = 1.0  # Exact match
            elif desc_lower.startswith(search_lower):
                confidence = 0.9  # Starts with
            elif f" {search_lower}" in desc_lower or f"{search_lower} " in desc_lower:
                confidence = 0.85  # Word match
            else:
                confidence = 0.7  # Partial match

            results.append(
                HTSSearchResult(
                    hts_code=hts_code,
                    description=description,
                    general_rate=general_rate or "Not specified",
                    chapter=chapter,
                    confidence=confidence,
                )
            )

        logger.info("search_hts_found", query=query, count=len(results))
        return results

    finally:
        conn.close()
