import os
from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import snowflake.connector
import structlog

logger = structlog.get_logger()
router = APIRouter()


class HTSCodeInChapter(BaseModel):
    hts_code: str
    description: str
    general_rate: str


@router.get("/hts/chapter", response_model=List[HTSCodeInChapter])
async def get_codes_in_chapter(chapter: str, limit: int = 50):
    """
    GET /tools/hts/chapter?chapter=85&limit=50

    Get all HTS codes in a specific chapter.
    Useful for scoping ChromaDB retrieval to relevant chapter.
    """
    logger.info("get_codes_in_chapter_called", chapter=chapter, limit=limit)

    # Validate chapter format (should be 2-4 digits)
    if not chapter or not chapter.isdigit() or len(chapter) > 4 or len(chapter) < 2:
        raise HTTPException(
            status_code=400,
            detail="Chapter must be 2-4 digits (e.g., '85' or '8517')"
        )

    conn = get_snowflake_conn()
    try:
        cur = conn.cursor()

        # Get all codes that start with the chapter
        chapter_pattern = f"{chapter}%"

        cur.execute(
            """
            SELECT hts_code, description, general_rate
            FROM HTS_CODES
            WHERE hts_code LIKE %s
            AND is_header_row = FALSE
            ORDER BY hts_code ASC
            LIMIT %s
            """,
            (chapter_pattern, limit),
        )
        rows = cur.fetchall()

        if not rows:
            logger.info("get_codes_in_chapter_no_results", chapter=chapter)
            return []

        results = [
            HTSCodeInChapter(
                hts_code=hts_code,
                description=description,
                general_rate=general_rate or "Not specified",
            )
            for hts_code, description, general_rate in rows
        ]

        logger.info("get_codes_in_chapter_found", chapter=chapter, count=len(results))
        return results

    finally:
        conn.close()


def get_snowflake_conn():
    return snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
    )
