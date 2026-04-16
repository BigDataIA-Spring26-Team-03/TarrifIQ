import os
import re
from typing import Optional

import snowflake.connector
import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = structlog.get_logger()
router = APIRouter()


class HTSRateResponse(BaseModel):
    hts_code: str
    description: str
    general_rate: str
    special_rate: Optional[str] = None
    other_rate: Optional[str] = None
    chapter: str
    is_chapter99: bool


def get_snowflake_conn():
    return snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
    )


def normalize_hts_code(code: str) -> str:
    """Normalize HTS code to standard format with dots (8 or 10 digits)."""
    # Remove all non-digit characters
    digits = re.sub(r'\D', '', code)

    if len(digits) == 4:  # Chapter only (e.g., "8517")
        return digits
    elif len(digits) == 6:  # Heading (e.g., "851713")
        return f"{digits[:4]}.{digits[4:]}"
    elif len(digits) == 8:  # 8-digit (e.g., "85171300")
        return f"{digits[:4]}.{digits[4:6]}.{digits[6:]}"
    elif len(digits) == 10:  # 10-digit (e.g., "8517130000")
        return f"{digits[:4]}.{digits[4:6]}.{digits[6:8]}.{digits[8:]}"
    else:
        return code  # Return as-is if format unclear


@router.get("/rate", response_model=HTSRateResponse)
async def resolve_hts_rate(hts_code: str):
    """
    GET /tools/rate?hts_code=8517.13.00

    Look up HTS code rate information. Handles multiple formats:
    - 8517.13.00 (8 digit with dots)
    - 8517.13.00.00 (10 digit with dots)
    - 851713 (6 digits, no dots)
    - 8517 (4 digits, chapter only)
    """
    logger.info("resolve_hts_rate_called", hts_code=hts_code)

    normalized = normalize_hts_code(hts_code)
    conn = get_snowflake_conn()

    try:
        cur = conn.cursor()

        # Try exact match first
        cur.execute(
            """
            SELECT hts_code, description, general_rate, special_rate, other_rate, is_chapter99
            FROM HTS_CODES
            WHERE hts_code = %s
            LIMIT 1
            """,
            (normalized,),
        )
        row = cur.fetchone()

        # Fallback: LIKE query if exact match fails
        if not row:
            cur.execute(
                """
                SELECT hts_code, description, general_rate, special_rate, other_rate, is_chapter99
                FROM HTS_CODES
                WHERE hts_code LIKE %s
                LIMIT 1
                """,
                (f"{normalized}%",),
            )
            row = cur.fetchone()

        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"HTS code {hts_code} not found"
            )

        db_code, description, general_rate, special_rate, other_rate, is_chapter99 = row
        chapter = db_code[:4]

        logger.info(
            "resolve_hts_rate_found",
            hts_code=db_code,
            description=description,
            general_rate=general_rate,
        )

        return HTSRateResponse(
            hts_code=db_code,
            description=description,
            general_rate=general_rate or "Not specified",
            special_rate=special_rate,
            other_rate=other_rate,
            chapter=chapter,
            is_chapter99=is_chapter99 or False,
        )

    finally:
        conn.close()
