import os

import snowflake.connector
import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

logger = structlog.get_logger()
router = APIRouter()


class AliasResult(BaseModel):
    hts_code: str
    confidence: float
    alias_matched: str


def get_snowflake_conn():
    return snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
    )


def lookup_alias(product: str) -> Optional[AliasResult]:
    """
    Looks up plain English product name in PRODUCT_ALIASES table.
    Returns hts_code and confidence if found, None if not found.
    """
    conn = get_snowflake_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT alias, hts_code, confidence
            FROM PRODUCT_ALIASES
            WHERE LOWER(alias) = LOWER(%s)
            LIMIT 1
            """,
            (product.strip(),),
        )
        row = cur.fetchone()
        if not row:
            return None

        alias, hts_code, confidence = row
        logger.info(
            "alias_lookup_hit",
            product=product,
            hts_code=hts_code,
            confidence=float(confidence),
        )
        return AliasResult(
            hts_code=hts_code,
            confidence=float(confidence),
            alias_matched=alias,
        )
    finally:
        conn.close()


@router.get("/alias", response_model=AliasResult)
async def get_product_alias(q: str):
    """
    GET /tools/alias?q=laptop
    Returns {hts_code, confidence, alias_matched} or 404 if not found.
    """
    logger.info("lookup_alias_called", product=q)
    result = lookup_alias(q)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No alias found for product: {q}",
        )
    return result