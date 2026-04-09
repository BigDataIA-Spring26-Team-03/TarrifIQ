import os
import uuid
from datetime import datetime, timezone

import snowflake.connector
import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from api.schemas import HITLRecord

logger = structlog.get_logger()
router = APIRouter()


class HITLCreateRequest(BaseModel):
    query_text: str
    trigger_reason: str
    classifier_hts: Optional[str] = None
    classifier_conf: Optional[float] = None


class HITLCreateResponse(BaseModel):
    hitl_id: str
    status: str


def get_snowflake_conn():
    return snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
    )


def write_hitl_record(record: HITLCreateRequest) -> HITLCreateResponse:
    """
    Writes a new HITL escalation record to HITL_RECORDS table.
    Generates a unique hitl_id and timestamps the creation.
    """
    hitl_id = f"hitl_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc)

    conn = get_snowflake_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO HITL_RECORDS (
                hitl_id,
                query_text,
                trigger_reason,
                classifier_hts,
                classifier_conf,
                human_decision,
                adjudicated_at,
                created_at
            ) VALUES (%s, %s, %s, %s, %s, NULL, NULL, %s)
            """,
            (
                hitl_id,
                record.query_text,
                record.trigger_reason,
                record.classifier_hts,
                record.classifier_conf,
                now,
            ),
        )
        conn.commit()

        logger.info(
            "hitl_record_created",
            hitl_id=hitl_id,
            trigger_reason=record.trigger_reason,
            classifier_hts=record.classifier_hts,
            classifier_conf=record.classifier_conf,
        )

        return HITLCreateResponse(hitl_id=hitl_id, status="created")

    except Exception as e:
        logger.error("hitl_record_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write HITL record: {str(e)}",
        )
    finally:
        conn.close()


@router.post("/hitl", response_model=HITLCreateResponse)
async def log_hitl(record: HITLCreateRequest):
    """
    POST /tools/hitl
    Logs a HITL escalation event to Snowflake.
    Returns {hitl_id, status: "created"}.
    """
    logger.info("log_hitl_called", trigger_reason=record.trigger_reason)
    return write_hitl_record(record)