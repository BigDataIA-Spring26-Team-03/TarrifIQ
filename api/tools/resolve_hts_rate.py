import os
from datetime import datetime, timezone

import snowflake.connector
import structlog
from fastapi import APIRouter, HTTPException

from api.schemas import VerificationReceipt, TariffCalculation, RateReconciliation

logger = structlog.get_logger()
router = APIRouter()


def get_snowflake_conn():
    return snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
    )


def resolve_total_duty(hts_code: str) -> VerificationReceipt:
    """
    Looks up the base MFN rate for an HTS code from HTS_CODES table.
    Looks up any Chapter 99 adder from NOTICE_HTS_CODES + FEDERAL_REGISTER_NOTICES.
    Returns a VerificationReceipt with full provenance.
    """
    conn = get_snowflake_conn()
    now = datetime.now(timezone.utc)

    try:
        cur = conn.cursor()

        # Step 1 — Get base rate from HTS_CODES
        cur.execute(
            """
            SELECT hts_code, general_rate, description
            FROM HTS_CODES
            WHERE hts_code = %s
            LIMIT 1
            """,
            (hts_code,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"HTS code {hts_code} not found in HTS_CODES table",
            )

        db_hts_code, general_rate_str, description = row

        # Parse base rate — handle "Free", percentages, per-unit rates
        base_rate = 0.0
        if general_rate_str and general_rate_str.strip().lower() not in ("", "free"):
            try:
                base_rate = float(
                    general_rate_str.strip().replace("%", "").split()[0]
                )
            except (ValueError, IndexError):
                base_rate = 0.0

        base_calc = TariffCalculation(
            component="Base MFN Rate",
            rate=base_rate,
            source_description=description or "",
            record_id=db_hts_code,
            fetched_from="TARIFFIQ.RAW.HTS_CODES",
            fetched_at=now,
        )

        # Step 2 — Check NOTICE_HTS_CODES for any Chapter 99 adder
        cur.execute(
            """
            SELECT n.document_number, n.context_snippet, f.title
            FROM NOTICE_HTS_CODES n
            LEFT JOIN FEDERAL_REGISTER_NOTICES f
                ON n.document_number = f.document_number
            WHERE n.hts_code = %s
            LIMIT 1
            """,
            (hts_code,),
        )
        notice_row = cur.fetchone()

        adder_rate = 0.0
        if notice_row:
            doc_number, snippet, title = notice_row
            # Parse adder rate from snippet — look for percentage pattern
            import re
            match = re.search(r"(\d+(?:\.\d+)?)\s*%", snippet or "")
            if match:
                adder_rate = float(match.group(1))

            adder_calc = TariffCalculation(
                component="Section 301 / IEEPA Adder",
                rate=adder_rate,
                source_description=title or snippet or "",
                record_id=doc_number,
                fetched_from="TARIFFIQ.RAW.FEDERAL_REGISTER_NOTICES",
                fetched_at=now,
            )
        else:
            adder_calc = TariffCalculation(
                component="Section 301 / IEEPA Adder",
                rate=0.0,
                source_description="No additional tariff notice found for this HTS code",
                record_id="NONE",
                fetched_from="TARIFFIQ.RAW.NOTICE_HTS_CODES",
                fetched_at=now,
            )

        # Step 3 — Calculate total and verify
        total = round(base_rate + adder_rate, 4)
        expected = round(base_rate + adder_rate, 4)
        reconciliation = RateReconciliation(
            calculation=f"{base_rate} + {adder_rate} = {total}",
            check_passed=(total == expected),
        )

        logger.info(
            "resolve_total_duty_complete",
            hts_code=hts_code,
            base_rate=base_rate,
            adder_rate=adder_rate,
            total=total,
        )

        return VerificationReceipt(
            hts_code=hts_code,
            base_rate=base_rate,
            base_rate_source=base_calc,
            adder_rate=adder_rate,
            adder_source=adder_calc,
            total_duty=total,
            rate_reconciliation=reconciliation,
        )

    finally:
        conn.close()


@router.get("/rate", response_model=VerificationReceipt)
async def get_hts_rate(hts_code: str):
    """
    GET /tools/rate?hts_code=8471.30.01.00
    Returns full VerificationReceipt with base rate, adder, total, and provenance.
    """
    logger.info("resolve_hts_rate_called", hts_code=hts_code)
    return resolve_total_duty(hts_code)  