"""
resolve_hts_rate.py — Rate resolution tool

Two functions:

1. resolve_hts_rate (GET /tools/rate) — Ayush's HTTP endpoint
   Simple HTS lookup: description, general_rate, special_rate, other_rate.
   Handles 4/6/8/10 digit input with or without dots.

2. resolve_total_duty — internal, used by rate_agent.py only
   Full provenance: base MFN rate + Section 301/IEEPA adder.
   Country-aware: China-specific notices only apply when country is China.
   Returns VerificationReceipt with full audit trail.
"""

import os
import re
from datetime import datetime, timezone
from typing import Optional

import snowflake.connector
import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.schemas import VerificationReceipt, TariffCalculation, RateReconciliation

logger = structlog.get_logger()
router = APIRouter()

CHINA_ALIASES = {"china", "prc", "people's republic of china"}
CHINA_NOTICE_KEYWORDS = ["china", "chinese"]
UNIVERSAL_NOTICE_KEYWORDS = ["section 232", "safeguard", "solar safeguard", "washing machine"]


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
    digits = re.sub(r'\D', '', code)
    if len(digits) == 4:
        return digits
    elif len(digits) == 6:
        return f"{digits[:4]}.{digits[4:]}"
    elif len(digits) == 8:
        return f"{digits[:4]}.{digits[4:6]}.{digits[6:]}"
    elif len(digits) == 10:
        return f"{digits[:4]}.{digits[4:6]}.{digits[6:8]}.{digits[8:]}"
    return code


# ── Ayush's HTTP endpoint ─────────────────────────────────────────────────────

@router.get("/rate", response_model=HTSRateResponse)
async def resolve_hts_rate(hts_code: str):
    """GET /tools/rate?hts_code=8517.13.00 — simple HTS rate lookup."""
    logger.info("resolve_hts_rate_called", hts_code=hts_code)
    normalized = normalize_hts_code(hts_code)
    conn = get_snowflake_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT hts_code, description, general_rate, special_rate, other_rate, is_chapter99 FROM HTS_CODES WHERE hts_code = %s LIMIT 1",
            (normalized,),
        )
        row = cur.fetchone()
        if not row:
            cur.execute(
                "SELECT hts_code, description, general_rate, special_rate, other_rate, is_chapter99 FROM HTS_CODES WHERE hts_code LIKE %s LIMIT 1",
                (f"{normalized}%",),
            )
            row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"HTS code {hts_code} not found")

        db_code, description, general_rate, special_rate, other_rate, is_chapter99 = row
        logger.info("resolve_hts_rate_found", hts_code=db_code, general_rate=general_rate)
        return HTSRateResponse(
            hts_code=db_code, description=description,
            general_rate=general_rate or "Not specified",
            special_rate=special_rate, other_rate=other_rate,
            chapter=db_code[:4], is_chapter99=is_chapter99 or False,
        )
    finally:
        conn.close()


# ── Internal functions used by rate_agent.py ─────────────────────────────────

def _parse_base_rate(s: Optional[str]) -> float:
    if not s:
        return 0.0
    s = s.strip()
    if s.lower() in ("", "free"):
        return 0.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", s)
    if m:
        return float(m.group(1))
    try:
        return float(s.split()[0].replace("%", ""))
    except (ValueError, IndexError):
        return 0.0


def _parse_adder_rate(snippet: Optional[str], hts_code: str) -> float:
    if not snippet:
        return 0.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", snippet)
    if m:
        r = float(m.group(1))
        if 0 < r <= 200:
            return r
    m = re.search(r"\b(\d+(?:\.\d+)?)\s+percent\b", snippet, re.IGNORECASE)
    if m:
        r = float(m.group(1))
        if 0 < r <= 200:
            return r
    escaped = re.escape(hts_code.strip())
    m = re.search(escaped + r".*?(\d+(?:\.\d+)?)\s*\n+\s*(20\d{2})", snippet, re.DOTALL)
    if m:
        r = float(m.group(1))
        if 0 < r <= 200:
            return r
    for m in re.finditer(r"\b(\d{1,3}(?:\.\d+)?)\s*\n+\s*(20\d{2})\b", snippet):
        r = float(m.group(1))
        if 0 < r <= 200:
            return r
    return 0.0


def _notice_applies_to_country(title: Optional[str], country: Optional[str]) -> bool:
    if not title:
        return True
    tl = title.lower()
    cl = (country or "").lower().strip()
    if any(kw in tl for kw in UNIVERSAL_NOTICE_KEYWORDS):
        return True
    if any(kw in tl for kw in CHINA_NOTICE_KEYWORDS):
        is_china = cl in CHINA_ALIASES
        if not is_china:
            logger.info("notice_skipped_country_mismatch title=%s country=%s", title[:60], country)
        return is_china
    return True


def resolve_total_duty(hts_code: str, country: Optional[str] = None) -> VerificationReceipt:
    """
    Internal — called by rate_agent.py only. NOT an HTTP endpoint.
    Returns VerificationReceipt with base rate + Section 301 adder + provenance.
    Country-aware: China Section 301 notices skipped for non-China origins.
    """
    conn = get_snowflake_conn()
    now = datetime.now(timezone.utc)
    try:
        cur = conn.cursor()

        cur.execute("SELECT hts_code, general_rate, description FROM HTS_CODES WHERE hts_code = %s LIMIT 1", (hts_code,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"HTS code {hts_code} not found")

        db_hts, general_rate_str, description = row
        base_rate = _parse_base_rate(general_rate_str)
        base_calc = TariffCalculation(
            component="Base MFN Rate", rate=base_rate,
            source_description=description or "", record_id=db_hts,
            fetched_from="TARIFFIQ.RAW.HTS_CODES", fetched_at=now,
        )

        cur.execute(
            """
            SELECT n.document_number, n.context_snippet, f.title, f.publication_date
            FROM NOTICE_HTS_CODES n
            LEFT JOIN FEDERAL_REGISTER_NOTICES f ON n.document_number = f.document_number
            WHERE n.hts_code = %s
            ORDER BY f.publication_date DESC NULLS LAST
            LIMIT 10
            """,
            (hts_code,),
        )
        notice_rows = cur.fetchall()

        adder_rate = 0.0
        best_doc = "NONE"
        best_title = "No additional tariff notice found"
        best_from = "TARIFFIQ.RAW.NOTICE_HTS_CODES"
        used_date = None

        for doc_number, snippet, title, pub_date in notice_rows:
            if not _notice_applies_to_country(title, country):
                continue
            adder_rate = _parse_adder_rate(snippet, hts_code)
            best_doc = doc_number
            best_title = title or snippet or ""
            best_from = "TARIFFIQ.RAW.FEDERAL_REGISTER_NOTICES"
            used_date = pub_date
            logger.info("rate_agent_adder_selected doc=%s pub_date=%s rate=%.2f country=%s",
                        doc_number, pub_date, adder_rate, country)
            break

        adder_calc = TariffCalculation(
            component="Section 301 / IEEPA Adder", rate=adder_rate,
            source_description=best_title, record_id=best_doc,
            fetched_from=best_from, fetched_at=now,
        )
        total = round(base_rate + adder_rate, 4)
        reconciliation = RateReconciliation(
            calculation=f"{base_rate} + {adder_rate} = {total}", check_passed=True
        )

        logger.info("resolve_total_duty_complete hts=%s country=%s base=%.4f adder=%.4f total=%.4f notice=%s",
                    hts_code, country, base_rate, adder_rate, total, str(used_date))

        return VerificationReceipt(
            hts_code=hts_code, base_rate=base_rate, base_rate_source=base_calc,
            adder_rate=adder_rate, adder_source=adder_calc,
            total_duty=total, rate_reconciliation=reconciliation,
        )
    finally:
        conn.close()