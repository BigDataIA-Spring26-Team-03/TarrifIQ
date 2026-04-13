"""
Census Bureau import trade flow tool — live API + Redis cache.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Query

from api.schemas import TradeFlowResult
from ingestion.census_client import get_trade_trend

logger = logging.getLogger(__name__)

router = APIRouter()

SOURCE_STAMP = "U.S. Census Bureau International Trade API"


def _parse_gen_val(raw: Any) -> float | None:
    if raw is None:
        return None
    s = str(raw).strip().replace(",", "")
    if not s or s in ("(D)", "(X)", "(NA)", "N", "S", "Z"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _row_country_key(row: dict[str, Any]) -> str:
    code = str(row.get("CTY_CODE") or "").strip()
    name = str(row.get("CTY_NAME") or "").strip()
    return f"{code}|{name}" if code else name


def _commodity_hs6_from_rows(rows: list[dict[str, Any]], fallback: str) -> str:
    for row in rows:
        c = row.get("I_COMMODITY")
        if c is not None and str(c).strip():
            d = "".join(x for x in str(c) if x.isdigit())
            if d:
                return d.ljust(6, "0")[:6]
    d = "".join(c for c in fallback if c.isdigit())
    return d.ljust(6, "0")[:6] if d else fallback[:6].ljust(6, "0")


def trade_flow_results(hts_code: str, months: int) -> list[TradeFlowResult]:
    """
    Build one TradeFlowResult per country from Census monthly snapshots.
    """
    trend = get_trade_trend(hts_code, months)
    if not trend:
        return []

    # country_key -> { yyyy-mm: value }
    by_country: dict[str, dict[str, float]] = {}
    fallback_commodity = str(trend[-1].get("hts_code") or hts_code)

    for snap in trend:
        ym = str(snap.get("time") or "")
        for row in snap.get("rows") or []:
            if not isinstance(row, dict):
                continue
            val = _parse_gen_val(row.get("GEN_VAL_MO"))
            if val is None:
                continue
            key = _row_country_key(row)
            if not key:
                continue
            by_country.setdefault(key, {})[ym] = val

    if not by_country:
        return []

    hs6 = _commodity_hs6_from_rows(
        trend[-1].get("rows") or [],
        fallback_commodity,
    )
    last_ym = str(trend[-1].get("time") or "")
    now = datetime.now(timezone.utc)
    out: list[TradeFlowResult] = []

    for key, series in sorted(by_country.items(), key=lambda kv: -sum(kv[1].values())):
        name = key.split("|", 1)[-1] if "|" in key else key
        sorted_months = sorted(series.keys())
        if not sorted_months:
            continue

        yoy = 0.0
        if last_ym:
            try:
                y, m = map(int, last_ym.split("-"))
                prev_ym = f"{y - 1:04d}-{m:02d}"
                v_now = series.get(last_ym)
                v_prev = series.get(prev_ym)
                if (
                    v_now is not None
                    and v_prev is not None
                    and v_prev != 0
                ):
                    yoy = round((v_now - v_prev) / v_prev * 100, 4)
            except (ValueError, TypeError):
                yoy = 0.0

        v_first = series.get(sorted_months[0])
        v_last = series.get(sorted_months[-1])
        trend_label = "insufficient data"
        if v_first is not None and v_last is not None and v_first != 0:
            chg = (v_last - v_first) / v_first * 100
            if chg > 5.0:
                trend_label = "increasing"
            elif chg < -5.0:
                trend_label = "decreasing"
            else:
                trend_label = "stable"
        elif v_first is not None and v_last is not None:
            trend_label = "stable"

        out.append(
            TradeFlowResult(
                hs6_code=hs6,
                country=name,
                period_months=months,
                pct_change_yoy=yoy,
                trend=trend_label,
                source_stamp=SOURCE_STAMP,
                fetched_at=now,
            )
        )

    return out


@router.get("/trade_flow", response_model=list[TradeFlowResult])
def get_trade_flow_endpoint(
    hts_code: str = Query(..., description="HTS / HS commodity (e.g. 8471, 847130)"),
    months: int = Query(12, ge=1, le=36, description="Number of recent months to include"),
):
    """
    GET /tools/trade_flow?hts_code=8471&months=12
    Live Census import values by country; cached in Redis.
    """
    logger.info("trade_flow_request hts_code=%s months=%s", hts_code, months)
    return trade_flow_results(hts_code, months)
