"""
Live Census Bureau International Trade (imports / HS) API client.
Results are cached in Redis; nothing is written to Snowflake.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

BASE_URL = "https://api.census.gov/data/timeseries/intltrade/imports/hs"
REQUEST_TIMEOUT = 30
CACHE_TTL_SEC = 86400
_GET_VARS = (
    "GEN_VAL_MO,GEN_VAL_YR,CON_VAL_MO,CAL_DUT_MO,CAL_DUT_YR,"
    "DUT_VAL_MO,CTY_CODE,CTY_NAME,I_COMMODITY,I_COMMODITY_LDESC"
)


def _redis_client():
    try:
        import redis

        url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        return redis.Redis.from_url(url, decode_responses=True)
    except Exception as e:
        logger.debug("redis_unavailable: %s", e)
        return None


def _cache_get(key: str) -> dict[str, Any] | None:
    r = _redis_client()
    if r is None:
        return None
    try:
        raw = r.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as e:
        logger.debug("census_cache_get_failed key=%s err=%s", key, e)
        return None


def _cache_set(key: str, payload: dict[str, Any]) -> None:
    r = _redis_client()
    if r is None:
        return
    try:
        r.setex(key, CACHE_TTL_SEC, json.dumps(payload))
    except Exception as e:
        logger.debug("census_cache_set_failed key=%s err=%s", key, e)


def _digits_only(hts_code: str) -> str:
    return "".join(c for c in (hts_code or "") if c.isdigit())


def _infer_commodity_and_level(hts_code: str) -> tuple[str, str]:
    """
    Map HTS input to (I_COMMODITY string, COMM_LVL).
    8-10 digit codes use HS10 (pad to 10 if needed).
    10+ digit codes are truncated to 10 for HS10.
    """
    digits = _digits_only(hts_code)
    if not digits:
        return "", "HS10"
    n = len(digits)
    if n == 2:
        return digits, "HS2"
    if n == 4:
        return digits, "HS4"
    if n == 6:
        return digits, "HS6"
    if 8 <= n <= 10:
        return digits.ljust(10, "0"), "HS10"
    if n > 10:
        return digits[:10], "HS10"
    # For odd lengths like 1/3/5/7, use the closest supported lower level.
    if n < 2:
        return digits, "HS2"
    if n < 4:
        return digits[:2], "HS2"
    if n < 6:
        return digits[:4], "HS4"
    return digits[:6], "HS6"


def _default_time_month() -> str:
    """Most recent month accounting for Census publication lag."""
    lagged = date.today() - timedelta(days=60)
    return f"{lagged.year:04d}-{lagged.month:02d}"


def _add_months(year: int, month: int, delta: int) -> tuple[int, int]:
    m = month + delta
    y = year
    while m > 12:
        m -= 12
        y += 1
    while m < 1:
        m += 12
        y -= 1
    return y, m


def _month_range_back_from(end_ym: str, count: int) -> list[str]:
    y, m = map(int, end_ym.split("-"))
    months: list[str] = []
    for i in range(count):
        yy, mm = _add_months(y, m, -i)
        months.append(f"{yy:04d}-{mm:02d}")
    return list(reversed(months))


def _unique_headers(headers: list[Any]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for h in headers:
        s = str(h)
        if s not in seen:
            seen[s] = 0
            out.append(s)
        else:
            seen[s] += 1
            out.append(f"{s}__{seen[s]}")
    return out


def _parse_census_table(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, list) or len(data) < 2:
        return []
    header_row = data[0]
    if not isinstance(header_row, list):
        return []
    headers = _unique_headers(header_row)
    rows: list[dict[str, Any]] = []
    for raw in data[1:]:
        if not isinstance(raw, list) or len(raw) != len(header_row):
            continue
        row = dict(zip(headers, raw))
        effective_tariff_rate = None
        try:
            dut_val = row.get("DUT_VAL_MO", row.get("dut_val_mo"))
            cal_dut = row.get("CAL_DUT_MO", row.get("cal_dut_mo"))
            if dut_val not in (None, "", "0", 0) and cal_dut not in (None, ""):
                dut = float(str(dut_val).replace(",", ""))
                cal = float(str(cal_dut).replace(",", ""))
                if dut > 0:
                    effective_tariff_rate = round((cal / dut) * 100, 2)
        except Exception:
            effective_tariff_rate = None
        row["effective_tariff_rate"] = effective_tariff_rate
        rows.append(row)
    filtered: list[dict[str, Any]] = []
    for row in rows:
        cty_code = str(row.get("CTY_CODE", "")).strip()
        cty_name = str(row.get("CTY_NAME", "")).strip().upper()
        if cty_code == "-" or cty_name == "TOTAL FOR ALL COUNTRIES":
            continue
        filtered.append(row)
    return filtered


def format_rows_for_display(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a renamed copy of rows for UI display only."""
    display_map = {
        "time": "Month",
        "country_code": "Country Code",
        "country_name": "Country",
        "gen_val_mo": "Import Value USD (Monthly)",
        "gen_val_yr": "Import Value USD (YTD)",
        "con_val_mo": "Consumption Imports USD (Monthly)",
        "cal_dut_mo": "Duties Collected USD (Monthly)",
        "cal_dut_yr": "Duties Collected USD (YTD)",
        "dut_val_mo": "Dutiable Value USD (Monthly)",
        "effective_tariff_rate": "Effective Tariff Rate (%)",
        "commodity": "HTS Code",
        "description": "Product Description",
        "resolved_level": "Data Granularity",
        "comm_lvl": "HS Level",
        "note": "Note",
    }
    formatted: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        new_row: dict[str, Any] = {}
        for key, value in row.items():
            new_row[display_map.get(key, key)] = value
        formatted.append(new_row)
    return formatted


def _fetch_census_raw(
    commodity: str, comm_lvl: str, time: str, api_key: str
) -> tuple[int, Any]:
    params: dict[str, str] = {
        "get": _GET_VARS,
        "COMM_LVL": comm_lvl,
        "SUMMARY_LVL": "DET",
        "time": time,
        "I_COMMODITY": commodity,
    }
    if api_key:
        params["key"] = api_key
    try:
        resp = requests.get(
            BASE_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        if not resp.content:
            return resp.status_code, None
        try:
            return resp.status_code, resp.json()
        except ValueError:
            logger.error("census_non_json_response status=%s", resp.status_code)
            return resp.status_code, None
    except requests.Timeout:
        logger.error("census_request_timeout commodity=%s time=%s", commodity, time)
        return -1, None
    except Exception as e:
        logger.error("census_request_error commodity=%s time=%s err=%s", commodity, time, e)
        return -2, None


def get_trade_flow(hts_code: str, time: str | None = None) -> dict[str, Any]:
    """
    Fetch import values by country for one HS chapter/commodity and month.

    Returns dict with hts_code, comm_lvl, time, rows (list of country rows),
    and optionally ``note`` (e.g. no data at this resolution).
    Never raises.
    """
    commodity, comm_lvl = _infer_commodity_and_level(hts_code)
    if not commodity:
        logger.warning("get_trade_flow_empty_commodity hts_code=%r", hts_code)
        return {
            "hts_code": hts_code,
            "resolved_level": comm_lvl,
            "comm_lvl": comm_lvl,
            "time": time or "",
            "rows": [],
            "note": "invalid or empty HTS code",
        }

    resolved_time = time if time else _default_time_month()
    api_key = os.environ.get("CENSUS_API_KEY", "") or ""
    input_digits = _digits_only(hts_code)

    def _attempt(level_code: str, level: str) -> tuple[str, int, Any, list[dict[str, Any]]]:
        cache_key = f"census:{hts_code.strip()}:{resolved_time}:{level}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return "cache", 200, cached, cached.get("rows", []) or []

        status, payload = _fetch_census_raw(level_code, level, resolved_time, api_key)
        if status != 200 or payload is None:
            return "api", status, payload, []
        if isinstance(payload, dict) and payload.get("error"):
            logger.error("census_api_error %s", payload)
            return "api", status, payload, []
        rows = _parse_census_table(payload)
        out = {
            "hts_code": level_code,
            "resolved_level": level,
            "comm_lvl": level,
            "time": resolved_time,
            "rows": rows,
        }
        _cache_set(cache_key, out)
        return "api", status, out, rows

    source, status, payload, rows = _attempt(commodity, comm_lvl)
    if source == "cache" and isinstance(payload, dict):
        return payload

    # HS10 fallback chain: HS10 -> HS6 -> HS4
    if comm_lvl == "HS10" and (status == 204 or not rows):
        logger.info("census_fallback_try from_level=HS10 to_level=HS6 hts_code=%s", hts_code)
        hs6_code = input_digits[:6] if len(input_digits) >= 6 else commodity[:6]
        source6, status6, payload6, rows6 = _attempt(hs6_code, "HS6")
        if source6 == "cache" and isinstance(payload6, dict):
            payload6["note"] = "HS10 returned no data, fell back to HS6"
            return payload6
        if status6 == 200 and rows6:
            if isinstance(payload6, dict):
                payload6["note"] = "HS10 returned no data, fell back to HS6"
                return payload6

        logger.info("census_fallback_try from_level=HS6 to_level=HS4 hts_code=%s", hts_code)
        hs4_code = input_digits[:4] if len(input_digits) >= 4 else hs6_code[:4]
        source4, status4, payload4, rows4 = _attempt(hs4_code, "HS4")
        if source4 == "cache" and isinstance(payload4, dict):
            payload4["note"] = "HS10/HS6 returned no data, fell back to HS4"
            return payload4
        if status4 == 200 and rows4:
            if isinstance(payload4, dict):
                payload4["note"] = "HS10/HS6 returned no data, fell back to HS4"
                return payload4

        return {
            "hts_code": commodity,
            "resolved_level": "HS10",
            "comm_lvl": "HS10",
            "time": resolved_time,
            "rows": [],
            "note": "data not available at this resolution",
        }

    if status == 204:
        return {
            "hts_code": commodity,
            "resolved_level": comm_lvl,
            "comm_lvl": comm_lvl,
            "time": resolved_time,
            "rows": [],
            "note": "data not available at this resolution",
        }

    if status != 200 or payload is None:
        logger.error(
            "census_bad_response status=%s commodity=%s time=%s",
            status,
            commodity,
            resolved_time,
        )
        return {
            "hts_code": commodity,
            "resolved_level": comm_lvl,
            "comm_lvl": comm_lvl,
            "time": resolved_time,
            "rows": [],
        }

    if isinstance(payload, dict) and "rows" in payload:
        return payload

    return {
        "hts_code": commodity,
        "resolved_level": comm_lvl,
        "comm_lvl": comm_lvl,
        "time": resolved_time,
        "rows": rows,
    }


def get_trade_trend(hts_code: str, months: int = 12) -> list[dict[str, Any]]:
    """
    Parallel monthly snapshots via get_trade_flow, oldest-first.
    Never raises; failed months yield empty-row dicts with the same shape.
    """
    if months < 1:
        months = 1
    end_ym = _default_time_month()
    month_list = _month_range_back_from(end_ym, months)

    results: list[dict[str, Any] | None] = [None] * len(month_list)
    commodity, _ = _infer_commodity_and_level(hts_code)

    def _one(idx: int, ym: str) -> tuple[int, dict[str, Any]]:
        return idx, get_trade_flow(hts_code, ym)

    max_workers = min(8, len(month_list))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_one, i, ym): i for i, ym in enumerate(month_list)
        }
        for fut in as_completed(futures):
            try:
                idx, snap = fut.result()
                results[idx] = snap
            except Exception as e:
                logger.error("get_trade_trend_worker_error %s", e)

    out: list[dict[str, Any]] = []
    for i, ym in enumerate(month_list):
        snap = results[i]
        if snap is None:
            out.append(
                {
                    "hts_code": commodity or _digits_only(hts_code),
                    "comm_lvl": _infer_commodity_and_level(hts_code)[1],
                    "time": ym,
                    "rows": [],
                    "note": "request failed",
                }
            )
        else:
            out.append(snap)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    s1 = get_trade_flow("8471", "2024-01")
    logger.info(
        "test_hs4_8471_2024_01 rows=%s note=%s",
        len(s1.get("rows") or []),
        s1.get("note"),
    )
    s2 = get_trade_flow("847130", "2024-01")
    logger.info(
        "test_hs6_847130_2024_01 rows=%s note=%s",
        len(s2.get("rows") or []),
        s2.get("note"),
    )
    trend = get_trade_trend("847130", 6)
    logger.info("trend_6m_months=%s", [t.get("time") for t in trend])
    for t in trend:
        logger.info("  %s rows=%s", t.get("time"), len(t.get("rows") or []))
