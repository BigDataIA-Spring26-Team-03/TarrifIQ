"""
Trade Agent — TariffIQ Pipeline Step 5

Queries Census Bureau International Trade API live at query time.
HS6 only — HS8 is confirmed broken by Census Bureau.
No data stored in Snowflake.

API: https://api.census.gov/data/timeseries/intltrade/imports/hs
"""

import logging
import os
from typing import Dict, Any, Optional

import requests

from agents.state import TariffState

logger = logging.getLogger(__name__)

CENSUS_BASE_URL = "https://api.census.gov/data/timeseries/intltrade/imports/hs"
REQUEST_TIMEOUT = 15

# Country name to Census country code mapping for common trade partners
COUNTRY_CODE_MAP = {
    "china": "5700",
    "canada": "1220",
    "mexico": "2010",
    "japan": "5880",
    "germany": "4280",
    "south korea": "5800",
    "korea": "5800",
    "vietnam": "5880",
    "india": "5330",
    "taiwan": "5830",
    "united kingdom": "4120",
    "uk": "4120",
    "france": "4279",
    "italy": "4750",
    "brazil": "3510",
    "thailand": "5490",
}

# Most recent available Census data period
DEFAULT_PERIOD = os.environ.get("CENSUS_DEFAULT_PERIOD", "2024-12")


def _get_country_code(country: Optional[str]) -> Optional[str]:
    """Map country name to Census country code."""
    if not country:
        return None
    return COUNTRY_CODE_MAP.get(country.lower().strip())


def _parse_hs6(hts_code: str) -> Optional[str]:
    """
    Strip HTS code to HS6 format (6 digits, no dots).
    e.g. "8471.30.01.00" -> "847130"
    """
    if not hts_code:
        return None
    digits = hts_code.replace(".", "").replace(" ", "")
    if len(digits) >= 6:
        return digits[:6]
    return None


def _query_census(hs6: str, country_code: Optional[str], period: str) -> Dict[str, Any]:
    """
    Query Census Bureau API for import data.

    Returns dict with import_value_usd, import_quantity, suppressed flag.
    """
    params = {
        "get": "GEN_VAL_MO,GEN_QY1_MO,CTY_NAME,I_COMMODITY_LDESC",
        "COMM_LVL": "HS6",
        "I_COMMODITY": hs6,
        "time": period,
    }

    if country_code:
        params["CTY_CODE"] = country_code

    census_api_key = os.environ.get("CENSUS_API_KEY")
    if census_api_key:
        params["key"] = census_api_key

    try:
        resp = requests.get(CENSUS_BASE_URL, params=params, timeout=REQUEST_TIMEOUT)

        # 204 = data suppressed (below reporting threshold)
        if resp.status_code == 204:
            logger.info("census_data_suppressed hs6=%s period=%s", hs6, period)
            return {"suppressed": True}

        if resp.status_code == 400:
            logger.warning("census_bad_request hs6=%s status=400", hs6)
            return {"suppressed": True}

        resp.raise_for_status()
        data = resp.json()

        # Census returns [header_row, data_row, ...]
        if not data or len(data) < 2:
            return {"suppressed": True}

        headers = data[0]
        row = data[1]
        record = dict(zip(headers, row))

        val = record.get("GEN_VAL_MO")
        qty = record.get("GEN_QY1_MO")

        import_value = float(val) if val and val != "null" else None
        import_quantity = float(qty) if qty and qty != "null" else None

        if import_value is None and import_quantity is None:
            return {"suppressed": True}

        return {
            "suppressed": False,
            "import_value_usd": import_value,
            "import_quantity": import_quantity,
            "country_name": record.get("CTY_NAME", ""),
            "commodity_desc": record.get("I_COMMODITY_LDESC", ""),
        }

    except requests.RequestException as e:
        logger.error("census_request_failed hs6=%s error=%s", hs6, e)
        return {"suppressed": True, "error": str(e)}


def run_trade_agent(state: TariffState) -> Dict[str, Any]:
    """
    Fetch live import trade flow data from Census Bureau.

    Args:
        state: TariffState with hts_code and optionally country populated

    Returns:
        Dict with import_value_usd, import_quantity, trade_period,
        trade_country_code, trade_suppressed
    """
    hts_code = state.get("hts_code")
    country = state.get("country")

    hs6 = _parse_hs6(hts_code)
    if not hs6:
        logger.warning("trade_agent_skipped no valid hts_code")
        return {
            "import_value_usd": None,
            "import_quantity": None,
            "trade_period": None,
            "trade_country_code": None,
            "trade_suppressed": True,
        }

    country_code = _get_country_code(country)
    period = DEFAULT_PERIOD

    logger.info(
        "trade_agent_start hs6=%s country=%s country_code=%s period=%s",
        hs6, country, country_code, period,
    )

    result = _query_census(hs6, country_code, period)

    if result.get("suppressed"):
        logger.info("trade_agent_suppressed hs6=%s", hs6)
        return {
            "import_value_usd": None,
            "import_quantity": None,
            "trade_period": period,
            "trade_country_code": country_code,
            "trade_suppressed": True,
        }

    logger.info(
        "trade_agent_done hs6=%s value=%s qty=%s",
        hs6,
        result.get("import_value_usd"),
        result.get("import_quantity"),
    )

    return {
        "import_value_usd": result.get("import_value_usd"),
        "import_quantity": result.get("import_quantity"),
        "trade_period": period,
        "trade_country_code": country_code,
        "trade_suppressed": False,
    }