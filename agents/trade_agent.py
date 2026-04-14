"""
Trade Agent — TariffIQ Pipeline Step 5

Uses Ishaan's census_client for live Census Bureau data with:
- Redis caching
- HS10 -> HS6 -> HS4 fallback chain
- Effective tariff rate calculation
- Multi-month trend support

No data stored in Snowflake — queried live at query time.
"""

import logging
from typing import Dict, Any, Optional

from ingestion.census_client import get_trade_flow
from agents.state import TariffState

logger = logging.getLogger(__name__)

# Country name to Census country code mapping
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


def _get_country_code(country: Optional[str]) -> Optional[str]:
    if not country:
        return None
    return COUNTRY_CODE_MAP.get(country.lower().strip())


def _filter_by_country(
    rows: list, country: Optional[str], country_code: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Find the row matching the requested country."""
    if not rows or not country:
        return rows[0] if rows else None

    country_lower = country.lower()
    for row in rows:
        cty_name = str(row.get("CTY_NAME", "")).lower()
        cty_code = str(row.get("CTY_CODE", ""))
        if country_lower in cty_name or (country_code and cty_code == country_code):
            return row

    # Fallback to first row if no country match
    return rows[0] if rows else None


def run_trade_agent(state: TariffState) -> Dict[str, Any]:
    """
    Fetch live import trade flow data from Census Bureau via census_client.

    Args:
        state: TariffState with hts_code and optionally country populated

    Returns:
        Dict with import_value_usd, import_quantity, trade_period,
        trade_country_code, trade_suppressed
    """
    hts_code = state.get("hts_code")
    country = state.get("country")

    if not hts_code:
        logger.warning("trade_agent_skipped no hts_code")
        return {
            "import_value_usd": None,
            "import_quantity": None,
            "trade_period": None,
            "trade_country_code": None,
            "trade_suppressed": True,
        }

    country_code = _get_country_code(country)

    logger.info(
        "trade_agent_start hts=%s country=%s country_code=%s",
        hts_code, country, country_code,
    )

    try:
        result = get_trade_flow(hts_code)

        rows = result.get("rows") or []
        period = result.get("time", "")
        note = result.get("note", "")

        if not rows or note == "data not available at this resolution":
            logger.info("trade_agent_suppressed hts=%s note=%s", hts_code, note)
            return {
                "import_value_usd": None,
                "import_quantity": None,
                "trade_period": period,
                "trade_country_code": country_code,
                "trade_suppressed": True,
            }

        # Filter to requested country if specified
        row = _filter_by_country(rows, country, country_code)
        if not row:
            return {
                "import_value_usd": None,
                "import_quantity": None,
                "trade_period": period,
                "trade_country_code": country_code,
                "trade_suppressed": True,
            }

        # Parse import value
        val_raw = row.get("GEN_VAL_MO")
        try:
            import_value = float(str(val_raw).replace(",", "")) if val_raw not in (None, "", "(D)") else None
        except (ValueError, TypeError):
            import_value = None

        # Parse quantity
        qty_raw = row.get("GEN_QY1_MO") or row.get("CON_VAL_MO")
        try:
            import_qty = float(str(qty_raw).replace(",", "")) if qty_raw not in (None, "", "(D)") else None
        except (ValueError, TypeError):
            import_qty = None

        logger.info(
            "trade_agent_done hts=%s country=%s value=%s period=%s",
            hts_code, country, import_value, period,
        )

        return {
            "import_value_usd": import_value,
            "import_quantity": import_qty,
            "trade_period": period,
            "trade_country_code": country_code,
            "trade_suppressed": import_value is None,
        }

    except Exception as e:
        logger.error("trade_agent_failed hts=%s error=%s", hts_code, e)
        return {
            "import_value_usd": None,
            "import_quantity": None,
            "trade_period": None,
            "trade_country_code": country_code,
            "trade_suppressed": True,
        }