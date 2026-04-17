"""
Trade Agent — Pipeline Step 6

Fetches live Census Bureau import data via tools.census_trade_flow().
Computes year-over-year trade trend by comparing current month vs same month
prior year using explicit time parameters.

Fixes applied:
  - Vietnam: 5520 (correct Census Schedule C code)
  - _filter_row: returns None on no match, never wrong-country data
  - YoY: fetches prior year same month explicitly for accurate comparison
  - COUNTRY_CODE_MAP: complete official Census Schedule C list
"""

import logging
from datetime import date, timedelta
from typing import Any, Dict, Optional

from agents.state import TariffState
from agents import tools

logger = logging.getLogger(__name__)


# Complete Census Bureau Schedule C country code mapping
COUNTRY_CODE_MAP: Dict[str, str] = {
    # North America
    "united states": "1000", "us": "1000", "usa": "1000",
    "greenland": "1010",
    "canada": "1220",
    "saint pierre and miquelon": "1610",
    "mexico": "2010",

    # Central America
    "guatemala": "2050", "belize": "2080", "el salvador": "2110",
    "honduras": "2150", "nicaragua": "2190", "costa rica": "2230",
    "panama": "2250",

    # Caribbean
    "bermuda": "2320", "bahamas": "2360", "cuba": "2390",
    "jamaica": "2410", "turks and caicos islands": "2430",
    "cayman islands": "2440", "haiti": "2450",
    "dominican republic": "2470", "anguilla": "2481",
    "british virgin islands": "2482", "saint kitts and nevis": "2483",
    "antigua and barbuda": "2484", "montserrat": "2485",
    "dominica": "2486", "saint lucia": "2487",
    "saint vincent and the grenadines": "2488", "grenada": "2489",
    "barbados": "2720", "trinidad and tobago": "2740",
    "sint maarten": "2774", "curacao": "2777", "aruba": "2779",
    "guadeloupe": "2831", "martinique": "2839",

    # South America
    "colombia": "3010", "venezuela": "3070", "guyana": "3120",
    "suriname": "3150", "french guiana": "3170", "ecuador": "3310",
    "peru": "3330", "bolivia": "3350", "chile": "3370",
    "brazil": "3510", "paraguay": "3530", "uruguay": "3550",
    "argentina": "3570", "falkland islands": "3720",

    # Europe
    "iceland": "4000", "sweden": "4010", "norway": "4039",
    "finland": "4050", "faroe islands": "4091", "denmark": "4099",
    "united kingdom": "4120", "uk": "4120", "great britain": "4120",
    "ireland": "4190", "netherlands": "4210", "holland": "4210",
    "belgium": "4231", "luxembourg": "4239", "andorra": "4271",
    "monaco": "4272", "france": "4279", "germany": "4280",
    "austria": "4330", "czech republic": "4351", "czechia": "4351",
    "slovakia": "4359", "hungary": "4370", "liechtenstein": "4411",
    "switzerland": "4419", "estonia": "4470", "latvia": "4490",
    "lithuania": "4510", "poland": "4550", "russia": "4621",
    "russian federation": "4621", "belarus": "4622", "ukraine": "4623",
    "armenia": "4631", "azerbaijan": "4632", "georgia": "4633",
    "kazakhstan": "4634", "kyrgyzstan": "4635", "moldova": "4641",
    "tajikistan": "4642", "turkmenistan": "4643", "uzbekistan": "4644",
    "spain": "4700", "portugal": "4710", "gibraltar": "4720",
    "malta": "4730", "san marino": "4751", "italy": "4759",
    "croatia": "4791", "slovenia": "4792",
    "bosnia and herzegovina": "4793", "bosnia": "4793",
    "north macedonia": "4794", "macedonia": "4794",
    "serbia": "4801", "kosovo": "4803", "montenegro": "4804",
    "albania": "4810", "greece": "4840", "romania": "4850",
    "bulgaria": "4870", "turkey": "4890", "turkiye": "4890",
    "cyprus": "4910",

    # Middle East
    "syria": "5020", "lebanon": "5040", "iraq": "5050",
    "iran": "5070", "israel": "5081", "jordan": "5110",
    "kuwait": "5130", "saudi arabia": "5170", "qatar": "5180",
    "united arab emirates": "5200", "uae": "5200",
    "yemen": "5210", "oman": "5230", "bahrain": "5250",

    # South Asia
    "afghanistan": "5310", "india": "5330", "pakistan": "5350",
    "nepal": "5360", "bangladesh": "5380", "sri lanka": "5420",
    "bhutan": "5682", "maldives": "5683",

    # Southeast Asia
    "burma": "5460", "myanmar": "5460", "thailand": "5490",
    "vietnam": "5520", "viet nam": "5520",       # FIX 1: correct code
    "laos": "5530", "cambodia": "5550", "malaysia": "5570",
    "singapore": "5590", "indonesia": "5600",
    "timor-leste": "5601", "east timor": "5601",
    "brunei": "5610", "philippines": "5650",
    "macao": "5660", "macau": "5660",

    # East Asia
    "china": "5700", "prc": "5700",
    "peoples republic of china": "5700",
    "people's republic of china": "5700",
    "mongolia": "5740", "north korea": "5790",
    "south korea": "5800", "korea": "5800",
    "republic of korea": "5800", "rok": "5800",
    "hong kong": "5820", "taiwan": "5830",
    "japan": "5880",

    # Oceania
    "australia": "6021", "papua new guinea": "6040",
    "new zealand": "6141", "samoa": "6150", "western samoa": "6150",
    "solomon islands": "6223", "vanuatu": "6224", "kiribati": "6226",
    "tuvalu": "6227", "new caledonia": "6412",
    "french polynesia": "6414", "marshall islands": "6810",
    "micronesia": "6820", "palau": "6830", "nauru": "6862",
    "fiji": "6863", "tonga": "6864",

    # Africa - North
    "morocco": "7140", "algeria": "7210", "tunisia": "7230",
    "libya": "7250", "egypt": "7290", "sudan": "7321",
    "south sudan": "7323",

    # Africa - West
    "equatorial guinea": "7380", "mauritania": "7410",
    "cameroon": "7420", "senegal": "7440", "mali": "7450",
    "guinea": "7460", "sierra leone": "7470",
    "cote d'ivoire": "7480", "ivory coast": "7480",
    "ghana": "7490", "gambia": "7500", "niger": "7510",
    "togo": "7520", "nigeria": "7530",
    "central african republic": "7540", "gabon": "7550",
    "chad": "7560", "burkina faso": "7600", "benin": "7610",
    "angola": "7620", "congo": "7630",
    "republic of congo": "7630", "guinea-bissau": "7642",
    "cabo verde": "7643", "cape verde": "7643",
    "sao tome and principe": "7644", "liberia": "7650",
    "democratic republic of congo": "7660", "drc": "7660",
    "zaire": "7660",

    # Africa - East
    "burundi": "7670", "rwanda": "7690", "somalia": "7700",
    "eritrea": "7741", "ethiopia": "7749", "djibouti": "7770",
    "uganda": "7780", "kenya": "7790", "seychelles": "7800",
    "tanzania": "7830", "mauritius": "7850", "mozambique": "7870",
    "madagascar": "7880", "comoros": "7890",

    # Africa - South
    "reunion": "7904", "south africa": "7910", "namibia": "7920",
    "botswana": "7930", "zambia": "7940", "eswatini": "7950",
    "swaziland": "7950", "zimbabwe": "7960", "malawi": "7970",
    "lesotho": "7990",

    # US Territories
    "puerto rico": "9030", "virgin islands": "9110",
    "guam": "9350", "american samoa": "9510",
    "northern mariana islands": "9610",
}

_EMPTY = {
    "import_value_usd": None,
    "import_quantity": None,
    "trade_period": None,
    "trade_country_code": None,
    "trade_suppressed": True,
    "trade_trend_pct": None,
    "trade_trend_label": None,
}


def _country_code(country: Optional[str]) -> Optional[str]:
    if not country:
        return None
    return COUNTRY_CODE_MAP.get(country.lower().strip())


def _filter_row(rows: list, country: Optional[str], code: Optional[str]) -> Optional[Dict]:
    if not rows:
        return None
    if not country and not code:
        return rows[0]
    country_lower = (country or "").lower()
    for row in rows:
        if code and str(row.get("CTY_CODE", "")) == code:
            return row
        if country_lower and country_lower in str(row.get("CTY_NAME", "")).lower():
            return row
    # FIX 2: never return wrong-country data — caller handles as suppressed
    logger.info("trade_agent_no_country_match country=%s code=%s rows=%d",
                country, code, len(rows))
    return None


def _parse_float(raw: Any) -> Optional[float]:
    try:
        if raw in (None, "", "(D)", "0"):
            return None
        return float(str(raw).replace(",", ""))
    except (ValueError, TypeError):
        return None


def _prior_year_month(current_period: str) -> Optional[str]:
    """Given '2026-02' return '2025-02'."""
    try:
        year, month = current_period.split("-")
        return f"{int(year) - 1:04d}-{month}"
    except Exception:
        return None


def _yoy(
    current_value: float,
    hts_code: str,
    current_period: str,
    country: Optional[str],
    code: Optional[str],
) -> Optional[float]:
    """FIX 3: Fetch prior year same month explicitly for accurate YoY comparison."""
    prior_period = _prior_year_month(current_period)
    if not prior_period:
        return None
    try:
        prior_result = tools.census_trade_flow_timed(hts_code, prior_period)
        prior_rows = prior_result.get("rows") or []
        prior_row = _filter_row(prior_rows, country, code)
        if not prior_row:
            return None
        prior_val = _parse_float(prior_row.get("GEN_VAL_MO"))
        if prior_val is None or prior_val == 0:
            return None
        return round(((current_value - prior_val) / prior_val) * 100, 1)
    except Exception as e:
        logger.debug("trade_yoy_failed hts=%s error=%s", hts_code, e)
        return None


def run_trade_agent(state: TariffState) -> Dict[str, Any]:
    hts_code = state.get("hts_code")
    country = state.get("country")

    if not hts_code:
        logger.warning("trade_agent_skipped no hts_code")
        return dict(_EMPTY)

    code = _country_code(country)
    logger.info("trade_agent_start hts=%s country=%s code=%s", hts_code, country, code)

    try:
        result = tools.census_trade_flow(hts_code)
        rows = result.get("rows") or []
        period = result.get("time", "")
        note = result.get("note", "")

        if not rows or note == "data not available at this resolution":
            return {**_EMPTY, "trade_period": period, "trade_country_code": code}

        row = _filter_row(rows, country, code)
        if not row:
            return {**_EMPTY, "trade_period": period, "trade_country_code": code}

        import_value = _parse_float(row.get("GEN_VAL_MO"))
        import_qty = _parse_float(row.get("GEN_QY1_MO") or row.get("CON_VAL_MO"))

        trend_pct: Optional[float] = None
        trend_label: Optional[str] = None
        if import_value is not None and period:
            trend_pct = _yoy(import_value, hts_code, period, country, code)
            if trend_pct is not None:
                direction = "▲" if trend_pct >= 0 else "▼"
                trend_label = f"{direction} {abs(trend_pct):.1f}% YoY"

        logger.info("trade_agent_done hts=%s country=%s value=%s period=%s trend=%s",
                    hts_code, country, import_value, period, trend_label)

        return {
            "import_value_usd": import_value,
            "import_quantity": import_qty,
            "trade_period": period,
            "trade_country_code": code,
            "trade_suppressed": import_value is None,
            "trade_trend_pct": trend_pct,
            "trade_trend_label": trend_label,
        }

    except Exception as e:
        logger.error("trade_agent_failed hts=%s error=%s", hts_code, e)
        return {**_EMPTY, "trade_country_code": code}