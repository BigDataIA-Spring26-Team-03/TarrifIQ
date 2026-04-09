"""
USITC HTS exportList API client — fetch and normalize HTS rows for loading.
"""

from __future__ import annotations

import logging
from typing import Any, List

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://hts.usitc.gov/reststop/exportList"
REQUEST_TIMEOUT = 60
_DEFAULT_PARAMS = {"format": "JSON", "styles": "false"}


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def normalize_footnotes(value: Any) -> List[str]:
    """Always return a list of strings; empty if null."""
    if value is None:
        return []
    if isinstance(value, list):
        return [_coerce_str(x) for x in value]
    return [_coerce_str(value)]


def _units_to_unit12(units: Any) -> tuple[str, str]:
    """Map API `units` (usually a list) to unit1 / unit2."""
    if units is None:
        return "", ""
    if isinstance(units, str):
        return units.strip(), ""
    if isinstance(units, list):
        u1 = _coerce_str(units[0]).strip() if len(units) > 0 else ""
        u2 = _coerce_str(units[1]).strip() if len(units) > 1 else ""
        return u1, u2
    return "", ""


def parse_usitc_export_rows(raw_rows: List[dict], *, is_chapter99: bool) -> List[dict]:
    """
    Normalize raw USITC JSON objects into HTS_CODES-shaped dicts.
    Skips rows with empty or missing htsno.
    """
    out: List[dict] = []
    for raw in raw_rows:
        htsno = raw.get("htsno")
        if htsno is None or str(htsno).strip() == "":
            logger.debug("skip_empty_htsno")
            continue
        unit1, unit2 = _units_to_unit12(raw.get("units"))
        out.append(
            {
                "htsno": str(htsno).strip(),
                "indent": _coerce_str(raw.get("indent")),
                "description": _coerce_str(raw.get("description")),
                "unit1": unit1,
                "unit2": unit2,
                "general": _coerce_str(raw.get("general")),
                "special": _coerce_str(raw.get("special")),
                "other": _coerce_str(raw.get("other")),
                "footnotes": normalize_footnotes(raw.get("footnotes")),
                "is_chapter99": is_chapter99,
            }
        )
    return out


def _unwrap_json_array(data: Any) -> List[dict]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("data", "results", "records"):
            inner = data.get(key)
            if isinstance(inner, list):
                return [x for x in inner if isinstance(x, dict)]
        logger.warning("unexpected_json_shape", keys=list(data.keys()))
    else:
        logger.warning("unexpected_json_type", type_name=type(data).__name__)
    return []


def _get_export_list(from_chapter: str, to_chapter: str) -> List[dict]:
    params = {**_DEFAULT_PARAMS, "from": from_chapter, "to": to_chapter}
    logger.info(
        "usitc_export_list_request from=%s to=%s",
        from_chapter,
        to_chapter,
    )
    response = requests.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return _unwrap_json_array(response.json())


def fetch_hts_schedule() -> List[dict]:
    """
    Full HTS schedule (chapters 01–99 range via USITC chapter window).
    GET .../exportList?format=JSON&from=0101&to=9999&styles=false
    """
    raw = _get_export_list("0101", "9999")
    return parse_usitc_export_rows(raw, is_chapter99=False)


def fetch_chapter99() -> List[dict]:
    """
    Chapter 99 window (9903–9904).
    GET .../exportList?format=JSON&from=9903&to=9904&styles=false
    """
    raw = _get_export_list("9903", "9904")
    return parse_usitc_export_rows(raw, is_chapter99=True)
