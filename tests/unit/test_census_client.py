"""
Unit tests for Census Bureau import trade client (no live HTTP / Redis).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ingestion import census_client
from ingestion.census_client import (
    BASE_URL,
    REQUEST_TIMEOUT,
    _add_months,
    _infer_commodity_and_level,
    _month_range_back_from,
    _parse_census_table,
    _unique_headers,
    get_trade_flow,
    get_trade_trend,
)


def test_infer_commodity_hs2_hs4_hs6_and_hs10_behavior():
    assert _infer_commodity_and_level("84") == ("84", "HS2")
    assert _infer_commodity_and_level("8471") == ("8471", "HS4")
    assert _infer_commodity_and_level("847130") == ("847130", "HS6")
    assert _infer_commodity_and_level("84713012") == ("8471301200", "HS10")
    assert _infer_commodity_and_level("8471301234") == ("8471301234", "HS10")
    assert _infer_commodity_and_level("847130123456") == ("8471301234", "HS10")
    assert _infer_commodity_and_level("84.71") == ("8471", "HS4")


def test_infer_commodity_empty_returns_hs10_placeholder():
    assert _infer_commodity_and_level("") == ("", "HS10")
    assert _infer_commodity_and_level("abc") == ("", "HS10")


def test_unique_headers_duplicate_i_commodity():
    raw = ["GEN_VAL_MO", "I_COMMODITY", "I_COMMODITY", "time"]
    assert _unique_headers(raw) == [
        "GEN_VAL_MO",
        "I_COMMODITY",
        "I_COMMODITY__1",
        "time",
    ]


def test_parse_census_table_realistic_shape():
    data = [
        [
            "GEN_VAL_MO",
            "CTY_CODE",
            "CTY_NAME",
            "I_COMMODITY",
            "I_COMMODITY_LDESC",
            "I_COMMODITY",
            "COMM_LVL",
            "time",
        ],
        [
            "100",
            "5330",
            "INDIA",
            "8471",
            "COMPUTERS",
            "8471",
            "HS4",
            "2024-01",
        ],
    ]
    rows = _parse_census_table(data)
    assert len(rows) == 1
    assert rows[0]["GEN_VAL_MO"] == "100"
    assert rows[0]["CTY_NAME"] == "INDIA"
    assert rows[0]["I_COMMODITY__1"] == "8471"


def test_parse_census_table_malformed_returns_empty():
    assert _parse_census_table([]) == []
    assert _parse_census_table([[]]) == []
    assert _parse_census_table("not a list") == []


def test_add_months_and_month_range_back_from():
    assert _add_months(2024, 3, -1) == (2024, 2)
    assert _add_months(2024, 1, -1) == (2023, 12)
    assert _month_range_back_from("2024-03", 3) == ["2024-01", "2024-02", "2024-03"]


@patch.object(census_client, "_cache_get", return_value=None)
@patch.object(census_client, "_cache_set")
@patch.object(census_client, "_fetch_census_raw")
def test_get_trade_flow_success_parses_rows(mock_fetch, mock_cache_set, mock_cache_get):
    mock_fetch.return_value = (
        200,
        [
            ["GEN_VAL_MO", "CTY_CODE", "CTY_NAME"],
            ["50", "5200", "UAE"],
        ],
    )
    out = get_trade_flow("8471", "2024-01")
    assert out["hts_code"] == "8471"
    assert out["comm_lvl"] == "HS4"
    assert out["time"] == "2024-01"
    assert len(out["rows"]) == 1
    assert out["rows"][0]["CTY_NAME"] == "UAE"
    mock_fetch.assert_called_once()
    mock_cache_set.assert_called_once()


@patch.object(census_client, "_cache_get", return_value=None)
@patch.object(census_client, "_cache_set")
@patch.object(census_client, "_fetch_census_raw")
def test_get_trade_flow_204_sets_note(mock_fetch, mock_cache_set, mock_cache_get):
    mock_fetch.return_value = (204, None)
    out = get_trade_flow("8471", "2024-01")
    assert out["rows"] == []
    assert out["note"] == "data not available at this resolution"
    mock_cache_set.assert_not_called()


@patch.object(census_client, "_cache_get", return_value=None)
@patch.object(census_client, "_cache_set")
@patch.object(census_client, "_fetch_census_raw")
def test_get_trade_flow_non_200_empty_rows(mock_fetch, mock_cache_set, mock_cache_get):
    mock_fetch.return_value = (500, None)
    out = get_trade_flow("8471", "2024-01")
    assert out["rows"] == []
    assert "note" not in out
    mock_cache_set.assert_not_called()


@patch.object(census_client, "_cache_get", return_value=None)
@patch.object(census_client, "_cache_set")
@patch.object(census_client, "_fetch_census_raw")
def test_get_trade_flow_api_error_dict(mock_fetch, mock_cache_set, mock_cache_get):
    mock_fetch.return_value = (200, {"error": ["something"]})
    out = get_trade_flow("8471", "2024-01")
    assert out["rows"] == []
    mock_cache_set.assert_not_called()


@patch.object(census_client, "_cache_get")
@patch.object(census_client, "_fetch_census_raw")
def test_get_trade_flow_cache_hit_skips_fetch(mock_fetch, mock_cache_get):
    cached = {
        "hts_code": "8471",
        "comm_lvl": "HS4",
        "time": "2024-01",
        "rows": [{"x": 1}],
    }
    mock_cache_get.return_value = cached
    out = get_trade_flow("8471", "2024-01")
    assert out is cached
    mock_fetch.assert_not_called()


def test_get_trade_flow_invalid_hts_never_calls_fetch():
    with patch.object(census_client, "_fetch_census_raw") as mock_fetch:
        out = get_trade_flow("   ", None)
        assert out["rows"] == []
        assert "invalid" in (out.get("note") or "")
        mock_fetch.assert_not_called()


@patch.object(census_client, "get_trade_flow")
@patch.object(census_client, "_default_time_month", return_value="2024-03")
def test_get_trade_trend_parallel_months(mock_default_ym, mock_gtf):
    def _side_effect(hts: str, ym: str):
        return {
            "hts_code": "8471",
            "comm_lvl": "HS4",
            "time": ym,
            "rows": [{"GEN_VAL_MO": "1", "CTY_CODE": "5200", "CTY_NAME": "X"}],
        }

    mock_gtf.side_effect = _side_effect
    trend = get_trade_trend("8471", 3)
    assert [t["time"] for t in trend] == ["2024-01", "2024-02", "2024-03"]
    assert mock_gtf.call_count == 3


@patch("ingestion.census_client.requests.get")
def test_fetch_census_raw_passes_params_and_timeout(mock_get):
    mock_resp = MagicMock()
    mock_resp.content = b'[["a"],["b"]]'
    mock_resp.status_code = 200
    mock_resp.json.return_value = [["a"], ["b"]]
    mock_get.return_value = mock_resp

    status, payload = census_client._fetch_census_raw("8471", "HS4", "2024-01", "")
    assert status == 200
    assert payload == [["a"], ["b"]]
    mock_get.assert_called_once_with(
        BASE_URL,
        params={
            "get": census_client._GET_VARS,
            "COMM_LVL": "HS4",
            "SUMMARY_LVL": "DET",
            "time": "2024-01",
            "I_COMMODITY": "8471",
        },
        timeout=REQUEST_TIMEOUT,
    )


@patch("ingestion.census_client.requests.get")
def test_fetch_census_raw_includes_key_when_set(mock_get, monkeypatch):
    monkeypatch.setenv("CENSUS_API_KEY", "secret")
    mock_resp = MagicMock()
    mock_resp.content = b"[]"
    mock_resp.status_code = 200
    mock_resp.json.return_value = []
    mock_get.return_value = mock_resp

    census_client._fetch_census_raw("84", "HS2", "2024-01", "secret")
    call_kw = mock_get.call_args.kwargs
    assert call_kw["params"]["key"] == "secret"
