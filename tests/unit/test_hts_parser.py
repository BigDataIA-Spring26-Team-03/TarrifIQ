"""
Unit tests for USITC HTS export parsing (mocked HTTP).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ingestion.hts_idempotent_load import (
    INSERT_WHERE_NOT_EXISTS_SQL,
    load_batches_idempotent,
    merge_rows_two_passes,
)
from ingestion.usitc_client import (
    BASE_URL,
    REQUEST_TIMEOUT,
    fetch_chapter99,
    fetch_hts_schedule,
    normalize_footnotes,
    parse_usitc_export_rows,
)


@pytest.fixture
def minimal_usitc_row() -> dict:
    return {
        "htsno": "8507.60",
        "indent": "1",
        "description": "Electric storage batteries",
        "superior": None,
        "units": [],
        "general": "Free",
        "special": "",
        "other": "",
        "footnotes": [],
    }


@pytest.fixture
def mock_get_success():
    with patch("ingestion.usitc_client.requests.get") as mock_get:
        response = MagicMock()
        response.raise_for_status = MagicMock()
        mock_get.return_value = response
        yield mock_get, response


def _assert_schedule_call(mock_get: MagicMock) -> None:
    mock_get.assert_called_with(
        BASE_URL,
        params={"format": "JSON", "styles": "false", "from": "0101", "to": "9999"},
        timeout=REQUEST_TIMEOUT,
    )


def _assert_chapter99_call(mock_get: MagicMock) -> None:
    mock_get.assert_called_with(
        BASE_URL,
        params={"format": "JSON", "styles": "false", "from": "9903", "to": "9904"},
        timeout=REQUEST_TIMEOUT,
    )


def test_is_chapter99_true_for_chapter99_schedule_false_for_main(
    mock_get_success, minimal_usitc_row
):
    mock_get, response = mock_get_success
    ch99_row = {**minimal_usitc_row, "htsno": "9903.88.03"}
    response.json.return_value = [ch99_row]

    rows_ch99 = fetch_chapter99()
    _assert_chapter99_call(mock_get)
    assert len(rows_ch99) == 1
    assert rows_ch99[0]["htsno"] == "9903.88.03"
    assert rows_ch99[0]["is_chapter99"] is True

    response.json.return_value = [minimal_usitc_row]
    rows_main = fetch_hts_schedule()
    _assert_schedule_call(mock_get)
    assert len(rows_main) == 1
    assert rows_main[0]["htsno"] == "8507.60"
    assert rows_main[0]["is_chapter99"] is False


def test_skips_empty_htsno(mock_get_success):
    mock_get, response = mock_get_success
    base = {
        "indent": "0",
        "description": "x",
        "units": [],
        "general": "",
        "special": "",
        "other": "",
        "footnotes": [],
    }
    response.json.return_value = [
        {**base, "htsno": ""},
        {**base, "htsno": None},
        {**base, "htsno": "   "},
        {**base, "htsno": "8501.12.00"},
    ]
    rows = fetch_hts_schedule()
    assert len(rows) == 1
    assert rows[0]["htsno"] == "8501.12.00"


def test_footnotes_always_list_string_list_and_null(mock_get_success):
    mock_get, response = mock_get_success
    base = {
        "indent": "0",
        "description": "d",
        "units": [],
        "general": "",
        "special": "",
        "other": "",
    }
    response.json.return_value = [
        {**base, "htsno": "1111.11.11", "footnotes": "single"},
        {**base, "htsno": "2222.22.22", "footnotes": ["a", "b"]},
        {**base, "htsno": "3333.33.33", "footnotes": None},
    ]
    rows = fetch_hts_schedule()
    assert rows[0]["footnotes"] == ["single"]
    assert rows[1]["footnotes"] == ["a", "b"]
    assert rows[2]["footnotes"] == []


def test_normalize_footnotes_unit_helpers_directly():
    assert normalize_footnotes(None) == []
    assert normalize_footnotes("x") == ["x"]
    assert normalize_footnotes([1, 2]) == ["1", "2"]


def test_parse_units_to_unit1_unit2():
    raw = [
        {
            "htsno": "1",
            "indent": "0",
            "description": "",
            "units": ["kg", "No."],
            "general": "",
            "special": "",
            "other": "",
            "footnotes": [],
        }
    ]
    out = parse_usitc_export_rows(raw, is_chapter99=False)
    assert out[0]["unit1"] == "kg"
    assert out[0]["unit2"] == "No."


def test_idempotent_insert_sql_uses_where_not_exists():
    assert "WHERE NOT EXISTS" in INSERT_WHERE_NOT_EXISTS_SQL
    assert "HTS_CODES" in INSERT_WHERE_NOT_EXISTS_SQL


def test_merge_chapter99_pass_forces_is_chapter99_true():
    base = {
        "indent": "1",
        "description": "d",
        "superior": None,
        "units": [],
        "general": "",
        "special": "",
        "other": "",
        "footnotes": [],
    }
    main_raw = [{**base, "htsno": "9903.88.03", "description": "from main"}]
    ch99_raw = [{**base, "htsno": "9903.88.03", "description": "from ch99"}]
    merged, skipped = merge_rows_two_passes(main_raw, ch99_raw)
    assert skipped == 0
    assert len(merged) == 1
    assert merged[0]["is_chapter99"] is True
    assert merged[0]["description"] == "from ch99"


@patch("ingestion.hts_idempotent_load.get_connection")
def test_second_idempotent_run_inserts_zero_when_nothing_new(mock_get_conn):
    """Simulate table already containing codes: INSERT...WHERE NOT EXISTS inserts 0 rows."""
    from ingestion.load_hts_to_snowflake import raw_row_to_params

    raw = {
        "htsno": "8501.10.00",
        "indent": "0",
        "description": "Motors",
        "superior": None,
        "units": [],
        "general": "Free",
        "special": "",
        "other": "",
        "footnotes": [],
    }
    row = raw_row_to_params(raw)
    assert row is not None

    cur = MagicMock()

    def _execute(query, *args, **kwargs):
        q = query if isinstance(query, str) else ""
        if "WHERE NOT EXISTS" in q:
            cur.rowcount = 0
        return None

    cur.execute.side_effect = _execute
    conn = MagicMock()
    conn.cursor.return_value = cur
    mock_get_conn.return_value = conn

    loaded = load_batches_idempotent([row], batch_size=500)
    assert loaded == 0


@patch("ingestion.usitc_client.requests.get")
def test_fetch_chapter99_all_rows_have_is_chapter99_true(mock_get):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = [
        {
            "htsno": "9903.01.01",
            "indent": "1",
            "description": "a",
            "units": [],
            "general": "",
            "special": "",
            "other": "",
            "footnotes": [],
        },
        {
            "htsno": "9904.12.00.00",
            "indent": "2",
            "description": "b",
            "units": [],
            "general": "",
            "special": "",
            "other": "",
            "footnotes": [],
        },
    ]
    mock_get.return_value = resp
    rows = fetch_chapter99()
    assert len(rows) == 2
    assert all(r["is_chapter99"] for r in rows)
