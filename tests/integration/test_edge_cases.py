"""
Layer 6 — Edge Cases
The API must never 500, must handle malformed input gracefully,
and must escalate to HITL or clarification for ambiguous/unknown inputs.
"""

import pytest
from tests.conftest import api_required


@api_required
def test_empty_query_no_500(api):
    r = api.post("/query", json={"query": ""})
    assert r.status_code != 500, "Got 500 on empty query"


@api_required
def test_empty_query_no_hts_returned(api):
    data = api.post("/query", json={"query": ""}).json()
    assert not data.get("hts_code"), "HTS code should not be returned for empty query"


@api_required
def test_missing_query_field_returns_422(api):
    r = api.post("/query", json={})
    assert r.status_code == 422


@api_required
def test_gibberish_triggers_hitl_or_clarification(api):
    r = api.post("/query", json={"query": "xzqwerty123florp"})
    assert r.status_code != 500, "Got 500 on gibberish query"
    data = r.json()
    triggered = data.get("hitl_required") or data.get("clarification_needed")
    assert triggered, (
        f"Expected HITL or clarification for gibberish, got status={data.get('status')}, "
        f"hitl={data.get('hitl_required')}, clarification={data.get('clarification_needed')}"
    )


@api_required
def test_no_country_query_ok_or_clarification(api):
    r = api.post("/query", json={"query": "tariff on laptops"})
    assert r.status_code != 500
    data = r.json()
    assert data.get("status") in ("ok", "clarification_needed", "error"), (
        f"Unexpected status: {data.get('status')}"
    )


@api_required
def test_misspelled_query_still_classifies(api):
    data = api.post("/query", json={"query": "tarrif on solar panls from china"}).json()
    assert data.get("status") == "ok", (
        "Misspelled query should still complete pipeline — got: "
        + str(data.get("error"))
    )


@api_required
def test_sql_injection_no_500(api):
    r = api.post("/query", json={"query": "'; DROP TABLE HTS_CODES; --"})
    assert r.status_code != 500, "SQL injection string caused a 500"


@api_required
def test_very_long_query_no_500(api):
    long_query = "tariff on " + "solar panels " * 80 + "from China"
    r = api.post("/query", json={"query": long_query})
    assert r.status_code != 500, f"Very long query caused 500: {r.text[:200]}"


@api_required
def test_special_characters_no_500(api):
    for special in ["<script>alert(1)</script>", "𝓣𝓪𝓻𝓲𝓯𝓯", "tariff\x00on\x00things"]:
        r = api.post("/query", json={"query": special})
        assert r.status_code != 500, f"Special chars caused 500: {special!r}"


@api_required
def test_ambiguous_product_clarification(api):
    data = api.post("/query", json={"query": "thing"}).json()
    triggered = data.get("hitl_required") or data.get("clarification_needed")
    assert triggered, (
        "Deliberately vague query 'thing' should trigger HITL or clarification"
    )


@api_required
def test_hitl_response_has_reason(api):
    data = api.post("/query", json={"query": "xzqwerty123florp"}).json()
    if data.get("hitl_required"):
        assert data.get("hitl_reason"), "hitl_reason must be non-empty when hitl_required=True"


@api_required
def test_clarification_response_has_suggestions(api):
    data = api.post("/query", json={"query": "thing"}).json()
    if data.get("clarification_needed"):
        assert data.get("clarification_message"), (
            "clarification_message must be present when clarification_needed=True"
        )
