"""
Layer 2 — API Contract Tests
Verifies every endpoint returns correct HTTP codes, required fields, and correct types.
Does not assert on business logic — just that the API surface is stable.
"""

import pytest
from tests.conftest import api_required


# ── POST /query ──────────────────────────────────────────────────────────────────

QUERY_REQUIRED_FIELDS = [
    "status", "hts_code", "base_rate", "total_duty",
    "final_response", "citations", "pipeline_confidence",
    "hitl_required", "policy_chunks_count",
]


@api_required
def test_query_returns_200(api):
    r = api.post("/query", json={"query": "solar panels from China"})
    assert r.status_code == 200


@api_required
def test_query_required_fields_present(api):
    data = api.post("/query", json={"query": "solar panels from China"}).json()
    missing = [f for f in QUERY_REQUIRED_FIELDS if f not in data]
    assert not missing, f"Missing required fields: {missing}"


@api_required
def test_query_field_types(api):
    data = api.post("/query", json={"query": "steel pipes from Germany"}).json()
    if data.get("total_duty") is not None:
        assert isinstance(data["total_duty"], (int, float)), "total_duty must be numeric"
    if data.get("base_rate") is not None:
        assert isinstance(data["base_rate"], (int, float)), "base_rate must be numeric"
    assert isinstance(data.get("hitl_required"), bool), "hitl_required must be bool"
    assert data.get("pipeline_confidence") in ("HIGH", "MEDIUM", "LOW", None)
    assert data.get("status") in ("ok", "clarification_needed", "error")


@api_required
def test_query_missing_body_returns_422(api):
    r = api.post("/query", json={})
    assert r.status_code == 422


@api_required
def test_query_citations_is_list(api):
    data = api.post("/query", json={"query": "lithium batteries from China"}).json()
    assert isinstance(data.get("citations"), list), "citations must be a list"


# ── GET /health ───────────────────────────────────────────────────────────────────

@api_required
def test_health_contract(api):
    data = api.get("/health").json()
    assert isinstance(data.get("services"), dict)
    for svc, info in data["services"].items():
        assert "status" in info, f"Service {svc} missing 'status' field"


# ── GET /tools/rate ───────────────────────────────────────────────────────────────

@api_required
def test_tools_rate_valid_code(api):
    r = api.get("/tools/rate", params={"hts_code": "8507.60.00"})
    assert r.status_code == 200
    d = r.json()
    assert d.get("hts_code"), "hts_code missing"
    assert d.get("description"), "description missing"
    assert d.get("general_rate") is not None, "general_rate missing"
    assert d.get("chapter"), "chapter missing"
    assert isinstance(d.get("is_chapter99"), bool), "is_chapter99 must be bool"


@api_required
def test_tools_rate_unknown_code_returns_404(api):
    r = api.get("/tools/rate", params={"hts_code": "9999.99.99"})
    assert r.status_code == 404


@api_required
def test_tools_rate_missing_param_returns_422(api):
    r = api.get("/tools/rate")
    assert r.status_code == 422


# ── GET /tools/hts/search ─────────────────────────────────────────────────────────

@api_required
def test_tools_hts_search_returns_list(api):
    r = api.get("/tools/hts/search", params={"query": "lithium battery", "limit": 5})
    assert r.status_code == 200
    results = r.json()
    assert isinstance(results, list)


@api_required
def test_tools_hts_search_result_fields(api):
    results = api.get("/tools/hts/search", params={"query": "solar cell", "limit": 3}).json()
    if not results:
        pytest.skip("No HTS search results — ChromaDB may be empty")
    for item in results:
        assert "hts_code" in item
        assert "description" in item
        assert "confidence" in item
        assert 0.0 <= item["confidence"] <= 1.0


# ── GET /tools/hts/chapter ────────────────────────────────────────────────────────

@api_required
def test_tools_hts_chapter_returns_list(api):
    r = api.get("/tools/hts/chapter", params={"chapter": "85", "limit": 10})
    assert r.status_code == 200
    assert isinstance(r.json(), list)


@api_required
def test_tools_hts_chapter_codes_in_correct_chapter(api):
    results = api.get("/tools/hts/chapter", params={"chapter": "72", "limit": 20}).json()
    if not results:
        pytest.skip("No results for chapter 72")
    for item in results:
        assert (item.get("hts_code") or "").startswith("72"), (
            f"Code {item.get('hts_code')} not in chapter 72"
        )


# ── POST /tools/search/policy ─────────────────────────────────────────────────────

@api_required
def test_policy_search_returns_list(api):
    r = api.post(
        "/tools/search/policy",
        params={"query": "Section 301 China tariff electronics", "limit": 5},
    )
    assert r.status_code == 200
    assert isinstance(r.json(), list)


@api_required
def test_policy_search_result_fields(api):
    results = api.post(
        "/tools/search/policy",
        params={"query": "steel tariff Section 232", "limit": 3},
    ).json()
    if not results:
        pytest.skip("No policy search results — ChromaDB may be empty")
    for item in results:
        assert "chunk_text" in item, "chunk_text missing"
        assert "document_number" in item, "document_number missing"
        assert "source" in item, "source missing"
        assert item["source"] in ("USTR", "CBP", "USITC", "ITA", "EOP", ""), (
            f"Unexpected source: {item['source']}"
        )


# ── POST /tools/search/hts ────────────────────────────────────────────────────────

@api_required
def test_hts_vector_search_returns_list(api):
    r = api.post("/tools/search/hts", params={"query": "electric vehicle battery", "limit": 5})
    assert r.status_code == 200
    assert isinstance(r.json(), list)
