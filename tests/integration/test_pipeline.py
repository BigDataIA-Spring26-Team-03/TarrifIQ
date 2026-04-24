"""
Layer 3 — Pipeline Correctness
Known product+country combos → correct HTS prefix, FTA flag, adder presence.
Requires full docker-compose stack + populated Snowflake + ChromaDB.
"""

import pytest
from tests.conftest import api_required

# (query, expected_hts_prefix, expect_fta_applied, expect_adder_gt_zero)
PIPELINE_CASES = [
    ("lithium batteries from China",     "8507", False, True),
    ("solar panels from China",          "8541", False, True),
    ("electric vehicles from China",     "8703", False, True),
    ("semiconductors from China",        "8542", False, True),
    ("steel pipes from South Korea",     "7304", True,  False),  # KORUS FTA
    ("cotton t-shirts from Mexico",      "6109", True,  False),  # USMCA
    ("auto parts from Canada",           "8708", True,  False),  # USMCA
    ("steel wire from Germany",          "7217", False, True),   # Section 232
    ("wine from France",                 "2204", False, False),
    ("coffee from Brazil",               "0901", False, False),
]


@api_required
@pytest.mark.parametrize(
    "query,hts_prefix,expect_fta,expect_adder",
    PIPELINE_CASES,
    ids=[c[0][:45] for c in PIPELINE_CASES],
)
def test_pipeline_hts_prefix(api, query, hts_prefix, expect_fta, expect_adder):
    data = api.post("/query", json={"query": query}).json()
    assert data["status"] == "ok", f"Pipeline failed for '{query}': {data.get('error')}"
    hts = data.get("hts_code") or ""
    assert hts.startswith(hts_prefix), (
        f"Expected HTS prefix {hts_prefix!r}, got {hts!r} for query: {query}"
    )


@api_required
@pytest.mark.parametrize(
    "query,hts_prefix,expect_fta,expect_adder",
    [c for c in PIPELINE_CASES if c[2]],  # only FTA cases
    ids=[c[0][:45] for c in PIPELINE_CASES if c[2]],
)
def test_fta_applied_for_fta_countries(api, query, hts_prefix, expect_fta, expect_adder):
    data = api.post("/query", json={"query": query}).json()
    if data.get("status") != "ok":
        pytest.skip(f"Pipeline non-ok: {data.get('error')}")
    assert data.get("fta_applied") is True, (
        f"Expected fta_applied=True for FTA country query: {query}"
    )
    assert data.get("fta_program"), (
        f"fta_program must be non-empty when fta_applied=True: {query}"
    )
    assert (data.get("base_rate") or 0) <= (data.get("mfn_rate") or 999), (
        f"FTA rate should be ≤ MFN rate for: {query}"
    )


@api_required
@pytest.mark.parametrize(
    "query,hts_prefix,expect_fta,expect_adder",
    [c for c in PIPELINE_CASES if c[3]],  # only adder > 0 cases
    ids=[c[0][:45] for c in PIPELINE_CASES if c[3]],
)
def test_adder_rate_nonzero(api, query, hts_prefix, expect_fta, expect_adder):
    data = api.post("/query", json={"query": query}).json()
    if data.get("status") != "ok":
        pytest.skip(f"Pipeline non-ok: {data.get('error')}")
    adder = data.get("adder_rate") or 0
    assert adder > 0, (
        f"Expected adder_rate > 0 for '{query}', got {adder}. "
        f"adder_method={data.get('adder_method')}"
    )


@api_required
def test_classification_confidence_in_range(api):
    data = api.post("/query", json={"query": "lithium batteries from China"}).json()
    conf = data.get("classification_confidence")
    if conf is not None:
        assert 0.0 <= conf <= 1.0, f"classification_confidence {conf} out of [0,1]"


@api_required
def test_policy_chunks_returned_for_china_query(api):
    data = api.post("/query", json={"query": "solar panels from China"}).json()
    count = data.get("policy_chunks_count") or 0
    assert count > 0, (
        "Expected policy_chunks_count > 0 for China solar panels query — "
        "ChromaDB policy_notices may be empty or retrieval failed"
    )


@api_required
def test_comparison_query_cheapest_country(api):
    data = api.post(
        "/query",
        json={"query": "cheaper to import laptops from China or Vietnam?"},
    ).json()
    if data.get("status") == "clarification_needed":
        pytest.skip("Comparison query routed to clarification")
    assert data.get("cheapest_country") is not None, (
        "comparison query must return cheapest_country"
    )
    comparison = data.get("country_comparison") or []
    assert len(comparison) >= 2, (
        f"country_comparison must have ≥ 2 entries, got {len(comparison)}"
    )


@api_required
def test_fta_adder_is_zero(api):
    data = api.post("/query", json={"query": "steel pipes from South Korea"}).json()
    if data.get("status") != "ok":
        pytest.skip(f"Pipeline non-ok: {data.get('error')}")
    if data.get("fta_applied"):
        adder = data.get("adder_rate") or 0
        assert adder == 0.0, (
            f"adder_rate should be 0 when fta_applied=True, got {adder}"
        )
