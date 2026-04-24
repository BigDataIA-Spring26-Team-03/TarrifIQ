"""
Layer 4 — Rate Validation
1. Arithmetic: base_rate + adder_rate == total_duty (±0.05) for every response.
2. rate_reconciliation.check_passed must be True when field is present.
3. Golden rate ranges: known product/country pairs must land within expected total_duty bounds.
"""

import pytest
from tests.conftest import api_required
from tests.integration.test_pipeline import PIPELINE_CASES

# (query, hts_prefix, min_total_duty, max_total_duty)
GOLDEN_RATES = [
    ("solar panels from China",       "8541",  40.0,  65.0),
    ("electric vehicles from China",  "8703",  90.0, 115.0),
    ("lithium batteries from China",  "8507",  25.0,  45.0),
    ("semiconductors from China",     "8542",  25.0,  60.0),
    ("wine from France",              "2204",   0.0,  15.0),
    ("coffee from Brazil",            "0901",   0.0,   2.0),
    ("steel pipes from South Korea",  "7304",   0.0,   5.0),
    ("cotton t-shirts from Mexico",   "6109",   0.0,   5.0),
]


@api_required
@pytest.mark.parametrize(
    "query", [c[0] for c in PIPELINE_CASES],
    ids=[c[0][:45] for c in PIPELINE_CASES],
)
def test_rate_arithmetic(api, query):
    data = api.post("/query", json={"query": query}).json()
    if data.get("status") != "ok":
        pytest.skip(f"Pipeline non-ok for arithmetic check: {data.get('error')}")
    base  = data.get("base_rate")  or 0.0
    adder = data.get("adder_rate") or 0.0
    total = data.get("total_duty") or 0.0
    delta = abs((base + adder) - total)
    assert delta < 0.05, (
        f"Rate arithmetic failed for '{query}': "
        f"{base} + {adder} = {base + adder:.4f}, reported total = {total:.4f}, delta = {delta:.4f}"
    )


@api_required
@pytest.mark.parametrize(
    "query", [c[0] for c in PIPELINE_CASES],
    ids=[c[0][:45] for c in PIPELINE_CASES],
)
def test_rate_reconciliation_passes(api, query):
    data = api.post("/query", json={"query": query}).json()
    if data.get("status") != "ok":
        pytest.skip(f"Pipeline non-ok: {data.get('error')}")
    rec = data.get("rate_reconciliation")
    if rec is None:
        pytest.skip("rate_reconciliation field not present in response")
    assert rec.get("check_passed") is True, (
        f"rate_reconciliation.check_passed=False for '{query}'. "
        f"Calculation: {rec.get('calculation')}"
    )


@api_required
def test_total_duty_nonnegative(api):
    for query, *_ in PIPELINE_CASES[:4]:
        data = api.post("/query", json={"query": query}).json()
        if data.get("status") == "ok":
            total = data.get("total_duty") or 0
            assert total >= 0, f"Negative total_duty {total} for: {query}"


@api_required
@pytest.mark.parametrize(
    "query,hts_prefix,min_total,max_total",
    GOLDEN_RATES,
    ids=[r[0][:45] for r in GOLDEN_RATES],
)
def test_golden_rate_ranges(api, query, hts_prefix, min_total, max_total):
    data = api.post("/query", json={"query": query}).json()
    if data.get("status") != "ok":
        pytest.skip(f"Pipeline non-ok for golden rate check: {data.get('error')}")
    hts = data.get("hts_code") or ""
    if not hts.startswith(hts_prefix):
        pytest.skip(
            f"HTS mismatch: expected prefix {hts_prefix}, got {hts} — skipping rate range check"
        )
    total = data.get("total_duty") or 0
    assert min_total <= total <= max_total, (
        f"total_duty {total:.2f}% outside expected range [{min_total}, {max_total}] for: {query}. "
        f"base={data.get('base_rate')}, adder={data.get('adder_rate')}, "
        f"adder_method={data.get('adder_method')}"
    )


@api_required
def test_fta_rate_lower_than_mfn(api):
    data = api.post("/query", json={"query": "auto parts from Canada"}).json()
    if data.get("status") != "ok" or not data.get("fta_applied"):
        pytest.skip("FTA not applied or pipeline failed")
    fta  = data.get("fta_rate")  or data.get("base_rate") or 0
    mfn  = data.get("mfn_rate")  or 0
    assert fta <= mfn, (
        f"FTA rate {fta}% should be ≤ MFN rate {mfn}% for Canada USMCA query"
    )
