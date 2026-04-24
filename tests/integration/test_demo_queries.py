"""
Layer 7 — Demo Queries
The exact 5 professor demo queries from CLAUDE.md must all complete successfully
with the correct outputs. These are the queries demonstrated on April 25.
"""

import pytest
from tests.conftest import api_required

DEMO_QUERIES = [
    {
        "id": "lithium_batteries_china",
        "query": "What is the tariff on lithium batteries from China?",
        "checks": {
            "status": "ok",
            "hts_prefix": "8507",
            "adder_rate_gt": 0,
            "policy_chunks_gt": 0,
            "fta_applied": False,
            "rate_reconciliation": True,
            "final_response_nonempty": True,
        },
    },
    {
        "id": "steel_pipes_south_korea",
        "query": "steel pipes from South Korea",
        "checks": {
            "status": "ok",
            "hts_prefix": "7304",
            "fta_applied": True,
            "fta_program_contains": "KORUS",
            "adder_is_zero": True,
            "rate_reconciliation": True,
            "final_response_nonempty": True,
        },
    },
    {
        "id": "furniture_china_2018",
        "query": "why did tariffs on furniture from China increase in 2018",
        "checks": {
            "status": "ok",
            "policy_chunks_gt": 0,
            "final_response_nonempty": True,
        },
    },
    {
        "id": "solar_imports_china_2019",
        "query": "how have solar panel imports from China changed since 2019",
        "checks": {
            "status": "ok",
            "final_response_nonempty": True,
        },
    },
    {
        "id": "vague_thing",
        "query": "thing",
        "checks": {
            "hitl_or_clarification": True,
        },
    },
]


def _run_checks(data: dict, checks: dict, query: str) -> None:
    if "status" in checks:
        assert data.get("status") == checks["status"], (
            f"status={data.get('status')!r} (expected {checks['status']!r}) for: {query}\n"
            f"error: {data.get('error')}"
        )

    if "hts_prefix" in checks:
        hts = data.get("hts_code") or ""
        assert hts.startswith(checks["hts_prefix"]), (
            f"HTS {hts!r} does not start with {checks['hts_prefix']!r} for: {query}"
        )

    if "adder_rate_gt" in checks:
        adder = data.get("adder_rate") or 0
        assert adder > checks["adder_rate_gt"], (
            f"adder_rate {adder} not > {checks['adder_rate_gt']} for: {query}. "
            f"adder_method={data.get('adder_method')}"
        )

    if "adder_is_zero" in checks and checks["adder_is_zero"]:
        adder = data.get("adder_rate") or 0
        assert adder == 0.0, (
            f"adder_rate should be 0 (FTA exempt), got {adder} for: {query}"
        )

    if "policy_chunks_gt" in checks:
        count = data.get("policy_chunks_count") or 0
        assert count > checks["policy_chunks_gt"], (
            f"policy_chunks_count={count} not > {checks['policy_chunks_gt']} for: {query}"
        )

    if "fta_applied" in checks:
        assert data.get("fta_applied") == checks["fta_applied"], (
            f"fta_applied={data.get('fta_applied')!r} (expected {checks['fta_applied']!r}) for: {query}"
        )

    if "fta_program_contains" in checks:
        prog = data.get("fta_program") or ""
        assert checks["fta_program_contains"] in prog, (
            f"fta_program {prog!r} does not contain {checks['fta_program_contains']!r} for: {query}"
        )

    if checks.get("rate_reconciliation"):
        rec = data.get("rate_reconciliation")
        if rec is not None:
            assert rec.get("check_passed") is True, (
                f"rate_reconciliation.check_passed=False for: {query}. "
                f"Calc: {rec.get('calculation')}"
            )

    if checks.get("final_response_nonempty"):
        resp = data.get("final_response") or ""
        assert len(resp) > 50, (
            f"final_response too short ({len(resp)} chars) for: {query}"
        )

    if checks.get("hitl_or_clarification"):
        triggered = data.get("hitl_required") or data.get("clarification_needed")
        assert triggered, (
            f"Expected HITL or clarification for vague query: {query!r}. "
            f"status={data.get('status')}, hitl={data.get('hitl_required')}, "
            f"clarification={data.get('clarification_needed')}"
        )


@api_required
@pytest.mark.parametrize(
    "case",
    DEMO_QUERIES,
    ids=[c["id"] for c in DEMO_QUERIES],
)
def test_demo_query(api, case):
    data = api.post("/query", json={"query": case["query"]}).json()
    _run_checks(data, case["checks"], case["query"])


@api_required
def test_all_demo_queries_no_500(api):
    for case in DEMO_QUERIES:
        r = api.post("/query", json={"query": case["query"]})
        assert r.status_code != 500, (
            f"Demo query returned 500: {case['query']!r}\n{r.text[:300]}"
        )


@api_required
def test_demo_rate_arithmetic_consistent(api):
    rate_queries = [c for c in DEMO_QUERIES if "hts_prefix" in c["checks"]]
    for case in rate_queries:
        data = api.post("/query", json={"query": case["query"]}).json()
        if data.get("status") != "ok":
            continue
        base  = data.get("base_rate")  or 0.0
        adder = data.get("adder_rate") or 0.0
        total = data.get("total_duty") or 0.0
        delta = abs((base + adder) - total)
        assert delta < 0.05, (
            f"Rate arithmetic failed for demo query '{case['query']}': "
            f"{base} + {adder} ≠ {total:.4f} (delta={delta:.4f})"
        )
