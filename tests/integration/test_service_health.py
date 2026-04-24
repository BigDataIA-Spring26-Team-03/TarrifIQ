"""
Layer 1 — Service Health
Verifies all dependent services are reachable and respond within latency SLAs.
Requires: docker-compose up (fastapi, chromadb, redis, snowflake connection)
"""

import pytest
from tests.conftest import api_required

LATENCY_SLAS = [
    ("snowflake", 2000),
    ("redis",     200),
    ("chromadb",  1000),
]


@api_required
def test_health_endpoint_returns_200(api):
    r = api.get("/health")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"


@api_required
def test_health_response_has_required_fields(api):
    data = api.get("/health").json()
    assert "status" in data
    assert "timestamp" in data
    assert "services" in data
    assert isinstance(data["services"], dict)


@api_required
def test_overall_status_is_ok(api):
    data = api.get("/health").json()
    assert data["status"] == "ok", (
        f"Overall status is '{data['status']}' — one or more services degraded: "
        + str({k: v.get("error") for k, v in data.get("services", {}).items() if v.get("status") != "ok"})
    )


@api_required
@pytest.mark.parametrize("svc", ["snowflake", "redis", "chromadb"])
def test_individual_service_healthy(api, svc):
    data = api.get("/health").json()
    svc_data = data.get("services", {}).get(svc, {})
    assert svc_data.get("status") == "ok", (
        f"{svc} reports status='{svc_data.get('status')}', error={svc_data.get('error')}"
    )


@api_required
@pytest.mark.parametrize("svc,max_ms", LATENCY_SLAS)
def test_service_latency_within_sla(api, svc, max_ms):
    data = api.get("/health").json()
    svc_data = data.get("services", {}).get(svc, {})
    lat = svc_data.get("latency_ms")
    if lat is None:
        pytest.skip(f"latency_ms not reported for {svc}")
    assert lat < max_ms, f"{svc} latency {lat:.0f}ms exceeds SLA of {max_ms}ms"
