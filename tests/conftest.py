"""
Shared pytest fixtures and skip markers for TariffIQ integration tests.
All integration tests require a running docker-compose stack on localhost.
"""

import os
import pytest

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore

API_BASE = os.environ.get("TARIFFIQ_API_URL", "http://localhost:8001")


def _api_reachable() -> bool:
    if httpx is None:
        return False
    try:
        return httpx.get(f"{API_BASE}/health", timeout=3).status_code == 200
    except Exception:
        return False


def _snowflake_available() -> bool:
    try:
        import snowflake.connector
        conn = snowflake.connector.connect(
            user=os.environ["SNOWFLAKE_USER"],
            password=os.environ["SNOWFLAKE_PASSWORD"],
            account=os.environ["SNOWFLAKE_ACCOUNT"],
            warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
            database=os.environ["SNOWFLAKE_DATABASE"],
            schema=os.environ["SNOWFLAKE_SCHEMA"],
        )
        conn.cursor().execute("SELECT 1")
        conn.close()
        return True
    except Exception:
        return False


# ── Skip markers ────────────────────────────────────────────────────────────────

api_required = pytest.mark.skipif(
    not _api_reachable(),
    reason=f"FastAPI not running at {API_BASE} — run docker-compose up first",
)

snowflake_required = pytest.mark.skipif(
    not _snowflake_available(),
    reason="Snowflake unreachable — check .env credentials",
)


# ── Session-scoped HTTP client ───────────────────────────────────────────────────

@pytest.fixture(scope="session")
def api():
    if httpx is None:
        pytest.skip("httpx not installed — run: pip install httpx")
    with httpx.Client(base_url=API_BASE, timeout=90) as client:
        yield client
