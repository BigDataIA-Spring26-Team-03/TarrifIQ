"""
TariffIQ FastAPI — main.py

Ayush's tool endpoints:
  GET  /tools/rate              — HTS rate lookup (resolve_hts_rate)
  GET  /tools/hts/search        — HTS keyword search
  GET  /tools/hts/chapter       — HTS chapter lookup
  POST /tools/search/policy     — ChromaDB policy vector search (policy_notices)
  POST /tools/search/hts        — ChromaDB HTS vector search (hts_descriptions)

Vaishnavi's pipeline endpoint:
  POST /query                   — Full LangGraph 6-agent pipeline

Startup: initialize_chromadb() with skip-if-populated (no delete-rebuild).
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import boto3
import redis as redis_lib
import structlog
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from chromadb import HttpClient as ChromaHttpClient

from api.schemas import HealthResponse, ServiceHealth, QueryRequest
from api.tools.resolve_hts_rate import router as resolve_hts_rate_router
from api.tools.hts_search import router as hts_search_router
from api.tools.hts_chapter import router as hts_chapter_router
from api.tools.search_policy_vector import router as search_policy_router
from api.tools.search_hts_vector import router as search_hts_vector_router
from api.tools.debug_agents import router as debug_agents_router
from services.chromadb_init import initialize_chromadb
from agents.graph import run_pipeline

# Configure structlog so keyword-arg logging works consistently
# across router.py, main.py, and any other structlog callers.
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()

tags_metadata = [
    {"name": "Core", "description": "Health check and query pipeline"},
    {"name": "HTS & Rates", "description": "HTS code lookup and search"},
    {"name": "Vector Search", "description": "ChromaDB semantic search for policies and HTS"},
    {"name": "Debug", "description": "Per-agent debug endpoints — dev only"},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("chromadb_init_started")
        initialize_chromadb()
        logger.info("chromadb_init_complete")
    except Exception as e:
        logger.error("chromadb_init_failed", error=str(e))
    yield


app = FastAPI(
    title="TariffIQ API",
    description="Conversational US Import Tariff Intelligence Platform",
    version="0.1.0",
    lifespan=lifespan,
    openapi_tags=tags_metadata,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(resolve_hts_rate_router, prefix="/tools", tags=["HTS & Rates"])
app.include_router(hts_search_router, prefix="/tools", tags=["HTS & Rates"])
app.include_router(hts_chapter_router, prefix="/tools", tags=["HTS & Rates"])
app.include_router(search_policy_router, prefix="/tools", tags=["Vector Search"])
app.include_router(search_hts_vector_router, prefix="/tools", tags=["Vector Search"])
app.include_router(debug_agents_router, prefix="/debug/agents", tags=["Debug"])


@app.get("/health", response_model=HealthResponse, tags=["Core"])
async def health():
    services: dict = {}

    t0 = time.monotonic()
    try:
        import snowflake.connector
        conn = snowflake.connector.connect(
            user=os.environ["SNOWFLAKE_USER"], password=os.environ["SNOWFLAKE_PASSWORD"],
            account=os.environ["SNOWFLAKE_ACCOUNT"], warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
            database=os.environ["SNOWFLAKE_DATABASE"], schema=os.environ["SNOWFLAKE_SCHEMA"],
        )
        conn.cursor().execute("SELECT 1")
        conn.close()
        services["snowflake"] = ServiceHealth(status="ok", latency_ms=round((time.monotonic() - t0) * 1000, 1))
    except Exception as e:
        services["snowflake"] = ServiceHealth(status="error", error=str(e))

    t0 = time.monotonic()
    try:
        r = redis_lib.Redis(host=os.environ.get("REDIS_HOST", "redis"), port=int(os.environ.get("REDIS_PORT", 6379)))
        r.ping()
        services["redis"] = ServiceHealth(status="ok", latency_ms=round((time.monotonic() - t0) * 1000, 1))
    except Exception as e:
        services["redis"] = ServiceHealth(status="error", error=str(e))

    t0 = time.monotonic()
    try:
        s3 = boto3.client("s3",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
        s3.head_bucket(Bucket=os.environ.get("S3_BUCKET", "tariffiq-raw"))
        services["s3"] = ServiceHealth(status="ok", latency_ms=round((time.monotonic() - t0) * 1000, 1))
    except Exception as e:
        services["s3"] = ServiceHealth(status="error", error=str(e))

    t0 = time.monotonic()
    try:
        chroma = ChromaHttpClient(
            host=os.environ.get("CHROMA_HOST", "chromadb"),
            port=int(os.environ.get("CHROMA_PORT", 8000)),
        )
        chroma.list_collections()
        services["chromadb"] = ServiceHealth(status="ok", latency_ms=round((time.monotonic() - t0) * 1000, 1))
    except Exception as e:
        services["chromadb"] = ServiceHealth(status="error", error=str(e))

    overall = "ok" if all(s.status == "ok" for s in services.values()) else "degraded"
    return HealthResponse(status=overall, timestamp=datetime.now(timezone.utc), services=services)


def _sanitize_chunks(chunks) -> list:
    if not chunks:
        return []
    out = []
    for c in chunks:
        safe = {}
        for k, v in c.items():
            if k == "chunk_text":
                safe[k] = str(v)[:500] if v else ""
            elif hasattr(v, "item"):
                safe[k] = float(v)
            elif v is None or isinstance(v, (str, int, bool, float)):
                safe[k] = v
            else:
                safe[k] = str(v)
        out.append(safe)
    return out


@app.post("/query", response_model=dict, tags=["Core"])
async def query(request: QueryRequest):
    """Full LangGraph pipeline: query → classify → base_rate → policy → adder_rate → trade → synthesize."""
    logger.info("query_received", query=request.query)
    try:
        result = run_pipeline(request.query)

        # Short-circuit: query agent detected ambiguity
        if result.get("clarification_needed"):
            return {
                "status": "clarification_needed",
                "query": request.query,
                "product": result.get("product"),
                "country": result.get("country"),
                "message": result.get("clarification_message"),
                "suggestions": result.get("clarification_suggestions", []),
            }

        return {
            "status": "ok",
            "query": request.query,
            "product": result.get("product"),
            "country": result.get("country"),
            "hts_code": result.get("hts_code"),
            "hts_description": result.get("hts_description"),
            "classification_confidence": result.get("classification_confidence"),
            "base_rate": result.get("base_rate"),
            "mfn_rate": result.get("mfn_rate"),
            "fta_rate": result.get("fta_rate"),
            "fta_program": result.get("fta_program"),
            "fta_applied": result.get("fta_applied"),
            "hts_footnotes": result.get("hts_footnotes"),
            "adder_rate": result.get("adder_rate"),
            "adder_doc": result.get("adder_doc"),
            "adder_method": result.get("adder_method"),
            "total_duty": result.get("total_duty"),
            "policy_summary": result.get("policy_summary"),
            "policy_chunks": _sanitize_chunks(result.get("policy_chunks")),
            "policy_chunks_count": len(result.get("policy_chunks") or []),
            "pipeline_confidence": result.get("pipeline_confidence"),
            "import_value_usd": result.get("import_value_usd"),
            "trade_period": result.get("trade_period"),
            "trade_suppressed": result.get("trade_suppressed"),
            "trade_trend_pct": result.get("trade_trend_pct"),
            "trade_trend_label": result.get("trade_trend_label"),
            "final_response": result.get("final_response"),
            "citations": result.get("citations"),
            "country_comparison": result.get("country_comparison"),
            "top_importers": result.get("top_importers"),
            "rate_change_history": result.get("rate_change_history"),
            "query_intent": result.get("query_intent"),
            "hitl_required": result.get("hitl_required"),
            "hitl_reason": result.get("hitl_reason"),
            "error": result.get("error"),
        }
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error("query_pipeline_failed", error=str(e), traceback=tb)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/hitl/feedback", response_model=dict, tags=["Core"])
async def hitl_feedback(hitl_id: str, correct_hts: str, notes: str = ""):
    """
    Submit human decision for a HITL escalation.
    Writes correct HTS back to PRODUCT_ALIASES for self-improvement.
    """
    from agents import tools as agent_tools
    success = agent_tools.hitl_feedback_write(hitl_id, correct_hts, notes)
    if success:
        return {"status": "ok", "hitl_id": hitl_id, "correct_hts": correct_hts}
    raise HTTPException(status_code=500, detail="Failed to write HITL feedback")