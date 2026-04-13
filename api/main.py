import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import HealthResponse, QueryRequest, CitedTariffResponse
from api.tools.resolve_hts_rate import router as resolve_hts_rate_router
from api.tools.lookup_product_alias import router as lookup_product_alias_router
from api.tools.log_hitl_record import router as log_hitl_record_router
from api.tools.trade_flow import router as trade_flow_router
from ingestion.chroma_loader import load_federal_register_to_chroma
from agents.graph import run_pipeline

logger = structlog.get_logger()


def rebuild_on_startup() -> None:
    logger.info("chromadb_rebuild_started")
    try:
        total = load_federal_register_to_chroma()
        logger.info("chromadb_rebuild_complete", total_chunks=total)
    except Exception as e:
        logger.error("chromadb_rebuild_failed", error=str(e))


@asynccontextmanager
async def lifespan(app: FastAPI):
    rebuild_on_startup()
    yield


app = FastAPI(
    title="TariffIQ API",
    description="Conversational US Import Tariff Intelligence Platform",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Tool routers ---
app.include_router(resolve_hts_rate_router, prefix="/tools")
app.include_router(lookup_product_alias_router, prefix="/tools")
app.include_router(log_hitl_record_router, prefix="/tools")
app.include_router(trade_flow_router, prefix="/tools")

# --- Stub routers for remaining tool endpoints ---
from fastapi import APIRouter

stubs = APIRouter()

@stubs.get("/tools/search_policy")
async def search_policy_stub():
    return {"status": "ok", "message": "stub — not implemented yet"}

@stubs.get("/tools/classify_query")
async def classify_query_stub():
    return {"status": "ok", "message": "stub — not implemented yet"}

@stubs.post("/tools/synthesize")
async def synthesize_stub():
    return {"status": "ok", "message": "stub — not implemented yet"}

@stubs.get("/tools/fta_rates")
async def fta_rates_stub():
    return {"status": "ok", "message": "stub — not implemented yet"}

app.include_router(stubs)


# --- Health check ---
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc),
    )


# --- Query endpoint ---
@app.post("/query", response_model=dict)
async def query(request: QueryRequest):
    logger.info("query_received", query=request.query)
    try:
        result = run_pipeline(request.query)
        return {
            "status": "ok",
            "query": request.query,
            "product": result.get("product"),
            "country": result.get("country"),
            "hts_code": result.get("hts_code"),
            "hts_description": result.get("hts_description"),
            "classification_confidence": result.get("classification_confidence"),
            "total_duty": result.get("total_duty"),
            "base_rate": result.get("base_rate"),
            "adder_rate": result.get("adder_rate"),
            "policy_summary": result.get("policy_summary"),
            "import_value_usd": result.get("import_value_usd"),
            "trade_period": result.get("trade_period"),
            "trade_suppressed": result.get("trade_suppressed"),
            "final_response": result.get("final_response"),
            "citations": result.get("citations"),
            "hitl_required": result.get("hitl_required"),
            "hitl_reason": result.get("hitl_reason"),
            "error": result.get("error"),
        }
    except Exception as e:
        logger.error("query_pipeline_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))