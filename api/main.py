import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import HealthResponse, QueryRequest, CitedTariffResponse
from api.tools.resolve_hts_rate import router as resolve_hts_rate_router
from api.tools.lookup_product_alias import router as lookup_product_alias_router
from api.tools.log_hitl_record import router as log_hitl_record_router

logger = structlog.get_logger()


def rebuild_on_startup() -> None:
    """
    Rebuild ChromaDB vector index from Snowflake on startup.
    Stub for now — replace with real embedding logic in later milestone.
    """
    logger.info("chromadb_rebuild_started")
    # TODO: pull chunks from FEDERAL_REGISTER_NOTICES, embed, load into ChromaDB
    logger.info("chromadb_rebuild_complete")


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

# --- Stub routers for remaining 5 tool endpoints ---
from fastapi import APIRouter

stubs = APIRouter()

@stubs.get("/tools/search_policy")
async def search_policy_stub():
    return {"status": "ok", "message": "stub — not implemented yet"}

@stubs.get("/tools/trade_flow")
async def trade_flow_stub():
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


# --- Query endpoint (stub) ---
@app.post("/query", response_model=dict)
async def query(request: QueryRequest):
    logger.info("query_received", query=request.query)
    return {
        "status": "received",
        "query": request.query,
        "message": "Agent pipeline not wired yet — stub response",
    }