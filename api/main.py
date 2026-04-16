import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import boto3
import redis as redis_lib
import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from chromadb import HttpClient as ChromaHttpClient

from api.schemas import HealthResponse, ServiceHealth
from api.tools.resolve_hts_rate import router as resolve_hts_rate_router
from api.tools.hts_search import router as hts_search_router
from api.tools.hts_chapter import router as hts_chapter_router
from api.tools.search_policy_vector import router as search_policy_router
from api.tools.search_hts_vector import router as search_hts_vector_router
from services.chromadb_init import initialize_chromadb

logger = structlog.get_logger()

tags_metadata = [
    {"name": "Core", "description": "Health check endpoint"},
    {"name": "HTS & Rates", "description": "HTS code lookup and search"},
    {"name": "Vector Search", "description": "ChromaDB semantic search for policies and HTS"},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize ChromaDB on startup
    try:
        logger.info("Initializing ChromaDB...")
        initialize_chromadb()
        logger.info("ChromaDB initialized successfully")
    except Exception as e:
        logger.error("ChromaDB initialization failed", error=str(e))
    yield


app = FastAPI(
    title="TariffIQ API",
    description="Conversational US Import Tariff Intelligence Platform",
    version="0.1.0",
    lifespan=lifespan,
    openapi_tags=tags_metadata,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Tool routers ---
app.include_router(resolve_hts_rate_router, prefix="/tools", tags=["HTS & Rates"])
app.include_router(hts_search_router, prefix="/tools", tags=["HTS & Rates"])
app.include_router(hts_chapter_router, prefix="/tools", tags=["HTS & Rates"])
app.include_router(search_policy_router, prefix="/tools", tags=["Vector Search"])
app.include_router(search_hts_vector_router, prefix="/tools", tags=["Vector Search"])


# --- Health check ---
@app.get("/health", response_model=HealthResponse, tags=["Core"])
async def health():
    services: dict = {}

    # --- Snowflake ---
    t0 = time.monotonic()
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
        services["snowflake"] = ServiceHealth(
            status="ok", latency_ms=round((time.monotonic() - t0) * 1000, 1)
        )
    except Exception as e:
        services["snowflake"] = ServiceHealth(status="error", error=str(e))

    # --- Redis ---
    t0 = time.monotonic()
    try:
        r = redis_lib.Redis(
            host=os.environ.get("REDIS_HOST", "redis"),
            port=int(os.environ.get("REDIS_PORT", 6379)),
        )
        r.ping()
        services["redis"] = ServiceHealth(
            status="ok", latency_ms=round((time.monotonic() - t0) * 1000, 1)
        )
    except Exception as e:
        services["redis"] = ServiceHealth(status="error", error=str(e))

    # --- S3 ---
    t0 = time.monotonic()
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
        s3.head_bucket(Bucket=os.environ.get("S3_BUCKET", "tariffiq-raw"))
        services["s3"] = ServiceHealth(
            status="ok", latency_ms=round((time.monotonic() - t0) * 1000, 1)
        )
    except Exception as e:
        services["s3"] = ServiceHealth(status="error", error=str(e))

    # --- ChromaDB ---
    t0 = time.monotonic()
    try:
        chroma = ChromaHttpClient(
            host=os.environ.get("CHROMA_HOST", "chromadb"),
            port=int(os.environ.get("CHROMA_PORT", 8000)),
        )
        chroma.list_collections()
        services["chromadb"] = ServiceHealth(
            status="ok", latency_ms=round((time.monotonic() - t0) * 1000, 1)
        )
    except Exception as e:
        services["chromadb"] = ServiceHealth(status="error", error=str(e))

    overall = "ok" if all(s.status == "ok" for s in services.values()) else "degraded"
    return HealthResponse(
        status=overall,
        timestamp=datetime.now(timezone.utc),
        services=services,
    )