"""
Vector search for policy notices in ChromaDB.
Uses HybridRetriever for dense + sparse + RRF fusion search.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import structlog

from services.retrieval.hybrid import HybridRetriever

logger = structlog.get_logger()
router = APIRouter()


class PolicySearchResult(BaseModel):
    chunk_id: str
    document_number: str
    chunk_text: str
    source: str  # "USTR", "CBP", or "USITC"
    hts_chapter: str
    hts_code: str
    score: float
    retrieval_method: str


@router.post("/search/policy", response_model=List[PolicySearchResult])
async def search_policy_notices(
    query: str,
    hts_chapter: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 5
):
    """
    POST /tools/search/policy

    Vector search for policy notices across USTR, CBP, and USITC documents.

    Query params:
    - query: Search query (e.g., "tariff on semiconductors")
    - hts_chapter: Optional filter by HTS chapter (e.g., "85")
    - source: Optional filter by source ("USTR", "CBP", "USITC")
    - limit: Number of results (default 5, max 20)
    """
    logger.info(
        "search_policy_notices_called",
        query=query,
        hts_chapter=hts_chapter,
        source=source,
        limit=limit
    )

    if not query or len(query.strip()) < 3:
        raise HTTPException(
            status_code=400,
            detail="Query must be at least 3 characters"
        )

    if limit < 1 or limit > 20:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 20"
        )

    try:
        retriever = HybridRetriever()
        results = retriever.search_policy(
            query=query,
            hts_chapter=hts_chapter,
            source=source,
            top_k=limit
        )

        logger.info(
            "search_policy_notices_complete",
            query=query,
            result_count=len(results),
            retrieval_method="hybrid_rrf"
        )

        return [PolicySearchResult(**r) for r in results]

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("search_policy_notices_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Search failed. ChromaDB may not be initialized."
        )
