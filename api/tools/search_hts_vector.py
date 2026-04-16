"""
Vector search for HTS descriptions in ChromaDB.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import structlog

from services.chromadb_init import search_hts

logger = structlog.get_logger()
router = APIRouter()


class HTSSearchVectorResult(BaseModel):
    hts_code: str
    description: str
    general_rate: str
    chapter: str
    distance: float


@router.post("/search/hts", response_model=List[HTSSearchVectorResult])
async def search_hts_vector(
    query: str,
    chapter: Optional[str] = None,
    limit: int = 5
):
    """
    POST /tools/search/hts

    Vector search for HTS codes by description.
    Useful for finding product classifications by semantic similarity.

    Query params:
    - query: Search query (e.g., "laptop computers" or "electronic devices")
    - chapter: Optional filter by HTS chapter (e.g., "84")
    - limit: Number of results (default 5, max 20)
    """
    logger.info(
        "search_hts_vector_called",
        query=query,
        chapter=chapter,
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
        results = search_hts(
            query=query,
            chapter=chapter,
            limit=limit
        )

        logger.info(
            "search_hts_vector_complete",
            query=query,
            result_count=len(results)
        )

        return [HTSSearchVectorResult(**r) for r in results]

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("search_hts_vector_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Search failed. ChromaDB may not be initialized."
        )
