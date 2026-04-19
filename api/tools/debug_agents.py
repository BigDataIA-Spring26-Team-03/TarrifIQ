"""
Debug Endpoints for Per-Agent Testing

Provides `/debug/agents/{agent}` endpoints to test individual agents in isolation.
Each endpoint accepts only the minimum state fields the agent reads.

Use for isolating which agent in the 7-step pipeline is failing.
"""

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Any, Dict

from agents.query_agent import run_query_agent
from agents.classification_agent import run_classification_agent
from agents.base_rate_agent import run_base_rate_agent
from agents.policy_agent import run_policy_agent
from agents.adder_rate_agent import run_adder_rate_agent
from agents.trade_agent import run_trade_agent
from agents.synthesis_agent import run_synthesis_agent

logger = structlog.get_logger()
router = APIRouter()


# ── Request Models ─────────────────────────────────────────────────────────

class QueryAgentRequest(BaseModel):
    """Query agent: parse raw query → {product, country}"""
    query: str


class ClassifyRequest(BaseModel):
    """Classification agent: product → HTS code"""
    product: str


class BaseRateRequest(BaseModel):
    """Base rate agent: HTS code + country → MFN/FTA rates"""
    hts_code: str
    country: Optional[str] = None


class PolicyRequest(BaseModel):
    """Policy agent: query + product + country + HTS → FR policy context"""
    query: str
    product: str
    country: str
    hts_code: str


class AdderRateRequest(BaseModel):
    """Adder rate agent: HTS + country + base_rate + policy → Section 301 rate"""
    hts_code: str
    country: str
    base_rate: float
    policy_chunks: Optional[List[Dict[str, Any]]] = None
    hts_footnotes: Optional[List[str]] = None


class TradeRequest(BaseModel):
    """Trade agent: HTS + country → Census Bureau import data"""
    hts_code: str
    country: str


class SynthesisRequest(BaseModel):
    """Synthesis agent: all intermediate results → final answer"""
    query: str
    product: Optional[str] = None
    country: Optional[str] = None
    hts_code: Optional[str] = None
    hts_description: Optional[str] = None
    classification_confidence: Optional[float] = None
    base_rate: Optional[float] = None
    mfn_rate: Optional[float] = None
    adder_rate: Optional[float] = None
    total_duty: Optional[float] = None
    fta_applied: Optional[bool] = None
    fta_program: Optional[str] = None
    adder_doc: Optional[str] = None
    rate_record_id: Optional[str] = None
    policy_summary: Optional[str] = None
    policy_chunks: Optional[List[Dict[str, Any]]] = None
    hts_footnotes: Optional[List[str]] = None
    import_value_usd: Optional[float] = None
    trade_period: Optional[str] = None
    trade_trend_label: Optional[str] = None
    trade_suppressed: Optional[bool] = None
    hitl_required: Optional[bool] = None
    hitl_reason: Optional[str] = None


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post("/query")
async def debug_query_agent(req: QueryAgentRequest):
    """
    Step 1: Query Agent — Parse raw user query.

    Input: {query: str}
    Output: {product: str, country: Optional[str], error: Optional[str]}
    """
    try:
        logger.info("debug_query_agent", query=req.query[:60])
        state = {"query": req.query}
        result = run_query_agent(state)
        return {"agent": "query", "input": req.model_dump(), "output": result}
    except Exception as e:
        logger.error("debug_query_agent_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/classify")
async def debug_classify_agent(req: ClassifyRequest):
    """
    Step 2: Classification Agent — Product → HTS code.

    Input: {product: str}
    Output: {hts_code: str, hts_description: str, classification_confidence: float, ...}
    """
    try:
        logger.info("debug_classify_agent", product=req.product)
        state = {"product": req.product}
        result = run_classification_agent(state)
        return {"agent": "classify", "input": req.model_dump(), "output": result}
    except Exception as e:
        logger.error("debug_classify_agent_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/base-rate")
async def debug_base_rate_agent(req: BaseRateRequest):
    """
    Step 3: Base Rate Agent — HTS code + country → MFN/FTA rates.

    Input: {hts_code: str, country: Optional[str]}
    Output: {base_rate: float, mfn_rate: float, fta_rate: Optional[float], fta_program: Optional[str], ...}
    """
    try:
        logger.info("debug_base_rate_agent", hts_code=req.hts_code, country=req.country)
        state = {
            "hts_code": req.hts_code,
            "country": req.country or "",
        }
        result = run_base_rate_agent(state)
        return {"agent": "base_rate", "input": req.model_dump(), "output": result}
    except Exception as e:
        logger.error("debug_base_rate_agent_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/policy")
async def debug_policy_agent(req: PolicyRequest):
    """
    Step 4: Policy Agent — Query + product + country + HTS → Federal Register context.

    Input: {query: str, product: str, country: str, hts_code: str}
    Output: {policy_chunks: List[Dict], policy_summary: str}
    """
    try:
        logger.info("debug_policy_agent", hts_code=req.hts_code, product=req.product)
        state = {
            "query": req.query,
            "product": req.product,
            "country": req.country,
            "hts_code": req.hts_code,
        }
        result = run_policy_agent(state)
        return {"agent": "policy", "input": req.model_dump(), "output": result}
    except Exception as e:
        logger.error("debug_policy_agent_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/adder-rate")
async def debug_adder_rate_agent(req: AdderRateRequest):
    """
    Step 5: Adder Rate Agent — HTS + country + base_rate + policy → Section 301 duty.

    Input: {hts_code: str, country: str, base_rate: float, policy_chunks: Optional[List], hts_footnotes: Optional[List]}
    Output: {adder_rate: float, adder_doc: Optional[str], adder_method: str, total_duty: float}
    """
    try:
        logger.info("debug_adder_rate_agent", hts_code=req.hts_code, country=req.country)
        state = {
            "hts_code": req.hts_code,
            "country": req.country,
            "base_rate": req.base_rate,
            "policy_chunks": req.policy_chunks or [],
            "hts_footnotes": req.hts_footnotes or [],
        }
        result = run_adder_rate_agent(state)
        return {"agent": "adder_rate", "input": req.model_dump(), "output": result}
    except Exception as e:
        logger.error("debug_adder_rate_agent_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trade")
async def debug_trade_agent(req: TradeRequest):
    """
    Step 6: Trade Agent — HTS + country → Census Bureau import data.

    Input: {hts_code: str, country: str}
    Output: {import_value_usd: Optional[float], trade_period: Optional[str], trade_trend_label: Optional[str], ...}
    """
    try:
        logger.info("debug_trade_agent", hts_code=req.hts_code, country=req.country)
        state = {
            "hts_code": req.hts_code,
            "country": req.country,
        }
        result = run_trade_agent(state)
        return {"agent": "trade", "input": req.model_dump(), "output": result}
    except Exception as e:
        logger.error("debug_trade_agent_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/synthesis")
async def debug_synthesis_agent(req: SynthesisRequest):
    """
    Step 7: Synthesis Agent — All intermediate results → final citation-grounded answer.

    Input: {query, product, country, hts_code, classification_confidence, base_rate,
            policy_summary, policy_chunks, adder_rate, total_duty, import_value_usd, ...}
    Output: {final_response: str, citations: List[Dict], pipeline_confidence: str, ...}
    """
    try:
        logger.info("debug_synthesis_agent", hts_code=req.hts_code, product=req.product)
        # Build state with all fields from request
        state = req.model_dump(exclude_none=False)
        result = run_synthesis_agent(state)
        return {"agent": "synthesis", "input": req.model_dump(), "output": result}
    except Exception as e:
        logger.error("debug_synthesis_agent_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
