from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from typing import Literal
from datetime import datetime


class TariffCalculation(BaseModel):
    component: str
    rate: float
    source_description: str
    record_id: str
    fetched_from: str
    fetched_at: datetime


class RateReconciliation(BaseModel):
    calculation: str
    check_passed: bool


class VerificationReceipt(BaseModel):
    hts_code: str
    base_rate: float
    base_rate_source: TariffCalculation
    adder_rate: float
    adder_source: TariffCalculation
    total_duty: float
    rate_reconciliation: RateReconciliation


class CitedTariffResponse(BaseModel):
    product: str
    hts_code: str
    hts_source: str
    tariff_calculation: List[TariffCalculation]
    total_effective_duty: float
    rate_reconciliation: RateReconciliation
    policy_summary: str
    policy_citations: List[str]
    trade_flow_summary: str
    trade_flow_source: str
    answer_prose: str


class TradeFlowResult(BaseModel):
    hs6_code: str
    country: str
    period_months: int
    pct_change_yoy: float
    trend: str
    source_stamp: str
    fetched_at: datetime


class HITLRecord(BaseModel):
    hitl_id: str
    query_text: str
    trigger_reason: str
    classifier_hts: Optional[str] = None
    classifier_conf: Optional[float] = None
    human_decision: Optional[str] = None
    created_at: datetime


class NormalizedQuery(BaseModel):
    product: str
    country: Optional[str] = None
    query_type: Literal[
        "rate_lookup",
        "policy_context",
        "sourcing_comparison",
        "trade_flow"
    ]


class QueryRequest(BaseModel):
    query: str


class ServiceHealth(BaseModel):
    status: str
    latency_ms: Optional[float] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    services: Dict[str, ServiceHealth]