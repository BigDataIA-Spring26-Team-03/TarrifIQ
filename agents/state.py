"""
LangGraph state definition for TariffIQ six-agent pipeline.
Flow: Query -> Classification -> Rate -> Policy -> Trade -> Synthesis
HITL triggers: classification confidence < 0.80, citation validation failure
"""

from typing import TypedDict, Optional, List, Dict, Any


class TariffState(TypedDict):
    # Input
    query: str

    # Query Agent
    product: Optional[str]
    country: Optional[str]

    # Classification Agent
    hts_code: Optional[str]
    hts_description: Optional[str]
    classification_confidence: Optional[float]

    # Rate Agent
    base_rate: Optional[float]
    adder_rate: Optional[float]
    total_duty: Optional[float]
    rate_record_id: Optional[str]

    # Policy Agent
    policy_chunks: Optional[List[Dict[str, Any]]]
    policy_summary: Optional[str]

    # Trade Agent
    import_value_usd: Optional[float]
    import_quantity: Optional[float]
    trade_period: Optional[str]
    trade_country_code: Optional[str]
    trade_suppressed: Optional[bool]

    # Synthesis Agent
    final_response: Optional[str]
    citations: Optional[List[Dict[str, Any]]]
    pipeline_confidence: Optional[str]  # "HIGH" | "MEDIUM" | "LOW"

    # HITL
    hitl_required: Optional[bool]
    hitl_reason: Optional[str]  # "low_confidence" | "citation_failure"

    # Pipeline
    error: Optional[str]