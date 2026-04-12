"""
LangGraph state definition for TariffIQ six-agent pipeline.

Flow: Query -> Classification -> Rate -> Policy -> Trade -> Synthesis
HITL triggers: classification confidence < 0.80, citation validation failure
"""

from typing import TypedDict, Optional, List, Dict, Any


class TariffState(TypedDict):
    # --- Input ---
    query: str                          # raw user query

    # --- Query Agent output ---
    product: Optional[str]              # extracted product name
    country: Optional[str]              # extracted country name

    # --- Classification Agent output ---
    hts_code: Optional[str]             # resolved HTS code e.g. "8471.30"
    hts_description: Optional[str]      # HTS description from Snowflake
    classification_confidence: Optional[float]  # 0.0 - 1.0

    # --- Rate Agent output ---
    base_rate: Optional[float]          # MFN base rate %
    adder_rate: Optional[float]         # Section 301 / IEEPA adder %
    total_duty: Optional[float]         # base + adder
    rate_record_id: Optional[str]       # Snowflake record ID for citation

    # --- Policy Agent output ---
    policy_chunks: Optional[List[Dict[str, Any]]]   # retrieved FR chunks with doc numbers
    policy_summary: Optional[str]                   # LLM-generated policy context

    # --- Trade Agent output ---
    import_value_usd: Optional[float]   # Census Bureau import value
    import_quantity: Optional[float]    # Census Bureau import quantity
    trade_period: Optional[str]         # e.g. "2024-12"
    trade_country_code: Optional[str]   # Census country code
    trade_suppressed: Optional[bool]    # True if Census returned no data

    # --- Synthesis Agent output ---
    final_response: Optional[str]       # cited natural language answer
    citations: Optional[List[Dict[str, Any]]]  # list of {type, id, text}

    # --- HITL control ---
    hitl_required: Optional[bool]
    hitl_reason: Optional[str]          # "low_confidence" | "citation_failure"

    # --- Pipeline control ---
    error: Optional[str]                # set if any agent fails hard