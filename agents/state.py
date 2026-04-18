"""
TariffIQ — LangGraph state definition

Pipeline (7 steps):
  1. query_agent       → product, country
  2. classification    → hts_code, hts_description, classification_confidence
  3. base_rate         → base_rate, rate_record_id            (pure SQL)
  4. policy            → policy_chunks, policy_summary        (HyDE + hybrid retrieval)
  5. adder_rate        → adder_rate, adder_doc, total_duty    (LLM reads policy chunks)
  6. trade             → import_value_usd, trade_trend_*      (Census API)
  7. synthesis         → final_response, citations            (final LLM call)

HITL:
  - after classification: confidence < 0.80
  - after synthesis:      citation validation failure
"""

from typing import TypedDict, Optional, List, Dict, Any


class TariffState(TypedDict):
    # Input
    query: str

    # Step 1 — Query Agent
    product: Optional[str]
    country: Optional[str]
    clarification_needed: Optional[bool]       # True if product too broad
    clarification_message: Optional[str]       # Human-readable message
    clarification_suggestions: Optional[List[Dict[str, str]]]  # [{label, query}]

    # Step 2 — Classification Agent
    hts_code: Optional[str]
    hts_description: Optional[str]
    classification_confidence: Optional[float]

    # Step 3 — Base Rate Agent  (pure SQL, no LLM, no regex)
    base_rate: Optional[float]         # effective rate (FTA if applicable, else MFN)
    mfn_rate: Optional[float]          # raw MFN general_rate from HTS_CODES
    fta_rate: Optional[float]          # FTA rate if country qualifies, else None
    fta_program: Optional[str]         # e.g. "USMCA", "KORUS FTA", None
    fta_applied: Optional[bool]        # True if base_rate is the FTA rate
    rate_record_id: Optional[str]      # hts_code that resolved (may be shortened)
    hts_footnotes: Optional[List[str]] # footnote strings from HTS_CODES

    # Step 4 — Policy Agent  (HyDE + HybridRetriever)
    policy_chunks: Optional[List[Dict[str, Any]]]
    policy_summary: Optional[str]

    # Step 5 — Adder Rate Agent  (LLM reads policy_chunks)
    adder_rate: Optional[float]       # Section 301 / IEEPA / Section 232 adder
    adder_doc: Optional[str]          # FR document that sourced the adder
    adder_method: Optional[str]       # "llm_policy" | "regex_fallback" | "none"
    total_duty: Optional[float]       # base_rate + adder_rate

    # Step 6 — Trade Agent  (Census Bureau)
    import_value_usd: Optional[float]
    import_quantity: Optional[float]
    trade_period: Optional[str]
    trade_country_code: Optional[str]
    trade_suppressed: Optional[bool]
    trade_trend_pct: Optional[float]  # YoY % change
    trade_trend_label: Optional[str]  # e.g. "▼ 62.0% YoY"

    # Step 7 — Synthesis Agent
    final_response: Optional[str]
    citations: Optional[List[Dict[str, Any]]]
    pipeline_confidence: Optional[str]  # "HIGH" | "MEDIUM" | "LOW"

    # HITL
    hitl_required: Optional[bool]
    hitl_reason: Optional[str]          # "low_confidence" | "citation_failure"

    # Internal bookkeeping
    _product_for_feedback: Optional[str]
    error: Optional[str]