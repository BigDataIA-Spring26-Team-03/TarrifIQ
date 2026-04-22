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
    query_intent: Optional[str]   # "rate_change" | "country_compare" | "exemption_check" | None
    query_intent_note: Optional[str]      # Context note for synthesis based on intent
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

    # Step 4 — Chapter 99 Lookup
    chapter99_adder: Optional[float]  # Rate from HTS Chapter 99 codes (highest priority)
    chapter99_doc: Optional[str]      # HTS Chapter 99 code e.g. "9903.88.03"

    # Step 5 — Notice HTS Codes Lookup
    notice_adder: Optional[float]     # Rate from NOTICE_HTS_CODES LLM extraction
    notice_doc: Optional[str]         # FR document number from notice snippets
    notice_basis: Optional[str]       # One-sentence explanation from LLM

    # Step 7 — Rate Stacking (final adder computation)
    adder_rate: Optional[float]       # Final Section 301 / IEEPA / Section 232 adder (after stacking priority)
    adder_specific_duty: Optional[str]  # e.g. "66.6¢/kg" for specific duty codes
    adder_doc: Optional[str]          # FR document that sourced the final adder
    adder_basis: Optional[str]        # "chapter99" | "notice_llm" | "regex_fallback" | "fta_exempt" | "none"
    adder_method: Optional[str]       # Deprecated, kept for backwards compat: maps to adder_basis
    section122_adder: Optional[float] # Section 122 universal surcharge (stacked on top)
    section122_doc: Optional[str]     # FR document for Section 122 surcharge
    total_duty: Optional[float]       # base_rate + adder_rate (or just base_rate if FTA exempt)

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

    # Step 7 extras
    country_comparison: Optional[List[Dict[str, Any]]]
    top_importers: Optional[List[Dict[str, Any]]]
    rate_change_history: Optional[List[Dict[str, Any]]]

    # HITL
    hitl_required: Optional[bool]
    hitl_reason: Optional[str]          # "low_confidence" | "citation_failure"

    # Internal bookkeeping
    _product_for_feedback: Optional[str]
    error: Optional[str]