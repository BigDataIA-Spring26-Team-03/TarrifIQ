# Run: streamlit run streamlit/app.py --server.port 8501

from __future__ import annotations

import html
import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.graph import run_pipeline, run_comparison_pipeline, run_pipeline_auto

st.set_page_config(
    layout="wide",
    page_title="TariffIQ — US Import Intelligence",
    page_icon="⚖️",
    initial_sidebar_state="expanded",
)

_PENDING_PIPELINE_QUERY = "pending_pipeline_query"


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

            /* ── Base ── */
            html, body, .stApp {
                background: #0a0e27 !important;
                color: #e0e7ff;
                font-family: 'Poppins', sans-serif;
            }

            /* ── Sidebar ── */
            [data-testid="stSidebar"] {
                background: #0f1535 !important;
                border-right: 1px solid #1e3a5f !important;
            }
            [data-testid="stSidebar"] * { color: #94a3b8 !important; }
            [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3 {
                color: #e0e7ff !important;
                font-weight: 600 !important;
            }

            /* ── Header ── */
            .tiq-header {
                padding: 2rem 0 1.5rem;
                background: transparent;
                border-bottom: 1px solid #1e3a5f;
                margin-bottom: 2rem;
            }
            .tiq-title {
                font-family: 'Space Mono', monospace;
                font-size: 2.2rem;
                font-weight: 700;
                background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                letter-spacing: -0.02em;
            }
            .tiq-title span { color: #60a5fa; }
            .tiq-subtitle {
                font-size: 0.75rem;
                color: #64748b;
                font-family: 'Poppins', sans-serif;
                letter-spacing: 0.1em;
                text-transform: uppercase;
                margin-top: 0.5rem;
                font-weight: 600;
            }

            /* ── Cards ── */
            .tiq-card {
                background: #0f1535;
                border: 1px solid #1e3a5f;
                border-radius: 10px;
                padding: 18px 22px;
                margin-bottom: 14px;
                transition: all 0.2s ease;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
            }
            .tiq-card:hover {
                border-color: #3b82f6;
                box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
            }

            .tiq-card-accent {
                background: #1a2553;
                border: 1px solid #3b82f6;
                border-radius: 10px;
                padding: 18px 22px;
                margin-bottom: 14px;
                box-shadow: 0 2px 8px rgba(59, 130, 246, 0.1);
            }

            /* ── HTS Code block ── */
            .tiq-hts {
                background: #0f1535;
                border: 1px solid #1e3a5f;
                border-radius: 10px;
                padding: 16px 20px;
                font-family: 'Space Mono', monospace;
                font-size: 0.9rem;
                color: #60a5fa;
                line-height: 1.6;
                font-weight: 600;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
                letter-spacing: 0.5px;
            }

            /* ── Duty gauge ── */
            .duty-gauge-wrap {
                display: flex;
                align-items: center;
                gap: 20px;
                padding: 24px 28px;
                background: #0f1535;
                border: 1px solid #1e3a5f;
                border-radius: 12px;
                margin-bottom: 18px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
            }
            .duty-number {
                font-family: 'Space Mono', monospace;
                font-size: 2.8rem;
                font-weight: 700;
                line-height: 1;
                min-width: 110px;
                letter-spacing: -0.02em;
            }
            .duty-number.high { color: #dc2626; }
            .duty-number.medium { color: #f59e0b; }
            .duty-number.low { color: #059669; }
            .duty-number.zero { color: #0284c7; }
            .duty-label {
                font-size: 0.7rem;
                color: #64748b;
                font-family: 'Poppins', sans-serif;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                margin-top: 6px;
                font-weight: 600;
            }
            .duty-bar-track {
                flex: 1;
                height: 8px;
                background: #1a2553;
                border-radius: 4px;
                overflow: hidden;
                box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.3);
            }
            .duty-bar-fill {
                height: 100%;
                border-radius: 4px;
                transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
            }

            /* ── Rate breakdown chips ── */
            .rate-chip {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                background: #1a2553;
                border: 1px solid #3b82f6;
                border-radius: 20px;
                padding: 6px 14px;
                font-family: 'Space Mono', monospace;
                font-size: 0.8rem;
                color: #60a5fa;
                margin: 4px;
                transition: all 0.2s ease;
                font-weight: 500;
            }
            .rate-chip:hover {
                border-color: #60a5fa;
                background: #0f1535;
                box-shadow: 0 2px 8px rgba(59, 130, 246, 0.15);
            }
            .rate-chip .dot {
                width: 6px; height: 6px;
                border-radius: 50%;
                display: inline-block;
            }

            /* ── Citation cards ── */
            .cit-card {
                background: #0f1535;
                border: 1px solid #1e3a5f;
                border-radius: 8px;
                padding: 14px 16px;
                margin-bottom: 10px;
                display: flex;
                gap: 12px;
                align-items: flex-start;
                transition: all 0.2s ease;
            }
            .cit-card:hover {
                border-color: #3b82f6;
                background: #1a2553;
                box-shadow: 0 2px 8px rgba(59, 130, 246, 0.1);
            }
            .cit-badge {
                font-family: 'Space Mono', monospace;
                font-size: 0.65rem;
                font-weight: 600;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                padding: 4px 10px;
                border-radius: 6px;
                white-space: nowrap;
                flex-shrink: 0;
                margin-top: 2px;
            }
            .cit-badge.ustr { background: #1a2553; color: #60a5fa; }
            .cit-badge.cbp  { background: #1a3a2a; color: #10b981; }
            .cit-badge.hts  { background: #2d1f3a; color: #a78bfa; }
            .cit-badge.eop  { background: #3a1f3a; color: #d946ef; }
            .cit-badge.census { background: #1a2332; color: #94a3b8; }
            .cit-badge.default { background: #1a2332; color: #94a3b8; }
            .cit-body { flex: 1; min-width: 0; }
            .cit-title {
                font-size: 0.82rem;
                color: #e0e7ff;
                line-height: 1.4;
                margin-bottom: 4px;
                overflow: hidden;
                display: -webkit-box;
                -webkit-line-clamp: 2;
                -webkit-box-orient: vertical;
            }
            .cit-meta {
                font-family: 'Space Mono', monospace;
                font-size: 0.7rem;
                color: #64748b;
            }
            .cit-link {
                font-family: 'Space Mono', monospace;
                font-size: 0.72rem;
                color: #60a5fa;
                text-decoration: none;
                display: inline-flex;
                align-items: center;
                gap: 4px;
                margin-top: 6px;
                opacity: 0.8;
            }
            .cit-link:hover { opacity: 1; }

            /* ── Metric row ── */
            .metric-row {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 10px;
                margin-bottom: 12px;
            }
            .metric-box {
                background: #0f1535;
                border: 1px solid #1e3a5f;
                border-radius: 10px;
                padding: 16px 18px;
                text-align: center;
                transition: all 0.2s ease;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
            }
            .metric-box:hover {
                border-color: #3b82f6;
                box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1);
            }
            .metric-val {
                font-family: 'Space Mono', monospace;
                font-size: 1.3rem;
                font-weight: 700;
                color: #60a5fa;
                letter-spacing: -0.01em;
            }
            .metric-lbl {
                font-size: 0.68rem;
                color: #64748b;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                margin-top: 6px;
                font-family: 'Poppins', sans-serif;
                font-weight: 600;
            }

            /* ── Intent tag ── */
            .intent-tag {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                background: #1a2553;
                border: 1px solid #3b82f6;
                border-radius: 8px;
                padding: 6px 12px;
                font-family: 'Space Mono', monospace;
                font-size: 0.75rem;
                color: #60a5fa;
                margin-bottom: 12px;
                font-weight: 600;
                letter-spacing: 0.05em;
                box-shadow: 0 1px 3px rgba(59, 130, 246, 0.1);
            }

            /* ── FTA badge ── */
            .fta-badge {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                background: #1a3a2a;
                border: 1px solid #10b981;
                border-radius: 8px;
                padding: 8px 14px;
                font-family: 'Space Mono', monospace;
                font-size: 0.8rem;
                color: #10b981;
                margin-bottom: 12px;
                font-weight: 600;
                box-shadow: 0 1px 3px rgba(16, 185, 129, 0.1);
            }

            /* ── Chat messages ── */
            [data-testid="stChatMessage"] {
                background: transparent !important;
                border: none !important;
                padding: 14px 0 !important;
            }
            .stChatMessage [data-testid="stMarkdownContainer"] p {
                color: #e0e7ff;
                line-height: 1.6;
                font-size: 0.95rem;
            }

            /* ── Chat input ── */
            [data-testid="stChatInput"] {
                background: transparent !important;
                border-top: none !important;
                padding: 16px 0 0 0 !important;
                margin-top: 16px !important;
            }
            [data-testid="stChatInputTextArea"] {
                background: #0f1535 !important;
                border: 1px solid #1e3a5f !important;
                border-radius: 10px !important;
                color: #e0e7ff !important;
                font-family: 'Poppins', sans-serif !important;
                font-size: 0.95rem !important;
                padding: 12px 16px !important;
                transition: all 0.2s ease !important;
            }
            [data-testid="stChatInputTextArea"]:focus {
                border-color: #3b82f6 !important;
                box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1), 0 1px 3px rgba(0, 0, 0, 0.3) !important;
            }

            /* ── Expanders ── */
            [data-testid="stExpander"] {
                background: #0f1535 !important;
                border: 1px solid #1e3a5f !important;
                border-radius: 10px !important;
                transition: all 0.2s ease !important;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3) !important;
            }
            [data-testid="stExpander"]:hover {
                border-color: #3b82f6 !important;
                box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1) !important;
            }
            [data-testid="stExpander"] summary {
                color: #e0e7ff !important;
                font-family: 'Poppins', sans-serif !important;
                font-size: 0.85rem !important;
                font-weight: 600 !important;
                padding: 12px 16px !important;
            }

            /* ── Dataframe ── */
            .stDataFrame { border-radius: 8px; overflow: hidden; }
            [data-testid="stDataFrameResizable"] {
                border: 1px solid #1a2332 !important;
                border-radius: 8px !important;
            }

            /* ── Buttons ── */
            .stButton button {
                background: #0f1535 !important;
                border: 1px solid #3b82f6 !important;
                color: #60a5fa !important;
                border-radius: 8px !important;
                font-family: 'Poppins', sans-serif !important;
                font-size: 0.9rem !important;
                font-weight: 600 !important;
                transition: all 0.2s ease !important;
                padding: 10px 18px !important;
                box-shadow: 0 1px 3px rgba(59, 130, 246, 0.1) !important;
            }
            .stButton button:hover {
                border-color: #60a5fa !important;
                background: #1a2553 !important;
                box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2) !important;
            }

            /* ── Confidence badge ── */
            .conf-high { color: #059669; }
            .conf-medium { color: #d97706; }
            .conf-low { color: #dc2626; }

            /* ── Section headers ── */
            .section-header {
                font-family: 'Space Mono', monospace;
                font-size: 0.7rem;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                color: #64748b;
                margin: 18px 0 12px;
                display: flex;
                align-items: center;
                gap: 10px;
                font-weight: 700;
            }
            .section-header::after {
                content: '';
                flex: 1;
                height: 1px;
                background: #1e3a5f;
            }

            /* ── Scrollbar ── */
            ::-webkit-scrollbar { width: 4px; height: 4px; }
            ::-webkit-scrollbar-track { background: #060a10; }
            ::-webkit-scrollbar-thumb { background: #1a2332; border-radius: 2px; }

            /* ── Clarification chips ── */
            .clarify-chip button {
                background: #0f1e2e !important;
                border: 1px solid #1e3a5f !important;
                color: #60a5fa !important;
                font-size: 0.78rem !important;
                font-family: 'IBM Plex Mono', monospace !important;
            }

            /* ── Warning / Info ── */
            [data-testid="stAlert"] {
                border-radius: 8px !important;
                border: 1px solid #1a2332 !important;
                background: #0d1520 !important;
            }

            /* hide streamlit branding */
            #MainMenu, footer, header { visibility: hidden !important; display: none !important; }
            .stDeployButton { display: none !important; }

            /* Hide all chat avatars */
            [data-testid="stChatMessageAvatarUser"],
            [data-testid="stChatMessageAvatarAssistant"],
            [data-testid*="ChatMessageAvatar"],
            .stChatMessage img,
            div[data-testid="stChatMessage"] img {
                display: none !important;
            }

            /* ── Kill ALL white backgrounds ── */
            .stApp > div, .main, .block-container,
            [data-testid="stAppViewContainer"],
            [data-testid="stVerticalBlock"],
            [data-testid="stHorizontalBlock"],
            .element-container, .stMarkdown,
            div[data-testid="stChatMessageContent"],
            div[data-testid="stChatMessage"],
            [data-baseweb="tab-panel"],
            .stTabs [data-baseweb="tab-list"] {
                background: transparent !important;
                background-color: transparent !important;
            }

            /* ── Dataframe dark theme ── */
            .stDataFrame, .stDataFrame > div,
            [data-testid="stDataFrameResizable"],
            .dvn-scroller, .dvn-scroller > div,
            .glideDataEditor, .gdg-style {
                background: #0d1520 !important;
                background-color: #0d1520 !important;
                color: #c9d1d9 !important;
                border: 1px solid #1a2332 !important;
                border-radius: 8px !important;
            }

            /* ── Expander full dark ── */
            details, details > summary,
            [data-testid="stExpander"],
            [data-testid="stExpander"] > div,
            .streamlit-expanderContent {
                background: #0d1520 !important;
                background-color: #0d1520 !important;
                border-color: #1a2332 !important;
            }

            /* ── Metric boxes ── */
            [data-testid="stMetric"],
            [data-testid="stMetricValue"],
            [data-testid="stMetricLabel"] {
                background: transparent !important;
                color: #e2e8f0 !important;
            }

            /* ── Alert/info boxes ── */
            [data-testid="stAlert"],
            .stInfo, .stWarning, .stError, .stSuccess {
                background: #0f1535 !important;
                border: 1px solid #1e3a5f !important;
                border-radius: 8px !important;
                color: #e0e7ff !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ensure_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def _new_conversation() -> None:
    st.session_state.messages = []
    st.session_state.pop(_PENDING_PIPELINE_QUERY, None)


def _last_state() -> Optional[Dict[str, Any]]:
    for msg in reversed(st.session_state.messages):
        if msg.get("role") == "assistant" and msg.get("state"):
            return msg["state"]
    return None


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.2f}%"
    except Exception:
        return "—"


def _fmt_usd(value: Any) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
        if v >= 1_000_000_000:
            return f"${v/1_000_000_000:.1f}B"
        if v >= 1_000_000:
            return f"${v/1_000_000:.1f}M"
        if v >= 1_000:
            return f"${v/1_000:.1f}K"
        return f"${v:,.0f}"
    except Exception:
        return "—"


def _duty_color_class(pct: float) -> str:
    if pct == 0:
        return "zero"
    if pct < 10:
        return "low"
    if pct < 30:
        return "medium"
    return "high"


def _bar_color(pct: float) -> str:
    if pct == 0:
        return "#3b82f6"
    if pct < 10:
        return "#10b981"
    if pct < 30:
        return "#f59e0b"
    return "#ef4444"


def _stream_text(text: str) -> Iterable[str]:
    for token in text.split(" "):
        yield token + " "
        time.sleep(0.006)


def _confidence_color(conf: Optional[str]) -> str:
    mapping = {"HIGH": "conf-high", "MEDIUM": "conf-medium", "LOW": "conf-low"}
    return mapping.get((conf or "").upper(), "")


def _html_table(rows: list, cols: list) -> str:
    """Render a dark-themed HTML table (replaces st.dataframe which goes invisible on dark CSS)."""
    th = (
        "padding:8px 12px;font-family:'IBM Plex Mono',monospace;font-size:0.72rem;"
        "text-transform:uppercase;letter-spacing:0.06em;color:#4a5568;"
        "border-bottom:1px solid #1a2332;text-align:left;white-space:nowrap"
    )
    td = (
        "padding:8px 12px;font-family:'IBM Plex Mono',monospace;font-size:0.8rem;"
        "color:#c9d1d9;border-bottom:1px solid #0d1520"
    )
    header = "".join(f'<th style="{th}">{html.escape(str(c))}</th>' for c in cols)
    body = ""
    for row in rows:
        cells = "".join(
            f'<td style="{td}">{html.escape(str(row.get(c) if row.get(c) is not None else "—"))}</td>'
            for c in cols
        )
        body += f'<tr style="background:#080d14">{cells}</tr>'
    return (
        f'<div style="overflow-x:auto;border:1px solid #1a2332;border-radius:6px;margin-bottom:10px">'
        f'<table style="width:100%;border-collapse:collapse">'
        f'<thead style="background:#0a0f18"><tr>{header}</tr></thead>'
        f'<tbody>{body}</tbody>'
        f'</table></div>'
    )


def _parse_json_response(text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    raw = str(text).strip()
    if not (raw.startswith("{") and raw.endswith("}")):
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def _is_low_confidence_stop(state: Dict[str, Any]) -> bool:
    return bool(state.get("hitl_required") and state.get("hitl_reason") == "low_confidence")


# ── Duty Gauge ────────────────────────────────────────────────────────────────

def _render_duty_gauge(state: Dict[str, Any]) -> None:
    total = state.get("total_duty")
    base = state.get("base_rate") or 0.0
    adder = state.get("adder_rate") or 0.0
    fta_applied = state.get("fta_applied", False)
    fta_program = state.get("fta_program") or ""
    adder_method = state.get("adder_method") or ""

    try:
        pct = float(total) if total is not None else 0.0
    except Exception:
        pct = 0.0

    bar_w = min(pct, 150) / 150 * 100
    color_cls = _duty_color_class(pct)
    bar_col = _bar_color(pct)

    st.markdown(
        f"""
        <div class="duty-gauge-wrap">
            <div>
                <div class="duty-number {color_cls}">{pct:.1f}%</div>
                <div class="duty-label">Total Effective Duty</div>
            </div>
            <div class="duty-bar-track">
                <div class="duty-bar-fill" style="width:{bar_w:.1f}%;background:{bar_col};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_comparison_result(state: Dict[str, Any]) -> None:
    """Render side-by-side country comparison pipeline results."""
    product = state.get("product") or "Product"
    hts_code = state.get("hts_code") or "—"
    hts_desc = state.get("hts_description") or ""
    comparison = state.get("comparison") or []
    cheapest = state.get("cheapest_country")

    st.markdown('<div class="section-header">Country Comparison</div>', unsafe_allow_html=True)
    st.markdown(
        f"<div class='tiq-hts'><span style='color:#60a5fa'>{html.escape(str(product))}</span>"
        f"<span style='color:#4a5568'> · </span>{html.escape(str(hts_code))}"
        f"<span style='color:#4a5568'> · </span>{html.escape(str(hts_desc))}</div>",
        unsafe_allow_html=True,
    )

    if not comparison:
        st.info("No comparison results returned.")
        return

    rows = []
    for c in comparison:
        rows.append({
            "Country": c.get("country", "—"),
            "Base MFN": f"{float(c.get('mfn_rate', c.get('base_rate', 0.0)) or 0.0):.1f}%",
            "Adder": f"+{float(c.get('adder_rate', 0.0) or 0.0):.1f}%",
            "Section122": f"+{float(c.get('section122_adder', 0.0) or 0.0):.1f}%",
            "Total Duty": f"{float(c.get('total_duty', 0.0) or 0.0):.1f}%",
            "Doc": c.get("adder_doc") or "—",
            "Method": c.get("adder_method") or "—",
            "FTA": c.get("fta_program") if c.get("fta_applied") else "No",
        })

    cols = ["Country", "Base MFN", "Adder", "Section122", "Total Duty", "Doc", "Method", "FTA"]
    st.markdown(_html_table(rows, cols), unsafe_allow_html=True)

    if cheapest:
        st.success(f"Cheapest country: {cheapest}")

    # Show policy summary for each country
    for c in comparison:
        if c.get("policy_summary"):
            with st.expander(f"📄 Policy Summary — {c['country']}", expanded=False):
                cleaned = re.sub(r"https?://\S+", "", str(c["policy_summary"])).strip()
                st.markdown(cleaned)

# ── Citation renderer ──────────────────────────────────────────────────────────

def _citation_badge_class(c: Dict[str, Any]) -> str:
    ctype = (c.get("type") or "").lower()
    agency = (c.get("agency_short") or c.get("agency") or "").upper()
    if "hts" in ctype:
        return "hts"
    if agency == "USTR" or "ustr" in agency.lower():
        return "ustr"
    if agency == "CBP":
        return "cbp"
    if agency == "EOP":
        return "eop"
    if "census" in ctype or "census" in agency.lower():
        return "census"
    return "default"


def _render_citations(citations: list) -> None:
    if not citations:
        return
    st.markdown('<div class="section-header">Citations</div>', unsafe_allow_html=True)
    all_cit_html = ""
    for c in citations:
        badge_cls = _citation_badge_class(c)
        ctype = html.escape(str(c.get("type", "unknown")))
        agency = html.escape(str(c.get("agency_short") or c.get("agency") or ""))
        title = c.get("title") or c.get("id") or ""
        title_h = html.escape(str(title))
        doc_id = html.escape(str(c.get("id") or c.get("document_number") or ""))
        date_s = html.escape(str(c.get("effective_date") or c.get("publication_date") or ""))
        url = c.get("url") or c.get("html_url") or ""
        link_html = ""
        if url and str(url).strip().startswith("http"):
            safe_u = html.escape(str(url).strip(), quote=True)
            link_html = f'<a class="cit-link" href="{safe_u}" target="_blank" rel="noopener noreferrer">&#8599; Open source</a>'

        meta_parts = []
        if doc_id:
            meta_parts.append(doc_id)
        if date_s:
            meta_parts.append(date_s)
        meta_str = " &middot; ".join(meta_parts)

        all_cit_html += f'<div class="cit-card"><span class="cit-badge {badge_cls}">{agency or ctype}</span><div class="cit-body"><div class="cit-title">{title_h}</div><div class="cit-meta">{meta_str}</div>{link_html}</div></div>'

    st.markdown(all_cit_html, unsafe_allow_html=True)


# ── Main answer renderer ───────────────────────────────────────────────────────

def _render_answer_text(state: Dict[str, Any]) -> str:
    if state.get("clarification_needed"):
        return state.get("clarification_message") or "Which type of product did you mean?"
    if _is_low_confidence_stop(state):
        return _low_confidence_copy(state)
    final = state.get("final_response")
    if _parse_json_response(final):
        return json.dumps(_parse_json_response(final), indent=2)
    if final:
        # Strip raw URLs from response text (handled by citations panel)
        cleaned = re.sub(r'https?://\S+', '', str(final)).strip()
        return cleaned
    if state.get("hitl_required"):
        return "Flagged for human review — please refine your query."
    return "No response generated."


def _low_confidence_copy(state: Dict[str, Any]) -> str:
    product = (state.get("product") or "").strip()
    hts = state.get("hts_code")
    desc = (state.get("hts_description") or "").strip()
    conf_raw = state.get("classification_confidence")
    try:
        conf_s = f"{float(conf_raw):.0%}" if conf_raw is not None else "very low"
    except Exception:
        conf_s = "very low"
    parts = ["Classification confidence is below threshold — duty rates not computed."]
    if product:
        parts.append(f'Specify the exact form, grade, or end use of "{product}" for a precise classification.')
    if hts and desc:
        parts.append(f"Best automated match: **{hts}** ({desc}) at **{conf_s}** confidence.")
    return " ".join(parts)


def _emit_answer(content: str) -> None:
    if content.lstrip().startswith("## ") or "\n## " in content:
        st.markdown(content)
    else:
        st.write_stream(_stream_text(content))


def _render_assistant_details(
    state: Dict[str, Any],
    *,
    show_clarification_actions: bool = False,
    widget_key_prefix: str = "assist",
) -> None:

    if state.get("error"):
        st.error(f"Pipeline error: {state.get('error')}")
        return

    # ── Clarification flow ──
    if state.get("clarification_needed"):
        st.markdown('<div class="section-header">Clarification Needed</div>', unsafe_allow_html=True)
        suggestions = state.get("clarification_suggestions") or []
        if suggestions and show_clarification_actions:
            for i, s in enumerate(suggestions[:6]):
                label = (s.get("label") or s.get("query") or "Suggestion").strip()
                q = (s.get("query") or label).strip()
                if not q:
                    continue
                st.markdown('<div class="clarify-chip">', unsafe_allow_html=True)
                if st.button(label, key=f"{widget_key_prefix}_clarify_{i}", use_container_width=True):
                    _queue_followup_pipeline(q)
                st.markdown('</div>', unsafe_allow_html=True)
        return

    # ── Low confidence stop ──
    if _is_low_confidence_stop(state):
        hts_code = state.get("hts_code")
        hts_desc = state.get("hts_description")
        if hts_code or hts_desc:
            st.markdown('<div class="section-header">Tentative Classification</div>', unsafe_allow_html=True)
            st.markdown(
                f"<div class='tiq-hts'>{hts_code or 'N/A'} — {hts_desc or ''}</div>",
                unsafe_allow_html=True,
            )
        return

    # ── Intent tag ──
    intent = state.get("query_intent")
    if intent:
        intent_labels = {
            "rate_change": "⏱ Rate Change Analysis",
            "country_compare": "⚖ Country Comparison",
            "exemption_check": "🔍 Exemption Check",
        }
        label = intent_labels.get(intent, intent.replace("_", " ").title())
        st.markdown(f'<div class="intent-tag">◈ {label}</div>', unsafe_allow_html=True)

    # ── HTS Classification ──
    hts_code = state.get("hts_code")
    hts_desc = state.get("hts_description")
    conf = state.get("classification_confidence")
    if hts_code:
        st.markdown('<div class="section-header">Classification</div>', unsafe_allow_html=True)
        conf_pct = f"{float(conf):.0%}" if conf is not None else "—"
        st.markdown(
            f"<div class='tiq-hts'>"
            f"<span style='color:#60a5fa'>{hts_code}</span>"
            f"<span style='color:#4a5568'> · </span>"
            f"<span style='color:#9ca3af'>{hts_desc or ''}</span>"
            f"<span style='float:right;color:#4a5568;font-size:0.75rem'>{conf_pct} confidence</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Duty Gauge ──
    if state.get("total_duty") is not None:
        st.markdown('<div class="section-header">Duty Breakdown</div>', unsafe_allow_html=True)
        _render_duty_gauge(state)

    # ── HITL warning ──
    if state.get("hitl_required") and not _is_low_confidence_stop(state):
        st.warning(f"⚑ Flagged for review: {state.get('hitl_reason') or ''}")

    # ── Trade Metrics ──
    period = state.get("trade_period")
    import_val = state.get("import_value_usd")
    trend = state.get("trade_trend_label")
    if any(v is not None for v in [period, import_val, trend]):
        st.markdown('<div class="section-header">Trade Data</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"<div class='metric-box'><div class='metric-val'>{period or '—'}</div><div class='metric-lbl'>Trade Period</div></div>",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"<div class='metric-box'><div class='metric-val'>{_fmt_usd(import_val)}</div><div class='metric-lbl'>Import Value</div></div>",
                unsafe_allow_html=True,
            )
        with col3:
            trend_color = "#10b981" if trend and "▲" in trend else "#ef4444" if trend and "▼" in trend else "#9ca3af"
            st.markdown(
                f"<div class='metric-box'><div class='metric-val' style='color:{trend_color}'>{trend or '—'}</div><div class='metric-lbl'>YoY Trend</div></div>",
                unsafe_allow_html=True,
            )

    # ── Country Comparison — disabled (hardcoded adder tiers removed) ──
    # country_comparison = state.get("country_comparison") or []
    country_comparison = []  # disabled
    if country_comparison:
        with st.expander("🌍 Country Comparison — Estimated Total Duty", expanded=False):
            # Build display table
            rows = []
            for c in country_comparison:
                total = c.get("estimated_total", c.get("base_rate", 0.0))
                adder = c.get("adder_rate", 0.0)
                fta = c.get("fta_program") or ""
                note = c.get("note") or ""
                rows.append({
                    "Country": c["country"],
                    "Base MFN": f"{c.get('base_rate', 0.0):.1f}%",
                    "Adder": f"+{adder:.0f}%" if adder > 0 else "—",
                    "Est. Total": f"{total:.1f}%",
                    "Program": fta or c.get("adder_program") or "—",
                    "Note": note,
                })
            _cc_cols = ["Country", "Base MFN", "Adder", "Est. Total", "Program", "Note"]
            st.markdown(_html_table(rows, _cc_cols), unsafe_allow_html=True)

            # Cheapest recommendation
            if rows:
                cheapest = min(country_comparison, key=lambda x: x.get("estimated_total", 999))
                total = cheapest.get("estimated_total", 0.0)
                st.markdown(
                    f'<div style="margin-top:10px;padding:10px 14px;background:#0a1f18;border:1px solid #065f46;border-radius:6px;font-family:\'IBM Plex Mono\',monospace;font-size:0.78rem;color:#34d399">'
                    f'✓ Cheapest alternative: <strong>{cheapest["country"]}</strong> at <strong>{total:.1f}%</strong> estimated total duty'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Top Importers ──
    top_importers = state.get("top_importers") or []
    if top_importers:
        with st.expander("📊 Top Import Partners (Census, 24-month)", expanded=False):
            _KEY_LABELS = {
                "census_country_name":  "Country",
                "imports_usd_trailing": "Import Value",
                "base_rate":            "Base Rate",
                "mfn_rate":             "MFN Rate",
                "fta_program":          "FTA Program",
            }
            _PREF_KEYS = ["census_country_name", "imports_usd_trailing", "base_rate", "mfn_rate", "fta_program"]
            avail_keys = [k for k in _PREF_KEYS if any(k in r for r in top_importers)]
            disp_cols = [_KEY_LABELS[k] for k in avail_keys]
            rows_disp = []
            for r in top_importers:
                row_d = {}
                for k in avail_keys:
                    val = r.get(k)
                    if k == "imports_usd_trailing" and val is not None:
                        try:
                            val = _fmt_usd(float(val))
                        except Exception:
                            pass
                    elif k in ("base_rate", "mfn_rate") and val is not None:
                        try:
                            val = f"{float(val):.2f}%"
                        except Exception:
                            pass
                    row_d[_KEY_LABELS[k]] = val if val is not None else "—"
                rows_disp.append(row_d)
            st.markdown(_html_table(rows_disp, disp_cols), unsafe_allow_html=True)

    # ── Citations ──
    citations = state.get("citations") or []
    _render_citations(citations)


# ── Pipeline runner ────────────────────────────────────────────────────────────

_PIPELINE_STEPS = [
    (3,  "Parsing query and detecting intent"),
    (8,  "Classifying product to HTS code"),
    (4,  "Looking up base MFN and FTA rates"),
    (10, "Fetching Federal Register policy context"),
    (5,  "Computing Section 301 / 232 / IEEPA adders"),
    (7,  "Pulling Census Bureau trade data"),
    (12, "Synthesizing final answer"),
]


def _query_header_html(query: str) -> str:
    """Render query text above the answer."""
    return (
        f'<div style="font-style:italic;color:#4a5568;font-size:0.85rem;'
        f'padding:0 0 10px;border-bottom:1px solid #1a2332;margin-bottom:14px;'
        f'font-family:\'IBM Plex Sans\',sans-serif;line-height:1.4">'
        f'{html.escape(query)}'
        f'</div>'
    )


def _query_footer_html(query: str) -> str:
    """Render query text below the answer."""
    return (
        f'<div style="margin-top:14px;padding-top:8px;border-top:1px solid #1a2332;'
        f'font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;color:#4a5568;'
        f'letter-spacing:0.04em">↑ {html.escape(query)}</div>'
    )


def _run_pipeline_with_status(text: str, status_ph) -> Dict[str, Any]:
    """Run pipeline in a background thread; animate step labels in the main thread."""
    result_holder: list = [None]
    error_holder: list = [None]

    def _worker() -> None:
        try:
            result_holder[0] = run_pipeline_auto(text)
        except Exception as exc:
            error_holder[0] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    # Build cumulative tick boundaries for each step
    cum: list = []
    acc = 0
    for ticks, lbl in _PIPELINE_STEPS:
        acc += ticks
        cum.append((acc, lbl))

    tick = 0
    while thread.is_alive():
        step_label = _PIPELINE_STEPS[-1][1]
        for threshold, lbl in cum:
            if tick < threshold:
                step_label = lbl
                break
        dots = "·" * ((tick % 3) + 1)
        status_ph.markdown(
            f'<div style="padding:8px 0;font-family:\'IBM Plex Mono\',monospace;'
            f'font-size:0.82rem;color:#60a5fa">'
            f'<span style="color:#4a5568;margin-right:8px">▶</span>'
            f'{step_label}{dots}'
            f'</div>',
            unsafe_allow_html=True,
        )
        time.sleep(0.3)
        tick += 1

    thread.join()
    status_ph.empty()

    if error_holder[0]:
        raise error_holder[0]
    return result_holder[0]


def _run_pipeline_response(text: str) -> None:
    with st.chat_message("assistant"):
        st.markdown(_query_header_html(text), unsafe_allow_html=True)
        status_ph = st.empty()
        try:
            state = _run_pipeline_with_status(text, status_ph)
        except Exception as exc:
            err = f"Pipeline error: {exc}"
            st.error(err)
            st.session_state.messages.append(
                {"role": "assistant", "content": err, "state": {"error": str(exc)}, "query": text}
            )
            return

        if isinstance(state, dict) and state.get("comparison") is not None:
            content = "Country comparison generated."
            _emit_answer(content)
            _render_comparison_result(state)
            st.markdown(_query_footer_html(text), unsafe_allow_html=True)
            st.session_state.messages.append(
                {"role": "assistant", "content": content, "state": state, "query": text}
            )
            return

        content = _render_answer_text(state)
        _emit_answer(content)

        show_followups = bool(
            state.get("clarification_needed") or _is_low_confidence_stop(state)
        )
        if not show_followups:
            conf = (state.get("pipeline_confidence") or "UNKNOWN").upper()
            color_cls = _confidence_color(conf)
            st.markdown(
                f'<span class="{color_cls}" style="font-family:\'IBM Plex Mono\',monospace;font-size:0.72rem;letter-spacing:0.06em">◆ Confidence: {conf}</span>',
                unsafe_allow_html=True,
            )

        _render_assistant_details(
            state,
            show_clarification_actions=show_followups,
            widget_key_prefix=f"live_{len(st.session_state.messages)}",
        )
        st.markdown(_query_footer_html(text), unsafe_allow_html=True)
        st.session_state.messages.append(
            {"role": "assistant", "content": content, "state": state, "query": text}
        )


def _append_user_and_run(text: str) -> None:
    st.session_state.messages.append({"role": "user", "content": text, "state": None})
    _run_pipeline_response(text)


def _queue_followup_pipeline(text: str) -> None:
    st.session_state.messages.append({"role": "user", "content": text, "state": None})
    st.session_state[_PENDING_PIPELINE_QUERY] = text
    st.rerun()


def _render_message(msg: Dict[str, Any], *, show_clarification_actions: bool = False, widget_key_prefix: str = "msg") -> None:
    role = msg["role"]
    with st.chat_message(role):
        query = msg.get("query", "") if role == "assistant" else ""
        if query:
            st.markdown(_query_header_html(query), unsafe_allow_html=True)
        st.markdown(msg.get("content", ""))
        if role == "assistant" and msg.get("state"):
            _render_assistant_details(
                msg["state"],
                show_clarification_actions=show_clarification_actions,
                widget_key_prefix=widget_key_prefix,
            )
        if query:
            st.markdown(_query_footer_html(query), unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div style="padding:16px 0 24px">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:1.1rem;font-weight:600;color:#e2e8f0">
                    ⚖ TariffIQ
                </div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#4a5568;text-transform:uppercase;letter-spacing:0.12em;margin-top:4px">
                    US Import Intelligence
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.button("＋ New Conversation", use_container_width=True, on_click=_new_conversation)

        state = _last_state()
        if not state:
            st.markdown(
                "<div style='margin-top:24px;padding:16px;background:#0d1520;border:1px solid #1a2332;border-radius:8px;font-family:\"IBM Plex Mono\",monospace;font-size:0.75rem;color:#4a5568;line-height:1.6'>"
                "Ask about any US import product to get:<br><br>"
                "· HTS classification<br>"
                "· MFN + Section 301 rates<br>"
                "· FTA preferential rates<br>"
                "· Policy notice trail<br>"
                "· Census trade data<br>"
                "· Country comparison"
                "</div>",
                unsafe_allow_html=True,
            )

            return

        # Show last query stats
        conf = (state.get("pipeline_confidence") or "UNKNOWN").upper()
        color_cls = _confidence_color(conf)
        hts = state.get("hts_code") or "—"
        total = state.get("total_duty")
        product = state.get("product") or "—"
        country = state.get("country") or "—"

        st.markdown(
            f"""
            <div style="margin-top:16px">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;color:#4a5568;margin-bottom:12px">Last Query</div>
                <div class="tiq-card">
                    <div style="font-size:0.78rem;color:#e2e8f0;margin-bottom:8px">{html.escape(str(product))} from {html.escape(str(country))}</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#60a5fa">{html.escape(hts)}</div>
                    <div style="margin-top:8px;display:flex;justify-content:space-between;align-items:center">
                        <span style="font-family:'IBM Plex Mono',monospace;font-size:1rem;font-weight:600;color:#e2e8f0">{_fmt_pct(total)}</span>
                        <span class="{color_cls}" style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem">{conf}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Pipeline JSON", expanded=False):
            st.json(state)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _inject_styles()
    _ensure_state()
    _render_sidebar()

    # Header
    st.markdown(
        """
        <div class="tiq-header">
            <div class="tiq-title">Tariff<span>IQ</span></div>
            <div class="tiq-subtitle">US Import Tariff Intelligence · Multi-Agent RAG</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    msgs = st.session_state.messages
    last_idx = len(msgs) - 1
    for i, msg in enumerate(msgs):
        st_obj = msg.get("state")
        show_actions = (
            msg.get("role") == "assistant"
            and st_obj
            and i == last_idx
            and (st_obj.get("clarification_needed") or _is_low_confidence_stop(st_obj))
        )
        _render_message(msg, show_clarification_actions=show_actions, widget_key_prefix=f"hist_{i}")

    pending = st.session_state.pop(_PENDING_PIPELINE_QUERY, None)
    if pending:
        _run_pipeline_response(pending)

    prompt = st.chat_input("Ask a tariff question — e.g. 'semiconductors from China'")
    if not prompt:
        return

    _append_user_and_run(prompt)


if __name__ == "__main__":
    main()