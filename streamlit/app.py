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

# ── NEW: Categorized example queries ─────────────────────────────────────────
EXAMPLE_QUERIES = {
    "🇨🇳 China Tariffs": [
        "electric vehicles from China",
        "semiconductors from China",
        "solar panels from China",
        "lithium batteries from China",
    ],
    "🤝 FTA Countries": [
        "washing machines from South Korea",
        "cotton t-shirts from Mexico",
        "auto parts from Canada",
        "laptops from Vietnam",
    ],
    "⚖️ Compare Sources": [
        "cheaper to import laptops from China or Vietnam?",
        "steel wire from China vs Germany",
    ],
    "📈 Rate Changes": [
        "has the tariff on lithium batteries changed?",
        "steel tariff history",
    ],
}


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
                font-weight: 700;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                padding: 4px 10px;
                border-radius: 6px;
                white-space: nowrap;
                flex-shrink: 0;
                margin-top: 2px;
            }
            /* NEW: distinct agency colors */
            .cit-badge.ustr   { background: #1e3a5f; color: #60a5fa; border: 1px solid #2d5a8f; }
            .cit-badge.cbp    { background: #1a3a2a; color: #10b981; border: 1px solid #065f46; }
            .cit-badge.hts    { background: #2d1f3a; color: #a78bfa; border: 1px solid #4c1d95; }
            .cit-badge.eop    { background: #3a2a1a; color: #f59e0b; border: 1px solid #92400e; }
            .cit-badge.census { background: #1a2332; color: #94a3b8; border: 1px solid #334155; }
            .cit-badge.usitc  { background: #1f2d3a; color: #38bdf8; border: 1px solid #0369a1; }
            .cit-badge.ita    { background: #1a2a3a; color: #7dd3fc; border: 1px solid #0c4a6e; }
            .cit-badge.default{ background: #1a2332; color: #94a3b8; border: 1px solid #334155; }
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

            /* ── Markdown headings ── */
            .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
            .stMarkdown h4, .stMarkdown h5, .stMarkdown h6,
            [data-testid="stMarkdownContainer"] h1,
            [data-testid="stMarkdownContainer"] h2,
            [data-testid="stMarkdownContainer"] h3 {
                color: #93c5fd !important;
                font-family: 'Poppins', sans-serif !important;
                font-weight: 600 !important;
            }
            [data-testid="stMarkdownContainer"] p,
            [data-testid="stMarkdownContainer"] li {
                color: #cbd5e1 !important;
                line-height: 1.7 !important;
            }
            [data-testid="stMarkdownContainer"] strong {
                color: #e0e7ff !important;
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
                background: #0a0e27 !important;
                border-top: 1px solid #1e3a5f !important;
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

            /* ── HTS Search input ── */
            [data-testid="stTextInput"] input {
                background: #0f1535 !important;
                border: 1px solid #1e3a5f !important;
                border-radius: 8px !important;
                color: #e0e7ff !important;
                font-family: 'Space Mono', monospace !important;
                font-size: 0.82rem !important;
            }
            [data-testid="stTextInput"] input:focus {
                border-color: #3b82f6 !important;
            }

            /* ── NEW: Timeline ── */
            .timeline-item {
                display: flex;
                gap: 14px;
                padding: 10px 0;
                border-bottom: 1px solid #1e3a5f;
            }
            .timeline-item:last-child { border-bottom: none; }
            .timeline-dot {
                width: 10px; height: 10px;
                border-radius: 50%;
                background: #3b82f6;
                flex-shrink: 0;
                margin-top: 5px;
                box-shadow: 0 0 8px rgba(59,130,246,0.5);
            }
            .timeline-title { font-size: 0.82rem; color: #e0e7ff; line-height: 1.4; }

            /* ── NEW: Example query category labels ── */
            .eq-category {
                font-family: 'Space Mono', monospace;
                font-size: 0.6rem;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                color: #4a5568;
                margin: 14px 0 6px;
                font-weight: 700;
            }

            /* ── Scrollbar ── */

            [data-testid="stBottom"], [data-testid="stBottom"] > div {
                background: #0a0e27 !important;
                background-color: #0a0e27 !important;
            }
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
    """Render a dark-themed HTML table."""
    th = (
        "padding:8px 12px;font-family:'Space Mono',monospace;font-size:0.68rem;"
        "text-transform:uppercase;letter-spacing:0.06em;color:#64748b;"
        "border-bottom:1px solid #1e3a5f;text-align:left;white-space:nowrap"
    )
    td = (
        "padding:8px 12px;font-family:'Space Mono',monospace;font-size:0.78rem;"
        "color:#e0e7ff;border-bottom:1px solid #0f1535"
    )
    header = "".join(f'<th style="{th}">{html.escape(str(c))}</th>' for c in cols)
    body = ""
    for row in rows:
        cells = "".join(
            f'<td style="{td}">{html.escape(str(row.get(c) if row.get(c) is not None else "—"))}</td>'
            for c in cols
        )
        body += f'<tr style="background:#0a0e27">{cells}</tr>'
    return (
        f'<div style="overflow-x:auto;border:1px solid #1e3a5f;border-radius:8px;margin-bottom:10px">'
        f'<table style="width:100%;border-collapse:collapse">'
        f'<thead style="background:#0f1535"><tr>{header}</tr></thead>'
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

    # Rate chips
    chips_html = f'<span class="rate-chip"><span class="dot" style="background:#3b82f6"></span>Base MFN {_fmt_pct(base)}</span>'
    if adder and float(adder) > 0:
        src = adder_method.upper().replace("_", " ")
        chips_html += f'<span class="rate-chip"><span class="dot" style="background:#ef4444"></span>Adder {_fmt_pct(adder)} · {src}</span>'
    if fta_applied and fta_program:
        chips_html += f'<span class="rate-chip"><span class="dot" style="background:#10b981"></span>{fta_program} applied</span>'
    st.markdown(f"<div style='margin-bottom:12px'>{chips_html}</div>", unsafe_allow_html=True)

    if fta_applied and fta_program:
        st.markdown(
            f'<div class="fta-badge">✓ {fta_program} preferential rate applied — {_fmt_pct(state.get("fta_rate"))}</div>',
            unsafe_allow_html=True,
        )


# ── NEW FEATURE 1: Duty Breakdown Donut Chart ─────────────────────────────────

def _render_duty_donut(state: Dict[str, Any]) -> None:
    """SVG donut chart showing base vs adder breakdown side-by-side with bars."""
    import math
    base = float(state.get("base_rate") or 0.0)
    adder = float(state.get("adder_rate") or 0.0)
    total = base + adder
    if total == 0:
        return

    r, cx, cy, stroke = 52, 68, 68, 16
    circumference = 2 * math.pi * r
    base_pct = (base / total) * 100
    adder_pct = (adder / total) * 100
    base_dash = (base_pct / 100) * circumference
    adder_dash = (adder_pct / 100) * circumference

    base_arc = (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#3b82f6" '
                f'stroke-width="{stroke}" stroke-dasharray="{base_dash:.2f} {circumference:.2f}" '
                f'stroke-dashoffset="0" transform="rotate(-90 {cx} {cy})"/>') if base > 0 else ""
    adder_arc = (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#ef4444" '
                 f'stroke-width="{stroke}" stroke-dasharray="{adder_dash:.2f} {circumference:.2f}" '
                 f'stroke-dashoffset="{-base_dash:.2f}" transform="rotate(-90 {cx} {cy})"/>') if adder > 0 else ""

    adder_bars = "" if adder == 0 else f"""
        <div style="margin-top:12px">
            <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                <span style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#ef4444">● 301/232/IEEPA Adder</span>
                <span style="font-family:'Space Mono',monospace;font-size:0.75rem;font-weight:700;color:#ef4444">{adder:.2f}%</span>
            </div>
            <div style="height:4px;background:#1a2553;border-radius:2px">
                <div style="height:100%;width:{adder_pct:.1f}%;background:#ef4444;border-radius:2px"></div>
            </div>
        </div>"""

    st.markdown(f"""
        <div style="display:flex;align-items:center;gap:20px;padding:16px 20px;
                    background:#0f1535;border:1px solid #1e3a5f;border-radius:10px;margin-bottom:12px">
            <svg width="136" height="136" viewBox="0 0 136 136">
                <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#1a2553" stroke-width="{stroke}"/>
                {base_arc}{adder_arc}
                <text x="{cx}" y="{cy-5}" text-anchor="middle"
                      font-family="Space Mono, monospace" font-size="15" font-weight="700" fill="#e0e7ff">{total:.1f}%</text>
                <text x="{cx}" y="{cy+12}" text-anchor="middle"
                      font-family="Poppins, sans-serif" font-size="7.5" fill="#64748b" letter-spacing="1">TOTAL DUTY</text>
            </svg>
            <div style="flex:1">
                <div>
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                        <span style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#60a5fa">● Base MFN</span>
                        <span style="font-family:'Space Mono',monospace;font-size:0.75rem;font-weight:700;color:#60a5fa">{base:.2f}%</span>
                    </div>
                    <div style="height:4px;background:#1a2553;border-radius:2px">
                        <div style="height:100%;width:{base_pct:.1f}%;background:#3b82f6;border-radius:2px"></div>
                    </div>
                </div>
                {adder_bars}
            </div>
        </div>""", unsafe_allow_html=True)


# ── NEW FEATURE 2: Top Importers Bar Chart ────────────────────────────────────

def _render_importers_chart(top_importers: list) -> None:
    """Horizontal bar chart for top import partners by USD value."""
    if not top_importers:
        return
    max_val = max(float(r.get("imports_usd_trailing") or 0) for r in top_importers)
    if max_val == 0:
        return

    colors = ["#3b82f6", "#6366f1", "#8b5cf6", "#a855f7", "#ec4899", "#f43f5e", "#f97316", "#eab308"]
    bars_html = ""
    for i, r in enumerate(top_importers[:8]):
        name = (r.get("census_country_name") or r.get("lookup_country") or "").title()
        val = float(r.get("imports_usd_trailing") or 0)
        pct_w = (val / max_val) * 100
        color = colors[i % len(colors)]
        fta = r.get("fta_program") or ""
        fta_tag = (f'<span style="font-size:0.58rem;background:#1a3a2a;color:#10b981;'
                   f'border:1px solid #065f46;padding:1px 5px;border-radius:3px;'
                   f'margin-left:6px;font-family:Space Mono,monospace">{html.escape(fta)}</span>'
                   if fta and r.get("fta_applied") else "")
        bars_html += f"""
        <div style="margin-bottom:10px">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
                <span style="font-family:'Poppins',sans-serif;font-size:0.8rem;color:#e0e7ff;font-weight:500">{html.escape(name)}{fta_tag}</span>
                <span style="font-family:'Space Mono',monospace;font-size:0.72rem;color:{color};font-weight:700">{_fmt_usd(val)}</span>
            </div>
            <div style="height:5px;background:#1a2553;border-radius:3px;overflow:hidden">
                <div style="height:100%;width:{pct_w:.1f}%;background:{color};border-radius:3px"></div>
            </div>
        </div>"""

    st.markdown(
        f'<div style="background:#0f1535;border:1px solid #1e3a5f;border-radius:10px;padding:18px 20px;margin-bottom:12px">'
        f'<div style="font-family:\'Space Mono\',monospace;font-size:0.6rem;text-transform:uppercase;'
        f'letter-spacing:0.1em;color:#64748b;margin-bottom:14px;font-weight:700">US Import Value by Partner — 24-Month</div>'
        f'{bars_html}</div>',
        unsafe_allow_html=True,
    )


# ── NEW FEATURE 3: Rate Change Timeline ──────────────────────────────────────

def _render_rate_timeline(rate_history: list, citations: list) -> None:
    """Chronological timeline of policy notices with agency color badges."""
    if not rate_history:
        return

    url_map = {(c.get("id") or c.get("document_number")): c["url"]
               for c in (citations or []) if c.get("url")}

    source_styles = {
        "USTR":  ("#1e3a5f", "#60a5fa"),
        "CBP":   ("#1a3a2a", "#10b981"),
        "EOP":   ("#3a2a1a", "#f59e0b"),
        "USITC": ("#1f2d3a", "#38bdf8"),
        "ITA":   ("#1a2a3a", "#7dd3fc"),
    }

    sorted_rows = sorted(rate_history, key=lambda x: x.get("publication_date") or "")
    items_html = ""
    for row in sorted_rows:
        doc = row.get("document_number") or ""
        title = (row.get("title") or "").replace("\n", " ")[:110]
        pub = row.get("publication_date") or ""
        src = (row.get("source") or "USTR").upper()
        url = url_map.get(doc, f"https://www.federalregister.gov/documents/{doc}" if doc else "")
        bg, fg = source_styles.get(src, ("#1a2332", "#94a3b8"))
        link_html = (f'<a href="{html.escape(url)}" target="_blank" '
                     f'style="color:#64748b;font-family:\'Space Mono\',monospace;'
                     f'font-size:0.62rem;text-decoration:none;margin-top:3px;display:block">'
                     f'↗ {html.escape(doc)}</a>') if url else ""
        items_html += f"""
        <div class="timeline-item">
            <div class="timeline-dot"></div>
            <div style="flex:1">
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;flex-wrap:wrap">
                    <span style="font-family:'Space Mono',monospace;font-size:0.66rem;color:#64748b">{html.escape(pub)}</span>
                    <span style="background:{bg};color:{fg};font-family:'Space Mono',monospace;
                                 font-size:0.58rem;font-weight:700;padding:2px 6px;border-radius:4px">{html.escape(src)}</span>
                </div>
                <div class="timeline-title">{html.escape(title)}</div>
                {link_html}
            </div>
        </div>"""

    st.markdown(
        f'<div style="background:#0f1535;border:1px solid #1e3a5f;border-radius:10px;padding:18px 20px;margin-bottom:12px">'
        f'<div style="font-family:\'Space Mono\',monospace;font-size:0.6rem;text-transform:uppercase;'
        f'letter-spacing:0.1em;color:#64748b;margin-bottom:14px;font-weight:700">Policy Notice Timeline</div>'
        f'{items_html}</div>',
        unsafe_allow_html=True,
    )


# ── NEW FEATURE 4 & 5: Enhanced citations with distinct agency colors ─────────

def _citation_badge_class(c: Dict[str, Any]) -> str:
    ctype = (c.get("type") or "").lower()
    agency = (c.get("agency_short") or c.get("agency") or "").upper()
    if "hts" in ctype:
        return "hts"
    if "census" in ctype or "census" in agency.lower():
        return "census"
    if "USTR" in agency or "TRADE REPRESENTATIVE" in agency:
        return "ustr"
    if "CBP" in agency or "CUSTOMS" in agency:
        return "cbp"
    if "EOP" in agency or "EXECUTIVE OFFICE" in agency:
        return "eop"
    if "USITC" in agency or "ITC" in agency:
        return "usitc"
    if "ITA" in agency or "INTERNATIONAL TRADE ADMINISTRATION" in agency:
        return "ita"
    return "default"


def _render_citations(citations: list) -> None:
    if not citations:
        return
    st.markdown('<div class="section-header">Sources & Citations</div>', unsafe_allow_html=True)
    all_cit_html = ""
    for c in citations:
        badge_cls = _citation_badge_class(c)
        agency = html.escape(str(c.get("agency_short") or c.get("agency") or ""))
        title = c.get("title") or c.get("id") or ""
        title_h = html.escape(str(title))
        doc_id = html.escape(str(c.get("id") or c.get("document_number") or ""))
        date_s = html.escape(str(c.get("effective_date") or c.get("publication_date") or ""))
        url = c.get("url") or c.get("html_url") or ""
        link_html = ""
        if url and str(url).strip().startswith("http"):
            safe_u = html.escape(str(url).strip(), quote=True)
            link_html = f'<a class="cit-link" href="{safe_u}" target="_blank" rel="noopener noreferrer">↗ Open source</a>'
        meta_parts = [p for p in [doc_id, date_s] if p]
        meta_str = " · ".join(meta_parts)
        all_cit_html += (f'<div class="cit-card"><span class="cit-badge {badge_cls}">{agency}</span>'
                         f'<div class="cit-body"><div class="cit-title">{title_h}</div>'
                         f'<div class="cit-meta">{meta_str}</div>{link_html}</div></div>')
    st.markdown(all_cit_html, unsafe_allow_html=True)


# ── NEW VIZ 1: Tariff Composition Stacked Bar (for comparison) ───────────────

def _render_tariff_stacked_bar(comp: list) -> None:
    """Stacked bar showing base MFN + adder = total per country."""
    if not comp or len(comp) < 2:
        return
    max_total = max(float(c.get("total_duty") or 0) for c in comp) or 1
    bar_w = 420
    bar_h = 36
    gap = 14
    label_w = 90
    svg_h = (bar_h + gap) * len(comp) + 40

    bars = ""
    legend = (
        f'<rect x="0" y="{svg_h-18}" width="12" height="12" fill="#3b82f6" rx="2"/>'
        f'<text x="16" y="{svg_h-8}" font-family="Space Mono,monospace" font-size="9" fill="#64748b">Base MFN</text>'
        f'<rect x="90" y="{svg_h-18}" width="12" height="12" fill="#ef4444" rx="2"/>'
        f'<text x="106" y="{svg_h-8}" font-family="Space Mono,monospace" font-size="9" fill="#64748b">Section 301/232/IEEPA Adder</text>'
    )
    for i, c in enumerate(comp):
        y = i * (bar_h + gap) + 20
        base = float(c.get("base_rate") or c.get("mfn_rate") or 0)
        adder = float(c.get("adder_rate") or 0)
        total = float(c.get("total_duty") or 0)
        base_w = (base / max_total) * bar_w
        adder_w = (adder / max_total) * bar_w
        country = c.get("country", "")
        fta = c.get("fta_program") if c.get("fta_applied") else ""
        fta_tag = f' · {fta}' if fta else ""
        bars += (
            f'<text x="{label_w-4}" y="{y+bar_h//2+4}" text-anchor="end" '
            f'font-family="Poppins,sans-serif" font-size="11" fill="#e0e7ff" font-weight="500">{html.escape(str(country))}</text>'
            f'<rect x="{label_w}" y="{y}" width="{bar_w}" height="{bar_h}" fill="#1a2553" rx="4"/>'
        )
        if base_w > 0:
            bars += f'<rect x="{label_w}" y="{y}" width="{base_w:.1f}" height="{bar_h}" fill="#3b82f6" rx="4"/>'
        if adder_w > 0:
            rx = "0" if base_w > 0 else "4"
            bars += f'<rect x="{label_w+base_w:.1f}" y="{y}" width="{adder_w:.1f}" height="{bar_h}" fill="#ef4444" rx="0" style="border-radius:0 4px 4px 0"/>'
        bars += (
            f'<text x="{label_w+base_w+adder_w+6:.1f}" y="{y+bar_h//2+4}" '
            f'font-family="Space Mono,monospace" font-size="11" fill="#e0e7ff" font-weight="700">{total:.1f}%{html.escape(fta_tag)}</text>'
        )
    svg = f'<svg viewBox="0 0 {label_w+bar_w+80} {svg_h}" xmlns="http://www.w3.org/2000/svg">{bars}{legend}</svg>'
    st.markdown(
        f'<div style="background:#0f1535;border:1px solid #1e3a5f;border-radius:10px;padding:18px 20px;margin-bottom:12px">'
        f'<div style="font-family:Space Mono,monospace;font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748b;margin-bottom:14px;font-weight:700">Tariff Composition by Country</div>'
        f'{svg}</div>',
        unsafe_allow_html=True
    )


# ── NEW VIZ 2: Top Importers Treemap ─────────────────────────────────────────

def _render_importers_treemap(top_importers: list) -> None:
    """Treemap visualization of top import partners by USD value."""
    if not top_importers or len(top_importers) < 2:
        return
    items = []
    for r in top_importers[:8]:
        name = (r.get("census_country_name") or r.get("lookup_country") or "").title()
        val = float(r.get("imports_usd_trailing") or 0)
        fta = bool(r.get("fta_applied"))
        if val > 0:
            items.append({"name": name, "val": val, "fta": fta})
    if not items:
        return
    total = sum(i["val"] for i in items)
    colors = ["#2563eb","#1d4ed8","#1e40af","#1e3a8a","#1d4ed8","#2563eb","#3b82f6","#60a5fa"]
    # Simple squarified layout — row-based
    W, H = 500, 220
    rects = ""
    x, y = 0, 0
    row_h = H // 2
    row_items = [items[:4], items[4:]]
    for ri, row in enumerate(row_items):
        if not row:
            continue
        row_total = sum(i["val"] for i in row)
        cx = 0
        rh = row_h - 2
        for j, item in enumerate(row):
            w = int((item["val"] / row_total) * W) - 2
            color = colors[(ri*4+j) % len(colors)]
            fta_mark = "★ " if item["fta"] else ""
            pct = item["val"]/total*100
            name_short = item["name"][:12]
            rects += (
                f'<rect x="{cx+1}" y="{ri*row_h+1}" width="{w}" height="{rh}" fill="{color}" '
                f'fill-opacity="0.85" rx="4" stroke="#0a0e27" stroke-width="2"/>'
                f'<text x="{cx+w//2+1}" y="{ri*row_h+rh//2-6}" text-anchor="middle" '
                f'font-family="Poppins,sans-serif" font-size="11" fill="#ffffff" font-weight="600">{html.escape(fta_mark+name_short)}</text>'
                f'<text x="{cx+w//2+1}" y="{ri*row_h+rh//2+10}" text-anchor="middle" '
                f'font-family="Space Mono,monospace" font-size="9" fill="rgba(255,255,255,0.7)">{_fmt_usd(item["val"])}</text>'
                f'<text x="{cx+w//2+1}" y="{ri*row_h+rh//2+24}" text-anchor="middle" '
                f'font-family="Space Mono,monospace" font-size="9" fill="rgba(255,255,255,0.5)">{pct:.1f}%</text>'
            )
            cx += w + 2
    svg = f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto">{rects}</svg>'
    st.markdown(
        f'<div style="background:#0f1535;border:1px solid #1e3a5f;border-radius:10px;padding:18px 20px;margin-bottom:12px">'
        f'<div style="font-family:Space Mono,monospace;font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748b;margin-bottom:12px;font-weight:700">Import Market Share — Top Partners (24-Month) · ★ FTA</div>'
        f'{svg}</div>',
        unsafe_allow_html=True
    )


# ── NEW VIZ 3: Duty Escalation Ladder ────────────────────────────────────────

def _render_duty_escalation(state: Dict[str, Any]) -> None:
    """Visual ladder showing how duty builds up: Free → Base MFN → +Adder → Total."""
    base = float(state.get("base_rate") or 0)
    adder = float(state.get("adder_rate") or 0)
    total = float(state.get("total_duty") or 0)
    fta = state.get("fta_applied", False)
    fta_rate = float(state.get("fta_rate") or 0) if fta else None
    fta_program = state.get("fta_program") or ""
    adder_doc = state.get("adder_doc") or ""
    adder_method = (state.get("adder_method") or "").upper().replace("_"," ")
    if total == 0 and not fta:
        return

    steps = [("Free (0%)", 0, "#4a5568", "Starting point — no duty")]
    if fta and fta_rate is not None:
        steps.append((f"{fta_program} Rate", fta_rate, "#10b981", f"FTA preferential rate"))
    steps.append((f"Base MFN", base, "#3b82f6", f"Standard most-favored-nation rate"))
    if adder > 0:
        steps.append((f"+{adder_method or 'Adder'}", base+adder, "#ef4444", f"{adder_doc} — Section 301/232/IEEPA"))
    
    max_val = max(s[1] for s in steps) or 1
    W, H_step = 560, 52
    svg_h = H_step * len(steps) + 20
    items_svg = ""
    bar_x, bar_w_max = 160, 260

    for i, (label, val, color, note) in enumerate(steps):
        y = i * H_step + 10
        filled_w = (val / max_val) * bar_w_max if max_val > 0 else 0
        connector = (f'<line x1="{bar_x + (steps[i-1][1]/max_val)*bar_w_max:.0f}" y1="{y}" '
                     f'x2="{bar_x + (steps[i-1][1]/max_val)*bar_w_max:.0f}" y2="{y+4}" '
                     f'stroke="#1e3a5f" stroke-width="1" stroke-dasharray="3,2"/>') if i > 0 else ""
        items_svg += (
            f'{connector}'
            f'<text x="{bar_x-6}" y="{y+H_step//2+4}" text-anchor="end" font-family="Poppins,sans-serif" '
            f'font-size="11" fill="#94a3b8" font-weight="500">{html.escape(label)}</text>'
            f'<rect x="{bar_x}" y="{y+8}" width="{bar_w_max}" height="20" fill="#1a2553" rx="3"/>'
            f'<rect x="{bar_x}" y="{y+8}" width="{filled_w:.1f}" height="20" fill="{color}" rx="3"/>'
            f'<text x="{bar_x+bar_w_max+8}" y="{y+H_step//2+4}" font-family="Space Mono,monospace" '
            f'font-size="11" fill="{color}" font-weight="700">{val:.1f}%</text>'
            f'<text x="{bar_x}" y="{y+H_step-2}" font-family="Space Mono,monospace" '
            f'font-size="8" fill="#4a5568">{html.escape(note)}</text>'
        )
    svg = f'<svg viewBox="0 0 {W} {svg_h}" xmlns="http://www.w3.org/2000/svg">{items_svg}</svg>'
    st.markdown(
        f'<div style="background:#0f1535;border:1px solid #1e3a5f;border-radius:10px;padding:18px 20px;margin-bottom:12px">'
        f'<div style="font-family:Space Mono,monospace;font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748b;margin-bottom:12px;font-weight:700">Duty Escalation Ladder</div>'
        f'{svg}</div>',
        unsafe_allow_html=True
    )


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
        f"<span style='color:#4a5568'> · </span><span style='color:#94a3b8'>{html.escape(str(hts_desc))}</span></div>",
        unsafe_allow_html=True,
    )

    if not comparison:
        st.info("No comparison results returned.")
        return

    # Bar chart for comparison
    max_duty = max(float(c.get("total_duty") or 0) for c in comparison) or 1
    comp_bars = ""
    for c in comparison:
        duty = float(c.get("total_duty") or 0)
        w = (duty / max_duty) * 100
        col = _bar_color(duty)
        comp_bars += f"""
        <div style="margin-bottom:10px">
            <div style="display:flex;justify-content:space-between;margin-bottom:3px">
                <span style="font-family:'Poppins',sans-serif;font-size:0.82rem;color:#e0e7ff">{html.escape(c.get('country',''))}</span>
                <span style="font-family:'Space Mono',monospace;font-size:0.82rem;color:{col};font-weight:700">{duty:.1f}%</span>
            </div>
            <div style="height:6px;background:#1a2553;border-radius:3px">
                <div style="height:100%;width:{w:.1f}%;background:{col};border-radius:3px"></div>
            </div>
        </div>"""
    st.markdown(
        f'<div style="background:#0f1535;border:1px solid #1e3a5f;border-radius:10px;padding:18px 20px;margin-bottom:12px">{comp_bars}</div>',
        unsafe_allow_html=True,
    )

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
        st.success(f"✓ Cheapest source: **{cheapest}**")

    for c in comparison:
        if c.get("policy_summary"):
            with st.expander(f"📄 Policy Summary — {c['country']}", expanded=False):
                cleaned = re.sub(r"https?://\S+", "", str(c["policy_summary"])).strip()
                st.markdown(cleaned)


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
            st.markdown(f"<div class='tiq-hts'>{hts_code or 'N/A'} — {hts_desc or ''}</div>", unsafe_allow_html=True)
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
            f"<span style='color:#94a3b8'>{hts_desc or ''}</span>"
            f"<span style='float:right;color:#64748b;font-size:0.75rem'>{conf_pct} confidence</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Duty Gauge + NEW Donut ──
    if state.get("total_duty") is not None:
        st.markdown('<div class="section-header">Duty Breakdown</div>', unsafe_allow_html=True)
        _render_duty_gauge(state)
        _render_duty_donut(state)
        _render_duty_escalation(state)

    # ── Exposure Score ──
    if state.get("total_duty") is not None:
        _render_exposure_score(state)

    # ── Sourcing Recommendation ──
    _render_sourcing_recommendation(state)

    # ── HITL warning ──
    if state.get("hitl_required") and not _is_low_confidence_stop(state):
        st.warning(f"⚑ Flagged for review: {state.get('hitl_reason') or ''}")

    # ── Country Comparison — only from comparison pipeline result ──
    _raw_comp = state.get("country_comparison") if state.get("_is_comparison") else None
    # Filter out bad country entries (e.g. "from" captured by regex)
    comp = [c for c in (_raw_comp or []) if len(c.get("country","")) > 3 and c.get("country","").lower() not in ("from","none","null","all")] or None
    if comp:
        st.markdown('<div class="section-header">Country Comparison</div>', unsafe_allow_html=True)
        cheapest = state.get("cheapest_country") or (min(comp, key=lambda x: x.get("total_duty") or x.get("estimated_total") or 0).get("country") if comp else None)
        max_duty = max((float(c.get("total_duty") or c.get("estimated_total") or 0)) for c in comp) or 1
        bars_html = ""
        for c in comp:
            duty = float(c.get("total_duty") or c.get("estimated_total") or 0)
            w = (duty / max_duty) * 100
            col = _bar_color(duty)
            country_name = c.get("country", "—")
            bars_html += f"""<div style="margin-bottom:10px">
                <div style="display:flex;justify-content:space-between;margin-bottom:3px">
                    <span style="font-family:'Poppins',sans-serif;font-size:0.82rem;color:#e0e7ff">{html.escape(str(country_name))}</span>
                    <span style="font-family:'Space Mono',monospace;font-size:0.82rem;color:{col};font-weight:700">{duty:.1f}%</span>
                </div>
                <div style="height:6px;background:#1a2553;border-radius:3px"><div style="height:100%;width:{w:.1f}%;background:{col};border-radius:3px"></div></div>
            </div>"""
        st.markdown(f'<div style="background:#0f1535;border:1px solid #1e3a5f;border-radius:10px;padding:18px 20px;margin-bottom:12px">{bars_html}</div>', unsafe_allow_html=True)
        rows = []
        for c in comp:
            rows.append({
                "Country": c.get("country", "—"),
                "Base MFN": f"{float(c.get('base_rate') or c.get('mfn_rate') or 0):.1f}%",
                "Adder": f"+{float(c.get('adder_rate') or 0):.1f}%",
                "Total Duty": f"{float(c.get('total_duty') or c.get('estimated_total') or 0):.1f}%",
                "FTA": c.get("fta_program") if c.get("fta_applied") else "—",
            })
        st.markdown(_html_table(rows, ["Country", "Base MFN", "Adder", "Total Duty", "FTA"]), unsafe_allow_html=True)
        _render_tariff_stacked_bar(comp)
        if cheapest:
            st.success(f"✓ Cheapest source: **{cheapest}**")

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
            trend_color = "#10b981" if trend and "▲" in trend else "#ef4444" if trend and "▼" in trend else "#94a3b8"
            st.markdown(
                f"<div class='metric-box'><div class='metric-val' style='color:{trend_color}'>{trend or '—'}</div><div class='metric-lbl'>YoY Trend</div></div>",
                unsafe_allow_html=True,
            )

    # ── NEW: Top Importers Bar Chart ──
    top_importers = state.get("top_importers") or []
    if top_importers:
        st.markdown('<div class="section-header">Top Import Partners</div>', unsafe_allow_html=True)
        _render_importers_treemap(top_importers)
        with st.expander("📊 Full Partner Table", expanded=False):
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
                        try: val = _fmt_usd(float(val))
                        except Exception: pass
                    elif k in ("base_rate", "mfn_rate") and val is not None:
                        try: val = f"{float(val):.2f}%"
                        except Exception: pass
                    row_d[_KEY_LABELS[k]] = val if val is not None else "—"
                rows_disp.append(row_d)
            st.markdown(_html_table(rows_disp, disp_cols), unsafe_allow_html=True)

    # ── NEW: Rate Change Timeline ──
    rate_history = state.get("rate_change_history") or []
    citations_list = state.get("citations") or []
    if rate_history:
        st.markdown('<div class="section-header">Policy Timeline</div>', unsafe_allow_html=True)
        _render_rate_timeline(rate_history, citations_list)

    # ── Policy Context ──
    policy_summary = state.get("policy_summary")
    if policy_summary:
        with st.expander("📋 Policy Context", expanded=False):
            st.markdown(
                f"<div style='font-size:0.85rem;color:#94a3b8;line-height:1.7'>{policy_summary}</div>",
                unsafe_allow_html=True,
            )

    # ── What-If Calculator ──
    _render_whatif_slider(state, widget_key_prefix=widget_key_prefix)

    # ── Citations (now with agency colors) ──
    _render_citations(citations_list)


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
    return (
        f'<div style="font-style:italic;color:#64748b;font-size:0.85rem;'
        f'padding:0 0 10px;border-bottom:1px solid #1e3a5f;margin-bottom:14px;'
        f'font-family:\'Poppins\',sans-serif;line-height:1.4">'
        f'{html.escape(query)}'
        f'</div>'
    )


def _query_footer_html(query: str) -> str:
    return (
        f'<div style="margin-top:14px;padding-top:8px;border-top:1px solid #1e3a5f;'
        f'font-family:\'Space Mono\',monospace;font-size:0.7rem;color:#4a5568;'
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
            f'<div style="padding:8px 0;font-family:\'Space Mono\',monospace;'
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

        # Merge comparison data into state for _render_assistant_details
        if state.get("comparison") is not None:
            comp_list = state.get("comparison", [])
            cheapest = state.get("cheapest_country")
            state = dict(state)
            state["country_comparison"] = comp_list
            state["cheapest_country"] = cheapest
            state["query_intent"] = "country_compare"
            state["_is_comparison"] = True

        content = _render_answer_text(state)
        _emit_answer(content)

        show_followups = bool(
            state.get("clarification_needed") or _is_low_confidence_stop(state)
        )
        if not show_followups:
            conf = (state.get("pipeline_confidence") or "UNKNOWN").upper()
            color_cls = _confidence_color(conf)
            st.markdown(
                f'<span class="{color_cls}" style="font-family:\'Space Mono\',monospace;font-size:0.72rem;letter-spacing:0.06em">◆ Confidence: {conf}</span>',
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
    # Track query history
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    if text not in st.session_state.query_history:
        st.session_state.query_history.insert(0, text)
        st.session_state.query_history = st.session_state.query_history[:5]
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


# ── FEATURE 1: Sourcing Recommendation Card ─────────────────────────────────

def _render_sourcing_recommendation(state: Dict[str, Any]) -> None:
    """Best alternative source card based on top_importers."""
    country = (state.get("country") or "").lower()
    total_duty = float(state.get("total_duty") or 0)
    top_importers = state.get("top_importers") or []
    if not top_importers or total_duty == 0:
        return
    # Find best alternative (lowest base_rate, not same country)
    alternatives = [
        r for r in top_importers
        if (r.get("census_country_name") or "").lower() != country
        and (r.get("lookup_country") or "").lower() != country
    ]
    if not alternatives:
        return
    best = min(alternatives, key=lambda r: float(r.get("base_rate") or r.get("mfn_rate") or 0))
    best_name = (best.get("census_country_name") or best.get("lookup_country") or "").title()
    best_rate = float(best.get("base_rate") or best.get("mfn_rate") or 0)
    best_fta = best.get("fta_program") or ""
    savings = total_duty - best_rate
    if savings <= 0:
        return
    fta_note = f" · {best_fta} preferential rate" if best_fta else ""
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0a2a1a,#0f1535);border:1px solid #10b981;
                border-radius:10px;padding:16px 20px;margin-bottom:12px">
        <div style="font-family:Space Mono,monospace;font-size:0.6rem;text-transform:uppercase;
                    letter-spacing:0.1em;color:#10b981;margin-bottom:8px;font-weight:700">
            ✦ Sourcing Recommendation
        </div>
        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px">
            <div>
                <div style="font-size:1rem;color:#e0e7ff;font-weight:600;margin-bottom:2px">
                    Consider sourcing from <span style="color:#10b981">{html.escape(best_name)}</span>
                </div>
                <div style="font-size:0.78rem;color:#64748b;font-family:Poppins,sans-serif">
                    {best_rate:.1f}% effective rate{html.escape(fta_note)} vs {total_duty:.1f}% current
                </div>
            </div>
            <div style="text-align:right">
                <div style="font-family:Space Mono,monospace;font-size:1.4rem;font-weight:700;color:#10b981">
                    -{savings:.1f}%
                </div>
                <div style="font-size:0.65rem;color:#64748b;font-family:Space Mono,monospace">DUTY SAVING</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)


# ── FEATURE 2: Section 301 Exposure Score ────────────────────────────────────

def _render_exposure_score(state: Dict[str, Any]) -> None:
    """0-100 risk score combining adder rate, country risk, and trade volume."""
    total = float(state.get("total_duty") or 0)
    adder = float(state.get("adder_rate") or 0)
    country = (state.get("country") or "").lower()
    import_val = float(state.get("import_value_usd") or 0)
    trend = state.get("trade_trend_pct") or 0

    if total == 0:
        return

    # Score components (0-100)
    rate_score = min(adder / 1.5, 40)  # max 40 pts from adder rate
    country_risk = 35 if "china" in country else 20 if country in ("russia","iran","north korea") else 5
    volume_score = min(import_val / 5_000_000 * 15, 15)  # max 15 pts
    trend_score = 10 if float(trend or 0) < -20 else 0  # declining = risky supply

    score = int(rate_score + country_risk + volume_score + trend_score)
    score = min(score, 100)

    if score >= 70:
        color, label, desc = "#ef4444", "HIGH RISK", "Significant tariff exposure — consider diversifying supply chain"
    elif score >= 40:
        color, label, desc = "#f59e0b", "MEDIUM RISK", "Moderate exposure — monitor policy changes closely"
    else:
        color, label, desc = "#10b981", "LOW RISK", "Manageable tariff exposure at current rates"

    arc_pct = score / 100
    import math
    r, cx, cy = 40, 50, 50
    circ = 2 * math.pi * r
    filled = arc_pct * circ * 0.75  # 270 deg arc
    gap = circ - filled

    st.markdown(f"""
    <div style="background:#0f1535;border:1px solid #1e3a5f;border-radius:10px;
                padding:16px 20px;margin-bottom:12px;display:flex;align-items:center;gap:20px">
        <svg width="100" height="100" viewBox="0 0 100 100">
            <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#1a2553" stroke-width="10"
                    stroke-dasharray="{circ*0.75:.1f} {circ*0.25:.1f}"
                    stroke-dashoffset="-{circ*0.125:.1f}" transform="rotate(0 {cx} {cy})"/>
            <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="10"
                    stroke-linecap="round"
                    stroke-dasharray="{filled:.1f} {circ:.1f}"
                    stroke-dashoffset="-{circ*0.125:.1f}" transform="rotate(0 {cx} {cy})"/>
            <text x="{cx}" y="{cy-4}" text-anchor="middle" font-family="Space Mono,monospace"
                  font-size="18" font-weight="700" fill="{color}">{score}</text>
            <text x="{cx}" y="{cy+12}" text-anchor="middle" font-family="Poppins,sans-serif"
                  font-size="7" fill="#64748b">/100</text>
        </svg>
        <div style="flex:1">
            <div style="font-family:Space Mono,monospace;font-size:0.7rem;font-weight:700;
                        color:{color};margin-bottom:4px">TARIFF EXPOSURE · {label}</div>
            <div style="font-size:0.78rem;color:#94a3b8;font-family:Poppins,sans-serif;
                        line-height:1.4;margin-bottom:8px">{html.escape(desc)}</div>
            <div style="display:flex;gap:6px;flex-wrap:wrap">
                <span style="font-family:Space Mono,monospace;font-size:0.6rem;color:#64748b;
                             background:#1a2553;padding:2px 6px;border-radius:3px">
                    Rate: {int(rate_score)}/40
                </span>
                <span style="font-family:Space Mono,monospace;font-size:0.6rem;color:#64748b;
                             background:#1a2553;padding:2px 6px;border-radius:3px">
                    Country: {country_risk}/35
                </span>
                <span style="font-family:Space Mono,monospace;font-size:0.6rem;color:#64748b;
                             background:#1a2553;padding:2px 6px;border-radius:3px">
                    Volume: {int(volume_score)}/15
                </span>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)


# ── FEATURE 3: What-If Tariff Slider ─────────────────────────────────────────

def _render_whatif_slider(state: Dict[str, Any], widget_key_prefix: str = "whatif") -> None:
    """Interactive what-if slider to recalculate duty at different adder rates."""
    base = float(state.get("base_rate") or 0)
    adder = float(state.get("adder_rate") or 0)
    total = float(state.get("total_duty") or 0)
    if adder == 0 or total == 0:
        return

    with st.expander("🔧 What-If Calculator — Adjust Tariff Rate", expanded=False):
        new_adder = st.slider(
            "Hypothetical adder rate (%)",
            min_value=0.0, max_value=float(max(adder * 2, 100)),
            value=float(adder), step=2.5,
            key=f"{widget_key_prefix}_slider"
        )
        new_total = base + new_adder
        delta = new_total - total
        delta_color = "#ef4444" if delta > 0 else "#10b981"
        delta_sign = "+" if delta > 0 else ""

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""<div class='metric-box'>
                <div class='metric-val' style='color:#64748b;text-decoration:line-through'>{total:.1f}%</div>
                <div class='metric-lbl'>Current Total</div></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class='metric-box'>
                <div class='metric-val' style='color:#60a5fa'>{new_total:.1f}%</div>
                <div class='metric-lbl'>New Total</div></div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class='metric-box'>
                <div class='metric-val' style='color:{delta_color}'>{delta_sign}{delta:.1f}%</div>
                <div class='metric-lbl'>Change</div></div>""", unsafe_allow_html=True)

        if st.session_state.get("calc_import_val"):
            import_v = float(st.session_state.get("calc_import_val", 10000))
            old_duty = import_v * (total / 100)
            new_duty = import_v * (new_total / 100)
            saving = old_duty - new_duty
            if abs(saving) > 0:
                s_color = "#10b981" if saving > 0 else "#ef4444"
                s_label = f"saves ${abs(saving):,.0f}" if saving > 0 else f"costs ${abs(saving):,.0f} more"
                st.markdown(f"<div style='font-family:Space Mono,monospace;font-size:0.78rem;color:{s_color};margin-top:8px'>On ${import_v:,.0f} import value → {s_label}</div>", unsafe_allow_html=True)


# ── Recent Policy Changes Sidebar Widget ─────────────────────────────────────

@st.cache_data(ttl=3600)
def _fetch_recent_fr_notices() -> list:
    try:
        import requests as _req
        params = {
            "per_page": "6",
            "order": "newest",
            "conditions[term]": "tariff",
            "conditions[agencies][]": [
                "trade-representative-office-of-united-states",
                "executive-office-of-the-president",
                "u-s-customs-and-border-protection",
            ],
            "fields[]": ["document_number","title","publication_date","agencies","html_url"],
        }
        resp = _req.get("https://www.federalregister.gov/api/v1/documents.json", params=params, timeout=8)
        if resp.ok:
            return resp.json().get("results", [])[:6]
    except Exception:
        pass
    return []


def _render_recent_policy_main() -> None:
    """Render recent policy changes in main pane with descriptions and links."""
    st.markdown('<div class="section-header">Recent Trade Policy Changes</div>', unsafe_allow_html=True)
    notices = _fetch_recent_fr_notices()
    if not notices:
        st.info("Could not load recent notices — Federal Register API unavailable.")
        return
    colors = {
        "trade-representative-office-of-united-states": ("#1e3a5f","#60a5fa","USTR"),
        "executive-office-of-the-president": ("#3a2a1a","#f59e0b","EOP"),
        "u-s-customs-and-border-protection": ("#1a3a2a","#10b981","CBP"),
    }
    for n in notices:
        title = (n.get("title") or "")
        pub = (n.get("publication_date") or "")[:10]
        url = n.get("html_url") or ""
        doc_num = n.get("document_number") or ""
        abstract = (n.get("abstract") or n.get("excerpt") or "")[:300]
        slug = ((n.get("agencies") or [{}])[0].get("slug") or "")
        agency_name = ((n.get("agencies") or [{}])[0].get("name") or "Federal Register")
        bg, fg, label = colors.get(slug, ("#1a2332","#94a3b8","FR"))
        link_html = f'<a href="{html.escape(url)}" target="_blank" style="font-family:Space Mono,monospace;font-size:0.72rem;color:#60a5fa;text-decoration:none">↗ View on Federal Register</a>' if url else ""
        st.markdown(f"""
        <div style="background:#0f1535;border:1px solid #1e3a5f;border-radius:10px;padding:16px 20px;margin-bottom:12px">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;flex-wrap:wrap">
                <span style="background:{bg};color:{fg};font-family:Space Mono,monospace;font-size:0.65rem;font-weight:700;padding:3px 8px;border-radius:5px">{label}</span>
                <span style="font-family:Space Mono,monospace;font-size:0.7rem;color:#64748b">{html.escape(pub)}</span>
                <span style="font-family:Space Mono,monospace;font-size:0.65rem;color:#4a5568">{html.escape(doc_num)}</span>
            </div>
            <div style="font-size:0.9rem;color:#e0e7ff;font-weight:600;margin-bottom:6px">{html.escape(title)}</div>
            <div style="font-size:0.78rem;color:#64748b;font-family:Poppins,sans-serif;margin-bottom:8px;line-height:1.5">{html.escape(agency_name)}</div>
            {link_html}
        </div>""", unsafe_allow_html=True)


def _render_recent_policy_sidebar() -> None:
    notices = _fetch_recent_fr_notices()
    if not notices:
        st.markdown("<div style='font-size:0.72rem;color:#4a5568;padding:4px 0'>Unable to load notices.</div>", unsafe_allow_html=True)
        return
    colors = {
        "trade-representative-office-of-united-states": ("#1e3a5f","#60a5fa","USTR"),
        "executive-office-of-the-president": ("#3a2a1a","#f59e0b","EOP"),
        "u-s-customs-and-border-protection": ("#1a3a2a","#10b981","CBP"),
    }
    items_html = ""
    for n in notices:
        title = (n.get("title") or "")[:72]
        pub = (n.get("publication_date") or "")[:10]
        url = n.get("html_url") or ""
        doc_num = n.get("document_number") or ""
        slug = ((n.get("agencies") or [{}])[0].get("slug") or "")
        bg, fg, label = colors.get(slug, ("#1a2332","#94a3b8","FR"))
        link_open = f'<a href="{html.escape(url)}" target="_blank" style="text-decoration:none">' if url else ""
        link_close = "</a>" if url else ""
        items_html += f"""<div style="padding:8px 0;border-bottom:1px solid #1e3a5f">
            <div style="display:flex;align-items:center;gap:6px;margin-bottom:3px">
                <span style="background:{bg};color:{fg};font-family:Space Mono,monospace;font-size:0.55rem;font-weight:700;padding:2px 5px;border-radius:3px">{label}</span>
                <span style="font-family:Space Mono,monospace;font-size:0.6rem;color:#4a5568">{html.escape(pub)}</span>
            </div>
            {link_open}<div style="font-size:0.72rem;color:#94a3b8;line-height:1.3;margin-bottom:2px">{html.escape(title)}</div>{link_close}
            <div style="font-family:Space Mono,monospace;font-size:0.6rem;color:#4a5568">{html.escape(doc_num)}</div>
        </div>"""
    st.markdown(f'<div>{items_html}</div>', unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div style="padding:16px 0 20px">
                <div style="font-family:'Space Mono',monospace;font-size:1.1rem;font-weight:700;color:#60a5fa">
                    ⚖ TariffIQ
                </div>
                <div style="font-family:'Space Mono',monospace;font-size:0.6rem;color:#4a5568;text-transform:uppercase;letter-spacing:0.12em;margin-top:4px">
                    US Import Intelligence
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.button("＋ New Conversation", use_container_width=True, on_click=_new_conversation)

        # ── Query History ──
        if st.session_state.get("query_history"):
            st.markdown("<div style='font-family:Space Mono,monospace;font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;color:#4a5568;margin:12px 0 6px;font-weight:700'>Recent Queries</div>", unsafe_allow_html=True)
            for i, q in enumerate(st.session_state.get("query_history", [])):
                short = q[:38] + "…" if len(q) > 38 else q
                if st.button(short, key=f"hist_q_{i}", use_container_width=True):
                    _append_user_and_run(q)

        state = _last_state()
        if state:
            conf = (state.get("pipeline_confidence") or "UNKNOWN").upper()
            color_cls = _confidence_color(conf)
            hts = state.get("hts_code") or "—"
            total = state.get("total_duty")
            product = state.get("product") or "—"
            country = state.get("country") or "—"
            st.markdown(
                f"""
                <div style="margin-top:16px">
                    <div style="font-family:'Space Mono',monospace;font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;color:#4a5568;margin-bottom:12px">Last Query</div>
                    <div class="tiq-card">
                        <div style="font-size:0.78rem;color:#e0e7ff;margin-bottom:8px">{html.escape(str(product))} from {html.escape(str(country))}</div>
                        <div style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#60a5fa">{html.escape(hts)}</div>
                        <div style="margin-top:8px;display:flex;justify-content:space-between;align-items:center">
                            <span style="font-family:'Space Mono',monospace;font-size:1rem;font-weight:600;color:#e0e7ff">{_fmt_pct(total)}</span>
                            <span class="{color_cls}" style="font-family:'Space Mono',monospace;font-size:0.7rem">{conf}</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("Pipeline JSON", expanded=False):
                st.json(state)

            # ── Tariff Calculator ──
            total_rate = state.get("total_duty")
            if total_rate is not None:
                try:
                    rate_f = float(total_rate) / 100
                    st.markdown("<div style='font-family:Space Mono,monospace;font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;color:#4a5568;margin:14px 0 6px;font-weight:700'>Duty Calculator</div>", unsafe_allow_html=True)
                    import_val = st.number_input("Import value (USD)", min_value=0.0, value=10000.0, step=1000.0, format="%.0f", key="calc_import_val", label_visibility="collapsed")
                    if import_val > 0:
                        duty_amt = import_val * rate_f
                        total_landed = import_val + duty_amt
                        st.markdown(f"""<div style='background:#0f1535;border:1px solid #1e3a5f;border-radius:8px;padding:12px 14px;font-family:Space Mono,monospace;font-size:0.75rem'>
                            <div style='color:#64748b;margin-bottom:4px'>Import Value</div>
                            <div style='color:#e0e7ff;font-weight:700'>${import_val:,.0f}</div>
                            <div style='color:#64748b;margin:6px 0 4px'>Duty ({total_rate:.1f}%)</div>
                            <div style='color:#ef4444;font-weight:700'>+${duty_amt:,.0f}</div>
                            <div style='border-top:1px solid #1e3a5f;margin-top:8px;padding-top:8px;color:#64748b'>Landed Cost</div>
                            <div style='color:#60a5fa;font-weight:700;font-size:0.9rem'>${total_landed:,.0f}</div>
                        </div>""", unsafe_allow_html=True)
                except Exception:
                    pass
            st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

        # Recent policy changes button at bottom
        st.markdown("<div style='margin-top:auto;padding-top:16px'></div>", unsafe_allow_html=True)
        if st.button("📋 Recent Policy Changes", use_container_width=True, key="recent_policy_btn"):
            st.session_state["show_recent_policy"] = True
            st.rerun()

        # HTS Code Search
        st.markdown(
            "<div style='font-family:Space Mono,monospace;font-size:0.6rem;text-transform:uppercase;"
            "letter-spacing:0.1em;color:#4a5568;margin:16px 0 8px;font-weight:700'>HTS Code Search</div>",
            unsafe_allow_html=True,
        )
        hts_query = st.text_input(
            "", placeholder="e.g. washing machine, 8703, solar...",
            key="hts_search_input", label_visibility="collapsed"
        )
        if hts_query and len(hts_query.strip()) >= 2:
            try:
                import requests as _req
                resp = _req.get(
                    "http://fastapi:8001/tools/hts/search",
                    params={"query": hts_query.strip(), "limit": 8},
                    timeout=5
                )
                hts_results = resp.json() if resp.ok else []
                if hts_results:
                    for item in hts_results[:8]:
                        code = item.get("hts_code", "")
                        desc = (item.get("description") or "")[:80]
                        label = f"{code} — {desc}"
                        if st.button(label, key=f"hts_{code}", use_container_width=True):
                            _append_user_and_run(desc)
                else:
                    st.markdown("<div style='font-size:0.75rem;color:#4a5568;padding:4px 0'>No results found</div>", unsafe_allow_html=True)
            except Exception:
                st.markdown("<div style='font-size:0.75rem;color:#ef4444'>Search unavailable</div>", unsafe_allow_html=True)



        # ── Example Queries ──
        st.markdown("<div style='font-family:Space Mono,monospace;font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;color:#4a5568;margin:16px 0 8px;font-weight:700'>Example Queries</div>", unsafe_allow_html=True)
        for category, queries in EXAMPLE_QUERIES.items():
            if category == "📈 Rate Changes":
                continue
            st.markdown(f"<div class='eq-category'>{html.escape(category)}</div>", unsafe_allow_html=True)
            for q in queries:
                if st.button(q, key=f"eq_{q}", use_container_width=True):
                    _append_user_and_run(q)

        # ── Demo Scenarios at bottom ──
        st.markdown("<div style='border-top:1px solid #1e3a5f;margin:20px 0 12px'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:Space Mono,monospace;font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;color:#4a5568;margin-bottom:8px;font-weight:700'>🎯 Demo Scenarios</div>", unsafe_allow_html=True)

        st.markdown("<div class='eq-category'>🔍 Ambiguous Product Classification</div>", unsafe_allow_html=True)
        _AMB = {
            "What tariff applies to batteries?": "batteries",
            "Import duty on steel products?": "steel",
            "Tariff rate for electric motors?": "motors",
            "What is the duty on pipes?": "pipes",
        }
        for label, q in _AMB.items():
            if st.button(label, key=f"demo_amb_{q}", use_container_width=True):
                _append_user_and_run(q)

        st.markdown("<div class='eq-category'>⚠️ HITL — Human Review Triggered</div>", unsafe_allow_html=True)
        _HITL = {
            "What is the tariff on metal parts from Germany?": "metal parts from Germany",
            "Import duty on industrial machinery parts?": "machinery parts",
        }
        for label, q in _HITL.items():
            if st.button(label, key=f"demo_hitl_{q}", use_container_width=True):
                _append_user_and_run(q)


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

    # Recent policy changes panel
    if st.session_state.get("show_recent_policy"):
        _render_recent_policy_main()
        if st.button("✕ Close", key="close_policy"):
            st.session_state["show_recent_policy"] = False
            st.rerun()

    pending = st.session_state.pop(_PENDING_PIPELINE_QUERY, None)
    if pending:
        _run_pipeline_response(pending)

    prompt = st.chat_input("Ask a tariff question — e.g. 'semiconductors from China'")
    if not prompt:
        return

    _append_user_and_run(prompt)


if __name__ == "__main__":
    main()