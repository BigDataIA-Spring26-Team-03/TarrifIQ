# Run: streamlit run streamlit/app.py --server.port 8501

from __future__ import annotations

import html
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.graph import run_pipeline

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
            @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

            /* ── Base ── */
            html, body, .stApp {
                background: #060a10 !important;
                color: #c9d1d9;
                font-family: 'IBM Plex Sans', sans-serif;
            }

            /* ── Sidebar ── */
            [data-testid="stSidebar"] {
                background: #0a0f18 !important;
                border-right: 1px solid #1a2332 !important;
            }
            [data-testid="stSidebar"] * { color: #8b96a5 !important; }
            [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3 { color: #e2e8f0 !important; }

            /* ── Header ── */
            .tiq-header {
                padding: 2rem 0 1.5rem;
                border-bottom: 1px solid #1a2332;
                margin-bottom: 2rem;
            }
            .tiq-title {
                font-family: 'IBM Plex Mono', monospace;
                font-size: 1.6rem;
                font-weight: 600;
                color: #e2e8f0;
                letter-spacing: -0.02em;
            }
            .tiq-title span { color: #3b82f6; }
            .tiq-subtitle {
                font-size: 0.8rem;
                color: #4a5568;
                font-family: 'IBM Plex Mono', monospace;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-top: 0.25rem;
            }

            /* ── Cards ── */
            .tiq-card {
                background: #0d1520;
                border: 1px solid #1a2332;
                border-radius: 8px;
                padding: 16px 20px;
                margin-bottom: 12px;
                transition: border-color 0.2s;
            }
            .tiq-card:hover { border-color: #2d3f55; }

            .tiq-card-accent {
                background: linear-gradient(135deg, #0d1520 0%, #0f1e2e 100%);
                border: 1px solid #1e3a5f;
                border-radius: 8px;
                padding: 16px 20px;
                margin-bottom: 12px;
            }

            /* ── HTS Code block ── */
            .tiq-hts {
                background: #080d14;
                border: 1px solid #1a2332;
                border-left: 3px solid #3b82f6;
                border-radius: 6px;
                padding: 12px 16px;
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.88rem;
                color: #93c5fd;
                line-height: 1.5;
            }

            /* ── Duty gauge ── */
            .duty-gauge-wrap {
                display: flex;
                align-items: center;
                gap: 16px;
                padding: 16px 20px;
                background: #0d1520;
                border: 1px solid #1a2332;
                border-radius: 8px;
                margin-bottom: 12px;
            }
            .duty-number {
                font-family: 'IBM Plex Mono', monospace;
                font-size: 2.4rem;
                font-weight: 600;
                line-height: 1;
                min-width: 100px;
            }
            .duty-number.high { color: #ef4444; }
            .duty-number.medium { color: #f59e0b; }
            .duty-number.low { color: #10b981; }
            .duty-number.zero { color: #3b82f6; }
            .duty-label {
                font-size: 0.72rem;
                color: #4a5568;
                font-family: 'IBM Plex Mono', monospace;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                margin-top: 4px;
            }
            .duty-bar-track {
                flex: 1;
                height: 6px;
                background: #1a2332;
                border-radius: 3px;
                overflow: hidden;
            }
            .duty-bar-fill {
                height: 100%;
                border-radius: 3px;
                transition: width 0.8s ease;
            }

            /* ── Rate breakdown chips ── */
            .rate-chip {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                background: #111827;
                border: 1px solid #1f2937;
                border-radius: 20px;
                padding: 4px 12px;
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.78rem;
                color: #9ca3af;
                margin: 3px;
            }
            .rate-chip .dot {
                width: 6px; height: 6px;
                border-radius: 50%;
                display: inline-block;
            }

            /* ── Citation cards ── */
            .cit-card {
                background: #080d14;
                border: 1px solid #1a2332;
                border-radius: 6px;
                padding: 12px 14px;
                margin-bottom: 8px;
                display: flex;
                gap: 12px;
                align-items: flex-start;
            }
            .cit-badge {
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.65rem;
                font-weight: 600;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                padding: 3px 8px;
                border-radius: 4px;
                white-space: nowrap;
                flex-shrink: 0;
                margin-top: 2px;
            }
            .cit-badge.ustr { background: #1e3a5f; color: #60a5fa; }
            .cit-badge.cbp  { background: #1a3326; color: #34d399; }
            .cit-badge.hts  { background: #1e2a4a; color: #818cf8; }
            .cit-badge.eop  { background: #2d1f3d; color: #c084fc; }
            .cit-badge.census { background: #1f2937; color: #9ca3af; }
            .cit-badge.default { background: #1f2937; color: #9ca3af; }
            .cit-body { flex: 1; min-width: 0; }
            .cit-title {
                font-size: 0.82rem;
                color: #cbd5e1;
                line-height: 1.4;
                margin-bottom: 4px;
                overflow: hidden;
                display: -webkit-box;
                -webkit-line-clamp: 2;
                -webkit-box-orient: vertical;
            }
            .cit-meta {
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.7rem;
                color: #4a5568;
            }
            .cit-link {
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.72rem;
                color: #3b82f6;
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
                background: #0d1520;
                border: 1px solid #1a2332;
                border-radius: 8px;
                padding: 14px 16px;
                text-align: center;
            }
            .metric-val {
                font-family: 'IBM Plex Mono', monospace;
                font-size: 1.2rem;
                font-weight: 600;
                color: #e2e8f0;
            }
            .metric-lbl {
                font-size: 0.7rem;
                color: #4a5568;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin-top: 3px;
                font-family: 'IBM Plex Mono', monospace;
            }

            /* ── Intent tag ── */
            .intent-tag {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                background: #0f1e2e;
                border: 1px solid #1e3a5f;
                border-radius: 4px;
                padding: 3px 10px;
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.72rem;
                color: #60a5fa;
                margin-bottom: 10px;
            }

            /* ── FTA badge ── */
            .fta-badge {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                background: #0a1f18;
                border: 1px solid #065f46;
                border-radius: 4px;
                padding: 5px 12px;
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.78rem;
                color: #34d399;
                margin-bottom: 10px;
            }

            /* ── Chat messages ── */
            [data-testid="stChatMessage"] {
                background: transparent !important;
                border: none !important;
            }
            .stChatMessage [data-testid="stMarkdownContainer"] p {
                color: #c9d1d9;
                line-height: 1.7;
            }

            /* ── Chat input ── */
            [data-testid="stChatInput"] {
                background: #0a0f18 !important;
                border-top: 1px solid #1a2332 !important;
            }
            [data-testid="stChatInputTextArea"] {
                background: #0d1520 !important;
                border: 1px solid #1a2332 !important;
                border-radius: 8px !important;
                color: #e2e8f0 !important;
                font-family: 'IBM Plex Sans', sans-serif !important;
            }

            /* ── Expanders ── */
            [data-testid="stExpander"] {
                background: #0d1520 !important;
                border: 1px solid #1a2332 !important;
                border-radius: 8px !important;
            }
            [data-testid="stExpander"] summary {
                color: #8b96a5 !important;
                font-family: 'IBM Plex Mono', monospace !important;
                font-size: 0.82rem !important;
            }

            /* ── Dataframe ── */
            .stDataFrame { border-radius: 8px; overflow: hidden; }
            [data-testid="stDataFrameResizable"] {
                border: 1px solid #1a2332 !important;
                border-radius: 8px !important;
            }

            /* ── Buttons ── */
            .stButton button {
                background: #0d1520 !important;
                border: 1px solid #1a2332 !important;
                color: #8b96a5 !important;
                border-radius: 6px !important;
                font-family: 'IBM Plex Mono', monospace !important;
                font-size: 0.78rem !important;
                transition: all 0.2s !important;
            }
            .stButton button:hover {
                border-color: #3b82f6 !important;
                color: #60a5fa !important;
                background: #0f1e2e !important;
            }

            /* ── Confidence badge ── */
            .conf-high { color: #10b981; }
            .conf-medium { color: #f59e0b; }
            .conf-low { color: #ef4444; }

            /* ── Section headers ── */
            .section-header {
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.7rem;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                color: #4a5568;
                margin: 16px 0 8px;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            .section-header::after {
                content: '';
                flex: 1;
                height: 1px;
                background: #1a2332;
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

            /* ── Kill ALL white backgrounds ── */
            .stApp > div, .main, .block-container,
            [data-testid="stAppViewContainer"],
            [data-testid="stVerticalBlock"],
            [data-testid="stHorizontalBlock"],
            section[data-testid="stSidebar"] > div,
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
                background: #0d1520 !important;
                border: 1px solid #1a2332 !important;
                border-radius: 8px !important;
                color: #c9d1d9 !important;
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
    st.rerun()


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
    chips_html = ""
    chips_html += f'<span class="rate-chip"><span class="dot" style="background:#3b82f6"></span>Base MFN {_fmt_pct(base)}</span>'
    if adder and float(adder) > 0:
        src = adder_method.upper().replace("_", " ")
        chips_html += f'<span class="rate-chip"><span class="dot" style="background:#ef4444"></span>Adder {_fmt_pct(adder)} · {src}</span>'
    if fta_applied and fta_program:
        chips_html += f'<span class="rate-chip"><span class="dot" style="background:#10b981"></span>{fta_program} applied</span>'

    st.markdown(f"<div>{chips_html}</div>", unsafe_allow_html=True)

    if fta_applied and fta_program:
        st.markdown(
            f'<div class="fta-badge">✓ {fta_program} preferential rate applied — {_fmt_pct(state.get("fta_rate"))}</div>',
            unsafe_allow_html=True,
        )


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

    # ── Policy Context ──
    policy_summary = state.get("policy_summary")
    if policy_summary:
        with st.expander("📋 Policy Context", expanded=False):
            st.markdown(
                f"<div style='font-size:0.85rem;color:#9ca3af;line-height:1.7'>{policy_summary}</div>",
                unsafe_allow_html=True,
            )

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

    # ── Country Comparison ──
    country_comparison = state.get("country_comparison") or []
    if country_comparison:
        with st.expander("🌍 Country Comparison", expanded=False):
            df = pd.DataFrame(country_comparison)
            cols_order = [c for c in ["country", "base_rate", "mfn_rate", "fta_program", "fta_applied", "note"] if c in df.columns]
            st.dataframe(df[cols_order] if cols_order else df, use_container_width=True, hide_index=True)

    # ── Top Importers ──
    top_importers = state.get("top_importers") or []
    if top_importers:
        with st.expander("📊 Top Import Partners (Census, 24-month)", expanded=False):
            df = pd.DataFrame(top_importers)
            display_cols = [c for c in ["census_country_name", "imports_usd_trailing", "base_rate", "mfn_rate", "fta_program"] if c in df.columns]
            st.dataframe(df[display_cols] if display_cols else df, use_container_width=True, hide_index=True)

    # ── Rate Change History ──
    rate_history = state.get("rate_change_history") or []
    if rate_history:
        with st.expander("⏱ Rate Change History", expanded=False):
            df = pd.DataFrame(rate_history)
            st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Citations ──
    citations = state.get("citations") or []
    _render_citations(citations)


# ── Pipeline runner ────────────────────────────────────────────────────────────

def _run_pipeline_response(text: str) -> None:
    with st.chat_message("assistant"):
        try:
            state = run_pipeline(text)
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
            st.session_state.messages.append(
                {"role": "assistant", "content": content, "state": state}
            )
        except Exception as exc:
            err = f"Pipeline error: {exc}"
            st.error(err)
            st.session_state.messages.append(
                {"role": "assistant", "content": err, "state": {"error": str(exc)}}
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
        st.markdown(msg.get("content", ""))
        if role == "assistant" and msg.get("state"):
            _render_assistant_details(
                msg["state"],
                show_clarification_actions=show_clarification_actions,
                widget_key_prefix=widget_key_prefix,
            )


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

            # Example queries
            st.markdown(
                "<div style='margin-top:16px;font-family:\"IBM Plex Mono\",monospace;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;color:#4a5568;margin-bottom:8px'>Example queries</div>",
                unsafe_allow_html=True,
            )
            examples = [
                "semiconductors from China",
                "electric vehicles from China",
                "washing machines from South Korea",
                "has the tariff on lithium batteries changed?",
                "cheaper to import laptops from China or Vietnam?",
            ]
            for ex in examples:
                if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                    _append_user_and_run(ex)
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