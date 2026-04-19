# Run: streamlit run streamlit/app.py --server.port 8501

from __future__ import annotations

import json
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


st.set_page_config(layout="wide", page_title="TariffIQ", page_icon="🛃")


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp { background: #0b1220; color: #e5e7eb; }
            [data-testid="stSidebar"] { background: #111827; }
            .tiq-card {
                background: #111827;
                border: 1px solid #1f2937;
                border-radius: 12px;
                padding: 12px;
                margin-bottom: 10px;
            }
            .tiq-kv { color: #9ca3af; font-size: 0.86rem; }
            .tiq-hts {
                background: #0f172a;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 10px;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
                font-size: 0.95rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _ensure_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def _new_conversation() -> None:
    st.session_state.messages = []
    st.rerun()


def _last_state() -> Optional[Dict[str, Any]]:
    for msg in reversed(st.session_state.messages):
        if msg.get("role") == "assistant" and msg.get("state"):
            return msg["state"]
    return None


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.2f}%"
    except Exception:
        return "N/A"


def _fmt_usd(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"${float(value):,.0f}"
    except Exception:
        return "N/A"


def _safe_badge(label: str, color: str = "blue") -> None:
    if hasattr(st, "badge"):
        st.badge(label, color=color)
    else:
        st.markdown(f"**{label}**")


def _confidence_color(conf: Optional[str]) -> str:
    mapping = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red"}
    return mapping.get((conf or "").upper(), "blue")


def _stream_text(text: str) -> Iterable[str]:
    for token in text.split(" "):
        yield token + " "
        time.sleep(0.01)


def _render_assistant_details(state: Dict[str, Any]) -> None:
    hts_code = state.get("hts_code")
    hts_desc = state.get("hts_description")
    if hts_code or hts_desc:
        st.markdown("**Classification**")
        st.markdown(
            f"<div class='tiq-hts'>{hts_code or 'N/A'} — {hts_desc or 'No description available'}</div>",
            unsafe_allow_html=True,
        )

    duty_df = pd.DataFrame(
        [
            {
                "Base Rate": _fmt_pct(state.get("base_rate")),
                "Adder Rate": _fmt_pct(state.get("adder_rate")),
                "Total Duty": _fmt_pct(state.get("total_duty")),
            }
        ]
    )
    st.markdown("**Duty Breakdown**")
    st.dataframe(duty_df, use_container_width=True, hide_index=True)

    if state.get("fta_applied"):
        st.success(
            f"FTA Applied: {state.get('fta_program') or 'Program not specified'} "
            f"at {_fmt_pct(state.get('fta_rate'))}"
        )

    policy_summary = state.get("policy_summary")
    if policy_summary:
        with st.expander("Policy Context", expanded=False):
            st.markdown(policy_summary)

    citations = state.get("citations") or []
    if citations:
        st.markdown("**Citations**")
        for c in citations:
            ctype = c.get("type", "unknown")
            cid = c.get("id", c.get("document_number", "N/A"))
            source = c.get("source", "N/A")
            st.markdown(
                f"<div class='tiq-card'><div><b>{ctype}</b></div>"
                f"<div class='tiq-kv'>id: {cid}</div>"
                f"<div class='tiq-kv'>source: {source}</div></div>",
                unsafe_allow_html=True,
            )

    col1, col2, col3 = st.columns(3)
    col1.metric("Trade Period", state.get("trade_period") or "N/A")
    col2.metric("Import Value (USD)", _fmt_usd(state.get("import_value_usd")))
    col3.metric("Trade Trend", state.get("trade_trend_label") or "N/A")

    if state.get("hitl_required"):
        st.warning(
            f"Flagged for Human Review: {state.get('hitl_reason') or 'No reason provided'}"
        )


def _render_sidebar() -> None:
    with st.sidebar:
        st.title("🛃 TariffIQ")
        st.caption("US Import Tariff Intelligence")
        st.button("New Conversation", use_container_width=True, on_click=_new_conversation)

        state = _last_state()
        if not state:
            st.info("Ask a question to see pipeline details.")
            return

        st.markdown("### Query Status")
        conf = (state.get("pipeline_confidence") or "UNKNOWN").upper()
        _safe_badge(f"Confidence: {conf}", color=_confidence_color(conf))

        if state.get("hitl_required"):
            st.error(f"HITL required: {state.get('hitl_reason') or 'No reason provided'}")

        with st.expander("Pipeline Details", expanded=False):
            st.json(state)


def _render_message(msg: Dict[str, Any]) -> None:
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(msg.get("content", ""))
        if role == "assistant" and msg.get("state"):
            _render_assistant_details(msg["state"])


def _pipeline_to_assistant_content(state: Dict[str, Any]) -> str:
    if state.get("error"):
        return f"Pipeline error: {state['error']}"
    final_response = state.get("final_response")
    if final_response:
        return str(final_response)
    if not state.get("hitl_required"):
        return "Pipeline returned no response"
    return "Flagged for human review."


def main() -> None:
    _inject_styles()
    _ensure_state()
    _render_sidebar()

    st.title("TariffIQ Chat")
    st.caption("Conversational multi-agent RAG for US import tariff analysis")

    for msg in st.session_state.messages:
        _render_message(msg)

    prompt = st.chat_input("Ask a tariff question...")
    if not prompt:
        return

    user_msg = {"role": "user", "content": prompt, "state": None}
    st.session_state.messages.append(user_msg)
    _render_message(user_msg)

    with st.chat_message("assistant"):
        try:
            state = run_pipeline(prompt)
            content = _pipeline_to_assistant_content(state)
            st.write_stream(_stream_text(content))

            if state.get("error"):
                st.error(state["error"])

            _render_assistant_details(state)

            conf = (state.get("pipeline_confidence") or "UNKNOWN").upper()
            _safe_badge(f"Confidence: {conf}", color=_confidence_color(conf))

            st.session_state.messages.append(
                {"role": "assistant", "content": content, "state": state}
            )
            st.rerun()
        except Exception as exc:
            err = f"Pipeline crashed: {exc}"
            st.error(err)
            st.session_state.messages.append(
                {"role": "assistant", "content": err, "state": {"error": str(exc)}}
            )


if __name__ == "__main__":
    main()
