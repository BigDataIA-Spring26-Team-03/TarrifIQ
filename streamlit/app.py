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

st.set_page_config(layout="wide", page_title="TariffIQ", page_icon="🛃")

# Follow-up chips run inside a parent `st.chat_message`; pipeline UI must run at root.
_PENDING_PIPELINE_QUERY = "pending_pipeline_query"


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
    st.session_state.pop(_PENDING_PIPELINE_QUERY, None)
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
        time.sleep(0.008)


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


def _json_to_markdown(payload: Dict[str, Any]) -> str:
    parts: list[str] = []
    query = payload.get("query")
    if query:
        parts.append(f"**Query:** {query}")

    product = payload.get("product")
    country = payload.get("country_of_origin") or payload.get("country")
    if product or country:
        parts.append(f"**Product/Country:** {product or 'N/A'} | {country or 'N/A'}")

    duty = payload.get("duty_rates") or {}
    if duty:
        parts.append(
            f"**Total Effective Duty:** {duty.get('total_effective_duty', 'N/A')}"
        )

    recommendation = payload.get("recommendation")
    if recommendation:
        parts.append(f"**Recommendation:** {recommendation}")

    if not parts:
        return json.dumps(payload, indent=2)
    return "\n\n".join(parts)


def _is_low_confidence_classification_stop(state: Dict[str, Any]) -> bool:
    """True when the graph ended at HITL after classify (duty/policy never ran)."""
    return bool(
        state.get("hitl_required") and state.get("hitl_reason") == "low_confidence"
    )


def _low_confidence_assistant_copy(state: Dict[str, Any]) -> str:
    product = (state.get("product") or "").strip()
    hts = state.get("hts_code")
    desc = (state.get("hts_description") or "").strip()
    conf_raw = state.get("classification_confidence")
    try:
        conf_s = f"{float(conf_raw):.0%}" if conf_raw is not None else "very low"
    except (TypeError, ValueError):
        conf_s = "very low"

    parts: list[str] = [
        "I do not have enough confidence in the HTS match to quote duty rates from that wording alone.",
    ]
    if product:
        parts.append(
            f'For “{product}”, name the exact article (species, form such as ground or whole, packaging, or end use), '
            "or tap a narrower example below if one fits."
        )
    else:
        parts.append(
            "Name the exact article (species, form, packaging, or end use), "
            "or tap a narrower example below if one fits."
        )
    if hts and desc:
        parts.append(
            f"The best automated match was **{hts}** — {desc} — at **{conf_s}** confidence, "
            "which is below the threshold for automatic duty lookup."
        )
    elif hts:
        parts.append(
            f"The best automated match was **{hts}** at **{conf_s}** confidence, "
            "which is below the threshold for automatic duty lookup."
        )
    parts.append(
        "A human-review flag is recorded internally; you can still refine your message here and I will run the analysis again."
    )
    return " ".join(parts)


def _suggestions_for_uncertain_classification(state: Dict[str, Any]) -> list[dict[str, str]]:
    """Build chip follow-ups from a broad heading description (e.g. chapter 0910 spices)."""
    desc = (state.get("hts_description") or "").strip()
    country = state.get("country")
    suffix = f" from {country}" if country else ""

    if not desc:
        p = (state.get("product") or "").strip()
        if p:
            return [
                {
                    "label": f"{p} — add form (ground, whole, …)",
                    "query": f"{p} dried ground{suffix}".strip(),
                }
            ]
        return []

    body = re.sub(r"^[\d.]+\s*[-–—]\s*", "", desc).strip()
    body = re.sub(r"\band other[^,;:]*$", "", body, flags=re.IGNORECASE).strip()
    parts = re.split(r",|\bor\b", body, flags=re.IGNORECASE)
    out: list[dict[str, str]] = []
    for raw in parts:
        p = re.sub(r"\([^)]*\)", "", raw).strip()
        p = re.sub(r"\s+", " ", p)
        if len(p) < 3:
            continue
        pl = p.lower()
        if pl in ("nesoi", "other", "not elsewhere specified or included"):
            continue
        if p.endswith(":"):
            p = p[:-1].strip()
        label = p if len(p) <= 80 else p[:77] + "..."
        out.append({"label": label, "query": f"{pl}{suffix}"})
        if len(out) >= 6:
            break

    if not out:
        pr = (state.get("product") or "").strip()
        if pr:
            out.append(
                {
                    "label": f"{pr} — specify grade or processing",
                    "query": f"{pr} food grade retail{suffix}".strip(),
                }
            )
    return out


def _render_structured_answer(state: Dict[str, Any]) -> str:
    if state.get("clarification_needed"):
        return state.get("clarification_message") or (
            "Your product description could map to several different tariff categories. "
            "Pick one of the options below or describe what you import in more detail."
        )
    if _is_low_confidence_classification_stop(state):
        return _low_confidence_assistant_copy(state)
    final_response = state.get("final_response")
    json_payload = _parse_json_response(final_response)
    if json_payload:
        with st.expander("Structured Response", expanded=False):
            st.json(json_payload)
        return _json_to_markdown(json_payload)
    if final_response:
        return str(final_response)
    if state.get("hitl_required"):
        return "Flagged for human review."
    return "Pipeline returned no response"


def _emit_assistant_body(content: str) -> None:
    """Structured tariff answers use ## headers — markdown renders correctly; stream breaks headers."""
    if content.lstrip().startswith("## ") or "\n## " in content:
        st.markdown(content)
    else:
        st.write_stream(_stream_text(content))


def _run_pipeline_response(text: str) -> None:
    """Render assistant turn for `text`. Call only at root (not inside another chat_message)."""
    with st.chat_message("assistant"):
        try:
            state = run_pipeline(text)
            content = _pipeline_to_assistant_content(state)
            _emit_assistant_body(content)
            show_followups = bool(
                state.get("clarification_needed")
                or _is_low_confidence_classification_stop(state)
            )
            if not show_followups:
                conf = (state.get("pipeline_confidence") or "UNKNOWN").upper()
                _safe_badge(f"Confidence: {conf}", color=_confidence_color(conf))
            _render_assistant_details(
                state,
                show_clarification_actions=show_followups,
                widget_key_prefix=f"live_{len(st.session_state.messages)}",
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": content, "state": state}
            )
        except Exception as exc:
            err = f"Pipeline crashed: {exc}"
            st.error(err)
            st.session_state.messages.append(
                {"role": "assistant", "content": err, "state": {"error": str(exc)}}
            )


def _append_user_and_run_pipeline(text: str) -> None:
    st.session_state.messages.append({"role": "user", "content": text, "state": None})
    _run_pipeline_response(text)


def _queue_followup_pipeline(text: str) -> None:
    """Used from buttons inside nested chat_message: defer assistant UI to next run."""
    st.session_state.messages.append({"role": "user", "content": text, "state": None})
    st.session_state[_PENDING_PIPELINE_QUERY] = text
    st.rerun()


def _render_clarification_actions(suggestions: list, key_prefix: str) -> None:
    st.caption("Tap a category to continue, or type a more specific product in the box below.")
    for i, s in enumerate(suggestions[:6]):
        label = (s.get("label") or s.get("query") or "Suggestion").strip()
        q = (s.get("query") or label).strip()
        if not q:
            continue
        if st.button(label, key=f"{key_prefix}_clarify_{i}", use_container_width=True):
            _queue_followup_pipeline(q)


def _render_assistant_details(
    state: Dict[str, Any],
    *,
    show_clarification_actions: bool = False,
    widget_key_prefix: str = "assist",
) -> None:
    if state.get("error"):
        st.error(f"Pipeline error: {state.get('error')}")

    if state.get("clarification_needed"):
        st.info(state.get("clarification_message") or "I need a bit more detail to classify this.")
        suggestions = state.get("clarification_suggestions") or []
        if suggestions and show_clarification_actions:
            st.markdown("**Which did you mean?**")
            _render_clarification_actions(suggestions, widget_key_prefix)
        elif suggestions:
            st.markdown("**Suggested follow-ups**")
            for s in suggestions[:5]:
                label = s.get("label") or s.get("query") or "Suggestion"
                st.markdown(f"- {label}")
        return

    if _is_low_confidence_classification_stop(state):
        st.info(
            "Duty and trade steps were not run because classification confidence was below the safe threshold."
        )
        hts_code = state.get("hts_code")
        hts_desc = state.get("hts_description")
        if hts_code or hts_desc:
            st.markdown("**Tentative HTS match (not used for rates)**")
            st.markdown(
                f"<div class='tiq-hts'>{hts_code or 'N/A'} - {hts_desc or 'No description available'}</div>",
                unsafe_allow_html=True,
            )
        suggestions = _suggestions_for_uncertain_classification(state)
        if suggestions and show_clarification_actions:
            st.markdown("**Try one of these narrower queries**")
            _render_clarification_actions(suggestions, widget_key_prefix)
        elif suggestions:
            st.markdown("**Suggested follow-ups**")
            for s in suggestions[:6]:
                label = s.get("label") or s.get("query") or "Suggestion"
                st.markdown(f"- {label}")
        return

    if state.get("hitl_required"):
        st.warning(
            f"Flagged for Human Review: {state.get('hitl_reason') or 'No reason provided'}"
        )

    intent = state.get("query_intent")
    if intent:
        _safe_badge(f"Intent: {intent}", color="blue")

    hts_code = state.get("hts_code")
    hts_desc = state.get("hts_description")
    if hts_code or hts_desc:
        st.markdown("**Classification**")
        st.markdown(
            f"<div class='tiq-hts'>{hts_code or 'N/A'} - {hts_desc or 'No description available'}</div>",
            unsafe_allow_html=True,
        )

    duty_df = pd.DataFrame(
        [
            {
                "Base Rate": _fmt_pct(state.get("base_rate")),
                "MFN Rate": _fmt_pct(state.get("mfn_rate")),
                "FTA Rate": _fmt_pct(state.get("fta_rate")),
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

    col1, col2, col3 = st.columns(3)
    col1.metric("Trade Period", state.get("trade_period") or "N/A")
    col2.metric("Import Value (USD)", _fmt_usd(state.get("import_value_usd")))
    col3.metric("Trade Trend", state.get("trade_trend_label") or "N/A")

    country_comparison = state.get("country_comparison") or []
    if country_comparison:
        with st.expander("Country Comparison", expanded=False):
            st.dataframe(pd.DataFrame(country_comparison), use_container_width=True, hide_index=True)

    top_importers = state.get("top_importers") or []
    if top_importers:
        with st.expander("Top import partners by country (Census, trailing window)", expanded=False):
            st.dataframe(pd.DataFrame(top_importers), use_container_width=True, hide_index=True)

    rate_change_history = state.get("rate_change_history") or []
    if rate_change_history:
        with st.expander("Rate Change History", expanded=False):
            st.dataframe(pd.DataFrame(rate_change_history), use_container_width=True, hide_index=True)

    citations = state.get("citations") or []
    if citations:
        st.markdown("**Citations**")
        for c in citations:
            ctype = html.escape(str(c.get("type", "unknown")))
            cid = html.escape(str(c.get("id", c.get("document_number", "N/A"))))
            source = html.escape(str(c.get("source", "N/A")))
            agency = html.escape(str(c.get("agency_short") or c.get("agency") or "N/A"))
            title = c.get("title")
            title_html = ""
            if title:
                title_html = f"<div class='tiq-kv'>title: {html.escape(str(title))}</div>"
            url = c.get("url") or c.get("html_url")
            link_html = ""
            if url and str(url).strip().startswith(("http://", "https://")):
                safe_u = html.escape(str(url).strip(), quote=True)
                link_html = (
                    f"<div style='margin-top:8px;'><a href='{safe_u}' "
                    f"target='_blank' rel='noopener noreferrer'>Open official source ↗</a></div>"
                )
            st.markdown(
                f"<div class='tiq-card'><div><b>{ctype}</b></div>"
                f"<div class='tiq-kv'>id: {cid}</div>"
                f"<div class='tiq-kv'>agency: {agency}</div>"
                f"<div class='tiq-kv'>source: {source}</div>"
                f"{title_html}{link_html}</div>",
                unsafe_allow_html=True,
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
            if _is_low_confidence_classification_stop(state):
                st.warning(
                    "Classification needs a clearer product description before duty rates can be shown."
                )
            else:
                st.error(f"HITL required: {state.get('hitl_reason') or 'No reason provided'}")

        with st.expander("Pipeline Details", expanded=False):
            st.json(state)


def _render_message(
    msg: Dict[str, Any],
    *,
    show_clarification_actions: bool = False,
    widget_key_prefix: str = "msg",
) -> None:
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(msg.get("content", ""))
        if role == "assistant" and msg.get("state"):
            _render_assistant_details(
                msg["state"],
                show_clarification_actions=show_clarification_actions,
                widget_key_prefix=widget_key_prefix,
            )


def _pipeline_to_assistant_content(state: Dict[str, Any]) -> str:
    if state.get("error"):
        return f"Pipeline error: {state.get('error')}"
    return _render_structured_answer(state)


def main() -> None:
    _inject_styles()
    _ensure_state()
    _render_sidebar()

    st.title("TariffIQ Chat")
    st.caption("Conversational multi-agent RAG for US import tariff analysis")

    msgs = st.session_state.messages
    last_idx = len(msgs) - 1
    for i, msg in enumerate(msgs):
        st_obj = msg.get("state")
        show_actions = (
            msg.get("role") == "assistant"
            and st_obj
            and i == last_idx
            and (
                st_obj.get("clarification_needed")
                or _is_low_confidence_classification_stop(st_obj)
            )
        )
        _render_message(
            msg,
            show_clarification_actions=show_actions,
            widget_key_prefix=f"hist_{i}",
        )

    pending = st.session_state.pop(_PENDING_PIPELINE_QUERY, None)
    if pending:
        _run_pipeline_response(pending)

    prompt = st.chat_input("Ask a tariff question...")
    if not prompt:
        return

    _append_user_and_run_pipeline(prompt)


if __name__ == "__main__":
    main()
