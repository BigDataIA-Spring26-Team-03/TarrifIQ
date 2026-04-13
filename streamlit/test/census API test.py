"""
Trial Streamlit app for exploring Census International Trade API data.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Ensure repo root is importable when run via Streamlit.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingestion.census_client import (
    _infer_commodity_and_level,
    format_rows_for_display,
    get_trade_trend,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Census API Test", layout="wide")
st.title("Census API Test")
st.caption("Quick explorer for HS2/HS4/HS6/HS10 Census import flows")

with st.sidebar:
    st.header("Inputs")
    hts_code = st.text_input("HTS / HS code", value="8471", help="Try 2, 4, or 6 digits")

    period_unit = st.radio("Period type", ["Months", "Years"], horizontal=True)
    if period_unit == "Months":
        period_count = st.number_input("Number of months", min_value=1, max_value=60, value=12, step=1)
        months = int(period_count)
    else:
        year_count = st.number_input("Number of years", min_value=1, max_value=5, value=1, step=1)
        months = int(year_count) * 12

    run = st.button("Fetch Census Data", type="primary")

commodity, comm_lvl = _infer_commodity_and_level(hts_code)
col1, col2, col3 = st.columns(3)
col1.metric("Input code", hts_code or "-")
col2.metric("Normalized commodity", commodity or "-")
col3.metric("Requested level", comm_lvl)

if run:
    with st.spinner("Fetching live Census data..."):
        trend = get_trade_trend(hts_code=hts_code, months=months)

    if not trend:
        st.warning("No trend data returned.")
        st.stop()

    st.success(f"Fetched {len(trend)} monthly snapshots.")

    monthly_summary: list[dict[str, Any]] = []
    country_rows: list[dict[str, Any]] = []
    fallback_count = 0

    for snap in trend:
        ym = snap.get("time", "")
        rows = snap.get("rows") or []
        note = snap.get("note", "")
        if note:
            fallback_count += 1
        monthly_summary.append(
            {
                "time": ym,
                "resolved_level": snap.get("resolved_level", snap.get("comm_lvl", "")),
                "comm_lvl": snap.get("comm_lvl", ""),
                "normalized_code": snap.get("hts_code", ""),
                "country_row_count": len(rows),
                "note": note,
            }
        )

        for r in rows:
            if not isinstance(r, dict):
                continue
            country_rows.append(
                {
                    "time": ym,
                    "country_code": r.get("CTY_CODE", ""),
                    "country_name": r.get("CTY_NAME", ""),
                    "gen_val_mo": r.get("GEN_VAL_MO", ""),
                    "gen_val_yr": r.get("GEN_VAL_YR", ""),
                    "con_val_mo": r.get("CON_VAL_MO", ""),
                    "cal_dut_mo": r.get("CAL_DUT_MO", ""),
                    "cal_dut_yr": r.get("CAL_DUT_YR", ""),
                    "dut_val_mo": r.get("DUT_VAL_MO", ""),
                    "effective_tariff_rate": r.get("effective_tariff_rate"),
                    "commodity": r.get("I_COMMODITY", ""),
                    "description": r.get("I_COMMODITY_LDESC", ""),
                    "resolved_level": snap.get("resolved_level", snap.get("comm_lvl", "")),
                    "comm_lvl": snap.get("comm_lvl", ""),
                    "note": note,
                }
            )

    resolved_levels = sorted(
        {
            str(snap.get("resolved_level", snap.get("comm_lvl", "")))
            for snap in trend
            if snap.get("resolved_level", snap.get("comm_lvl", ""))
        }
    )
    col_a, col_b = st.columns(2)
    col_a.metric("Levels returned", ", ".join(resolved_levels) if resolved_levels else "-")
    col_b.metric("Fallback months", fallback_count)

    st.subheader("Monthly Snapshot Summary")
    st.dataframe(monthly_summary, use_container_width=True)

    st.subheader("Country-Level Rows (flattened)")
    if country_rows:
        display_rows = format_rows_for_display(country_rows)
        df = pd.DataFrame(display_rows)
        rate_col = "Effective Tariff Rate (%)"

        def _highlight_rate(row: pd.Series) -> list[str]:
            if rate_col in row and pd.notna(row[rate_col]):
                return ["background-color: #e8f5e9"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df.style.apply(_highlight_rate, axis=1),
            use_container_width=True,
            height=420,
        )

        rates = [
            float(v)
            for v in pd.to_numeric(df.get(rate_col), errors="coerce").dropna().tolist()
        ] if rate_col in df.columns else []
        if rates:
            avg_rate = round(sum(rates) / len(rates), 2)
            st.caption(
                f"Average effective tariff rate across all countries this period: {avg_rate}%"
            )
        else:
            st.caption(
                "Average effective tariff rate across all countries this period: N/A"
            )

        st.info(
            "Effective Tariff Rate = Duties Actually Collected / Dutiable Value × 100.\n"
            "This is the real tariff being paid at the border — not the stated HTS rate.\n"
            "A gap between the stated rate (from USITC) and this number indicates\n"
            "importers are using exclusions, FTA preferences, or other duty reduction mechanisms."
        )
    else:
        st.info("No country rows returned for this input.")

    with st.expander("Raw response payload"):
        st.json(trend)
else:
    st.info("Set code + period, then click 'Fetch Census Data'.")
