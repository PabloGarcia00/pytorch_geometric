"""Streamlit app: DataHub API -> SQLite cache -> CVAE disaggregation ->
interactive comparison of model-predicted PV against earn-e's own processed
PV estimate and raw inverter telemetry.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import os

# Must be set before numpy/pandas/torch are first imported (below) to have
# any effect -- they're read once by the native OpenMP/MKL runtime at
# load time. Part of the mitigation for a segfault observed under a live
# `streamlit run` server (never reproduced standalone or under AppTest);
# see disaggregate_pv.py's torch.set_num_threads(1) for the other half.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# `import custom_graphgym` (via disaggregate_pv, below) transitively pulls
# in lightning -> torchmetrics -> torchmetrics/utilities/plot.py, which
# imports matplotlib at module level -- confirmed via import tracing.
# matplotlib then auto-selects an interactive GUI backend (TkAgg on this
# machine). We never use matplotlib ourselves (all charts are Plotly), and
# Tkinter is not thread-safe -- a classic segfault source when touched off
# the main thread, which is exactly what Streamlit's script-runner is.
# Force a headless backend before matplotlib is ever imported.
os.environ.setdefault("MPLBACKEND", "Agg")

# faulthandler: if this still crashes, dumps each thread's Python stack to
# stderr at the moment of the fatal signal -- turns a silent segfault into
# an actual lead instead of another guess.
import faulthandler
faulthandler.enable()

from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import cache
import disaggregate_pv as dpv
import fetch_datahub as dh

# Categorical palette, dark-mode steps (validated, fixed order -- see dataviz
# skill reference). Entity -> color is fixed per chart, not cycled.
PV_COLOR = "#d95926"       # orange -- model-predicted PV
PV_FILL = "rgba(217, 89, 38, 0.15)"
TRUE_PV_COLOR = "#e66767"  # red -- true PV (raw inverter telemetry; the only
                           # ground-truth source this pipeline trusts -- see
                           # disaggregate_pv.run_disaggregation's docstring)
IMPORT_COLOR = "#3987e5"   # blue -- P1 import register
EXPORT_COLOR = "#d95926"   # orange -- P1 export register (separate chart, no clash with PV_COLOR)

# Chrome (backgrounds/gridlines) matches the app's grey/black theme in
# .streamlit/config.toml -- only the data-series colors above carry hue.
SURFACE = "#1a1a19"
GRID = "#3a3a38"


def _dark_chart_layout(fig: go.Figure, **kwargs) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(color="#c3c2b7"),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        **kwargs,
    )
    return fig


st.set_page_config(page_title="PV Disaggregation", layout="wide")
st.markdown(
    "<style>.block-container { padding-top: 1.5rem; }</style>",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model + transform...")
def get_model():
    model, device = dpv.load_model()
    transform = dpv.Transform.load(dpv.TRANSFORM_PATH)
    return model, device, transform


@st.cache_resource
def get_client():
    return dh.DatahubClient(dh.load_api_key())


@st.cache_data(ttl=3600, show_spinner="Loading site list...")
def get_sites() -> list[dict]:
    client = get_client()
    sites = dh.build_sites(cache.get_devices(client))
    return [{"p1_id": s.p1_id, "n_inverters": len(s.inverter_ids)} for s in sites]


def pv_comparison_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.concat([df["timestamp"], df["timestamp"][::-1]]),
        y=pd.concat([df["pv_pred_q90_w"], df["pv_pred_q10_w"][::-1]]),
        fill="toself", fillcolor=PV_FILL, line=dict(color="rgba(0,0,0,0)"),
        name="PV (model) q10-q90", hoverinfo="skip", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["pv_pred_q50_w"], mode="lines",
        line=dict(color=PV_COLOR, width=2), name="PV (model)",
    ))
    # True PV is optional (site may have no inverter) -- omit the trace
    # entirely rather than plotting an empty/all-NaN scatter with a dead
    # legend entry. Raw solar telemetry is the only ground truth used here
    # (see run_disaggregation's docstring for why the derived earn-e figure
    # was dropped) -- kept as unconnected markers since it's genuinely
    # sparse/irregular; a line would falsely imply interpolation.
    if df["raw_solar_w"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["raw_solar_w"], mode="markers",
            marker=dict(color=TRUE_PV_COLOR, size=5, symbol="x"), name="PV (true, raw solar)",
        ))
    return _dark_chart_layout(
        fig,
        yaxis_title="PV output (W)", xaxis_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=40, b=10), height=440,
        hovermode="x unified",
    )


def input_signal_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["consumption_w"], mode="lines", line=dict(color=IMPORT_COLOR, width=1.5), name="Import"))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["generation_w"], mode="lines", line=dict(color=EXPORT_COLOR, width=1.5), name="Export"))
    return _dark_chart_layout(
        fig,
        yaxis_title="P1 meter (W)", xaxis_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=40, b=10), height=300,
        hovermode="x unified",
    )


def render_results(merged: pd.DataFrame, site_label: str, key_prefix: str) -> None:
    st.subheader(f"Results -- {site_label}")

    mae_raw, n_raw = dpv.comparison_metrics(merged)
    c1, c2 = st.columns(2)
    c1.metric("MAE vs true PV (raw solar)", f"{mae_raw:.0f} W" if n_raw else "n/a", help=f"{n_raw} matched samples")
    c2.metric("Predicted timesteps", f"{len(merged)}")

    st.plotly_chart(pv_comparison_chart(merged), use_container_width=True, key=f"{key_prefix}_pv_chart")

    st.subheader("P1 input signals (Import / Export)")
    st.plotly_chart(input_signal_chart(merged), use_container_width=True, key=f"{key_prefix}_input_chart")

    st.subheader("Raw comparison table")
    # Internal column names (consumption_w/generation_w) match the
    # checkpoint's own Transform registry keys -- renamed only here, at
    # the display boundary, to the earn-e field vocabulary (import/export).
    display_df = merged.rename(columns={"consumption_w": "import_w", "generation_w": "export_w"})
    st.dataframe(display_df, use_container_width=True, key=f"{key_prefix}_table")
    st.download_button(
        "Download CSV", display_df.to_csv(index=False).encode(),
        file_name=f"{site_label}_pv_disaggregation.csv", mime="text/csv",
        key=f"{key_prefix}_download",
    )


@st.fragment(run_every="60s")
def live_panel(p1_id: str, sleep: float, mc_samples: int) -> None:
    """Reruns itself every 60s (matching the SM stream's own 1-min native
    cadence -- polling faster wouldn't surface new real data) via
    st.fragment, independent of the rest of the page. Always a rolling 24h
    window ending "now" -- the incremental cache (cache.py's
    _missing_ranges) means each tick only fetches the new tail from the
    live API, not the whole 24h again."""
    eval_end = pd.Timestamp.now(tz="UTC").as_unit("ns")
    eval_start = (eval_end - pd.Timedelta(hours=24)).as_unit("ns")
    try:
        client = get_client()
        site = dpv.get_site(client, p1_id)
        model, device, transform = get_model()
        merged = dpv.run_disaggregation(client, model, device, transform, site, eval_start, eval_end, sleep, mc_samples)
    except ValueError as e:
        st.error(str(e))
        return
    st.caption(f"Live -- last updated {pd.Timestamp.now(tz='UTC').strftime('%H:%M:%S')} UTC, refreshes every 60s")
    render_results(merged, p1_id, key_prefix="live")


st.title("SM -> PV Disaggregation")
st.caption("res_eval_cvae_5min_dualmask -- P1 net-metering registers in, disaggregated PV out.")

with st.sidebar:
    st.header("Controls")
    sites = get_sites()
    if not sites:
        st.error("No sites found.")
        st.stop()
    labels = {
        s["p1_id"]: (
            f'{s["p1_id"]}  ({s["n_inverters"]} inverter{"s" if s["n_inverters"] != 1 else ""})'
            if s["n_inverters"] else f'{s["p1_id"]}  (no PV ground truth)'
        )
        for s in sites
    }
    p1_id = st.selectbox("Household / node", options=list(labels), format_func=lambda k: labels[k])

    # Read before the toggle itself runs (below) so Run/date-picker can be
    # disabled based on last known state without waiting a render behind.
    live = st.session_state.get("live", False)

    today = datetime.now(timezone.utc).date()
    selected_day = st.date_input("Day (UTC)", value=today, max_value=today, disabled=live)

    sleep = st.slider("Seconds between API requests", 0.0, 2.0, 0.5, 0.1)

    mc_on = st.toggle("Monte Carlo uncertainty", value=False)
    # Measured cost: the BiLSTM encode (the expensive part) runs once
    # regardless of sample count -- only the small decode heads + sampling
    # scale with it, so 200 samples cost about the same as the default
    # analytic path. See disaggregate_pv.run_disaggregation's mc_samples doc.
    mc_samples = st.number_input("Samples", min_value=2, max_value=2000, value=200, step=50, disabled=not mc_on) if mc_on else 0

    run = st.button("Run", type="primary", use_container_width=True, disabled=live)
    live = st.toggle("Live (refresh every 60s)", value=live, key="live")
    if live:
        st.caption("Live ignores the date picker -- always the rolling last 24h ending now.")

if live:
    live_panel(p1_id, sleep, mc_samples)
else:
    if run:
        # Always exactly a 24h window (.as_unit("ns") pins a fixed resolution --
        # a bare datetime.date otherwise constructs a 's'-resolution Timestamp,
        # which mismatched the 'us'-resolution Timestamps coming back through
        # the SQLite cache and broke comparisons/merges downstream).
        if selected_day == today:
            # Today isn't complete yet -- roll the window back from the most
            # recent available timestep instead of anchoring to midnight.
            eval_end = pd.Timestamp.now(tz="UTC").as_unit("ns")
            eval_start = (eval_end - pd.Timedelta(hours=24)).as_unit("ns")
        else:
            eval_start = pd.Timestamp(selected_day, tz="UTC").as_unit("ns")
            eval_end = (eval_start + pd.Timedelta(hours=24)).as_unit("ns")
        with st.spinner("Fetching + running inference..."):
            try:
                client = get_client()
                site = dpv.get_site(client, p1_id)
                model, device, transform = get_model()
                merged = dpv.run_disaggregation(client, model, device, transform, site, eval_start, eval_end, sleep, mc_samples)
            except ValueError as e:
                st.error(str(e))
                st.stop()
        st.session_state["result"] = merged
        st.session_state["result_site"] = p1_id

    result = st.session_state.get("result")
    if result is not None:
        render_results(result, st.session_state.get("result_site"), key_prefix="run")
    else:
        st.info("Pick a household and date range, then click Run.")
