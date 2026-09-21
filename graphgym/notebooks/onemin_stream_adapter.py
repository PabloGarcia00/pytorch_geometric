"""
onemin_stream_adapter.py

Data source for the real-time (1-minute) cadence robustness study -- see
~/.claude/plans/squishy-wobbling-bunny.md ("Robustness Test Plan for Input
Cadence Mismatch at Inference Time"). Exposes one function per data path so
custom_graphgym/eval/streaming_harness.py can consume either without caring
which is active:

  - synthesize_oneminute(): a synthetic 1-min-cadence CONSUMPTION stream,
    upsampled from the existing 5-min gold-layer data via linear
    interpolation. A smoothness PROXY only -- kept for households that
    don't have a matching raw file (see below), and for quick harness
    iteration. Never report its output as a final finding.

  - load_real_oneminute(): the REAL-data path -- now implemented (Phase 0
    resolved 2026-09-18). The raw source behind `data_v2/Smartmeter-P1-data`
    (referenced in exploratory-data-analysis/config.yaml) was located at
    <EFS>/raw-data-earne/earne_v2-data/Smartmeter-P1-data/ (uploaded by the
    user) -- 119 files, ";"-delimited, comma-decimal, columns
    [MessageTimestamp, IMPORT_KW, EXPORT_KW, GAS_USAGE_M3]. Confirmed
    genuinely 1-minute-native two ways: (1) raw timestamps are ~60s apart
    (spot-checked), and (2) exploratory-data-analysis/config.yaml's own
    `energy.streams.V2_sm.native_cadence_minutes: 1` documents this
    explicitly -- it predates resample_to_cadence() as required.

    Reuses exploratory-data-analysis/src/ingestion.py's ingest() directly
    (unit conversion, KWH-register aggregation, best-signal selection,
    resampling) rather than re-deriving watts from IMPORT_KW/EXPORT_KW by
    hand -- that logic is already written, tested, and is what actually
    built the gold layer this project trains on.

    IMPORTANT CORRECTION to this module's earlier assumption: the plan
    text says "PV/generation is natively 5-minute... never refreshed at
    1-minute cadence." That's true of the PV *target* (inverter_w, read
    from the separate Solar-data/V2_inv stream -- genuinely 5-min native,
    confirmed via config.yaml's `V2_inv.native_cadence_minutes: 5` and the
    raw file's own 5-min-spaced timestamps). It is NOT true of dual-read's
    *generation input channel* (generation_w, derived from this SAME
    V2_sm file's EXPORT_KW column) -- that's 1-minute-native too, per the
    same config entry used for consumption. So a fully-real mixed-cadence
    window should refresh BOTH input channels at 1-min in their fine
    positions, not just consumption -- only the PV *target* is capped at
    5-min. cadence_robustness_eval_real.py does this; the synthetic-only
    harness in cadence_robustness_eval.py still holds generation constant
    since synthesize_oneminute() only ever modeled consumption.

    KNOWN GAP, not yet resolved: which of these 119 raw devices
    corresponds to which integer `user_id` in fleet_gold_layer_5min.parquet
    is not recoverable from currently-available artifacts (the gold layer
    keeps only a 2-digit `zipcode`, no MAC/device-id column; the Snakemake
    workflow that assigns user_id from the raw file list isn't present in
    this checkout). So results using this real data are reported against
    an out-of-training-population household -- valid for architectures
    with node-agnostic shared weights (MLP/LSTM/CVAE), not for GNN
    (fixed training-population graph) or Linear/SVR/KNN (one fit per
    training household, undefined for a novel one).

Only the CONSUMPTION+GENERATION channels are ever loaded at 1-min from
this module. PV ground truth for accuracy scoring always comes from the
separate 5-min inverter stream (never from here) -- see
load_real_pv_ground_truth().

Usage:
    from onemin_stream_adapter import load_real_oneminute, load_real_pv_ground_truth

    one_min = load_real_oneminute("08F9E07960A5-P1", cadence_minutes=1)
    five_min = load_real_oneminute("08F9E07960A5-P1", cadence_minutes=5)
    pv_5min = load_real_pv_ground_truth("FDF49722")
"""
import sys
from pathlib import Path

import pandas as pd

EDA_ROOT = Path("/home/sagemaker-user/exploratory-data-analysis")
sys.path.insert(0, str(EDA_ROOT))
from src.ingestion import ingest  # noqa: E402

GOLD_LAYER_5MIN = (
    "/mnt/custom-file-systems/efs/fs-0e26e28a2c1df7f40/eda-gold-layer/"
    "fleet_gold_layer_5min.parquet"
)

RAW_ROOT = Path(
    "/home/sagemaker-user/custom-file-systems/efs/fs-0e26e28a2c1df7f40/"
    "raw-data-earne/earne_v2-data"
)
SMARTMETER_DIR = RAW_ROOT / "Smartmeter-P1-data"
SOLAR_DIR = RAW_ROOT / "Solar-data"
META_DATA_PATH = RAW_ROOT / "meta_data.csv"

# exploratory-data-analysis/config.yaml's energy.streams entries -- copied
# here rather than re-parsing that YAML, since only these two stream_cfg
# dicts are needed and re-parsing would pull in unrelated weather/cleaning
# config this module doesn't use.
_V2_SM_CFG = {
    "native_cadence_minutes": 1,
    "kw_cols": {"consumption": "IMPORT_KW", "generation": "EXPORT_KW"},
}
_V2_INV_CFG = {
    "native_cadence_minutes": 5,
    "w_cols": {"inverter": "EXPORT_W"},
    "kwh_cols": {"inverter": ["EXPORT_KWH"]},
}
_MAX_PLAUSIBLE_W = 55000.0
_MIN_NONZERO_W = 15.0


def synthesize_oneminute(
    five_min_df: pd.DataFrame,
    value_col: str = "consumption_w",
    time_col: str = "timestamp",
    user_col: str = "user_id",
) -> pd.DataFrame:
    """Upsample one or more households' 5-min consumption series to 1-min
    via linear interpolation. Synthetic smoothness proxy only -- see module
    docstring. Returns a DataFrame with [time_col, user_col, value_col] at
    1-min spacing, spanning the same date range as the input.

    Deliberately linear, not cubic/shape-preserving: cubic risks overshoot
    on step-like meter reads that real 1-min data may actually have, which
    would misrepresent the synthetic stream as smoother/more well-behaved
    than reality.
    """
    if five_min_df.empty:
        return five_min_df.copy()

    parts = []
    for user_id, group in five_min_df.groupby(user_col, sort=False):
        g = group[[time_col, value_col]].sort_values(time_col).set_index(time_col)
        one_min_index = pd.date_range(
            g.index.min(), g.index.max(), freq="1min", tz=g.index.tz
        )
        g_upsampled = g.reindex(g.index.union(one_min_index)).interpolate(
            method="linear"
        )
        g_upsampled = g_upsampled.reindex(one_min_index)
        g_upsampled[user_col] = user_id
        parts.append(g_upsampled.reset_index(names=time_col))

    return pd.concat(parts, ignore_index=True)


def _smartmeter_path(device_id: str) -> Path:
    matches = list(SMARTMETER_DIR.glob(f"{device_id}_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No Smartmeter-P1-data file found for {device_id!r}")
    return matches[0]


def _solar_path(solar_id: str) -> Path:
    matches = list(SOLAR_DIR.glob(f"({solar_id})-*.csv"))
    if not matches:
        raise FileNotFoundError(f"No Solar-data file found for {solar_id!r}")
    return matches[0]


def solar_id_for(device_id: str) -> str:
    """Look up the paired Solar-ID for a Smartmeter-ID via meta_data.csv
    (tab-delimited: Solar-ID, Smartmeter-ID, Regiocode)."""
    meta = pd.read_csv(META_DATA_PATH, sep="\t")
    row = meta[meta["Smartmeter-ID"] == device_id]
    if row.empty:
        raise KeyError(f"{device_id!r} not found in {META_DATA_PATH}")
    return row.iloc[0]["Solar-ID"]


def load_real_oneminute(device_id: str, cadence_minutes: int = 1) -> pd.DataFrame:
    """Real smartmeter consumption+generation for one raw device, ingested
    via exploratory-data-analysis's own ingest() (Stage-1 pipeline: KWH
    register aggregation, unit conversion, best-signal selection,
    median-resampling with phase alignment -- identical logic to what
    built fleet_gold_layer_5min.parquet, applied here to a device outside
    that population). See module docstring for the known device_id ->
    training-population user_id gap.

    cadence_minutes=1 returns the genuinely-native, un-resampled series
    (resample_to_cadence() with a 1-min rule is a no-op reshape, not a
    real aggregation, since the source is already ~1-min-spaced);
    cadence_minutes=5 returns the same underlying data resampled the
    same way the gold layer's 5-min columns were built.

    Returns columns [timestamp, consumption_w, generation_w].
    """
    path = _smartmeter_path(device_id)
    df, _report = ingest(
        paths=[str(path)],
        stream_key="V2_sm",
        stream_cfg=_V2_SM_CFG,
        time_col="MessageTimestamp",
        global_time_col="timestamp",
        cadence_minutes=cadence_minutes,
        user_id=device_id,
        max_plausible_w=_MAX_PLAUSIBLE_W,
        min_nonzero_w=_MIN_NONZERO_W,
    )
    return df


def load_real_pv_ground_truth(solar_id: str, cadence_minutes: int = 5) -> pd.DataFrame:
    """Real PV (inverter) ground truth for one raw device, from the
    separate Solar-data/V2_inv stream -- genuinely 5-min native (confirmed
    via config.yaml and the raw file's own 5-min-spaced timestamps), so
    there is no finer-than-5-min version of this to request. Returns
    columns [timestamp, inverter_w]."""
    path = _solar_path(solar_id)
    df, _report = ingest(
        paths=[str(path)],
        stream_key="V2_inv",
        stream_cfg=_V2_INV_CFG,
        time_col="MessageTimestamp",
        global_time_col="timestamp",
        cadence_minutes=cadence_minutes,
        user_id=solar_id,
        max_plausible_w=_MAX_PLAUSIBLE_W,
        min_nonzero_w=_MIN_NONZERO_W,
    )
    return df
