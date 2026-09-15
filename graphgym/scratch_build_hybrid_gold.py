"""
One-off script: builds TSTR/TRTS hybrid gold-layer parquet files.

Splices each synthetic generator's 2024 data with the real 2024 data for the
same 48-household population, at the exact raw-timestep cutoff that
_calc_splits (custom_graphgym/loader/graph_dataset.py) uses as the val/test
boundary under seq_len=96, train_split=0.7, val_split=0.85 -- so the test
split (used for final TSTR/TRTS evaluation) is 100% from one source with no
window contamination from the other, while the train/val split (used for
fitting/early-stopping) is close to 100% from the other source (a ~95-sample
tail of val straddles the cutoff, an accepted minor boundary effect since
seq_len=96 windows must span it somewhere).

TSTR-<gen>: train+val = synthetic <gen>, test = real.
TRTS-<gen>: train+val = real, test = synthetic <gen>.

Also builds trtr_gold.parquet: real data for the same 48 households, capped
to exactly the same 2024 calendar year as the synthetic files. Needed
because the loader hardcodes an internal start_date+3y window regardless of
any cfg.earne_data.end_date setting (dead/unread field) -- pointing the TRTR
baseline at the full real file with only a filter_ids restriction would give
it ~2.3 years of real data (2024-2026) instead of the 1 year every other
experiment gets, breaking the matched-baseline comparison.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

REAL_PATH = "/home/sagemaker-user/ssmd-validated/raw_data/fleet_gold_layer.parquet"
SYNTH_DIR = Path("/home/sagemaker-user/ssmd-validated/processed_data/fleet_synthetic_gold")
OUT_DIR = Path("/home/sagemaker-user/pytorch_geometric/tstr_trts_gold")
GENERATORS = ["faraday", "energydiff", "guide_vae"]

N_RAW, SEQ_LEN = 35136, 96
TRAIN_SPLIT, VAL_SPLIT = 0.7, 0.85
N_SAMPLES = N_RAW - SEQ_LEN - 1
VAL_END = int(N_SAMPLES * VAL_SPLIT)
START = datetime(2024, 1, 1, tzinfo=timezone.utc)
END = datetime(2025, 1, 1, tzinfo=timezone.utc)
CUTOFF = START + timedelta(minutes=15 * VAL_END)

print(f"n_samples={N_SAMPLES} val_end={VAL_END} cutoff={CUTOFF.isoformat()}")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# trtr baseline: real data only, but capped to the same 48 households + same
# 2024 calendar year as every TSTR/TRTS experiment (see module docstring).
_fara_ids = sorted(
    pl.scan_parquet(SYNTH_DIR / "faraday_gold.parquet").select("user_id").unique().collect()["user_id"].to_list()
)
trtr = pl.scan_parquet(REAL_PATH).filter(
    pl.col("user_id").is_in(_fara_ids) & (pl.col("timestamp") >= START) & (pl.col("timestamp") < END)
).sort(["timestamp", "user_id"])
trtr_out = OUT_DIR / "trtr_gold.parquet"
trtr.collect().write_parquet(trtr_out)
_df = pl.scan_parquet(trtr_out)
_counts = _df.group_by("user_id").agg(pl.len().alias("n")).collect()
print(f"  trtr: rows={_df.select(pl.len()).collect().item()} n_ids={_df.select('user_id').unique().collect().height} "
      f"all_ids_full_coverage={(_counts['n'] == N_RAW).all()}")

for gen in GENERATORS:
    synth_path = SYNTH_DIR / f"{gen}_gold.parquet"
    ids = sorted(
        pl.scan_parquet(synth_path).select("user_id").unique().collect()["user_id"].to_list()
    )
    assert len(ids) == 48, f"{gen}: expected 48 ids, got {len(ids)}"

    synth = pl.scan_parquet(synth_path).filter(
        pl.col("user_id").is_in(ids) & (pl.col("timestamp") >= START) & (pl.col("timestamp") < END)
    )
    real = pl.scan_parquet(REAL_PATH).filter(
        pl.col("user_id").is_in(ids) & (pl.col("timestamp") >= START) & (pl.col("timestamp") < END)
    )

    synth_early = synth.filter(pl.col("timestamp") < CUTOFF)
    synth_late = synth.filter(pl.col("timestamp") >= CUTOFF)
    real_early = real.filter(pl.col("timestamp") < CUTOFF)
    real_late = real.filter(pl.col("timestamp") >= CUTOFF)

    tstr = pl.concat([synth_early, real_late]).sort(["timestamp", "user_id"])
    trts = pl.concat([real_early, synth_late]).sort(["timestamp", "user_id"])

    tstr_out = OUT_DIR / f"tstr_{gen}_gold.parquet"
    trts_out = OUT_DIR / f"trts_{gen}_gold.parquet"
    tstr.collect().write_parquet(tstr_out)
    trts.collect().write_parquet(trts_out)

    for name, path in [("tstr", tstr_out), ("trts", trts_out)]:
        df = pl.scan_parquet(path)
        n_rows = df.select(pl.len()).collect().item()
        n_ids = df.select("user_id").unique().collect().height
        counts = df.group_by("user_id").agg(pl.len().alias("n")).collect()
        ok = (counts["n"] == N_RAW).all()
        print(f"  {gen} {name}: rows={n_rows} n_ids={n_ids} all_ids_full_coverage={ok}")

print("done")
