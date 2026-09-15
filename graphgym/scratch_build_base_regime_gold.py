"""
One-off script: builds tstr_guide_vae_base_gold.parquet / trts_guide_vae_base_gold.parquet
for the GUIDE-VAE fleet *base* conditioning regime (months+weekdays only, no
day_befores), to compare against the existing tstr_guide_vae_gold.parquet /
trts_guide_vae_gold.parquet (which reflect the day_befores regime, since the
canonical processed_data/fleet_synthetic_gold/guide_vae_gold.parquet had
already been promoted to day_befores by the time scratch_build_hybrid_gold.py
ran on 2026-07-28). Exact same splicing logic as that script, just pointed at
a different source parquet and a "_base" output suffix so nothing existing is
overwritten.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

REAL_PATH = "/home/sagemaker-user/ssmd-validated/raw_data/fleet_gold_layer.parquet"
BASE_SYNTH_PATH = Path(
    "/tmp/claude-1000/-home-sagemaker-user-ssmd-validated/dacf0ea7-110d-49d8-b8c0-f12fa7c22d34/"
    "scratchpad/base_regime/guide_vae_base_gold.parquet"
)
OUT_DIR = Path("/home/sagemaker-user/pytorch_geometric/tstr_trts_gold")

N_RAW, SEQ_LEN = 35136, 96
TRAIN_SPLIT, VAL_SPLIT = 0.7, 0.85
N_SAMPLES = N_RAW - SEQ_LEN - 1
VAL_END = int(N_SAMPLES * VAL_SPLIT)
START = datetime(2024, 1, 1, tzinfo=timezone.utc)
END = datetime(2025, 1, 1, tzinfo=timezone.utc)
CUTOFF = START + timedelta(minutes=15 * VAL_END)

print(f"n_samples={N_SAMPLES} val_end={VAL_END} cutoff={CUTOFF.isoformat()}")

OUT_DIR.mkdir(parents=True, exist_ok=True)

gen = "guide_vae_base"
ids = sorted(pl.scan_parquet(BASE_SYNTH_PATH).select("user_id").unique().collect()["user_id"].to_list())
assert len(ids) == 48, f"{gen}: expected 48 ids, got {len(ids)}"

synth = pl.scan_parquet(BASE_SYNTH_PATH).filter(
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
