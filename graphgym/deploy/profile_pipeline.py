"""One-off profiling script: times each phase of run_disaggregation()'s
pipeline exactly as Live mode calls it (rolling last-24h window), to find
where the ~30s refresh time actually goes. Not part of the app -- run
manually, e.g. inside the container:
    docker exec <container> .venv/bin/python deploy/profile_pipeline.py <p1_id>
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import cache
import disaggregate_pv as dpv
import fetch_datahub as dh

p1_id = sys.argv[1] if len(sys.argv) > 1 else None

timings = {}


def timed(label, fn, *a, **kw):
    t0 = time.time()
    result = fn(*a, **kw)
    timings[label] = time.time() - t0
    return result


client = dh.DatahubClient(dh.load_api_key())
sites = timed("get_devices (cached or live GET)", lambda: dh.build_sites(cache.get_devices(client)))
if p1_id is None:
    site = next(s for s in sites if s.inverter_ids)
    p1_id = site.p1_id
else:
    site = next(s for s in sites if s.p1_id == p1_id)
print(f"site: {p1_id}")

model, device = timed("load_model (cached after first call)", dpv.load_model)
transform = timed("Transform.load", dpv.Transform.load, dpv.TRANSFORM_PATH)

eval_end = pd.Timestamp.now(tz="UTC").as_unit("ns")
eval_start = pd.Timestamp(eval_end - pd.Timedelta(hours=24)).as_unit("ns").floor("5min")
fetch_start = eval_start - pd.Timedelta(hours=24)

p1_df = timed(
    "fetch p1_graph (API or cache)",
    cache.fetch_cached, client, "p1_graph", client.get_graph, [site.p1_id], site.p1_id, fetch_start, eval_end, 1, 0.5,
)
print(f"  -> {len(p1_df)} rows")

seed_5min = timed("resample_median_5min", dpv.resample_median_5min, p1_df)
one_min = timed("to_one_minute_grid", dpv.to_one_minute_grid, p1_df)

data_list, target_ts = timed(
    "build_data_list (window construction, per-tick Python loop)",
    dpv.build_data_list, seed_5min, one_min, transform, eval_start, eval_end,
)
print(f"  -> {len(data_list)} target windows")

pred_df = timed("run_inference (batched model forward pass)", dpv.run_inference, model, device, data_list, target_ts, transform)

if site.inverter_ids:
    raw_solar_df = timed(
        "fetch raw_solar_graph (API or cache)",
        cache.fetch_cached, client, "raw_solar_graph", client.get_raw_solar_graph, site.inverter_ids, site.p1_id, eval_start, eval_end, 1, 0.5,
    )
    merged = timed("merge_raw_solar", dpv.merge_raw_solar, pred_df, raw_solar_df)
else:
    merged = pred_df

merged = timed("final merge with one_min (display columns)", lambda: merged.merge(one_min[["timestamp", "consumption_w", "generation_w"]], on="timestamp", how="left"))

print("\n--- timings (seconds) ---")
total = sum(timings.values())
for label, secs in sorted(timings.items(), key=lambda kv: -kv[1]):
    pct = 100 * secs / total if total else 0
    print(f"  {secs:7.3f}s  ({pct:5.1f}%)  {label}")
print(f"  {total:7.3f}s  (100.0%)  TOTAL")
