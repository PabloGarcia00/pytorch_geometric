"""Transfer Sundael (physics disaggregator) onto the graphgym fleet gold layer.

Reproduces graphgym's node-filter + train/val/test split, maps the dual-read
meter channels onto Sundael's inputs, runs PVDisaggregator, and scores its
generation/consumption estimates against the gold-layer ground truth
(inverter_w = gross PV, load_w = gross load) on the test window.

Run with Sundael's venv (correct pandas<3, numba, etc.):
    /home/llan/projects/messm/sundael/.venv/bin/python scratch_sundael_transfer.py
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, "/home/llan/projects/messm/sundael/src")
from sundael import PVConfig, PVDisaggregator  # noqa: E402

GOLD = "/home/llan/projects/messm/pytorch_geometric/fleet_gold_layer.parquet"
COORDS = "/home/llan/projects/messm/exploratory-data-analysis/assets/zipcode_coordinate.csv"
OUT = Path("/home/llan/projects/messm/pytorch_geometric/graphgym/results/sundael_transfer")
OUT.mkdir(parents=True, exist_ok=True)

# graphgym CVAE-config knobs that define the test set
START = dt.datetime(2023, 4, 3, tzinfo=dt.timezone.utc)
END = START.replace(year=2026)
SEQ_LEN = 96
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.85
ACTIVITY_COLS = ["consumption_w", "generation_w", "inverter_w", "load_w"]
MAX_USERS = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # 0 = all


def log(*a):
    print("[sundael-transfer]", *a, flush=True)


# ----------------------------------------------------------------------------
# 1. Extract + reproduce graphgym node filter and split
# ----------------------------------------------------------------------------
def load_pivots(population: str = "seen"):
    """population: 'seen' = require_full_span survivors (graphgym's trained
    set); 'heldout' = the complement (native users dropped by the full-span
    filter, i.e. held_out_metrics.py's population)."""
    lf = (
        pl.scan_parquet(GOLD)
        .with_columns(pl.col("user_id").cast(pl.Utf8))
        .filter(pl.col("timestamp").is_between(START, END, closed="none"))
        .filter(pl.col("zipcode").is_not_null())
        .with_columns(pl.col("zipcode").cast(pl.Float64, strict=False).cast(pl.Int32))
    )
    # activity-trimmed last timestamp (matches graph_dataset._process_shared)
    last_ts = (
        lf.filter(pl.any_horizontal(pl.col(ACTIVITY_COLS).abs() > 0))
        .select(pl.col("timestamp").max())
        .collect()
        .item()
    )
    lf = lf.filter(pl.col("timestamp") <= last_ts)
    df = lf.sort(["timestamp", "user_id"]).collect()

    ids = sorted(df["user_id"].unique().to_list())
    ts = sorted(df["timestamp"].unique().to_list())
    log(f"raw: {len(ids)} users x {len(ts)} timesteps")

    def piv(col):
        return (
            df.pivot(values=col, index="timestamp", on="user_id")
            .sort("timestamp")
            .select(ids)
            .to_numpy()
        )  # [T, N]

    pivots = {c: piv(c) for c in ACTIVITY_COLS}
    # mask: every activity column present (True = supervised/available)
    mask = np.stack([~np.isnan(pivots[c]) for c in ACTIVITY_COLS]).all(0)  # [T,N]

    # require_full_span: active in first & last week (15-min -> 672 steps/week)
    T = mask.shape[0]
    steps_per_week = int(pd.Timedelta(weeks=1) / (pd.Timestamp(ts[1]) - pd.Timestamp(ts[0])))
    window = min(steps_per_week, T // 2)
    first_active = np.argmax(mask, axis=0)
    last_active = T - 1 - np.argmax(mask[::-1], axis=0)
    full_span = (first_active < window) & (last_active > T - 1 - window)
    # held-out users must still carry *some* supervised data to be scorable,
    # otherwise there's nothing to disaggregate/score against.
    has_data = mask.any(axis=0)
    if population == "heldout":
        keep = (~full_span) & has_data
    else:
        keep = full_span
    keep_idx = np.where(keep)[0]
    if MAX_USERS > 0:
        keep_idx = keep_idx[:MAX_USERS]
    log(f"[{population}] after span filter: {len(keep_idx)} users kept "
        f"(full-span={int(full_span.sum())}, native={int(has_data.sum())})")

    ids = [ids[i] for i in keep_idx]
    pivots = {c: v[:, keep_idx] for c, v in pivots.items()}
    mask = mask[:, keep_idx]

    # zipcode per user
    uz = (
        df.group_by("user_id").agg(pl.col("zipcode").first()).to_pandas().set_index("user_id")["zipcode"]
    )
    zips = uz.reindex(ids).fillna(-1).astype(int)

    # fleet-mean temperature series (Sundael uses one global temp series)
    temp = (
        df.group_by("timestamp")
        .agg(pl.col("temperature_2m_om_mean").mean())
        .sort("timestamp")
        .to_pandas()
        .set_index("timestamp")["temperature_2m_om_mean"]
        .to_numpy()
    )

    # split (mirrors graph_dataset._calc_splits): target index = idx + seq_len
    n_samples = T - SEQ_LEN - 1
    train_end = int(n_samples * TRAIN_SPLIT)
    val_end = int(n_samples * VAL_SPLIT)
    test_target_start = val_end + SEQ_LEN  # first supervised test *target* timestep
    log(f"split: train_end={train_end} val_end={val_end} test targets=[{test_target_start}:{T}) "
        f"({T - test_target_start} steps)")

    return dict(
        ids=ids, ts=pd.DatetimeIndex(ts), zips=zips, temp=temp, mask=mask,
        pivots=pivots, test_target_start=test_target_start,
    )


# ----------------------------------------------------------------------------
# 2. Build Sundael inputs
# ----------------------------------------------------------------------------
def build_inputs(D):
    ids, ts, zips = D["ids"], D["ts"], D["zips"]
    # [users x timesteps] frames
    net_con = pd.DataFrame(D["pivots"]["consumption_w"].T, index=ids, columns=ts)  # import
    net_gen = pd.DataFrame(D["pivots"]["generation_w"].T, index=ids, columns=ts)  # export
    area_mapping = {u: str(int(z)) for u, z in zips.items()}
    areas = sorted(set(area_mapping.values()))

    coords = pd.read_csv(COORDS).set_index("zipcode")[["latitude", "longitude"]]
    area_latlon = coords.reindex([int(a) for a in areas])
    area_latlon.index = areas
    area_latlon = area_latlon.dropna()
    log(f"{len(areas)} areas, {len(area_latlon)} with coords")

    temperature = pd.Series(D["temp"], index=ts)
    return net_con, net_gen, area_mapping, area_latlon, areas, temperature


def ones_reference(areas, ts):
    return pd.DataFrame(1.0, index=areas, columns=ts)  # clear-sky (no weather anchor)


# ----------------------------------------------------------------------------
# 3. Score
# ----------------------------------------------------------------------------
def metrics(pred, true, valid):
    p = pred[valid].astype(float)
    t = true[valid].astype(float)
    err = p - t
    rmse = np.sqrt(np.mean(err**2))
    mae = np.mean(np.abs(err))
    ss_res = np.sum(err**2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return rmse, mae, r2


def evaluate(D, result, tag):
    ts = D["ts"]
    ids = D["ids"]
    s = D["test_target_start"]
    inv = D["pivots"]["inverter_w"].T  # [N,T] gross PV truth
    load = D["pivots"]["load_w"].T  # [N,T] gross load truth
    valid = D["mask"].T  # [N,T]

    gen = result["generation"].reindex(index=ids).to_numpy()
    con = result["consumption"].reindex(index=ids).to_numpy()

    # restrict to test target window
    sl = slice(s, inv.shape[1])
    vmask = valid[:, sl]
    out = {}
    out["PV  (gen vs inverter_w)"] = metrics(gen[:, sl], inv[:, sl], vmask)
    out["Load(con vs load_w)"] = metrics(con[:, sl], load[:, sl], vmask)
    # naive baselines for context
    out["  naive PV=export"] = metrics(
        D["pivots"]["generation_w"].T[:, sl], inv[:, sl], vmask
    )
    out["  naive Load=import"] = metrics(
        D["pivots"]["consumption_w"].T[:, sl], load[:, sl], vmask
    )
    # daytime-only PV (where truth PV>0)
    day = vmask & (inv[:, sl] > 0)
    out["PV  daytime-only"] = metrics(gen[:, sl], inv[:, sl], day)

    print(f"\n===== METRICS [{tag}]  (test window, Watts, N={vmask.sum():,} obs) =====")
    print(f"{'target':32s} {'RMSE':>10s} {'MAE':>10s} {'R2':>8s}")
    for k, (rmse, mae, r2) in out.items():
        print(f"{k:32s} {rmse:10.1f} {mae:10.1f} {r2:8.3f}")
    return out


# ----------------------------------------------------------------------------
# per-population run
# ----------------------------------------------------------------------------
def run_population(population: str, suffix: str) -> list[str]:
    """Disaggregate one population (seen/heldout) under both reference regimes
    (clear-sky A, constructed-export B) and save generation predictions to
    pred_generation{suffix}_{clearsky,ref}.parquet. Returns the final
    coords-filtered user-id run set."""
    D = load_pivots(population=population)
    net_con, net_gen, area_mapping, area_latlon, areas, temperature = build_inputs(D)
    # drop areas without coords from the run set
    areas = list(area_latlon.index)
    keep_users = [u for u in net_con.index if area_mapping[u] in set(areas)]
    net_con, net_gen = net_con.loc[keep_users], net_gen.loc[keep_users]
    area_mapping = {u: area_mapping[u] for u in keep_users}
    D["ids"] = keep_users
    log(f"[{population}] final run set: {len(keep_users)} users")

    config = PVConfig(auto_infer_temp=False)

    # ---- Run A: clear-sky (all-ones) reference ----
    log(f"[{population}] RUN A: clear-sky reference (no weather anchor)")
    pvA = PVDisaggregator(
        pv_ratio=ones_reference(areas, D["ts"]),
        area_mapping=area_mapping,
        area_latlon=area_latlon,
        temperature=temperature,
        config=config,
    )
    resA = pvA.disaggregate(net_con, net_gen, use_sampling=True)
    evaluate(D, resA, f"{population} A: clear-sky ref")
    resA["generation"].to_parquet(OUT / f"pred_generation{suffix}_clearsky.parquet")

    # ---- Run B: constructed per-zip reference ratio (export vs theoretical) ----
    log(f"[{population}] RUN B: constructed reference from average export")
    # theoretical (clear-sky) generation per area from run A's fitted geometry:
    # ratio = area-avg actual export / area-avg clear-sky generation, clipped [0,1]
    genA = resA["generation"]
    ref_rows = {}
    for area in areas:
        members = [u for u in keep_users if area_mapping[u] == area]
        avg_export = net_gen.loc[members].mean(axis=0)
        avg_theo = genA.loc[members].mean(axis=0)
        ratio = (avg_export / avg_theo.replace(0, np.nan)).clip(0, 1).fillna(0)
        ref_rows[area] = ratio.values
    pv_ratioB = pd.DataFrame(ref_rows, index=D["ts"]).T  # [areas x ts]

    pvB = PVDisaggregator(
        pv_ratio=pv_ratioB,
        area_mapping=area_mapping,
        area_latlon=area_latlon,
        temperature=temperature,
        config=config,
    )
    resB = pvB.disaggregate(net_con, net_gen, use_sampling=True)
    evaluate(D, resB, f"{population} B: constructed ref")
    resB["generation"].to_parquet(OUT / f"pred_generation{suffix}_ref.parquet")

    return keep_users


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    import json

    # Seen = require_full_span survivors (graphgym's trained set) -> scored by
    # notebooks/calculate_metrics.py's val+test window. Suffix "" keeps the
    # original pred_generation_{clearsky,ref}.parquet filenames.
    seen_ids = run_population("seen", suffix="")
    # Held-out = native users the full-span filter dropped -> scored by
    # notebooks/held_out_metrics.py's population.
    heldout_ids = run_population("heldout", suffix="_heldout")

    (OUT / "sundael_membership.json").write_text(
        json.dumps({"seen_ids": seen_ids, "heldout_ids": heldout_ids}, indent=2)
    )
    log(f"done. predictions + membership saved to {OUT} "
        f"({len(seen_ids)} seen, {len(heldout_ids)} held-out)")


if __name__ == "__main__":
    main()
