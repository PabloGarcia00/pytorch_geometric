"""
Per-user variant of calculate_metrics.py.

calculate_metrics.py pools every node (user) together before scoring, which
hides how performance varies across the fleet. This script reuses the same
discovery/inference pipeline but scores each user's test split individually,
so we can see the distribution (mean/std/min/max) of each metric across
users rather than a single fleet-wide number.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import custom_graphgym  # noqa — registers loaders/losses/metrics
import pandas as pd
import tqdm
from calculate_metrics import (
    CACHE_FIT_MODEL_TYPES,
    METRIC_LABELS,
    METRIC_ORDER,
    discover_manifest,
    disaggregate_test_set,
)
from custom_graphgym.metric.regression import DisaggregationMetrics


def compute_metrics_per_user(row: pd.Series) -> list[dict] | None:
    """
    Disaggregate `row`'s test set and score each user (node) independently.

    Returns one record per (user, target, metric), or None if the
    dataset/checkpoint for this run isn't available on this machine.
    """
    result = disaggregate_test_set(row)
    if result is None:
        return None
    true, pred, user_ids, _ = result
    true_np, pred_np = true.numpy(), pred.numpy()

    records = []
    for i, user_id in enumerate(user_ids):
        user_true, user_pred = true_np[i : i + 1], pred_np[i : i + 1]
        if not user_true[:, DisaggregationMetrics.TRUE_MASK, :].any():
            # no labeled timesteps for this user in the test split — nothing to score
            continue
        try:
            metrics = DisaggregationMetrics.all(user_true, user_pred)
        except Exception as e:
            print(f"  user {user_id} FAILED: {e}")
            continue
        for target, target_metrics in metrics.items():
            for metric, value in target_metrics.items():
                records.append(
                    {
                        "sweep": row["sweep"],
                        "run_name": row["run_name"],
                        "seed": row["seed"],
                        "user_id": user_id,
                        "target": target,
                        "metric": metric,
                        "value": value,
                    }
                )
    return records


if __name__ == "__main__":
    manifest = discover_manifest()
    print(f"Discovered {len(manifest)} runs across {manifest['sweep'].nunique()} sweeps")

    records = []
    runnable = manifest[
        manifest["ckpt_path"].notna()
        | manifest["model_type"].isin(CACHE_FIT_MODEL_TYPES)
    ]
    skipped = len(manifest) - len(runnable)
    if skipped:
        print(f"Skipping {skipped} runs — no checkpoint")

    for _, row in tqdm.tqdm(runnable.iterrows(), total=len(runnable), desc="disaggregating test sets (per user)"):
        try:
            user_records = compute_metrics_per_user(row)
        except Exception as e:
            print(f"FAILED  {row['sweep']}/{row['run_name']}/seed={row['seed']}: {e}")
            continue
        if user_records is None:
            continue
        records.extend(user_records)

    df_long = pd.DataFrame(records)
    if df_long.empty:
        print("No runs were disaggregated — nothing to report.")
    else:
        df_long.to_csv("notebooks/metrics_per_user_long.csv", index=False)

        # summary: distribution of each metric across users (seeds pooled in too)
        summary = df_long.groupby(["target", "sweep", "run_name", "metric"])["value"].agg(
            ["mean", "std", "min", "max"]
        )
        metric_order = [m for m in METRIC_ORDER if m in summary.index.get_level_values("metric")]
        summary = summary.reindex(
            pd.MultiIndex.from_product(
                [
                    summary.index.get_level_values("target").unique(),
                    summary.index.get_level_values("sweep").unique(),
                    summary.index.get_level_values("run_name").unique(),
                    metric_order,
                ],
                names=["target", "sweep", "run_name", "metric"],
            )
        ).dropna(how="all")
        summary = summary.rename(index=METRIC_LABELS, level="metric")
        summary.to_csv("notebooks/metrics_per_user_summary.csv")

        pd.set_option("display.width", 200)
        pd.set_option("display.max_columns", None)
        print(summary)
