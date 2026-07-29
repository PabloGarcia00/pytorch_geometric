# Journal extension: grounding notes against the actual system

This corrects the generic outline against what we actually built and ran in this repo. The
outline was written without visibility into our code, so several of its proposals describe a
*different, hypothetical* system. Below: what's real, what needs correcting, and what's an open
decision before this becomes a paper plan.

## Anchor: what the SEST 2026 paper actually measured

`graphgym/results.tex` (checked into the repo since the first commit) is the real published
results table, and it's a useful anchor for terminology: columns are `W+`/`W-` (weather
on/off) × `MP0`/`MP3` (`physics_weight ∈ {0.0, 0.3}`), rows are metrics (RMSE, MAE, R², EFE,
Pinball Q50, Coverage, Sharpness, **Net RMSE**), stratified by "solar penetration bin"
(0/25/50/75/100%/global). The presence of **Net RMSE** (`load_pred - pv_pred` vs. true net
demand) confirms the published model predicted **load and PV jointly** — the physics constraint
was the load-pv=net_demand identity, which only makes sense when both targets are predicted.

This matters because our recent work (this session) predicts **PV only** by default
(`cfg.model.predict_targets = ["pv"]`, `custom_graphgym/target_utils.py`) across all 7 models —
a change made before this session, for the baseline-comparison campaign. Under PV-only, the old
load-pv=net_demand physics term is undefined (needs `load_pred`), so it silently became a no-op.
**We didn't just add physics constraints to new models — we found this gap and redesigned the
physics loss to work under PV-only prediction, which is a stronger, more general formulation than
the original paper's.** That's a real narrative point for the paper, not just an implementation
detail.

## Actual system (correcting the outline's generic "ST-GNN")

**Models** (all sharing one dataset, one metric suite, one train/eval harness):
- `earne_network` (the actual GNN — GATConv message passing, `custom_graphgym/network/earne_network.py`)
- 5 non-GNN baselines added this session, all per-node (no message passing), registered as
  standalone `@register_network`s: `baseline_mlp` (per-node MLP temporal encoder), `baseline_lstm`
  (BiLSTM), `baseline_cvae` (generative BiLSTM+VAE, Beta-distributed PV head), `baseline_knn`
  (per-node analog-ensemble lookup, non-gradient), `baseline_svr` (kernel quantile regression via
  Random Fourier Features approximating an RBF kernel — *not* exact dual-form SVR, which has no
  native quantile output; see `custom_graphgym/network/baseline_svr.py`'s docstring), `baseline_linear`
  (per-node linear quantile regression).
- **ST-SGCCaps is excluded from every result in this whole campaign** — its output is a point
  prediction `[N,2]`, incompatible with the quantile-shaped `[N,6]` reshape the whole metrics
  pipeline (`notebooks/calculate_metrics.py`) expects. If the paper wants to include it, that's a
  separate, not-yet-done integration effort, not something already covered by our results.

**Output**: quantile predictions (`cfg.model.quantiles = [0.1, 0.5, 0.9]`), one block per
selected target in `cfg.model.predict_targets` (default `["pv"]` only, can be `["load","pv"]`),
via a shared masked pinball loss (`custom_graphgym/loss/earne_loss.py`).

**Metrics** (`custom_graphgym/metric/regression.py`'s `DisaggregationMetrics`): RMSE, MAE, R²,
EFE, Pinball Q50, Coverage, Sharpness — same names as `results.tex` — plus `net_rmse` (only when
both load+pv predicted) and `export_violation_rate`/`count`/`mag` (only needs PV; this gate was a
bug we fixed this session, see Contribution 2).

**Weather**: the fleet gold-layer parquet recently gained clear-sky-index columns
(`clearsky_index_mean`, `clearsky_index_std`, `is_clear_sky_day`). We redid the weather-on/off
sweep using `clearsky_index_mean` in place of raw irradiance (`shortwave_radiation_om_mean`) —
this is what "W+/W-" means in our current runs, distinct from the original paper's weather
feature set.

## Contribution 1 — Dual-read vs single-read: correct the formulation

**The outline's proposal is wrong for our system.** It proposes a 3rd output target (export) with
a new "export head," multi-task masking per household, etc. **That's not what dual-read is in our
implementation, and we shouldn't retrofit the paper to match the outline — the outline should
match us.**

**WhDefine a taxonomy of metering configurations (single net, dual import/export, import + PV sub‑meter, etc.).