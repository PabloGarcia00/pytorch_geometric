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

---

## Methods section draft: baselines and training setup

Grounded directly against the current code (`custom_graphgym/network/`, `custom_graphgym/encoder/`,
`custom_graphgym/loss/`, `custom_graphgym/train/`, `custom_graphgym/config/`). All 7 models share one
dataset, one quantile-output convention, and one metrics suite — differences below are purely
architectural/loss-mechanical.

### Main model: ST-GNN (`earne_network`)
- Node encoder (`earne_temporal`): three parallel streams — a 4-layer dilated 1D CNN over the
  raw (single- or dual-read) history (dilations 1/4/8/16, covering local spikes → 15-min → 1-2hr →
  4hr patterns), an optional weather CNN (grouped conv keeps mean/std feature pairs coupled before
  cross-feature mixing), and a calendar CNN over the full cyclical time window. Weather stream
  gates the net-demand embedding via a learned sigmoid gate before fusion.
- Spatial body: `cfg.gnn.layers_mp` message-passing layers (GATv2Conv by default; GCNConv/SAGEConv
  also supported), each pre-LayerNorm → conv → GELU → dropout → residual.
- Graph construction, 3 modes: `spatial_knn` (fixed edges precomputed in the dataset),
  `static_corr` (precomputed household-pair correlation, thresholded by a learnable
  `lambda_threshold`), `learned_corr` (correlation recomputed on the fly from a rolling net-demand
  window, same learnable threshold).
- Shared quantile head (`earne_quantile`): one block of `n_quantiles` outputs per entry in
  `cfg.model.predict_targets` (default PV-only).

### MLP baseline
- Reuses `earne_network`'s spatial body/head/loss unmodified — isolates the effect of message
  passing by setting `gnn.layers_mp = 0` and swapping in the `baseline_mlp_temporal` node encoder.
- 2-layer feedforward trunk (`mlp_hidden_dim=128`) over the flattened `[T, C]` window (raw signal +
  operational flag [+ weather]), fused with the identical calendar-CNN time stream used by the GNN
  encoder, so only the temporal-encoding mechanism changes, not the calendar/weather handling.

### LSTM baseline
- Same shell as the MLP baseline (`layers_mp = 0`), node encoder swapped to
  `baseline_lstm_temporal`: a bidirectional LSTM (1 layer, `hidden_dim=32` by default) over the raw
  window, final timestep's hidden state fused with the same calendar-CNN stream.
- Tests whether recurrent temporal state adds anything over the MLP's flatten-and-project
  approach; still zero cross-household weight sharing beyond the one shared LSTM.

### CVAE baseline
- Full standalone `@register_network` — bypasses the encoder/GNN-body/head chain entirely (own
  forward/loss contract, ported from a stand-alone CVAE baseline script).
- BiLSTM encoder (+ shared calendar-CNN stream) → variational latent `z` (`cvae_latent_dim=32`,
  reparameterization trick during training; posterior mean used at eval).
- Decode heads, built only for whichever targets are active: a Gaussian head for load (mean +
  softplus std), and a zero-inflated Beta head for PV — a `gate_head` logit gives P(daytime), a
  `solar_head` gives Beta(α, β) for the daytime-conditional PV fraction in [0, 1].
- Quantiles derived analytically each forward call (not fit once at the end): `Normal.icdf` for
  load; a gated/clipped `Beta.ppf` (vectorized via scipy) for PV that shifts quantile mass across
  the day/night gate boundary to represent night-time zero output.
- Loss (`cvae_loss`): masked Gaussian NLL (load) + `solar_weight`·masked Beta NLL (PV,
  daytime-masked) + `gate_weight`·BCE (day/night gate) + `kl_weight`·KL(q(z|x) ‖ N(0,I)), plus the
  same Watt-scale PV-export physics penalty described below — added specifically because CVAE
  previously had no physics-loss path at all.

### Linear baseline
- Per-household independent linear quantile regression: one `[in_features → out_features]` weight
  matrix per household, batched via a grouped `PerNodeLinearQuantileHead` (einsum over a
  `[num_nodes, F, O]` parameter tensor) — no cross-household weight sharing, no message passing.
- Two feature modes (`cfg.baseline.linear_feature_mode`): `window` (default) — flattened
  `seq_len`-step raw history (+ operational flag [+ weather]) plus a 6-d cyclical calendar anchor
  (month / day-of-week / hour sin-cos) taken from the window's most recent step; `current` — the
  same, but using only the single most recent timestep's raw signal, motivated by PV being a
  near-instantaneous function of current conditions rather than a linear function of 96 lags.
- Trained with the identical masked pinball loss / optimizer / Lightning loop as the GNN — a fully
  standard `nn.Module`, no custom training code.

### SVR baseline
- Per-household kernel quantile regression, kept GPU-resident: a fixed Random Fourier Feature (RFF)
  map (`svr_rff_dim=256`) approximating an RBF kernel — `φ(x) = √(2/D)·cos(xW + b)`,
  `W ~ N(0, 2γ)` (Rahimi & Recht, 2007) — followed by the same `PerNodeLinearQuantileHead` used by
  the Linear baseline.
- Deliberately not exact dual-form SVR (e.g. sklearn's — a CPU-only QP solve with no native
  quantile output); kernel quantile regression (Takeuchi et al., 2006) generalizes the
  kernel-machine idea to the pinball loss directly, staying GPU-resident and reusing the shared
  training loop unmodified.
- γ auto-fit by default (`svr_gamma_auto=True`, sklearn's `scale` heuristic:
  `1 / (n_features · Var(X))`) from the first training batch actually seen. A flat fixed γ badly
  mismatched the ~300-d flattened-window input during development (RFF phase std blew up,
  degenerating the model to a per-node bias-only fit that ignored the input).
- Same `window`/`current` feature-mode toggle as Linear (`cfg.baseline.svr_feature_mode`); L2/ridge
  regularization comes from the standard `cfg.optim.weight_decay` knob, no bespoke code.

### KNN baseline
- Per-household analog-ensemble lookup — non-parametric, zero learnable weights (one placeholder
  parameter exists solely so Lightning's optimizer/scheduler setup doesn't fail on an empty
  parameter list).
- "Fitting" (`fit_cache`) is a single no-grad pass building a dense per-node neighbor bank
  (`[num_nodes, S_train, F]`: features + validity mask + target values) from the training split.
  Trained through a dedicated `knn_train` entry point (Lightning `max_epoch` forced to 0) rather
  than `earne_train`, since there's no gradient step — `.fit()` still runs to satisfy Lightning's
  optimizer/logger plumbing, but performs zero real training.
- 6 feature modes spanning window/current × calendar/weather inclusion (`window`,
  `window_calendar`, `window_calendar_weather`, `current`, `current_calendar`,
  `current_calendar_weather`) — the most granular feature-ablation surface of the three classical
  baselines.
- Prediction = empirical quantiles of the outcomes following the `k=20` (default) nearest training
  windows for that same household (Euclidean distance), computed vectorized (`cdist`/`topk`/
  `gather`/`quantile`, no Python loop over households). Distance search runs on
  `knn_cache_device` (GPU by default — profiled ~9.7GB at window-mode width on the full dataset,
  comfortably GPU-resident; CPU fallback available for larger populations).

### Training setup (shared across all 7 models)
- **Loss**: masked pinball (quantile) loss per selected target (`cfg.model.predict_targets`,
  default PV-only, optionally load+PV), summed across targets. GNN/MLP/LSTM/Linear/SVR/KNN all
  consume the shared `earne_loss`; CVAE uses its own likelihood-based `cvae_loss` (structurally
  different output distributions) but includes the identical physics term below.
- **Physics-informed penalty**: predicted PV must be ≥ the export implied by net demand (can't
  export more than you generate) — both quantities inverse-transformed to physical Watts, the
  shortfall `asinh`-compressed (keeps it on a comparable scale to the ihs-space quantile loss)
  before squaring. Weighted by `cfg.train.physics_weight` (0.0 or 0.3 across the sweep); scaled
  ×0.1 specifically for the three per-household canonical baselines (Linear/SVR/KNN), whose far
  smaller parameter count otherwise let this term dominate the objective and blow up PV MAE at the
  full weight.
- **Quantile-crossing penalty**: penalizes `q[i] > q[i+1]` for adjacent quantiles, per selected
  target, weighted by `cfg.train.crossing_weight` (default 0.05).
- **Optimizer**: Adam, `base_lr = 1e-3`; weight decay via the standard `cfg.optim.weight_decay`
  knob (doubles as SVR's ridge-style regularizer).
- **Schedule**: linear warmup (`cfg.train.warmup.epochs`, typically 10) into cosine annealing
  (`SequentialLR`: `LambdaLR` warmup → `CosineAnnealingLR` over the remaining epochs — the
  `warmup_cos` custom scheduler).
- **Early stopping**: monitors `val_loss`, patience 10; `min_delta` tuned per-config — set > 0 for
  fast-converging low-parameter configs (e.g. Linear/SVR under `current` feature-mode), which can
  plateau almost immediately and otherwise oscillate within float noise forever.
- **Data split**: temporal 70/15/15 train/val/test (`train_split=0.7`, `val_split=0.85`),
  `seq_len = 96` input windows (24h at 15-min resolution).
- **Seeds**: 3 random seeds per config for every gradient-trained model; KNN is deterministic given
  a fixed training split, so only 1 seed is run.
- **Harness**: every run — regardless of "training" mechanism (SGD, or KNN's single `fit_cache`
  pass) — goes through the same Lightning `Trainer` / wandb+CSV logging / `stats.json` output, so
  results across all 7 models are directly comparable through one shared metrics pipeline
  (`notebooks/calculate_metrics.py`).

---

## Methods section draft: real-world data-availability and data-quality robustness experiments

Four gaps motivate this cluster of experiments: published disaggregation methods routinely (1)
condition on per-household location, panel configuration, or explicit meteorological features
unavailable in practice (privacy, metadata-quality, infrastructure reasons); (2) assume net load as
the only observable input, when an increasing share of real metering infrastructure exposes import
and export as separate channels; (3) ignore calibration heterogeneity across inverter manufacturers,
which produces timesteps where recorded PV generation is physically inconsistent with the export the
household is known to have delivered (`PV < Export`); and (4) implicitly assume a temporally
complete observation window, when real populations have continuous customer/meter turnover. Each
below is grounded in an actual experiment already run or currently running in this codebase, not a
hypothetical design.

### Gap 1 — Incomplete information: location, panel configuration, weather
- **No model in this campaign ever receives per-household location or panel configuration as a
  predictive feature.** Household lat/lon (`zipcode_coordinate.csv`, joined via zip code) is used
  only to construct `earne_network`'s optional spatial graph topology (`spatial_knn`/`static_corr`
  edge construction, `custom_graphgym/loader/graph_dataset.py`) — it never reaches any baseline
  (MLP/LSTM/CVAE/KNN/Linear/SVR), and even for the GNN it only shapes *which households talk to
  which*, not an input feature value. Panel configuration (inverter rating, tilt/azimuth, etc.) is
  not part of the dataset schema at all.
- **Weather availability is the axis we do sweep**: `cfg.earne_data.weather_mode` (True/False, the
  `W+`/`W-` columns in `results.tex`) gates an entire encoder stream (or, for baselines, an input
  block) on/off; `weather_features` lists exactly which columns are used when on (currently
  `clearsky_index_mean`, `temperature_2m_om_mean`, `cloud_cover_low_om_mean` for the `_csi` config
  family). `W-` runs are the "metadata/weather unavailable" condition for all 7 models.

### Gap 2 — Signal-type mismatch: net load vs. import/export
- **Single-read** (`cfg.model.dim_in=1`, `earne_data.dual_read=False`): one channel, net demand
  (`net_scaled` — import minus export, the only observable signal the disaggregation literature
  assumes). **Dual-read** (`dim_in=2`, `dual_read=True`): two channels, consumption and generation
  reported separately (`_extract_raw_x` in `earne_network.py` branches purely on `batch.x`'s last
  dimension; every baseline's feature builders take the same branch via `cfg.model.dim_in`).
  Architecture, loss, and training loop are otherwise identical between the two — the contrast
  isolates the value of the richer channel split alone.
- This campaign runs **both** conditions across all 7 models: dual-read is the basis of the main
  baseline-comparison and TSTR/TRTS matrix; single-read is the basis of the resolution-eval and
  coverage-eval batches below, plus the existing `baseline_*_single_csi` / `earne_learned_corr_
  single_csi` weather sweep.

### Gap 3 — Calibration heterogeneity: physically inconsistent PV/export readings
- **Physics-informed penalty** (`custom_graphgym/loss/earne_loss.py`): predicted PV must be ≥ the
  export implied by net demand (`export = max(-net_demand, 0)`; you cannot export more than you
  generate). Both quantities are inverse-transformed to physical Watts before comparing (they live
  in different normalized spaces otherwise), the shortfall `asinh`-compressed, then squared and
  weighted by `cfg.train.physics_weight` (swept 0.0/0.3, the `MP0`/`MP3` columns). This is a direct,
  differentiable penalty on exactly the `PV < Export` inconsistency the calibration-heterogeneity
  gap describes — not a data-cleaning step, so the inconsistency remains visible in raw inputs and
  is instead attacked at the objective level.
- **Corresponding eval metric**: `export_violation_rate`/`count`/`mag`
  (`custom_graphgym/metric/regression.py`), which measures the same inconsistency at test time
  independent of whether the penalty was active — lets us report both "how much did the penalty
  help" and "how much inconsistency remains" separately.
- **Canonical-model correction**: the per-household baselines (Linear/SVR/KNN) have far fewer
  parameters than the shared-weight models, so at the full `physics_weight` the penalty dominated
  the objective and PV MAE exploded (confirmed empirically: composite loss descending smoothly while
  val MAE-PV rose into the thousands). Fixed by scaling `physics_weight` ×0.1 specifically for these
  three model types — a stronger, per-architecture-aware formulation than a single global weight.

### Gap 4 — Customer and meter turnover: incomplete observation windows
- **`cfg.earne_data.require_full_span`** (`custom_graphgym/loader/graph_dataset.py`'s
  `_get_node_mask`) controls whether a household must be active in both the first and last week of
  the dataset to be included. `True` ("optimal coverage") drops late-onboarded/early-offboarded
  households, keeping only temporally-complete series — the implicit assumption most published
  evaluations make. `False` ("sub-optimal coverage") keeps every household regardless of partial
  coverage — the realistic condition under continuous customer/meter turnover. On the reference
  15-min dataset this is 62 vs. 112 households (`st_sgc_caps_single_full_optimal/suboptimal.yaml`'s
  documented counts) — sub-optimal nearly doubles the population by including incomplete series.
- This toggle only affects post-hoc node filtering (runs after the cached dataset is loaded), never
  the underlying processed data — so both conditions reuse one dataset build, isolating the effect
  of population completeness from any confound with data processing.
- **Currently running**: all 7 models × {optimal, suboptimal}, single-read, 15-min data, 3 seeds
  (1 for KNN) — `coverage_eval_<model>_<optimal|suboptimal>.yaml`.
- One data-quality finding surfaced by this experiment: `baseline_knn`'s GPU-resident neighbor bank
  (`torch.cdist` over a `[num_nodes, S_train, F]` tensor) scales with household count, and OOM'd
  under sub-optimal coverage's larger population on a 22GB GPU — fixed via the existing
  `knn_cache_device: cpu` escape hatch for that one config. Worth noting in a methods write-up as a
  concrete resource-scaling consequence of turnover-inclusive populations, not just a modeling one.

### Related experiment (not one of the 4 gaps, same theme): metering interval heterogeneity
- Real fleets report at different intervals depending on hardware/provider (5/15/30/60-min are all
  in active use). We evaluate all 7 models, single-read, at 3 additional resolutions
  (`res_eval_<model>_{5min,30min,60min}.yaml`) against the existing 15-min baseline.
- `cfg.model.seq_len` is a pure row-count with no time-unit awareness in the loader, so it was scaled
  per resolution (288/48/24 vs. the baseline 96) to preserve an identical 24h real-world lookback
  window across resolutions — otherwise resolution and window-duration would be confounded.
- One latent bug this surfaced and fixed: the loader's household-onboarding "active in the first/last
  week" check (same `_get_node_mask` as Gap 4) hardcoded a 672-step window assuming 15-min spacing;
  now derived from the dataset's actual timestamp spacing, so it stays correct at any resolution.
- `baseline_knn`'s GPU bank also OOM'd at 5-min resolution specifically (3× more training rows ×
  3× wider window vs. 15-min) — same `knn_cache_device: cpu` fix as Gap 4, applied per-config rather
  than globally, since 30/60-min have *fewer* rows than the already-working 15-min baseline.

---

## Open items to confirm before the training-setup section is final

\needinfo{
  \item Final numeric values for weight decay, batch size, dropout rate, and maximum epoch cap --
  not yet confirmed against this paper's actual run configs, and may no longer match the values
  from EARNE's initial development.
  \item Confirmation that the warm-up length used for this paper's reported runs is exactly 10
  epochs (stated as ``typically 10'' in development notes) rather than a per-model value.
  \item The specific \texttt{min\_delta} value(s) used for the fast-converging low-parameter
  configs (Linear/SVR under \texttt{current} feature mode) versus the default used elsewhere.
}

**Resolved by grepping every `configs/pyg/*.yaml` actually used in this campaign** (excluding
GraphGym's own `example_*.yaml` templates and clearly-legacy/smoke configs —
`earne_exp2/exp3_mean_only/gat/weather_all_15min.yaml`, `earne_dual_read_smoke.yaml` — none of which
back any reported result):

- **Weight decay**: not set explicitly anywhere except SVR — meaning every other model
  (GNN/MLP/LSTM/CVAE/KNN/Linear) inherits GraphGym's stock default, `cfg.optim.weight_decay =
  0.0005`. SVR overrides it as its ridge-style regularizer: `0.001` under `window` feature mode,
  `1.0e-05` under `current` feature mode (16 config occurrences total, split evenly between the two
  modes — no other value appears).
- **Batch size**: uniformly `32` across all 171 real-campaign configs. (The only other values —
  `128`/`16`/`1` — belong exclusively to GraphGym's `example_*.yaml` files, one legacy smoke config,
  and `st_sgc_caps` respectively, none of which are part of the 7-model campaign.)
- **Dropout**: uniformly `0.1` — `gnn.dropout` (82 occurrences), `baseline.lstm_dropout` (23),
  `baseline.cvae_dropout` (19). The only `0.0` occurrences are in GraphGym's own `example_link.yaml`/
  `example_graph.yaml`, unrelated to this project.
- **Max epoch cap**: uniformly `200` (176 of 183 total occurrences); the handful of `100`/`150`/`5`
  values all belong to the same excluded legacy/example/smoke configs above. Early stopping
  (patience 10 on `val_loss`) generally halts training well before this cap in practice.
- **Warm-up length**: confirmed **exactly 10 epochs** across all 152 real-campaign occurrences of
  `train.warmup.epochs` — no config deviates. "Typically 10" in the earlier development notes was
  imprecise; it is not a per-model value, it is universal.
- **`min_delta`**: exactly `0.001`, set on exactly 16 configs — `baseline_linear_dual_current.yaml`,
  `baseline_svr_dual_current.yaml`, and all 14 `tstr_trts_{linear,svr}_current_*.yaml` variants (2
  base + 7 conditions × 2 models). Every other config in the campaign leaves `min_delta` unset,
  inheriting the library default of `0.0` (`custom_graphgym/config/earne.py`) — i.e. any
  non-negative improvement resets the early-stopping patience counter. The `0.001` override exists
  specifically because these fast-converging, low-parameter (`current`-mode Linear/SVR) configs
  plateau almost immediately and would otherwise oscillate within float noise indefinitely without
  ever triggering early stopping.

---

## Problem Formulation (LaTeX, ready for the paper)

Rewrite of the drafted Problem Formulation section — the draft still gave the joint load-and-PV
formulation roughly equal weight to the PV-only one, which misstates what this paper actually
reports: **every model, every experiment, every result in this whole campaign predicts PV only.**
Joint load-and-PV appears solely as historical context explaining two artifacts that persist in the
shared harness (the `Net RMSE` metric, the shape of the physics-informed loss) — it is prior work
being distinguished *from*, not a formulation this paper also uses.

**One factual correction, verified against git history, not assumed**: the draft claimed the earlier
(pre-refactor) model had *no* differentiable net-consistency loss term. That's false — I checked out
`custom_graphgym/loss/earne_loss.py` as it existed at commit `04bc53bd5` (before `cc5cc7c95`, "added
new baselines + removed load prediction head") and it has a genuine MSE-style physics penalty:
`loss_physics = MSE(q_load[0.5] - q_pv[0.5], y_net_demand)`, added to the composite loss as
`physics_weight * loss_physics` — i.e. exactly the `Load − PV = NetDemand` equality constraint,
directly tied to the `MP0`/`MP3` columns in `results.tex`. The corrected framing below states this
precisely: the old term was a real, trained equality constraint that happens to be **undefined**
under PV-only prediction (no `L̂` exists to subtract), which is *why* Section on physics-informed loss
had to design a new inequality constraint (`PV ≥ Export`) that needs only PV — a necessary redesign,
not a gap-filling addition to a system that previously had nothing.

```latex
\section{Problem Formulation}\label{sec:problem-formulation}

This paper's disaggregation target is PV generation alone (Section~\ref{sec:pv-only-formulation});
the joint load-and-PV formulation used in EARNE's earlier development is retained here only as
historical context that explains two artifacts still present in the shared harness -- the
\emph{Net RMSE} metric and the shape of the physics-informed loss (Section~\ref{sec:joint-formulation}).
Section~\ref{sec:metering-regimes} then formalises the input-availability taxonomy behind Barrier 2,
and Sections~\ref{sec:problem-weather}, \ref{sec:problem-calibration}, and \ref{sec:problem-coverage}
state Barriers 1, 3, and 4 of Section~\ref{sec:four-barriers} as concrete, testable problem statements.

\subsection{Joint Load-and-PV Formulation (prior work, not used in this paper)}\label{sec:joint-formulation}
EARNE's earlier development cast the behind-the-meter disaggregation problem into the identity
\begin{equation}\label{eq:disaggregation}
NL_t^i = L_t^i - PV_t^i,
\end{equation}
where the net load $NL_t^i$ measured AtM for household $i$ at time $t$ equals the unobserved BtM
load $L_t^i$ minus BtM PV generation $PV_t^i$. That earlier system predicted both $L_t^i$ and
$PV_t^i$ via two separate quantile heads and trained against a differentiable net-consistency
penalty on this same identity, $\left(\hat{L}_t^i(0.5) - \hat{PV}_t^i(0.5) - NL_t^i\right)^2$,
weighted by a physics coefficient swept over $\{0.0, 0.3\}$ -- the source of the \emph{MP0}/\emph{MP3}
columns and the \emph{Net RMSE} metric already present in the shared offline metrics suite. This
equality-constraint formulation is well-defined only when both targets are predicted; it has no
PV-only counterpart, since $\hat{L}_t^i$ does not exist under the default of
Section~\ref{sec:pv-only-formulation}. Section~\ref{sec:physics} replaces it with an inequality
constraint that needs only $\hat{PV}_t^i$, letting the physics term survive the move to PV-only
prediction rather than becoming vacuous.

\subsection{PV-Only Disaggregation}\label{sec:pv-only-formulation}
Every model and every result in this paper predicts a single target -- BtM PV generation -- at
quantile levels $\tau \in \{0.1, 0.5, 0.9\}$:
\begin{equation}
\hat{y}_i^t = \hat{PV}_i^t(\tau), \qquad \tau \in \{0.1, 0.5, 0.9\}.
\end{equation}
This is a configuration choice, not an architectural one: the prediction-target list is read at run
time by every model in the benchmark (Section~\ref{sec:models}), and \texttt{load} remains available
as an additional entry for anyone reproducing the joint formulation of
Section~\ref{sec:joint-formulation} -- but no config in this paper's experimental matrix
(Section~\ref{sec:experimental-design}) ever sets it. PV-only is not a reduced or simplified setting
relative to the joint case: it is the operationally relevant configuration wherever direct BtM load
supervision is unavailable or deliberately withheld -- e.g. where only inverter or PV-metadata-adjacent
signals establish ground truth and no companion household load meter exists.

\subsection{Barrier 2: Metering Regimes}\label{sec:metering-regimes}
Independently of the prediction-target choice above, the data pipeline distinguishes two metering
regimes that determine what is observed as input for a given household:
\begin{itemize}
    \item \textbf{Dual-read regime}: consumption and generation are available as separate input
    channels, letting the network condition on both underlying signals directly rather than only on
    their difference.
    \item \textbf{Single-read regime}: only the aggregate AtM net-demand signal is observed as
    input; no separate consumption or generation channel exists.
\end{itemize}
This distinction changes \emph{input dimensionality and data availability}, not model architecture:
the same graph model and non-graph baselines are trained and evaluated under both regimes, with only
the input width changing accordingly. Keeping metering regime a data toggle rather than a separate
model family is what lets Section~\ref{sec:experimental-design} treat it as one factorial axis among
several rather than as its own model family.

\subsection{Barrier 1: Weather-Agnostic Input}\label{sec:problem-weather}
The input feature vector $x_t^i$ either includes or excludes a meteorological feature block, on the
same toggle logic as the metering regime above: in weather-informed mode $x_t^i$ is augmented with
the clear-sky-index features of Section~\ref{sec:weather}; in weather-agnostic mode this block is
omitted entirely and $x_t^i$ contains only the raw (single- or dual-read) signal, the operational
flag, and the calendar-time encoding. This toggle is applied identically to every model in the
benchmark, so Section~\ref{sec:experimental-design} can test, as one factorial axis, whether the
meteorological metadata assumed available throughout Section~\ref{sec:related-information}'s
literature is actually necessary.

\subsection{Barrier 3: Physically Impossible Readings}\label{sec:problem-calibration}
Heterogeneous inverter calibration (Section~\ref{sec:related-calibration}) produces timesteps at
which the recorded PV generation $PV_t^i$ is inconsistent with the export $E_t^i$ implied by net
demand, $E_t^i = \max(-NL_t^i, 0)$ (Section~\ref{sec:physics}): a household cannot export more than
it generates, so $PV_t^i \geq E_t^i$ must hold at every timestep -- trivially when $E_t^i = 0$,
substantively when $E_t^i > 0$. Timesteps that violate this inequality in the raw data are
physically impossible, not merely noisy. Sections~\ref{sec:calibration-noise} and \ref{sec:physics}
evaluate two complementary interventions against them: excluding such timesteps from training, and
penalising violations of the inequality directly in the loss -- the PV-only-compatible successor to
the equality constraint of Section~\ref{sec:joint-formulation}.

\subsection{Barrier 4: Coverage and Turnover}\label{sec:problem-coverage}
Customer and meter turnover (Section~\ref{sec:related-turnover}) means the operational timeline
$\mathcal{T}_i$ over which household $i$ has valid measurements need not span the full global
timeline $\mathcal{T}$. Section~\ref{sec:coverage-regimes} formalises this with the same operational
indicator used for masking (Section~\ref{sec:operational-window}) and defines the optimal- and
sub-optimal-coverage regimes compared in this paper: the former retains only households observed
across the full timeline, the latter retains every household regardless of when it was onboarded or
offboarded. The question this section poses is whether a model trained on the smaller, temporally
complete optimal-coverage population generalises differently -- better on the households it kept,
worse on the ones it never saw -- than one trained on the larger, native sub-optimal-coverage
population.
```

---

## Elucidation of `remaining points` (the 17 open `\needinfo{}` items)

Going through every item in the `remaining points` file in reading order. The dominant blocker that
file names — the missing coverage-regime (Barrier 4) sweep — **is no longer missing**: the
`coverage_eval_<model>_{optimal,suboptimal}` batch (all 7 models, single-read, 15-min data, 3 seeds
except KNN) finished this session. That resolves or partially resolves 6 of the 17 items outright.
The rest split into genuine citation gaps (can't be resolved from code — literature search needed)
and authorial/scope decisions (informed by data below, but the call is yours).

### Manuscript — Abstract — **RESOLVED** (data now exists)
Best-seed-by-`val_loss` test MAE-PV, optimal vs. sub-optimal, all 7 models:

| model  | optimal (62 hh) | sub-optimal (112 hh) | Δ (degradation) |
|--------|-----------------|------------------------|------------------|
| GNN    | 133.2 W | 135.7 W | **+2.5 W** |
| MLP    | 121.2 W | 126.4 W | +5.2 W |
| CVAE   | 115.0 W | 121.7 W | +6.7 W |
| LSTM   | 128.0 W | 135.9 W | +7.9 W |
| SVR    | 208.1 W | 222.2 W | +14.1 W |
| KNN    | 150.6 W | 164.9 W | +14.3 W |
| Linear | 152.4 W | 165.4 W | +13.0 W |

Household counts confirmed directly from a run's own saved log (`grep num_nodes`), not just the
`st_sgc_caps_single_full_optimal/suboptimal.yaml` doc comments: **62 optimal, 112 sub-optimal.**
Headline framing this supports: every model degrades under sub-optimal coverage (as expected —
harder, incomplete-history households), but the four shared-weight models (GNN/MLP/CVAE/LSTM)
degrade by 2.5-7.9W while the three per-household canonical models (SVR/KNN/Linear) degrade by
13-14.3W — roughly 2-5x more sensitive to population completeness. Contribution ordering is now
answerable; I'd suggest leading with this shared-weight-vs-per-household split since it's the
sharpest, most defensible claim in the table.

### Related Work — Barriers 2/3/4 citations — **STILL OPEN, needs literature search**
Not something I can resolve from this codebase — these need actual citation sourcing (heterogeneous
metering-infrastructure studies, physics-informed learning in power systems, smart-meter calibration
literature, customer/meter-turnover studies). I won't fabricate references here; flagging this
explicitly rather than guessing is the safer path. Happy to help search/screen candidates if you want
to point me at a database or give me candidate titles to verify.

### Data and Preprocessing — Weather Features — **PARTIALLY RESOLVED**
Feature set confirmed: every `_csi` config in the current campaign uses
`weather_features: [temperature_2m_om_mean, clearsky_index_mean, cloud_cover_low_om_mean]` — CSI-based,
not raw shortwave radiation. This is the set that should populate the table; raw-irradiance
correlation is legacy (pre-CSI) and not what any reported result actually used. The Pearson
correlation figure itself (CSI vs. energy features) hasn't been generated — that's a plotting task,
not a code-grounding one; say the word and I'll write it.

### Data and Preprocessing — Graph Construction — **RESOLVED**
`earne_learned_corr_dual.yaml` / `earne_learned_corr_single_csi.yaml` — the actual templates behind
every reported GNN result this session (`coverage_eval_gnn_*`, `res_eval_gnn_*`) — use
`graph_mode: learned_corr`, confirmed directly in both the templates and the generated configs.
`spatial_knn` (geographic k-NN) appears only in the non-graph baseline configs (MLP/LSTM/CVAE/KNN),
where it's vestigial — those models have `gnn.layers_mp: 0`, so no message passing ever runs along
whatever edges that field would produce. **`learned_corr` is this paper's actual default graph
construction for the GNN**; `spatial_knn` and `static_corr` are the ablated alternatives (71 and 4
config occurrences respectively, vs. 103 for `learned_corr`).

### Data and Preprocessing — Training-Coverage Regimes — **RESOLVED** (counts) / **OPEN** (taxonomy decision)
Counts: 62 optimal / 112 sub-optimal, confirmed above. The finer-grained taxonomy question
(full/partial/reduced/constrained activation windows) is a scope decision only you can make — it
would need new data-pipeline work (`_get_node_mask` currently only supports a binary
`require_full_span` toggle, not graded windows) beyond what's built. I'd frame it as a "future work"
item rather than something this paper's results already support, given the binary toggle is all that
exists right now.

### Physics-Informed Loss — Loss-Level Ablation (PV-only export-consistency penalty) — **RESOLVED, remove the caveat**
Confirmed shipped: `export_violation_rate`/`count`/`mag` (EVR/EVM) in
`custom_graphgym/metric/regression.py`'s `DisaggregationMetrics.all()` is gated only on
`"pv" in q50_by_target` (line ~215), never on `"load"` also being present — this PV-only gate was a
bug fixed earlier this session (previously silently required both targets, dropping the metric
entirely under the PV-only default). The caveat describing this as an open gap should be deleted, not
just revised — the results already contradict it, as the `remaining points` note itself suspected.

### Physics-Informed Loss — Loss Composition (joint net-consistency penalty) — **OPEN, scope decision**
Genuine choice for you: implement a differentiable joint penalty for future work, or state joint-mode
physics constraints out of scope. One data point worth having before deciding: the *historical*
equality-constraint version of this (`Load - PV = NetDemand`, MSE-penalized) already existed in this
codebase pre-refactor (commit `04bc53bd5`, before load prediction was removed in `cc5cc7c95`) — it
isn't a from-scratch design effort if you do want it, more a matter of re-adding a load head and
reviving that loss term alongside the current PV-only inequality constraint.

### Results — Barrier 4 Results (Coverage and Turnover) — **RESOLVED**
Same table as the Abstract item above — this is the sweep result. No longer "the one barrier without
results."

### Discussion — Message Passing Does Not Clearly Earn Its Complexity Here — **RESOLVED, and it complicates the section title**
The coverage data gives a real, nuanced answer, and it's worth flagging that it partially **cuts
against** this section's current title: on raw accuracy, GNN is *not* the best model at either
coverage level (CVAE and MLP both beat it, 115W/121W and 121W/126W vs. GNN's 133W/136W) — consistent
with "message passing doesn't clearly earn its complexity" on accuracy alone. But on
**coverage-robustness specifically**, GNN degrades least of all seven models (+2.5W vs. +5-14W
elsewhere) — the smallest degradation in the whole table. A defensible synthesis: message passing
doesn't buy raw accuracy in this benchmark, but among the models that already share weights across
households (GNN/MLP/LSTM/CVAE), it does buy the most robustness to incomplete household coverage —
while the real dividing line for robustness is shared-weight vs. per-household, not graph vs.
non-graph. For minimum data-completeness requirements: per-household canonical models (SVR/KNN/Linear)
are the ones a DSO should worry about if their population has significant turnover; the shared-weight
family is comparatively forgiving. Barrier-payoff ranking is now answerable with data (see below) but
is still your editorial call on framing/emphasis.

### Limitations — **PARTIALLY RESOLVED, new items to add**
Two concrete, empirically-observed items from this session worth adding here rather than leaving
implicit:
- Canonical baselines (Linear/SVR) under `window` feature mode were found to genuinely diverge (not
  just underperform) under nonzero `physics_weight` — one config reached test MAE-PV in the billions
  of Watts while `val_loss` looked completely normal throughout. Fixed by moving Linear/SVR to
  `current` feature mode universally (see the current-mode policy section, if added above); worth
  stating as a known instability mode of window-mode canonical baselines under the physics penalty,
  not just a footnote fix.
- In the mask=False/pw=0.3 physics-loss cell specifically, GNN/MLP/LSTM all show substantially
  elevated MAE-PV (635-765W) relative to their healthy ~115-140W range, while CVAE and KNN in the
  *same* cell stay healthy (112W/144W) — the instability is architecture-specific, not universal to
  "no mask + physics loss," and isn't yet explained. Worth flagging as an open empirical limitation
  rather than silently omitting the cell.

### Conclusion — **RESOLVED** (coverage result) / **OPEN** (future-work list, informed by data above)
The fourth-barrier result is in (table above); the "may change the relative ranking" caveat can be
revisited now rather than left conditional. Future-work items beyond what's already flagged: the
graded coverage taxonomy (see Training-Coverage Regimes above), and — new, from this session — the
unexplained mask=False/pw=0.3 architecture-specific instability (GNN/MLP/LSTM vs. CVAE/KNN) is a
concrete, well-scoped future-work item that didn't exist when the original list was written.

### Net tally
Of 17 original items: **6 fully resolved** (abstract numbers, graph-construction default, EVR/EVM
gate confirmation, Barrier 4 results, household counts, conclusion's barrier-4 caveat), **2 partially
resolved** (weather feature-set confirmed / correlation figure still pending, limitations list
extended with 2 new concrete items), **3 remain genuine citation gaps** (Related Work, all three
barriers — literature search needed, not resolvable from code), and **3 remain authorial/scope
decisions** informed by data but not decidable by me (finer coverage taxonomy, joint-mode physics loss
future-work-vs-out-of-scope, barrier-payoff ranking for the closing synthesis paragraph).
  ever triggering early stopping.