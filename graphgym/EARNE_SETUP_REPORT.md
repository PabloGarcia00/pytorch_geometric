# EARNE Development Setup — Quality Report

_Generated: 2026-03-26_

---

## Overall Assessment

The setup is a well-scoped research prototype built on top of PyTorch Geometric's GraphGym.
It correctly separates data, model, loss, and metric concerns, and the shift to a
master-bundle architecture is the right engineering direction. However, several fragile
patterns and silent-failure risks make it brittle for anything beyond controlled
experimentation.

---

## Strengths

**Modular registration pattern**
Every component (encoder, head, loss, metric, loader) is independently registered with
GraphGym via decorators. Swapping a GCN for a GAT or changing the loss requires only a
config change, not code surgery.

**Master bundle approach** (`loader/graph_dataset.py`)
Pre-computing all tensors once and filtering dynamically at runtime eliminates repeated
DuckDB queries during epoch iteration and enables instant zero-shot transfer experiments
by changing `filter_zips` in config.

**Physics-aware loss**
Encoding the `Load - PV = NetDemand` constraint directly in the loss function is clean
and easy to ablate (set `physics_weight: 0.0`).

**Quantile outputs**
Probabilistic predictions via pinball loss are appropriate for energy forecasting —
point estimates would be insufficient for the disaggregation use case.

**Config-driven reproducibility**
The YACS config system, seeding in `main.py`, and `configs_gen.py` grid search make
experiments reproducible and sweepable.

---

## Complications

**Two loaders coexist without a clear handoff** (`loader/earne.py` vs `loader/graph_dataset.py`)
`earne_disagg` (legacy) and `earne_loader_new` (bundle) are both registered with overlapping
functionality. The legacy loader defines its own `GraphDataset`, creates full-graph edges via
`permutations`, and handles its own train/val/test splitting — all re-implemented more robustly
in the new loader. The continued presence of the legacy loader creates ambiguity and adds
maintenance surface.

**Global batch registration in the loss** (`loss/earne_loss.py`)
The loss accesses `register.batch` — a globally stored reference — rather than receiving the
batch as a function argument. If the head fails silently or execution order changes, the loss
reads stale or missing data and falls back to zero without warning. Silent failures of this
kind are very hard to diagnose during training.

**Head stores predictions as batch attributes** (`head/earne_head.py`)
`earne_head.py` attaches `batch.q_load` and `batch.q_pv` directly onto the batch object; the
loss then reads these off `register.batch`. This creates an implicit contract between two
unrelated modules that is invisible from either module alone and not enforced anywhere.

**DuckDB ETL complexity** (`loader/graph_dataset.py::process()`)
Multi-step SQL with a two-tier weather imputation chain (station → fleet average → zero) is
correct but hard to validate. A silent misalignment between energy and weather rows (e.g. a
timezone offset in hive partitioning) would produce plausible-looking but wrong weather
inputs, and nothing in the pipeline would catch it.

---

## Drawbacks

**Quantile index hardcoding in metrics** (`metric/earne_metric.py`)
The code assumes `quantiles[0.5]` is always at index `1` and `quantiles[0.9]` at
`n_quantiles + 1`. Changing the quantile list silently evaluates the wrong columns. The
median index should be derived from `cfg.model.quantiles.index(0.5)`.

**Relative path assumptions** (`config/earne.py`)
Paths like `../../exploratory-data-analysis/output/clean_data_hive` are relative to the
graphgym working directory and will break silently if the repo is cloned or the project
layout changes. An environment variable such as `EARNE_DATA_ROOT` would be more portable.

**Full-graph mode is O(N²)** (`loader/graph_dataset.py::_generate_topology()`)
With 136 nodes, full graph produces 18,496 edges — manageable. This is not guarded against
larger networks. If `n_user` is increased significantly, full graph will silently exhaust GPU
memory.

**Conv1D kernel fixed to 48 steps (12 hours)** (`encoder/temporal_encoder.py`)
The kernel size is 48 and `seq_len` is 96. If `seq_len` is reduced below 48 for a short-
lookback ablation, the convolution produces degenerate output without any error.

**`mask_weather` ablation uses zeros, not a learned null**
Setting weather inputs to exactly zero is a strong inductive bias — zero is a valid value in
the normalized range. A proper ablation would use a learned null embedding or masked attention.

---

## Optimization Opportunities

**`get(idx)` re-concatenates weather + temporal on every call**
The weather and temporal tensors are indexed and stacked into a new `Data` object on every
sample access. Pre-stacking the combined weather+temporal tensor once at `__init__` time would
reduce per-sample CPU overhead across thousands of epoch iterations.

**Single scaler for all targets**
Both load and PV are normalized with one `MinMaxScaler`. If their value ranges differ
significantly (e.g. a building with tiny PV but large load), one signal is compressed relative
to the other, unbalancing the two loss terms. Separate scalers per target (or per node) would
improve gradient balance.

**`export_baseline_data.py` recomputes hand-crafted features on every run**
Summary statistics (mean, max, last) are computed from the raw bundle each time. For repeated
baseline comparisons, pre-computing and caching them would avoid redundancy.

**No DataLoader `num_workers > 0`**
`get()` accesses tensors already in memory, so multi-worker loading is safe and would reduce
epoch time. This is configurable in GraphGym's dataloader but is likely left at default (0).

---

## Potential Improvements

**Retire `loader/earne.py` explicitly**
Mark it deprecated with a `DeprecationWarning` on registration, or remove it. Its presence
alongside `graph_dataset.py` creates confusion about which loader is active and can cause stale
experiments if the wrong config value is used.

**Make quantile indices config-derived throughout**
Replace all hardcoded `[:, 1]` / `[:, n_quantiles + 1]` slices with:
```python
median_idx = cfg.model.quantiles.index(0.5)
```
This makes the metric and head robust to any quantile list.

**Add early validation in `EARNeGraphDataset.__init__`**
Check that paths exist, that the bundle has the expected keys, and that node counts are
consistent before training starts. Currently a path misconfiguration surfaces as a crash
deep in epoch 1.

**Decouple loss from global `register.batch`**
GraphGym's loss signature is `loss_fn(pred, true)`. Both tensors can carry everything needed —
`true` already holds `[load, pv, mask]`. The physics loss additionally needs `net_demand`,
which could be appended to `true` in the head rather than reading it from global state.

**Add a node-level scaler option**
Fit one `MinMaxScaler` per node so that nodes with low PV generation are not dominated by
high-load nodes in the loss. One-line change in `process()` with a corresponding update in
the metric denormalization.

**Guard `seq_len < conv1d_kernel_size` in the encoder**
Add an assertion in `EARNeTemporalEncoder.__init__`:
```python
assert cfg.model.seq_len >= cfg.model.conv1d_kernel_size, \
    "seq_len must be >= conv1d_kernel_size"
```

**Expand `test_integrity.py` beyond the happy path**
The current suite verifies things work when all data is present. Missing tests include:
wrong path configs, mismatched node counts after filtering, `filter_zips` returning an
empty mask, and `n_quantiles` mismatch. These are exactly the failure modes that occur in
practice.

**Weather ablation: use a learned null token instead of zeros**
Replace:
```python
weather = torch.zeros_like(weather)
```
with a registered learnable `nn.Parameter` null embedding so the model is not given a
semantically meaningful value when the intent is "no weather information."

---

## Summary Table

| Dimension            | Rating  | Key Issue                              |
|----------------------|---------|----------------------------------------|
| Architecture clarity | Good    | Two coexisting loaders                 |
| Robustness           | Weak    | Global batch register, hardcoded indices |
| Reproducibility      | Good    | Config-driven, seeded                  |
| Scalability          | Limited | Full graph O(N²), single scaler        |
| Test coverage        | Partial | Happy path only                        |
| Extensibility        | Good    | Registration pattern works well        |
| ETL defensiveness    | Weak    | No alignment validation post-DuckDB   |
| Documentation        | Fair    | Sparse; no architecture guide          |

---

## Pipeline Summary

```
Raw Parquet Hives (Energy + Weather)
  ↓
[loader/graph_dataset.py] DuckDB ETL
  ├─ Join: Energy × MAC metadata
  ├─ Join: MAC zip code × Weather station
  ├─ Join: Weather measurements
  ├─ Impute: Station → Fleet avg → Zero
  └─ Pivot: [T, N] matrices for load/pv/net
  ↓
Temporal Encoding: [T, 6] sine/cosine (month, weekday, hour)
  ↓
MinMaxScaler (fitted on train split, applied to all)
  ↓
Master Bundle.pt
  ├─ net_scaled:     [T, N]
  ├─ load_scaled:    [T, N]
  ├─ pv_scaled:      [T, N]
  ├─ mask_raw:       [T, N]  (validity)
  ├─ weather_data:   [T, N, 5]
  ├─ temporal_data:  [T, 6]
  ├─ pos:            [N, 2]  (coordinates)
  ├─ macs:           [N]
  ├─ zips:           [N]
  └─ timestamps:     [T]
  ↓
[EARNeGraphDataset] Runtime Filtering
  ├─ Apply node mask (filter_zips / filter_macs)
  ├─ Slice temporal (limit_t)
  ├─ Generate topology (spatial_knn or full_graph)
  └─ Cache for epoch iteration
  ↓
[get(idx)] Sample Assembly → Data(x, y, weather, edge_index, pos, macs, zips)
  ↓
[EARNeTemporalEncoder] Conv1D → [N, 64]
  ↓
[EARNeNetwork] Integration → 2× GATConv + Residual + LayerNorm + Dropout → [N, 64]
  ↓
[earne_head] Dual linear heads → [N, 6]  (load q0.1/q0.5/q0.9 | pv q0.1/q0.5/q0.9)
  ↓
[earne_loss_complex] Pinball + Physics
  ↓
[earne_mae_load / earne_mae_pv] Denormalized MAE (Watts)
```
