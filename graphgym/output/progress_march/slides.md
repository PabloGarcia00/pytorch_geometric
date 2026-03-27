---
theme: seriph
background: ./background.png
class: text-center
highlighter: shiki
lineNumbers: true
drawings:
  persist: false
transition: slide-left
title: Spatio-Temporal Graph Neural Networks for Implicit Weather-aware Behind-the-Meter Load Disaggregation
---

## Spatio-Temporal Graph Neural Networks for Implicit Weather-aware Behind-the-Meter Load Disaggregation

**Llan Almendariz Garcia, Wilfried van Sark, Tarek Alskaif**

<div class="grid grid-cols-2 gap-4 text-xs mt-8 opacity-80">
  <div class="text-left">
    <strong>Wageningen University & Research</strong><br>
    Information Technology Group<br>
    llan.almendarizgarcia@wur.nl<br>
    Tarek.Alskaif@wur.nl
  </div>
  <div class="text-left border-l pl-4">
    <strong>Utrecht University</strong><br>
    Copernicus Institute of Sustainable Development<br>
    w.g.j.h.m.vansark@uu.nl
  </div>
</div>

<div class="abs-br m-6 text-xs italic text-gray-400">
Supported by the MESSM project, funded by RVO (Netherlands Enterprise Agency), Project No. 2321202.
</div>

---

# 1. Introduction & Motivation
### At-the-Meter (AtM) vs. Behind-the-Meter (BtM)
* **DSO Constraint:** Operators only see **At-the-Meter (AtM)** net load.
* **Privacy Barrier:** **Behind-the-Meter (BtM)** device data is restricted by privacy laws.
* **Data Gap:** Standard models require high-res weather and panel metadata (tilt/azimuth) rarely available.
* **ISP potential:** Use ISP-sourced datasets (**EARNE, BeNext**) containing both Inverter and SM data to train a robust BtM disaggregation model. 

**Objective:** Leverage spatial correlations among households to **implicitly** capture latent weather factors, eliminating the need for explicit irradiance measurements.



<div class="flex flex-col items-center justify-center"> 
<pre class="text-[10px] leading-tight font-mono">
[ At-the-Meter (AtM) ]  ──▶  [ The Inference Engine ]  ──▶  [ Behind-the-Meter (BtM) ]
          │                           │                           │
    Observed Net Load                 │                   Inferred Components
         (Known)                      │                         (Latent)
          ▼                           ▼                           ▼
    ┌────────────┐            ┌───────────────┐           ┌──────────────────┐
    │  P_net_t   │ ─────────▶ │    ST-GNN     │ ─────────▶│  🏠 P_load_t (?)  │
    └────────────┐            └───────────────┘           │  ☀️ P_pv_t   (?)  │
                                      ▲                   └──────────────────┘
                                      │                            │
                                      └────────────────────────────┘
                                         Physics-Aware Constraint
                                       "Ensuring BtM sum = AtM"
</pre>
</div>

<!--
**Energy balance identity (the fundamental constraint):**
P_net(t) = P_load(t) - P_pv(t)

- P_net: net demand at smart meter — the *only* observable (AtM)
- P_load: household consumption — latent (BtM)
- P_pv: solar generation at inverter — latent (BtM)

The task is underdetermined: one equation, two unknowns.
The GNN solves this by leveraging spatial correlations across neighbours,
implicitly encoding shared cloud-cover / irradiance without explicit weather data.

**Identity mismatch check** (src/time_series_cleanup.py):
  residual = P_load - (P_net + P_pv)
  Tolerance: 5 W (configured in config.yaml).
  Positive residual → uncaptured consumption; negative → sensor inconsistency.
-->

---
layout: section
---

# Part I: Data Infrastructure
## Pipeline Architecture & Pre-processing

---
layout: two-cols
---

# Data Engineering Pipeline
### 21.2M Record Pre-processing

```text
Raw CSVs (data/)
    │
    ▼
[Snakemake Pipeline]
    │
    ├─ Metadata extraction & validation
    ├─ Per-file cleaning (Hampel + Kalman)
    ├─ Device-level merge (SM + Inverter)
    ├─ Hive partitioning (MAC / year)
    ├─ Weather fetch (KNMI API → Hive)
    └─ Analysis pre-computation
            │
            ▼
    output/clean_data_hive/
    output/weather_hive/
    output/calculations/
```

::right::

<div class="mt-10 ml-10">

### Data Quality Metrics
Empirical results from the cleaning phase:

| Metric | Count / Result |
| :--- | :--- |
| **Total Records** | 21,272,304 |
| **Outliers Corrected** | 17,703,984 |
| **Missing Recovered** | 15,176,609 |
| **Devices (MACs)** | 136 paired AtM/BtM |

</div>

<!--
**Total Records — 21,272,304** (src/rules/calculate_pipeline_kpis.py)
  SELECT count(*) FROM energy_data
  The `energy_data` DuckDB view reads all cleaned Parquet files from
  output/clean_data_hive/**/*.parquet and aggregates raw 5-min cadence to
  15-min cadence via time_bucket('15 minutes', MessageTimestamp), averaging
  power within each bucket. Count = (15-min slots across all device-days) × 136.

**Outliers Corrected — 17,703,984** (src/rules/calculate_pipeline_kpis.py)
  Summed across all numerical columns (consumption_w, generation_w, inverter_w,
  EXPORT_KWH, EXPORT_W, IMPORT_KWH, IMPORT_W, …) and all 136 devices.
  Hampel filter algorithm (src/utils.py → apply_hampel_filter):
    1. Rolling median over centred window of 2×hampel_window=20 points.
    2. Rolling MAD over same window → robust std = 1.4826 × MAD.
    3. Flag if |value − rolling_median| > n_sigma × rolling_std  (n_sigma=3, config.yaml).
    4. Replace flagged points with rolling median.
  Per-file counts stored as {col}_outlier in .stats.json, aggregated in mac_overview_updated.csv.

**Missing Recovered — 15,176,609** (src/rules/calculate_pipeline_kpis.py)
  Counts values that WERE NaN and were successfully filled (not remaining NaNs).
  Kalman / RTS Smoother algorithm (src/time_series_cleanup.py):
    1. EM (Shumway-Stoffer, 5 iters, ≤2016-pt subsample) estimates per-column Q and R.
       Model: random walk x_t = x_{t-1} + w_t, observation y_t = x_t + v_t.
    2. Forward Kalman pass:
       - Predict: x_pred = x_hat[t-1], p_pred = p_hat[t-1] + Q
       - Nighttime prior for inverters (hour < 6 or ≥ 19): night_R = 0.01
       - Update if obs exists: K = p_pred/(p_pred+R), x_hat = x_pred + K×(y−x_pred)
       - If obs is NaN: propagate prediction (x_hat = x_pred)
    3. Backward RTS smoother for smoother gap fills.
    4. Clip to [0, ∞) — power cannot be negative.

**136 Paired Devices** (src/device_metadata.py)
  Paired ⟺ device has BOTH a Smart Meter (P1) file AND an Inverter (PV) file:
    f = lambda origins: set(origins) == {"inverter", "smart meter"}
  Unpaired devices excluded from modelling.
-->

---
layout: two-cols
colSpecs: "40/60"
---

# The Analytical Lake
### High-Performance Feature Engineering

- **Hive Partitioning:** MAC/Year indexing for device lookup.
- **Pre-computed Artefacts:** Parquet files for fingerprints, seasonal stats, and autocorrelation.
- **In-Memory Speed:** **DuckDB** view stack enables real-time exploration of 136+ devices.

<p class="mt-12 text-sm text-gray-500 italic">
  "The pipeline enables us to zone in on a group of MAC devices that have high residual load, indicating potential issues"
</p>

::right::

<div class="flex flex-col h-full justify-between items-center pl-6">

  <div class="w-full">
    <img src="./residuals.png" class="rounded shadow-lg border border-white/10 max-h-[250px] mx-auto" />
  </div>

  <div class="w-full">
    <img src="./umap.png" class="rounded shadow-lg border border-white/10 max-h-[250px] mx-auto" />
  </div>

</div>

---
layout: two-cols
colSpecs: "45/55"
---

# High-Throughput Modeling
### From DuckDB Hive to GraphGym

The DuckDB view stack serves as the high-speed ETL engine, converting relational Hive-partitions into a vectorized graph dataset.

- **Master Bundle.pt:** pre-compiled tensor.
- **GraphGym Compatibility:** enables modularized Neural Network architectures that can easily be reconfigured.


::right::

<div class="flex flex-col items-center justify-center"> 
<pre class="text-[11px] leading-tight font-mono">
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
</pre>
</div>

<!--
**MinMaxScaler:** fitted exclusively on the training split (chronological 85%),
then applied to validation and test to prevent data leakage.

**mask_raw [T, N]:** boolean validity mask; a cell is False if the original
reading was NaN or failed the identity check (|P_net − (P_load − P_pv)| > 5 W).

**Temporal encoding [T, 6]:** sine/cosine pairs for month, weekday, and hour —
captures cyclic seasonality and intra-day patterns without ordinal bias.

**Graph topology options:**
  - spatial_knn: K-nearest neighbours on GPS coordinates (pos [N, 2]).
  - full_graph: every node connected to every other node.
-->

---
layout: section
---

# Part II: Methodology & Results
### Time-Then-Space (TTS) ST-GNN Architecture & Weather Ablation Experiment

---
layout: two-cols
colSpecs: "45/55"
---

# Network Architecture: ST-GNN
### Temporal Convolution & Spatial Attention

We define a set of prosumers and solve disaggregation via a **Time-Then-Space (TTS)** architecture:

* **Temporal Block (1D-CNN):** Diurnal patterns 12h lookback (at 15min periods -> kernel=48). 
* **Spatial Block (GATv2):** Dynamic attention reweights adjacency matrix. 
* **Regression Heads:** Quantile head for PV and BTM-load for uncertainty-aware estimates.
* **Pinball loss:** Different penalties at different quantiles.

::right::


<div class="flex flex-col items-center justify-center"> 
<pre class="text-[11px] leading-tight font-mono">
[ Sample Assembly ]
Data(x, y, weather, edge_index, pos)
    ↓
[ EARNeTemporalEncoder ]
Conv1D (Kernel=48) → [N, 64]
    ↓
[ EARNeNetwork ]
2× GATConv (4 heads) + Res + LN 
Captures Spatial Correlations → [N, 64]
    ↓
[ Dual Quantile Heads ]
Linear Split
Out: [N, 6] (Quantiles: 10% | 50% | 90%)
    ↓
[ Physics-Aware Loss ] 
Pinball + Identity Constraint 
    ↓
[ Metrics: Watts MAE ]
Denormalized Fleet Performance
</pre>
</div>

<!--
**Kernel = 48 derivation:**
  12-hour lookback ÷ 15-min resolution = 48 time steps → Conv1D kernel size = 48.

**Output dimensionality — 6 values per node:**
  Dual quantile heads, each producing 3 quantiles:
    q_{0.1}, q_{0.5}, q_{0.9} for P_load
    q_{0.1}, q_{0.5}, q_{0.9} for P_pv
  → [N, 6] per graph snapshot.

**Pinball loss (quantile regression loss):**
  L(q, y, ŷ) = q × max(y − ŷ, 0) + (1−q) × max(ŷ − y, 0)
  - At q=0.5: reduces to MAE.
  - Asymmetric penalty: under-prediction is penalised at rate q,
    over-prediction at rate (1−q). Encourages calibrated uncertainty bounds.

**Physics-Aware Identity Constraint:**
  The loss includes a term penalising deviation from P_net = P_load − P_pv
  on the median (q=0.5) outputs.
-->

---
layout: two-cols
colSpecs: "45/55"
---

# Experiment: Ablation Matrix
### Spatial Topology vs. Weather Features

**Architecture Details**
- **Spatial GNN Body:** 
  - **Graph-Aware (MP=2):** 2-layer message passing.
  - **Local-Only (MP=0):** no message passing.

**Training Protocol**
- **Optimization:** Adam (LR=0.001), 100 Epochs.
- **Validation:** 15% split for checkpointing.
- **Primary Metrics:** Denormalized MAE (Watts).

::right::

<div class="mt-10 ml-10">

| Feature | Weather-Explicit | Weather-Implicit |
| :--- | :--- | :--- |
| **ST-GNN** (MP=2) | **Hybrid:** Full Topology + KNMI | **Proposed:** Topology Only |
| **Baseline** (MP=0) | **Classic:** Node Features + KNMI | **Ablation:** Net Load Only |

</div>

<!--
**MP=0 (Local-Only / Baseline):**
  GATConv layers are configured with add_self_loops=True but no edges between
  distinct neighbours — each node attends only to itself. Functionally equivalent
  to a per-node MLP. No spatial information crosses node boundaries.

**MP=2 (Graph-Aware / Proposed):**
  2 rounds of GATv2 message passing. At each layer every node aggregates
  weighted messages from its K spatial neighbours (KNN graph on GPS coordinates).
  Multi-head attention (4 heads) with residual connection and layer norm.
  After 2 hops a node has an effective receptive field of 2 edges in the graph.

**Training/validation split:** chronological 85% / 15% — NOT random.
  Chronological ordering prevents future data leaking into training
  (time-series leakage). Validation set used for checkpoint selection only.

**Optimiser:** Adam, LR=0.001, up to 100 epochs.
  Best checkpoint = epoch with lowest validation pinball loss.
-->

---
layout: two-cols
colSpecs: "40/60"
---

# Results: MAE
### Marginal Improvement Weather + GNN

The ST-GNN (MP=2) demonstrates a **"Proxy Effect"**, recovering accuracy lost when weather data is removed.

- **Load Prediction:** Consistent performance across regional clusters; slight decay at national scale.
- **PV Prediction:** High sensitivity to graph density; spatial attention captures cloud-cover better than local baselines.
- **Key Metric:** Denormalized Mean Absolute Error (Watts).

::right::

<div class="grid grid-cols-2 gap-3 h-full items-center">

<div>
  <img src="./visualization_results/mae_load_grid.png" class="rounded border border-white/10 shadow-lg w-full" />
  <p class="text-center text-[10px] text-gray-500 mt-2">Fig A: Best Load Error</p>
</div>

<div>
  <img src="./visualization_results/mae_pv_grid.png" class="rounded border border-white/10 shadow-lg w-full" />
  <p class="text-center text-[10px] text-gray-500 mt-2">Fig B: Best PV Error</p>
</div>

</div>

---
layout: two-cols
colSpecs: "40/60"
---

# Inference Trajectories
### Diurnal Profiles & Temporal Stability

Visualizing the "Unmixing" process at high temporal resolution for two 'dissimilar' days. Showing difficulties with accurately modelling peaks, and $q_{0.1}$ is always 0.

- **Dotted line:** Median value.
- **Quantile Coverage:** The $q_{0.1} \dots q_{0.9}$.
- **Resolution:** 15-minute intervals.


::right::

<div class="flex flex-col h-full justify-center pl-4">
<img src="./visualization_results/profile_daily.png" class="rounded border border-white/10 shadow-lg w-full" />
<p class="mt-4 text-[10px] text-center text-gray-500 italic">
  Two Dissimilar Daily Profiles: Load and PV signals with uncertainty bounds.
</p>
</div>

---
layout: section
---

# Part III: Discussion & Future Directions
## How To Maximize efficiency?

---
layout: two-cols
---

# Short-Term Priorities
### More Experiments or New Model?

Results show small effect between configurations. This begs the question if we proceed with the current model or perhaps switch to ST-GCCaps (by Lorance).
Key considerations:
- **Training Time**: 4x 4 models in parallel (my computer) 1.5 days.
- **Server Availability:** Still waiting on Alliander contract.
- **Scope of Paper:**: Whether the paper idea is compatible with new setup.
- **Integration:** Model has been integrated in the development environment (almost) ready.

<p class="mt-12 text-sm text-gray-500 italic">
    Otherwise we stick to plan.
</p>


::right::


| Experiment | Objective | Significance |
| :--- | :--- | :--- | 
| **Ablation** | Ablate Weather | Implicit Learning |
| **Cloud Passing** | Isolate GATv2 Response | Cloud Pass Response |
| **Transfer** | Train on ZIP A test ZIP B | Generalizability |

--- 
layout: two-cols
colSpecs: "50/50"
---

# Roadmap (Q2 2026)
### Scaling & High-Throughput Research

**Current Capability: The Engine**
- **Parallel Execution:** High-throughput testing of model variants using the **DuckDB-to-GraphGym** bridge.
- **Production Ready:** Architecture is ready for cloud deployment and national-scale datasets (EARN-E + BeNext).

::right::

**Future Roadmap: The Laboratory**
- **More Complex Models:**
  - Using Import and Export inplace of Net Demand
  - Integratting ST-GCCaps
  - Adapting ST-GCCaps using Siamese architecture
- **Explore Data Preprocessing Configurations**
  - Different Normalization Techniques (e.g. zero-preserved log normalization)

---
layout: center
---

# Thank You
## Questions?

---
layout: default
---

# Experiment 1: Best Validation Loss
### Impact of Message Passing (MP) and Weather Features

The results confirm that **Spatial Topology (MP=2)** provides a greater performance gain than explicit weather features.

| Configuration | Mean Best Loss | Avg. Convergence |
| :--- | :---: | :---: |
| **No Weather, MP=0** (Baseline) | 0.0124 | ~74 Epochs |
| **Weather, MP=0** | 0.0122 | ~80 Epochs |
| **No Weather, MP=2** (Proposed) | 0.0116 | ~94 Epochs |
| **Weather, MP=2** (Hybrid) | **0.0114** | ~93 Epochs |

<!--
**Loss metric:** Pinball loss summed over all 6 quantile outputs (3 quantiles × 2 heads:
  load and PV), averaged over the validation set.
  L(q, y, ŷ) = q × max(y − ŷ, 0) + (1−q) × max(ŷ − y, 0)

**Mean Best Loss:** average of the best validation loss across runs in each
  configuration (4 seeds or 4 graph-size variants — check run config).

**Avg. Convergence:** epoch at which the best validation checkpoint was reached.
  Later convergence for MP=2 variants reflects the additional complexity of
  neighbourhood aggregation requiring more gradient steps to stabilise.

**Training/validation split:** chronological 85% / 15%, not random,
  to prevent temporal data leakage.

**Key takeaway:** MP=2 gains (+0.008 loss reduction) exceed weather features
  (+0.002 loss reduction), confirming the spatial proxy effect hypothesis.
-->

---
