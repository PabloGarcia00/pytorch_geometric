# System Report: EDA Pipeline & Dashboard

**Project:** MESSM Exploratory Data Analysis
**Stack:** Snakemake · DuckDB · Streamlit · PyTorch · KNMI API

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Rule Dependency Graph](#2-rule-dependency-graph)
3. [Cleaning Pipeline — EarneCleaner](#3-cleaning-pipeline--earnecleaner)
4. [Analysis Rules](#4-analysis-rules)
5. [Dashboard Overview](#5-dashboard-overview)
6. [DuckDB View Hierarchy](#6-duckdb-view-hierarchy)
7. [Dashboard Sections](#7-dashboard-sections)
8. [Configuration Reference](#8-configuration-reference)

---

## 1. Pipeline Overview

The pipeline ingests raw CSV exports from smart meters and solar inverters, cleans and merges them into a Hive-partitioned dataset, joins them with meteorological observations from KNMI, then pre-computes analytical artefacts consumed by the dashboard.

```
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
    output/clean_data_hive/    ← partitioned energy data
    output/weather_hive/       ← partitioned weather data
    output/calculations/       ← pre-computed parquets + JSON
            │
            ▼
    [Streamlit Dashboard]
    DuckDB in-memory views → interactive exploration
```

---

## 2. Rule Dependency Graph

```mermaid
flowchart TD
    RAW["📁 data/\nRaw CSVs\n(smart meter + inverter)"]
    ZIP["📄 assets/\nzipcode_lookup.csv\nmac_lookup.csv"]

    subgraph CHECKPOINT_1["⚙️ checkpoint: generate_metadata"]
        GEN["device_metadata.py\n- Scan data/ for CSV files\n- Extract MAC via hex regex\n- Classify origin (P1=SM, else=INV)\n- Validate SM+INV pairs exist\n- Map MAC → 2-digit postcode\n- Set is_valid flag"]
    end

    subgraph CLEAN["🔁 rule: clean_file  ×278 (one per CSV file)"]
        CF["clean_file_rule.py\n+ EarneCleaner\n- Resample to 5-min cadence\n- Hampel outlier filter\n- Kalman EM imputation\n- Derive power metrics\n→ {stem}.parquet + stats.json"]
    end

    subgraph CHECKPOINT_2["⚙️ checkpoint: aggregate_metadata"]
        AGG["aggregate_stats_rule.py\n- Merge all stats.json → metadata\n- Adds: start_date, end_date,\n  {col}_missing, _imputed, _outlier\n→ mac_overview_updated.csv"]
    end

    subgraph MERGE["🔁 rule: merge_mac  ×136 (one per device)"]
        MM["merge_mac_rule.py\n- Concat SM parquets for MAC\n- Concat INV parquets for MAC\n- Outer join on MessageTimestamp\n- Reindex to global fleet range\n- Re-derive literature situation\n→ {mac}.parquet"]
    end

    subgraph HIVE["🔁 rule: write_mac_to_hive  ×136"]
        WH["write_mac_to_hive_rule.py\n- Filter to 7 power columns\n- Split by year\n→ clean_data_hive/MAC={mac}/year={y}/data.parquet"]
    end

    subgraph SENTINEL["rule: finalize_hive"]
        FH["touch clean_data_hive/.complete\n(dependency sentinel)"]
    end

    subgraph WEATHER["rule: fetch_weather_data  (parallel)"]
        WD["weather_data_fetcher.py\n- KNMI API per zip code\n- 10-min meteorological obs\n- Remap column names (config)\n→ weather_hive/station={id}/year={y}/\n→ zip_to_station_mapping.csv"]
    end

    subgraph ANALYSIS["Analysis Rules  (run after hive + weather)"]
        SS["calculate_seasonal_stats\n16 GB · 4 threads\n→ seasonal_stats.parquet"]
        FP["calculate_fingerprints\n16 GB · 4 threads\n→ fleet_fingerprints.parquet"]
        AC["calculate_autocorrelation\n16 GB · 4 threads\n→ autocorrelation_stats.parquet"]
        KP["calculate_pipeline_kpis\n8 GB · 4 threads\n→ pipeline_kpis.json"]
    end

    ALL["✅ rule: all\n(final target)"]

    RAW --> CHECKPOINT_1
    ZIP --> CHECKPOINT_1
    CHECKPOINT_1 --> CLEAN
    CLEAN --> CHECKPOINT_2
    CHECKPOINT_2 --> MERGE
    MERGE --> HIVE
    HIVE --> SENTINEL
    SENTINEL --> SS & FP & AC & KP
    WEATHER --> SS & FP & AC & KP
    CHECKPOINT_2 --> SS & FP & AC & KP
    SS & FP & AC & KP --> ALL
    SENTINEL --> ALL
    WEATHER --> ALL
    CHECKPOINT_2 --> ALL
```

**Resource scheduling:** The global `max_mem_mb = 25,538` cap (80 % of 31 GB) prevents the scheduler from running two 16 GB analysis rules simultaneously. The 8 GB KPI rule can overlap with one 16 GB rule.

---

## 3. Cleaning Pipeline — EarneCleaner

Each raw CSV goes through `EarneCleaner.process_single_file()` in `src/time_series_cleanup.py`.

```mermaid
flowchart TD
    CSV["📄 Raw CSV\nsep=';'  decimal=','"]

    subgraph LOAD["Step 1 — Load & Schematize"]
        L1["Read CSV\nDrop 100%-null columns"]
        L2["Parse MessageTimestamp → UTC\nCast numeric cols to float32"]
    end

    subgraph PREP["Step 2 — Prepare Time Series"]
        P1["De-duplicate timestamps\n(keep first)"]
        P2["Resample to 5-min cadence\n(mean aggregation)"]
        P3["Drop data before 2021-01-01"]
        P4["Reindex to continuous grid\n(pd.date_range, freq='5min')"]
    end

    subgraph STD["Step 3 — Standardize Power Metrics"]
        S1["Smart Meter path:\nconsumption_w = Δ(IMPORT_LOW + IMPORT_NORMAL) / Δt × 1000\ngeneration_w  = Δ(EXPORT_LOW + EXPORT_NORMAL) / Δt × 1000"]
        S2["Inverter path:\ninverter_w = EXPORT_KWH → kWh→W conversion\n(fallback: EXPORT_W × 1000, or EXPORT_KW × 1000)"]
        S3["Invalidate on:\n- time gap > cadence × 1.1\n- negative energy delta\n- jump from midnight reset"]
        S4["Hard clip: values > 25 kW → 25 kW"]
    end

    subgraph HAMPEL["Step 4 — Hampel Outlier Filter"]
        H1["Sliding window = 10 samples (50 min)\nThreshold = 3 × MAD\nReplace spike → window median\nTrack {col}_outlier count"]
    end

    subgraph KALMAN["Step 5 — Kalman EM Imputation"]
        K1["Per column:\nSubsample series to ≤ 2016 pts\n(≈1 week at 5-min cadence)"]
        K2["Warm-start Q, R\nvia Method of Moments:\nQ ≈ 0.5 × Var(Δx_obs)\nR ≈ Var(rolling residuals)"]
        K3["5 × EM iterations:\nE-step: Forward Kalman + RTS Backward smoother\nM-step: Update Q, R from state dynamics"]
        K4["Apply smoother on full series\nNighttime prior (6PM–6AM):\nBlend x̂ → 0 for generation/inverter"]
        K5["Clip imputed values ≥ 0\nRecord {col}_imputed mask"]
    end

    subgraph LIT["Step 6 — Literature Situation"]
        LT1["net_demand = consumption_w − generation_w"]
        LT2["self_consumption_w = (inverter_w − generation_w).clip(0)"]
        LT3["load = consumption_w + self_consumption_w"]
        LT4["Identity check:\n|net_demand − (load − inverter_w)| ≤ 5 W\n→ warn if violated"]
    end

    subgraph STATS["Step 7 — Stats Collection"]
        ST1["missing_stats(): count NaN per col\ndate_range_stats(): min/max timestamp\n→ written to {stem}.stats.json"]
    end

    OUT1["📦 {stem}.parquet\nIntermediate cleaned file"]
    OUT2["📄 {stem}.stats.json\nCleaning statistics"]

    CSV --> LOAD --> PREP --> STD --> HAMPEL --> KALMAN --> LIT --> STATS
    STATS --> OUT1 & OUT2
```

---

## 4. Analysis Rules

All four analysis rules share the same data setup via `setup_analysis_con()` in `src/analysis_utils.py`, which builds the DuckDB view stack described in §6.

```mermaid
flowchart LR
    HIVE["clean_data_hive/\nweather_hive/\nmac_overview_updated.csv"]

    subgraph SETUP["setup_analysis_con()"]
        V["Build DuckDB views:\nraw_energy_data\nraw_weather_data\nmac_meta\nzip_to_station\nenergy_data\nenergy_weather_joined\nenergy_enriched"]
    end

    subgraph SS["calculate_seasonal_stats"]
        SS1["GROUP BY (MAC, season, weekly_hour)\nMetrics: avg, med, q10, q90\nfor each power column"]
        SS2["→ seasonal_stats.parquet\n(MAC × 168 h × 3 seasons)"]
    end

    subgraph FP["calculate_fingerprints"]
        FP1["Per MAC:\n24-hour profiles (GROUP BY hour)\nAggregate stats: mean, CV, completeness\nGPU behavioral scores:\n  Solarity = NMI(solar_rad, consumption)\n  Thermal = OLS slope (temp → consumption)"]
        FP2["→ fleet_fingerprints.parquet\n(one row per device, ~60 features)"]
    end

    subgraph AC["calculate_autocorrelation"]
        AC1["Per MAC × per power metric:\nGPU FFT-based ACF\nSample at lags:\nhour · half_day · day · week"]
        AC2["→ autocorrelation_stats.parquet\n(MAC, Feature, Shift, Correlation)"]
    end

    subgraph KP["calculate_pipeline_kpis"]
        KP1["Sum {col}_outlier cols → total_outliers\nSum {col}_imputed cols → total_missing\nCOUNT(*) energy_data → total_records\nCOUNT(solar_radiation_avg)/COUNT(*) → weather_match_pct"]
        KP2["→ pipeline_kpis.json"]
    end

    HIVE --> SETUP --> SS & FP & AC & KP
    SS --> SS2
    FP --> FP2
    AC --> AC2
    KP --> KP2
```

---

## 5. Dashboard Overview

```mermaid
flowchart TD
    subgraph INIT["dashboard.py — Startup"]
        D1["load_config()\nget_db_connection() → @st.cache_resource\nget_mac_addresses()\nget_date_range()"]
    end

    subgraph SIDEBAR["Sidebar Controls"]
        SB1["MAC multiselect\n(default: all valid MACs)"]
        SB2["Date range preset\n(All / 3M / 1M / Custom)"]
        SB3["Clear disk cache button"]
    end

    subgraph MAIN["render_disaggregation_dashboard()"]
        S1["§1  Pipeline KPIs\npipeline_kpis.json"]
        S2["§2  Identity & Residual\nenergy_weather_joined filtered"]
        S3["§2b Seasonal & Weekly Profiles\nseasonal_stats.parquet"]
        S4["§3  Lifecycle Gantt\nenergy_data aggregated"]
        S5["§3b Weather Feature Heatmap\nraw_weather_data aggregated"]
        S6["§4  Autocorrelation ECDF\nautocorrelation_stats.parquet"]
        S7["§5  Weather Correlation\nsampled energy_enriched"]
        S8["§7  Fleet Topology (UMAP)\nfleet_fingerprints.parquet"]
        S9["§8  Filtering Table\nAgGrid + export"]
    end

    INIT --> SIDEBAR
    SIDEBAR --> MAIN
    S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7 --> S8 --> S9
```

---

## 6. DuckDB View Hierarchy

Every query in the dashboard and analysis rules operates through this layered view stack:

```mermaid
flowchart TD
    subgraph DISK["On-Disk Artefacts"]
        H1["clean_data_hive/MAC=*/year=*/data.parquet"]
        H2["weather_hive/station=*/year=*/data.parquet"]
        H3["mac_overview_updated.csv"]
        H4["zip_to_station_mapping.csv"]
    end

    subgraph L1["Layer 1 — Raw Hive Views"]
        V1["raw_energy_data\nMESSAGETIMESTAMP, MAC\nconsumption_w, generation_w\ninverter_w, net_demand, load"]
        V2["raw_weather_data\ntimestamp, station, year\nair_temperature, solar_radiation_avg\nrelative_humidity_pct, wind_speed_avg\n+ 9 more weather columns"]
        V3["mac_meta\nMAC, zip_code, is_valid\nstart_date, end_date\n{col}_missing, _imputed, _outlier"]
        V4["zip_to_station\ntwo_number_zip → weather_station_id"]
    end

    subgraph L2["Layer 2 — Time Bucketing"]
        V5["energy_data\ntime_bucket(15 min, MessageTimestamp)\nmean(consumption_w), mean(generation_w)\nmean(inverter_w), mean(net_demand), mean(load)\nGROUP BY (bucket, MAC)"]
    end

    subgraph L3["Layer 3 — Spatial-Temporal Join"]
        V6["energy_weather_joined\nenergy_data\n  ⋈ mac_meta ON MAC\n  ⋈ zip_to_station ON TRY_CAST(zip)\n  ⋈ raw_weather_data ON (timestamp, station)\n+ residual = load − (net_demand + inverter_w)"]
    end

    subgraph L4["Layer 4 — Time Feature Enrichment"]
        V7["energy_enriched\n+ month, hour, dayofweek\n+ weekly_hour = dayofweek × 24 + hour\n+ season (Winter/Shoulder/Summer)"]
    end

    H1 --> V1
    H2 --> V2
    H3 --> V3
    H4 --> V4
    V1 --> V5
    V5 --> V6
    V3 & V4 & V2 --> V6
    V6 --> V7
```

**Important:** Views are lazy (DuckDB defers execution until a query runs). Predicate pushdown through view layers is handled by DuckDB's optimizer, and Hive partitioning on `MAC` and `year` means filters like `WHERE MAC IN (...)` skip irrelevant Parquet files entirely.

---

## 7. Dashboard Sections

### §1 — Pipeline KPIs
```
Source: output/calculations/pipeline_kpis.json  (read once, no query)

┌───────────────────┬────────────────────────┬─────────────────────────┐
│  Total Records    │  Outliers Corrected    │  Missing Recovered      │
│  21,272,304       │  17,703,984 (Hampel)   │  15,176,609 (Kalman)    │
└───────────────────┴────────────────────────┴─────────────────────────┘
```

### §2 — Physical Identity & Residual Analysis
```mermaid
flowchart LR
    Q["SELECT * FROM energy_weather_joined\nWHERE MAC IN (...)\nAND ts BETWEEN date_range\nORDER BY ts"]
    P1["📈 Line chart\nresidual = load − (net_demand + inverter_w)\nzero = perfect balance"]
    P2["📊 Stacked area\nnet_demand + inverter_w → total load"]
    Q --> P1 & P2
```

### §2b — Seasonal & Weekly Profiles
```mermaid
flowchart LR
    SRC["seasonal_stats.parquet\n(pre-aggregated)"]
    CTRL["User selects:\nAvg / Median / Q10 / Q90"]
    PLOT["📈 Line chart per power metric\nX: 168 weekly hours (Mon 00:00 → Sun 23:00)\nY: Power (W)\nColor: Winter / Shoulder / Summer"]
    SRC --> CTRL --> PLOT
```

### §3 — MAC Lifecycle Gantt & Weather Heatmap
```mermaid
flowchart LR
    subgraph LEFT["Left panel"]
        LQ["SELECT MAC, min(ts), max(ts), count(*)\nFROM energy_data GROUP BY MAC"]
        LG["📊 Gantt timeline\nX: date span, Y: MAC\nColor: record_count"]
        LQ --> LG
    end
    subgraph RIGHT["Right panel"]
        RQ["SELECT station, count(feat_i) / count(*)\nFROM raw_weather_data\nGROUP BY station\n→ UNPIVOT"]
        RH["🗺️ Heatmap\nRows: station, Cols: weather feature\nColor: completeness % (0–100)"]
        RQ --> RH
    end
```

### §4 — Autocorrelation ECDF
```mermaid
flowchart LR
    SRC["autocorrelation_stats.parquet\nSingle batch query for all selected MACs"]
    PLOT["📈 ECDF plot (4 facets)\nX: correlation coefficient\nColor: power metric\nFacets: hour · half_day · day · week\n\nReads: what % of the fleet reaches\ncorrelation X at lag Y"]
    SRC --> PLOT
```
Interpretation: a steep ECDF near 1.0 at the day lag means the fleet has strong 24 h periodicity — good signal for disaggregation training.

### §5 — Weather & Energy Correlation
```mermaid
flowchart TD
    SAM["get_sampled_correlation_data()\nUSING SAMPLE 100,000 ROWS (reservoir)\nfrom energy_enriched\n→ cached to disk per MAC selection"]

    subgraph LEFT2["Left: OLS Linearity (GPU)"]
        OLS["calculate_ols_gpu()\nx = selected weather col\ny = selected energy col\n→ slope m, intercept b, R²"]
        SC["📊 Scatter plot (opacity 0.05)\n+ trendline y = mx + b\n+ R² annotation"]
        OLS --> SC
    end

    subgraph RIGHT2["Right: MI Ranking (GPU)"]
        MI["calculate_mi_gpu()\nFor each weather feature:\n50×50 2D histogram\nNMI = 2·I(X;Y) / (H(X)+H(Y))"]
        BAR["📊 Horizontal bar chart\nAll weather features ranked by NMI\nfor selected energy column"]
        MI --> BAR
    end

    SAM --> LEFT2 & RIGHT2
```

### §7 — Fleet Topology (UMAP)
```mermaid
flowchart LR
    FP["fleet_fingerprints.parquet\n(one row per MAC, ~60 features)"]
    BASIS["User selects basis:\nHolistic / Consumption Only\nGeneration Only / Inverter Only\nBehavioral Stats Only"]
    SCALE["StandardScaler\nnp.nan_to_num"]
    UMAP["UMAP\nn_neighbors = min(15, n_macs−1)\nmin_dist = 0.1\nmetric = euclidean\n→ 2D embedding"]
    COLOR["Color by:\nSolarity · Thermal Sensitivity\nVolatility · Completeness\nResidual · Net Demand"]
    MAP["📍 Scatter plot\nX: UMAP_1, Y: UMAP_2\nHover: MAC, stats, zip"]
    CARD["📋 Sidecar: active MAC\n- 24h holistic profile\n- Stats table\n- Behavioral scores"]

    FP --> BASIS --> SCALE --> UMAP --> COLOR --> MAP
    MAP --> CARD
```

### §8 — Filtering & Model-Ready Selection
```mermaid
flowchart LR
    A["mac_meta"] --> JOIN
    B["get_lifecycle_data()\nenergy_data aggregated"] --> JOIN
    C["get_quality_metrics()\nenergy_weather_joined aggregated"] --> JOIN
    D["fleet_fingerprints.parquet\n(solarity, thermal_sensitivity)"] --> JOIN
    E["zip_to_station mapping"] --> JOIN

    JOIN["Joined display_df\n(one row per MAC)"]
    GRID["📋 AgGrid interactive table\n- Excel-like column filters\n- Multi-row checkbox selection\n- Pagination (20/page)"]
    EXPORT["Export:\n💾 Save to mac_overview.csv\n📋 Copy MACs to clipboard\n📥 Download CSV"]

    JOIN --> GRID --> EXPORT
```

---

## 8. Configuration Reference

| Key | Default | Effect |
|-----|---------|--------|
| `cleaning.cadence_minutes` | `5` | Resampling frequency of raw CSVs |
| `cleaning.hampel_window` | `10` | Sliding window for outlier detection (samples) |
| `cleaning.hampel_sigma` | `3` | MAD threshold for spike detection |
| `cleaning.drop_values_before` | `01-01-2021` | Temporal cutoff for all devices |
| `cleaning.tolerance` | `5` | Identity check tolerance in Watts |
| `cleaning.kalman.use_em` | `true` | Enable EM Q/R estimation per column |
| `cleaning.kalman.transition_covariance` | `0.1` | Kalman Q init (process noise) |
| `cleaning.kalman.observation_covariance` | `1.0` | Kalman R init (measurement noise) |
| `cleaning.which_mac` | `"all"` | Run on subset or all devices |
| `dashboard.cadence_minutes` | `15` | Display resolution (time-bucketing in DuckDB) |
| `dashboard.power_metrics` | `[consumption_w, generation_w, inverter_w, net_demand, load]` | Columns loaded and visualized |
| `weather.start_year` | `2021` | First year to fetch from KNMI |
| `weather.request_delay` | `0.5` | Rate-limit delay between API calls (s) |
| `resources.max_mem_mb` | `25538` | Snakemake global memory cap (80 % of RAM) |
| `resources.max_cores` | `12` | Max parallel threads for Snakemake jobs |

---

*Generated 2026-03-26*
