# Experiment highlights

_Generated 2026-07-17 15:49 UTC. Best-seed-by-MAE per config. All training runs referenced here are complete (3/3 seeds); see full `metrics_report.md` for per-seed detail._

### 1. No physics constraint - all models land in the same range

_Load-disaggregation accuracy, single-read, no physics mask/loss, weather=True._

| Model | Load MAE (optimal) | Load MAE (sub-optimal) | PV MAE (optimal) | PV MAE (sub-optimal) |
|---|---|---|---|---|
| EARNE | 275.24 | 290.83 | 118.69 | 129.85 |
| MLP | 284.94 | 289.81 | 119.91 | 126.35 |
| LSTM | 275.00 | 286.18 | 117.69 | 128.32 |
| CVAE | 290.09 | 293.54 | 109.06 | 116.21 |

**Takeaway: without the physics constraint, all four models perform comparably at both optimal and sub-optimal coverage - no architecture stands out.**

### 2a. Physics mask only (physics_weight=0.0)

_Weather=True. Mask drops physically-impossible timesteps (generation > inverter capacity) from training/eval, but the loss has no physics constraint. CVAE has no physics-loss term to begin with, so its physics run already **is** mask-only._

| Model | Load MAE (W) | PV MAE (W) | PV<Export rate |
|---|---|---|---|
| EARNE | 278.67 | 105.45 | 0.2198 |
| MLP | 280.41 | 105.21 | 0.4853 |
| LSTM | 274.21 | 105.57 | 0.4084 |
| CVAE | 271.70 | 96.48 | 0.0146 |

**Takeaway: mask-only stays close to the no-physics numbers (table 1) for EARNE/MLP/LSTM - the ~7.5% of timesteps dropped by the mask costs little accuracy on its own.**

### 2b. Physics mask + physics loss (physics_weight=1.0)

_Weather=True. Same mask as 2a, plus the physics-loss term (load - PV = net demand) added to training. Not applicable to CVAE - its loss function has no physics_weight term, so there's no "+loss" variant for it._

| Model | Load MAE (W) | PV MAE (W) | PV<Export rate |
|---|---|---|---|
| EARNE | 610.90 | 145.60 | 0.0657 |
| MLP | 613.02 | 138.13 | 0.0667 |
| LSTM | 650.41 | 144.81 | 0.0728 |

**Takeaway: adding the loss term roughly doubles Load MAE vs. mask-only (2a) for EARNE/MLP/LSTM, but cuts the PV<Export violation rate substantially - the loss term trades accuracy for physical plausibility; the mask alone barely costs anything.**

### 3. Weather effect - minor, and inconsistent in direction

_No-physics runs, single-read, optimal coverage._

| Model | Load R² (wx=False) | Load R² (wx=True) | Load MAE (wx=False) | Load MAE (wx=True) | PV R² (wx=False) | PV R² (wx=True) | PV MAE (wx=False) | PV MAE (wx=True) |
|---|---|---|---|---|---|---|---|---|
| EARNE | 0.805 | 0.813 | 282.61 | 275.24 | 0.800 | 0.802 | 120.14 | 118.69 |
| MLP | 0.809 | 0.801 | 276.77 | 284.94 | 0.813 | 0.789 | 117.42 | 119.91 |
| LSTM | 0.796 | 0.800 | 279.97 | 275.00 | 0.792 | 0.764 | 119.99 | 117.69 |
| CVAE | 0.823 | 0.820 | 283.77 | 290.09 | 0.861 | 0.857 | 113.02 | 109.06 |

**Takeaway: adding weather features moves R² by only ~0.01-0.02 either way, depending on the model - no consistent, meaningful effect from weather alone.**

### 4. Coverage effect - optimal (62 nodes) beats sub-optimal (112 nodes) for every model

_No-physics runs, single-read, weather-averaged. Optimal = only nodes active in both the first and last week of data; sub-optimal = full native node set, including sparser/noisier nodes._

| Model | Load R² (optimal) | Load R² (sub-optimal) | Load MAE (optimal) | Load MAE (sub-optimal) | PV R² (optimal) | PV R² (sub-optimal) | PV MAE (optimal) | PV MAE (sub-optimal) |
|---|---|---|---|---|---|---|---|---|
| EARNE | 0.809 | 0.794 | 278.92 | 288.16 | 0.801 | 0.768 | 119.41 | 127.53 |
| MLP | 0.805 | 0.787 | 280.86 | 288.78 | 0.801 | 0.776 | 118.67 | 125.33 |
| LSTM | 0.798 | 0.769 | 277.48 | 286.07 | 0.778 | 0.755 | 118.84 | 127.44 |
| CVAE | 0.822 | 0.800 | 286.93 | 293.21 | 0.859 | 0.845 | 111.04 | 115.18 |

**Takeaway: optimal coverage consistently gives a small but real accuracy edge over sub-optimal across every model - the extra sparser/noisier nodes in the sub-optimal set add a bit of difficulty, but it's a much smaller effect than the physics-loss issue.**

### 5. Single-read vs. dual-read - dual-read edges out single-read

_No-physics runs, optimal coverage, weather-averaged. Dual-read gives the model separate consumption + generation channels instead of one pre-combined net-demand channel._

| Model | Load R² (single) | Load R² (dual) | Load MAE (single) | Load MAE (dual) | PV R² (single) | PV R² (dual) | PV MAE (single) | PV MAE (dual) |
|---|---|---|---|---|---|---|---|---|
| EARNE | 0.809 | 0.814 | 278.92 | 273.70 | 0.801 | 0.813 | 119.41 | 114.66 |
| MLP | 0.805 | 0.813 | 280.86 | 265.25 | 0.801 | 0.817 | 118.67 | 108.79 |
| LSTM | 0.798 | 0.810 | 277.48 | 261.58 | 0.778 | 0.806 | 118.84 | 108.39 |
| CVAE | 0.822 | 0.829 | 286.93 | 271.73 | 0.859 | 0.870 | 111.04 | 100.66 |

**Takeaway: dual-read (separate consumption/generation channels) gives a modest but consistent accuracy bump over single-read (pre-combined net demand) across all four models - most noticeably on PV.**
