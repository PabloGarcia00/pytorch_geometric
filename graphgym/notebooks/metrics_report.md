# Model comparison report

_Generated 2026-07-17 15:49 UTC from `notebooks/metrics_long.csv` (checkpoint-based test-set disaggregation via `calculate_metrics.py`, denormalized to physical units). Metrics use the best-performing seed by MAE per config, not an average. 'seeds' shows how many of 3 planned seeds have a saved checkpoint (all training is now complete)._

## Exp1 — no physics (learned_corr / no-mask)

### EARNE — single-read

Coverage (span=True=optimal/62 nodes, span=False=sub-optimal/112 nodes) × weather.

| config | seeds | load RMSE | load MAE | load R² | pv RMSE | pv MAE | pv R² | pv export-viol rate |
|---|---|---|---|---|---|---|---|---|
| optimal, wx=False | 3/3 ✓ | 737.11 | 282.61 | 0.805 | 417.38 | 120.14 | 0.800 | 0.4001 |
| optimal, wx=True | 3/3 ✓ | 707.84 | 275.24 | 0.813 | 413.65 | 118.69 | 0.802 | 0.5504 |
| sub-optimal, wx=False | 3/3 ✓ | 721.03 | 285.49 | 0.786 | 435.50 | 125.22 | 0.775 | 0.1098 |
| sub-optimal, wx=True | 3/3 ✓ | 714.90 | 290.83 | 0.801 | 451.12 | 129.85 | 0.761 | 0.5183 |

### EARNE — dual-read

Coverage × weather, dual-read (consumption+generation).

| config | seeds | load RMSE | load MAE | load R² | pv RMSE | pv MAE | pv R² | pv export-viol rate |
|---|---|---|---|---|---|---|---|---|
| optimal, wx=False | 3/3 ✓ | 711.44 | 270.68 | 0.811 | 393.59 | 112.08 | 0.825 | 0.5776 |
| optimal, wx=True | 3/3 ✓ | 699.20 | 276.72 | 0.817 | 437.42 | 117.24 | 0.801 | 0.1355 |
| sub-optimal, wx=False | 3/3 ✓ | 703.66 | 278.71 | 0.799 | 413.53 | 117.81 | 0.797 | 0.0870 |
| sub-optimal, wx=True | 3/3 ✓ | 699.18 | 273.94 | 0.798 | 436.17 | 121.17 | 0.775 | 0.1968 |

### MLP baseline — single-read

No spatial mixing, per-node MLP temporal encoder.

| config | seeds | load RMSE | load MAE | load R² | pv RMSE | pv MAE | pv R² | pv export-viol rate |
|---|---|---|---|---|---|---|---|---|
| optimal, wx=False | 3/3 ✓ | 724.01 | 276.77 | 0.809 | 402.16 | 117.42 | 0.813 | 0.5069 |
| optimal, wx=True | 3/3 ✓ | 732.01 | 284.94 | 0.801 | 425.61 | 119.91 | 0.789 | 0.5306 |
| sub-optimal, wx=False | 3/3 ✓ | 712.64 | 287.75 | 0.792 | 424.32 | 124.30 | 0.787 | 0.6150 |
| sub-optimal, wx=True | 3/3 ✓ | 728.66 | 289.81 | 0.781 | 444.61 | 126.35 | 0.765 | 0.1149 |

### MLP baseline — dual-read

No spatial mixing, per-node MLP temporal encoder, dual-read.

| config | seeds | load RMSE | load MAE | load R² | pv RMSE | pv MAE | pv R² | pv export-viol rate |
|---|---|---|---|---|---|---|---|---|
| optimal, wx=False | 3/3 ✓ | 710.63 | 264.97 | 0.811 | 398.01 | 108.99 | 0.820 | 0.5929 |
| optimal, wx=True | 3/3 ✓ | 703.81 | 265.52 | 0.815 | 401.49 | 108.59 | 0.813 | 0.1964 |
| sub-optimal, wx=False | 3/3 ✓ | 696.67 | 271.05 | 0.799 | 425.31 | 118.50 | 0.799 | 0.4546 |
| sub-optimal, wx=True | 3/3 ✓ | 707.70 | 276.19 | 0.793 | 418.59 | 117.07 | 0.790 | 0.1420 |

### LSTM baseline — single-read

BiLSTM per-node temporal encoder, no spatial mixing.

| config | seeds | load RMSE | load MAE | load R² | pv RMSE | pv MAE | pv R² | pv export-viol rate |
|---|---|---|---|---|---|---|---|---|
| optimal, wx=False | 3/3 ✓ | 738.26 | 279.97 | 0.796 | 422.20 | 119.99 | 0.792 | 0.2171 |
| optimal, wx=True | 3/3 ✓ | 737.35 | 275.00 | 0.800 | 467.70 | 117.69 | 0.764 | 0.0813 |
| sub-optimal, wx=False | 3/3 ✓ | 752.88 | 285.96 | 0.768 | 455.24 | 126.55 | 0.753 | 0.4155 |
| sub-optimal, wx=True | 3/3 ✓ | 746.59 | 286.18 | 0.771 | 448.56 | 128.32 | 0.757 | 0.1860 |

### CVAE baseline — single-read

Generative BiLSTM+VAE model.

| config | seeds | load RMSE | load MAE | load R² | pv RMSE | pv MAE | pv R² | pv export-viol rate |
|---|---|---|---|---|---|---|---|---|
| optimal, wx=False | 3/3 ✓ | 709.43 | 283.77 | 0.823 | 348.85 | 113.02 | 0.861 | 0.0370 |
| optimal, wx=True | 3/3 ✓ | 715.14 | 290.09 | 0.820 | 351.21 | 109.06 | 0.857 | 0.0305 |
| sub-optimal, wx=False | 3/3 ✓ | 708.94 | 292.87 | 0.804 | 354.73 | 114.16 | 0.851 | 0.0300 |
| sub-optimal, wx=True | 3/3 ✓ | 714.15 | 293.54 | 0.796 | 368.70 | 116.21 | 0.839 | 0.0406 |

### LSTM baseline — dual-read

BiLSTM per-node temporal encoder, no spatial mixing, dual-read.

| config | seeds | load RMSE | load MAE | load R² | pv RMSE | pv MAE | pv R² | pv export-viol rate |
|---|---|---|---|---|---|---|---|---|
| optimal, wx=False | 3/3 ✓ | 693.79 | 258.01 | 0.820 | 394.98 | 107.07 | 0.820 | 0.4204 |
| optimal, wx=True | 3/3 ✓ | 732.23 | 265.15 | 0.799 | 424.07 | 109.70 | 0.791 | 0.1568 |
| sub-optimal, wx=False | 3/3 ✓ | 717.72 | 272.02 | 0.787 | 419.43 | 116.85 | 0.790 | 0.5996 |
| sub-optimal, wx=True | 3/3 ✓ | 720.16 | 273.22 | 0.786 | 413.56 | 116.33 | 0.796 | 0.5918 |

### CVAE baseline — dual-read

Generative BiLSTM+VAE model, dual-read.

| config | seeds | load RMSE | load MAE | load R² | pv RMSE | pv MAE | pv R² | pv export-viol rate |
|---|---|---|---|---|---|---|---|---|
| optimal, wx=False | 3/3 ✓ | 687.80 | 274.74 | 0.832 | 318.62 | 99.07 | 0.883 | 0.0351 |
| optimal, wx=True | 3/3 ✓ | 692.17 | 268.72 | 0.825 | 355.51 | 102.25 | 0.856 | 0.0365 |
| sub-optimal, wx=False | 3/3 ✓ | 694.80 | 287.23 | 0.813 | 326.68 | 105.53 | 0.873 | 0.0413 |
| sub-optimal, wx=True | 3/3 ✓ | 694.36 | 283.63 | 0.808 | 345.35 | 104.68 | 0.859 | 0.0300 |

## Exp2 — physics (mask on, optimal coverage only)

### EARNE — single-read

physics_weight=1.0 (loss+mask), 0.3 (reduced loss), and mask-only/no-loss (physics_weight=0.0).

| config | seeds | load RMSE | load MAE | load R² | pv RMSE | pv MAE | pv R² | pv export-viol rate |
|---|---|---|---|---|---|---|---|---|
| pw=1.0, wx=False | 3/3 ✓ | 1398.54 | 623.35 | 0.358 | 764.93 | 152.52 | 0.713 | 0.1623 |
| pw=1.0, wx=True | 3/3 ✓ | 1333.70 | 610.90 | 0.412 | 708.29 | 145.60 | 0.713 | 0.0657 |
| pw=0.3, wx=False | 3/3 ✓ | 1330.98 | 572.19 | 0.429 | 685.24 | 136.34 | 0.757 | 0.0220 |
| pw=0.3, wx=True | 3/3 ✓ | 1320.05 | 564.66 | 0.422 | 769.67 | 146.86 | 0.738 | 0.1649 |
| mask-only (pw=0), wx=False | 3/3 ✓ | 721.75 | 277.32 | 0.816 | 379.22 | 105.75 | 0.833 | 0.5022 |
| mask-only (pw=0), wx=True | 3/3 ✓ | 728.64 | 278.67 | 0.814 | 397.69 | 105.45 | 0.819 | 0.2198 |

### EARNE — dual-read

physics_weight=1.0, 0.3, and mask-only/no-loss (physics_weight=0.0), dual-read.

| config | seeds | load RMSE | load MAE | load R² | pv RMSE | pv MAE | pv R² | pv export-viol rate |
|---|---|---|---|---|---|---|---|---|
| pw=1.0, wx=False | 3/3 ✓ | 1425.48 | 623.51 | 0.335 | 603.55 | 134.61 | 0.738 | 0.1153 |
| pw=1.0, wx=True | 3/3 ✓ | 1407.17 | 627.38 | 0.345 | 585.11 | 132.20 | 0.774 | 0.0509 |
| pw=0.3, wx=False | 3/3 ✓ | 1298.89 | 570.96 | 0.426 | 546.03 | 120.51 | 0.777 | 0.1074 |
| pw=0.3, wx=True | 3/3 ✓ | 1320.00 | 567.80 | 0.414 | 508.25 | 124.86 | 0.793 | 0.0681 |
| mask-only (pw=0), wx=False | 3/3 ✓ | 739.66 | 270.41 | 0.806 | 382.51 | 101.92 | 0.829 | 0.5671 |
| mask-only (pw=0), wx=True | 3/3 ✓ | 714.44 | 266.37 | 0.821 | 379.58 | 99.59 | 0.834 | 0.6170 |

### MLP baseline — single-read

physics_weight=1.0, 0.3, and mask-only/no-loss (physics_weight=0.0).

| config | seeds | load RMSE | load MAE | load R² | pv RMSE | pv MAE | pv R² | pv export-viol rate |
|---|---|---|---|---|---|---|---|---|
| pw=1.0, wx=False | 3/3 ✓ | 1339.07 | 610.52 | 0.417 | 723.33 | 141.87 | 0.720 | 0.0389 |
| pw=1.0, wx=True | 3/3 ✓ | 1337.18 | 613.02 | 0.398 | 686.99 | 138.13 | 0.662 | 0.0667 |
| pw=0.3, wx=False | 3/3 ✓ | 1246.22 | 557.58 | 0.475 | 696.57 | 135.19 | 0.727 | 0.0998 |
| pw=0.3, wx=True | 3/3 ✓ | 1324.66 | 555.91 | 0.416 | 588.91 | 129.36 | 0.727 | 0.0317 |
| mask-only (pw=0), wx=False | 3/3 ✓ | 721.22 | 273.67 | 0.812 | 373.86 | 100.58 | 0.843 | 0.0754 |
| mask-only (pw=0), wx=True | 3/3 ✓ | 719.95 | 280.41 | 0.818 | 414.31 | 105.21 | 0.803 | 0.4853 |

### MLP baseline — dual-read

physics_weight=1.0, 0.3, and mask-only/no-loss (physics_weight=0.0), dual-read.

| config | seeds | load RMSE | load MAE | load R² | pv RMSE | pv MAE | pv R² | pv export-viol rate |
|---|---|---|---|---|---|---|---|---|
| pw=1.0, wx=False | 3/3 ✓ | 1300.10 | 613.16 | 0.429 | 592.99 | 130.68 | 0.745 | 0.1508 |
| pw=1.0, wx=True | 3/3 ✓ | 1379.44 | 621.16 | 0.377 | 642.27 | 154.03 | 0.745 | 0.1616 |
| pw=0.3, wx=False | 3/3 ✓ | 1202.90 | 542.67 | 0.506 | 680.79 | 128.28 | 0.723 | 0.2864 |
| pw=0.3, wx=True | 3/3 ✓ | 1298.14 | 553.81 | 0.432 | 490.12 | 119.21 | 0.801 | 0.3091 |
| mask-only (pw=0), wx=False | 3/3 ✓ | 709.53 | 261.86 | 0.818 | 367.65 | 92.69 | 0.842 | 0.6346 |
| mask-only (pw=0), wx=True | 3/3 ✓ | 702.26 | 262.68 | 0.825 | 371.52 | 93.46 | 0.845 | 0.3988 |

### LSTM baseline — single-read

physics_weight=1.0, 0.3, and mask-only/no-loss (physics_weight=0.0).

| config | seeds | load RMSE | load MAE | load R² | pv RMSE | pv MAE | pv R² | pv export-viol rate |
|---|---|---|---|---|---|---|---|---|
| pw=1.0, wx=False | 3/3 ✓ | 1438.90 | 631.23 | 0.331 | 611.12 | 143.52 | 0.705 | 0.1211 |
| pw=1.0, wx=True | 3/3 ✓ | 1458.60 | 650.41 | 0.292 | 794.68 | 144.81 | 0.679 | 0.0728 |
| pw=0.3, wx=False | 3/3 ✓ | 1355.47 | 574.75 | 0.390 | 564.49 | 139.60 | 0.734 | 0.0789 |
| pw=0.3, wx=True | 3/3 ✓ | 1435.47 | 594.45 | 0.323 | 498.29 | 125.96 | 0.743 | 0.1213 |
| mask-only (pw=0), wx=False | 3/3 ✓ | 723.11 | 268.64 | 0.813 | 378.63 | 102.82 | 0.836 | 0.0492 |
| mask-only (pw=0), wx=True | 3/3 ✓ | 762.49 | 274.21 | 0.792 | 421.20 | 105.57 | 0.792 | 0.4084 |

### LSTM baseline — dual-read

physics_weight=1.0, 0.3, and mask-only/no-loss (physics_weight=0.0), dual-read.

| config | seeds | load RMSE | load MAE | load R² | pv RMSE | pv MAE | pv R² | pv export-viol rate |
|---|---|---|---|---|---|---|---|---|
| pw=1.0, wx=False | 3/3 ✓ | 1399.13 | 627.90 | 0.356 | 486.93 | 131.31 | 0.759 | 0.1761 |
| pw=1.0, wx=True | 3/3 ✓ | 1455.68 | 639.47 | 0.304 | 509.47 | 131.17 | 0.762 | 0.1261 |
| pw=0.3, wx=False | 3/3 ✓ | 1336.63 | 567.84 | 0.400 | 682.56 | 129.83 | 0.777 | 0.0826 |
| pw=0.3, wx=True | 3/3 ✓ | 1389.62 | 575.32 | 0.362 | 452.76 | 116.67 | 0.779 | 0.1992 |
| mask-only (pw=0), wx=False | 3/3 ✓ | 721.81 | 261.14 | 0.813 | 341.12 | 92.14 | 0.864 | 0.3238 |
| mask-only (pw=0), wx=True | 3/3 ✓ | 708.97 | 257.21 | 0.819 | 371.27 | 93.89 | 0.844 | 0.4553 |

### CVAE baseline — single-read

cvae_loss has no physics_weight term — this run is effectively "mask-only" by construction.

| config | seeds | load RMSE | load MAE | load R² | pv RMSE | pv MAE | pv R² | pv export-viol rate |
|---|---|---|---|---|---|---|---|---|
| mask-only, wx=False | 3/3 ✓ | 679.64 | 270.69 | 0.834 | 324.35 | 96.18 | 0.884 | 0.0131 |
| mask-only, wx=True | 3/3 ✓ | 685.77 | 271.70 | 0.836 | 349.66 | 96.48 | 0.862 | 0.0146 |

### CVAE baseline — dual-read

cvae_loss has no physics_weight term — this run is effectively "mask-only" by construction, dual-read.

| config | seeds | load RMSE | load MAE | load R² | pv RMSE | pv MAE | pv R² | pv export-viol rate |
|---|---|---|---|---|---|---|---|---|
| mask-only, wx=False | 3/3 ✓ | 664.87 | 260.66 | 0.843 | 307.94 | 86.63 | 0.892 | 0.0120 |
| mask-only, wx=True | 3/3 ✓ | 689.61 | 267.96 | 0.839 | 312.38 | 85.14 | 0.887 | 0.0140 |

## Notes

- **ST-SGCCaps is excluded** — deactivated per request, and its point-prediction output (`[N,2]`) doesn't fit `calculate_metrics.py`'s quantile-shaped (`[N,6]`) reshape anyway.
- Runs marked with fewer than 3/3 seeds are still training; treat those rows as preliminary.
- `export-viol rate` = fraction of test timesteps where predicted PV production is less than the export implied by observed net demand (a physically-impossible prediction).
