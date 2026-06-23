# Cold-Start Demand Forecasting with Chronos-2

**Accompanying code for the article:**
*"Cold-start demand forecasting with foundation models: when Chronos-2 outperforms traditional ML and why"*

---

## What this repo contains

We benchmark [Chronos-2-small](https://github.com/amazon-science/chronos-forecasting) (46M parameters, encoder-only)
against LightGBM on the [M5 competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy) dataset
(30,490 Walmart retail series, 3 seasonal test cutoffs, WRMSSE metric).

**Key finding:** LightGBM wins on full-history series (geo-mean WRMSSE 0.62 vs 0.80).
Chronos-2 wins for series with fewer than 28 days of active history — precisely
the period before `lag_28`, the single most important LightGBM feature, becomes available.
This is a mechanistically justified cold-start advantage, not just an empirical pattern.

| Model | Autumn | Winter | Spring | Geo-mean |
|---|---|---|---|---|
| Seasonal Naive (floor) | 2.085 | 1.422 | 1.464 | 1.631 |
| Chronos-2 ZS dept-seg xcl=True | 0.902 | 1.141 | 0.957 | 1.000 |
| Chronos-2 + per-dept OLS (v20) | — | — | — | **0.797** |
| LightGBM global | — | — | — | **0.620** |
| LightGBM × Chronos OLS blend | — | — | — | 0.625 (no gain) |

**Cold-start (<28d history):**

| Bucket | LightGBM | Chronos-2 | Winner |
|---|---|---|---|
| < 28 days | 1.059 | **1.019** | **Chronos-2** (59% of series) |
| 28 – 90 days | **0.812** | 0.847 | LightGBM |
| 90 – 365 days | **0.735** | 0.871 | LightGBM |
| 730+ days | **0.664** | 0.772 | LightGBM |

**Deployment rule:** use Chronos-2 for the first 28 days after SKU launch; switch to LightGBM once `lag_28` is available.

---

## Repository layout

```
tsfm-explo/
├── src/tsfm_m5/           # Installable Python package
│   ├── data.py            # M5DataPipeline — data prep, caching, WRMSSE weights
│   ├── eval.py            # M5Evaluator   — exact match to R benchmark script
│   └── benchmarks.py      # M5BenchmarkSuite — Naive/SES/MA/ETS/ARIMA baselines
├── experiments/           # Numbered scripts — the full reproduction pipeline
│   ├── 00_run_benchmarks.py          # M5 competition baselines
│   ├── 01_naive_baseline.py          # Seasonal Naive (WRMSSE floor)
│   ├── 02_chronos_zs_cl_sweep.py     # ZS context-length sweep per dept
│   ├── 03_chronos_xcl_ablation.py    # Cross-learning ON vs OFF
│   ├── 04_chronos_hpo_v20.py         # Chronos dept HPO (best: geo=0.797)
│   ├── 05_chronos_ols_v20.py         # Per-dept OLS blend
│   ├── 06_lgbm_build_features.py     # Build lag/rolling/price features
│   ├── 07_lgbm_hpo.py                # LightGBM Optuna HPO
│   ├── 08_lgbm_eval.py               # LightGBM evaluation (best: geo=0.620)
│   ├── 09_lgbm_chronos_blend.py      # OLS blend — negative result
│   ├── 10_coldstart_analysis.py      # Cold-start natural segmentation
│   ├── 11_ft_steps_curve.py          # Catastrophic forgetting curve
│   ├── 12_chronos_zs_standalone.py   # Standalone ZS baseline (no ensemble)
│   ├── 13_chronos_cl_sensitivity.py  # CL sensitivity analysis
│   ├── demo_peft_bug.py              # PeftModel + Chronos-2 bug & fix
│   └── lgbm_watchdog.sh              # OOM guardrail for parallel jobs
├── notebooks/
│   ├── m5_eda.ipynb                  # Data exploration
│   ├── report_zs_CL.ipynb            # CL sweep visualisation
│   ├── hpo_analysis_report.ipynb     # HPO trial analysis
│   └── run_benchmarks_v2.ipynb       # Benchmark comparison dashboard
├── reports/
│   ├── article_chronos2_retail_guide_v2.md    # Article (English)
│   ├── article_chronos2_retail_guide_v2_fr.md # Article (French)
│   ├── Progress_Report_M5_v2.md               # Full experimental log (EN)
│   ├── Rapport_Progression_M5_v2.md           # Full experimental log (FR)
│   ├── experimental_log.md                    # Condensed experiment history
│   └── figures/                               # PNG figures + plotting scripts
├── pyproject.toml
└── uv.lock
```

---

## Setup

Requires Python 3.10. We use [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Clone and install
git clone <repo-url> && cd tsfm-explo
uv sync                          # installs everything from uv.lock (exact versions)
uv pip install -e .              # installs tsfm_m5 as an editable package

# Verify
python -c "from tsfm_m5 import M5DataPipeline, M5Evaluator; print('OK')"
```

---

## Data

The M5 dataset is not included. Download from Kaggle and prepare a joined parquet:

```bash
# One-time download (requires kaggle credentials in kaggle.json)
kaggle competitions download -c m5-forecasting-accuracy

# Point experiments at your data location via environment variables:
export M5_DATA_PATH=/path/to/jointed_M5.parquet   # ~20 GB parquet
export M5_LAB_DIR=/path/to/output/dir             # experiment outputs land here
```

All experiments read `M5_DATA_PATH` and write outputs under `M5_LAB_DIR`.
If neither variable is set the scripts fall back to the original server paths
(`/mnt/lab/datasets/M5/jointed_M5.parquet` and `/mnt/lab/nmwamsojo`).

---

## Running the experiments

Scripts are numbered in execution order. Each is self-contained and idempotent
(re-running skips already-cached prepared data and existing Optuna trials).

```bash
source .venv/bin/activate   # or: uv run python experiments/...

# 1. Establish floors
python experiments/01_naive_baseline.py

# 2. Chronos-2 zero-shot pipeline
python experiments/02_chronos_zs_cl_sweep.py      # ~4 h on GPU
python experiments/03_chronos_xcl_ablation.py
python experiments/04_chronos_hpo_v20.py           # ~8 h, 150 Optuna trials
python experiments/05_chronos_ols_v20.py

# 3. LightGBM pipeline
python experiments/06_lgbm_build_features.py       # builds feature parquets
python experiments/07_lgbm_hpo.py                  # ~2 h, 100 Optuna trials
python experiments/08_lgbm_eval.py

# 4. Ensemble and cold-start analysis
python experiments/09_lgbm_chronos_blend.py
python experiments/10_coldstart_analysis.py        # key finding

# 5. Additional analyses referenced in the article
python experiments/11_ft_steps_curve.py            # catastrophic forgetting
python experiments/12_chronos_zs_standalone.py
python experiments/13_chronos_cl_sensitivity.py
python experiments/demo_peft_bug.py                # PeftModel bug reproduction
```

Long-running GPU jobs can be launched with nohup:
```bash
nohup python experiments/04_chronos_hpo_v20.py >> $M5_LAB_DIR/hpo_v20.log 2>&1 &
tail -f $M5_LAB_DIR/hpo_v20.log
```

For memory-intensive parallel jobs (LGBM feature build, CL sweep), launch the
OOM guardrail first:
```bash
bash experiments/lgbm_watchdog.sh &
python experiments/06_lgbm_build_features.py
```

---

## Hardware used

| Resource | Spec |
|---|---|
| CPU | 20 cores |
| RAM | 128 GB (watchdog at 60 GB) |
| GPU | NVIDIA (CUDA) for Chronos-2 inference; OpenCL for LightGBM GPU histograms |
| Storage | `/mnt/lab` — NFS mount, ~500 GB for all intermediate parquets |

---

## Key design decisions

**Evaluation metric:** WRMSSE (Weighted Root Mean Squared Scaled Error) — the
official M5 metric, averaged across 12 hierarchy levels. Exactly reproduces
`Point_Forecasts_-_Benchmarks.R` from the competition, including four known
bugs in the R script that must be preserved for comparability.

**Cold-start metric:** unweighted per-series median RMSSE by history-length
bucket. WRMSSE suppresses new-product signal (low revenue weight) so it is
not the right metric for the cold-start claim.

**Chronos signal for cold-start:** pure zero-shot, dept-segmented, xcl=True —
no tier OLS, no fine-tuning. This is what a practitioner deploys on a new SKU
with zero training data.

**Cross-learning:** enabled via `predict_quantiles(cross_learning=True)`,
semantic batches (same department together). Random batching with covariates
gives worse results than univariate — coherent grouping is required.

---

## Article

- English: [reports/article_chronos2_retail_guide_v2.md](reports/article_chronos2_retail_guide_v2.md)
- French: [reports/article_chronos2_retail_guide_v2_fr.md](reports/article_chronos2_retail_guide_v2_fr.md)
- Full experimental log: [reports/experimental_log.md](reports/experimental_log.md)
