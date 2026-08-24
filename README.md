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

## How we got here (experimental journey)

The numbered `experiments/` pipeline above is the clean, reproducible distillation.
The real path was ~25+ named Optuna studies (`v2`, `v5`–`v27`, `store_ensemble_v1`,
`category_weight_v1/v2`, `state_weight_v1`, …) and a lot of dead ends. Recording
the order and the negative results here, because the negative results are exactly
the part that's easy to forget and expensive to re-discover:

1. **Statistical baselines first.** Full reproduction of the M5 competition's own
   R benchmark script (Naive/SES/MA/ETS/ARIMA/Croston/TSB/ADIDA/…, 12 methods) as
   floors — `M5Evaluator` intentionally preserves 4 known bugs in the original R
   script so results stay comparable to the official benchmark.
2. **Chronos-2 zero-shot, manual context-length (CL) grid search.** Short context
   (CL 1–16) dominates — M5 series are highly intermittent, long context mostly
   adds noise.
3. **Segmentation/grouping sensitivity.** Tested grouping series by weight-quartile,
   store, state×weight, category×weight, and department for Chronos-2's
   cross-learning batching. **Department won** — cross-learning benefits more from
   product-category peers than geographic (store) peers. Store-grouped ZS (v14)
   was measurably worse than dept-grouped.
4. **`cross_learning=True` turned out to be the single biggest lever in the whole
   study** — enabling Chronos-2's group attention (same CL, same grouping, just
   flipping the flag) was worth ~0.025 geo-mean WRMSSE on its own, bigger than most
   of the HPO effort combined. This isn't obvious from the API — it's a group
   attention mechanism described in the Chronos-2 technical report, not something
   you'd find by tuning hyperparameters.
5. **A real bug found and fixed:** `PeftModel.forward()` (the LoRA fine-tuning path)
   was numerically incompatible with Chronos-2's quantile inference — fixed via
   `merge_and_unload()` (reproduced in `experiments/demo_peft_bug.py`). This voided
   every LoRA fine-tuning result from before the fix (roughly v15–v22).
6. **Fine-tuning abandoned as a lever.** A steps-curve experiment (FT steps ∈
   {0, 5, 10, 25, 50, 100, 500}) showed only a marginal gain at 5 steps
   (spring EW 0.8623 → 0.8581) followed by monotonic degradation —
   catastrophic forgetting. In hindsight this experiment should have been run
   *before* the many rounds of FT-hyperparameter HPO that preceded it, not after.
7. **A zero-leakage ensemble blend design catastrophically failed.** Calibrating
   OLS blend weights on 2014–2015 windows and applying them to the 2015–2016 test
   period gave geo 1.386 — far worse than the leaky, LOO-on-test approach used
   everywhere else. Chronos-2 zero-shot performance is **non-stationary across
   years** on this dataset, so the leaky approach, while methodologically
   imperfect, was judged the only workable way to learn ensemble weights here.
   This is a known, documented limitation, not an oversight.
8. **The tier ensemble turned out to be load-bearing.** Removing it (dept-ZS-only
   + OLS) degrades geo-mean from 0.7969 to 0.9829. The tier ensemble mixes
   zero-shot and minimal-fine-tune forecasts by sales-volume quartile — a
   different, complementary grouping axis from department.
9. **A systematic FOODS under-forecast bias emerged** in the OLS scale weights
   (FOODS_1=1.59, FOODS_2=1.18, FOODS_3=1.16) — the working hypothesis is that
   dept-grouped cross-learning reinforces the bias, since FOODS peers all
   under-forecast together and have nothing to correct against within the group.
10. **LightGBM (global + per-department, 37 lag/rolling/price/calendar features,
    tweedie objective, Optuna-tuned) became the final best result** — geo-mean
    0.6198, beating the best Chronos-2 ensemble (0.7969) by 22%. This is what
    forced the article's reframe (see below).
11. **LightGBM × Chronos-2 OLS blend was a negative result** — 0.6254, worse than
    LightGBM alone. The two models don't have enough diversity for blending to help
    once LightGBM has full-history lag features available.
12. **Cold-start natural-segmentation analysis became the headline finding** —
    bucketing series by *active history length* (not calendar time) revealed the
    28-day crossover described at the top of this README. A controlled-truncation
    version of this experiment was tried and dropped as methodologically unsound
    (the LightGBM checkpoint was trained on full history; NaN-masking short
    history at inference time is not the same as training on short history).
13. **An earlier AutoGluon-wrapped Chronos + fine-tuning approach was abandoned
    outright** once the project moved to calling the Chronos-2 API directly with
    explicit segmentation/OLS (roughly the v9 cutover point above). Its HPO
    artifacts (`category_weight_v1/v2`, `state_weight_v1`, `weight_v8`,
    `v6_perfs_validation`, `v7_selective_ft`, and a large `explorations/` directory
    — collectively ~890 GB) were deleted in August 2026 as pure regenerable dead
    weight from a superseded approach; nothing in the current `experiments/`
    pipeline depends on them.

**Article reframe (2026-06-23):** the original framing — "Chronos-2 beats
everything" — didn't survive finding #10 above (LightGBM wins on full history by
22%). The article was reframed from general superiority to the cold-start claim,
which is both true and more useful in practice: traditional ML needs months of
history before `lag_28`-style features exist; Chronos-2 is competitive from day
one with zero feature engineering and no retraining.

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

## Lessons learned / pitfalls to avoid re-hitting

- **PeftModel + Chronos-2 LoRA bug** — `PeftModel.forward()` is numerically
  incompatible with Chronos-2's quantile inference unless you call
  `merge_and_unload()` first. See `experiments/demo_peft_bug.py`. Any LoRA
  result you find without this fix applied is invalid.
- **Fine-tuning past ~5-10 steps causes catastrophic forgetting** on this
  dataset/model pair. Run the steps-curve check *first*, before any
  FT-hyperparameter search — we did it in the wrong order and paid for it in
  wasted HPO trials.
- **M5 `id` suffix bug:** forgetting to strip `_evaluation`/`_validation` from
  `id` before reindexing against the submission format silently produces an
  all-zero comparison and a WRMSSE of 5.17 — that number alone is now a red flag
  meaning "check the id join," not a real result.
- **Ensemble blend weights learned on one period don't transfer across years**
  on this dataset — Chronos-2 zero-shot performance is non-stationary
  year-to-year. A "zero-leakage" calibrate-then-apply design will look
  catastrophically worse than a leaky LOO design; that's a property of the
  data, not a bug in the zero-leakage code.
- **Cross-learning grouping axis matters more than most HPO knobs** — semantic
  (department/product-category) grouping beats geographic (store) grouping by a
  wide margin. If extending this work to a new grouping axis, department-style
  (semantic) groupings are the prior to start from.
- **OOM guardrail for parallel jobs:** always launch `experiments/lgbm_watchdog.sh`
  alongside any heavy parallel/multi-worker job — 3 parallel `M5DataPipeline`
  workers is enough to OOM the machine.
- **`notebooks/tmp/`, `bin/micromamba`, stray AutoGluon test-run logs, and a
  broken `notebooks/.venv` symlink were cleaned up (Aug 2026)** as pure cache —
  regenerate with `uv sync` / a fresh `bin/micromamba` download if ever needed,
  nothing there was unique.

---

## Article

- English: [reports/article_chronos2_retail_guide_v2.md](reports/article_chronos2_retail_guide_v2.md)
- French: [reports/article_chronos2_retail_guide_v2_fr.md](reports/article_chronos2_retail_guide_v2_fr.md)
- Full experimental log: [reports/experimental_log.md](reports/experimental_log.md)
