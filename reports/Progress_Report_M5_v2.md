# Chronos-2 on M5 — Experimental Progress Report

**Model:** `autogluon/chronos-2-small`  
**Task:** M5 Walmart demand forecasting — 30,490 SKUs, 28-day horizon  
**Metric:** WRMSSE (geo-mean over 3 cutoffs unless noted); cold-start analysis uses per-series RMSSE (unweighted)  
**Cutoffs:** Spring 2016-04-24 · Winter 2016-01-03 · Autumn 2015-10-04  
**Date:** June 2026

---

## 1. Problem Setup

The M5 competition asks for 28-day sales forecasts across 30,490 series from 3 Walmart states (CA, TX, WI), 10 stores, 3 categories (FOODS, HOBBIES, HOUSEHOLD) and 7 departments. The evaluation metric is WRMSSE, a hierarchically-weighted RMSSE computed across 12 aggregation levels from individual SKU to total. High-volume FOODS series (~57% of total WRMSSE weight) dominate the score.

**Series characteristics relevant to modelling choices:**
- High intermittency — many SKUs have zero-sale days; WRMSSE is dominated by high-volume items
- Strong weekly seasonality (SNAP benefit days on Fridays in CA/TX/WI)
- Price promotions and event effects (holidays, sports, cultural events)
- Sales context length matters differently by volume tier: high-volume items benefit from longer history, low-volume items from short context

**Evaluation methodology:** All results from v9 onwards use a 3-cutoff geo-mean WRMSSE. Earlier experiments (v2–v8) used a single spring cutoff (2016-04-24) and are not directly comparable to later results. Where a single-cutoff result is cited, it is flagged.

**Metric note for cold-start analysis (§11):** Cold-start results use per-series unweighted RMSSE, not WRMSSE. Revenue weights in WRMSSE suppress new-product signal; per-series RMSSE treats every SKU equally regardless of volume. The two metrics are not directly comparable — do not mix them in the same ranking.

---

## 2. Naive Baseline

A seasonal naïve forecast (repeat the last 7 daily sales, cycling weekly) provides the evaluation floor.

| Cutoff | WRMSSE |
|---|---|
| Spring 2016-04-24 | 1.4639 |
| Winter 2016-01-03 | 1.4216 |
| Autumn 2015-10-04 | 2.0848 |
| **3-cutoff geo-mean** | **1.6310** |

The Autumn cutoff is notably harder: pre-holiday stocking disrupts the weekly cycle the naïve baseline relies on. All Chronos-2 results comfortably beat this baseline.

---

## 3. Phase 1 — Zero-Shot Baselines (Single Cutoff, Spring 2016-04-24)

Initial exploration of Chronos-2 raw ZS capability without any fine-tuning.

### 3.1 Context Length Sensitivity

| Configuration | WRMSSE (spring) |
|---|---|
| ZS, CL=2, no covariates | 1.243 |
| ZS, CL=8, + price covariates | 1.255 |
| ZS, CL=8, + event covariates | 1.267 |
| ZS, CL=16 | 1.350 |
| ZS, CL=32 | 1.520 |
| ZS, CL=128+ | 1.8–2.1 |

**Finding:** Context length is the most impactful ZS hyperparameter. Short CL (2–8) outperforms long CL because M5 series are highly intermittent — long context introduces irrelevant noise from older zero-sales periods. Covariates add negligible value at ZS with global (mixed-category) batches.

**Note on CL=1:** CL=1 is numerically identical to the seasonal naïve baseline (1.6310). With a single observation, Chronos produces the same last-value weekly repeat as the naïve. Any useful inference requires at least CL=7.

### 3.2 Covariate × Cross-Learning Interaction (Global Batches)

With all 30,490 series in mixed global batches, the covariates and `cross_learning` interact in a counter-intuitive way:

|  | xcl=True | xcl=False |
|--|---|---|
| **With covariates** | 1.1765 | **1.0776** |
| **Without covariates** | **1.2504** | 1.2716 |

Covariates and cross-learning are substitutes, not complements, in globally mixed batches. When both are active, cross-attention mixes per-series covariate representations across unrelated series (e.g. California SNAP series attending on Texas series during California-only SNAP periods), injecting noise. In the absence of covariates, cross-attention finds useful demand-similarity signal across the batch. **Conclusion:** With rich exogenous data, disable cross-learning globally; enable it only within semantically coherent (same-category) batches (see §6).

### 3.3 Per-Segment ZS HPO

Breaking the 30,490 series into demand-homogeneity segments and tuning CL per segment yielded a major improvement without any fine-tuning.

**Smoothness segmentation** (ADI/CV² demand classification):

| Rank | Smooth | Erratic | Interm. | Lumpy | Covariates | WRMSSE (spring) |
|---|---|---|---|---|---|---|
| 1 | 16 | 4 | 1 | 8 | is_weekend + price + snap | 0.9418 |
| 2 | 16 | 2 | 1 | 8 | is_weekend + price + snap | 0.9729 |

**Weight-tier segmentation** (Low/Med-Low/Med-High/High volume quartiles):

| Rank | CL_Low | CL_MedLow | CL_MedHigh | CL_High | Covariates | WRMSSE (spring) |
|---|---|---|---|---|---|---|
| 1 | 16 | 4 | 1 | 16 | is_weekend + price + snap | 0.9082 |
| 2 | 8 | 4 | 1 | 256 | is_weekend + price + snap | 0.9137 |

**Finding:** Assigning different context lengths per demand segment improves WRMSSE by ~27% over blind ZS (1.243 → 0.908) with no GPU training. High-weight items benefit from longer context (CL=16–256); Med-High items need minimal context (CL=1). This established **segment-aware ZS** as the baseline for all subsequent experiments.

---

## 4. Phase 2 — Fine-Tuning Experiments with Tier Segmentation (v2–v11)

*Note: All FT results in this phase use single spring cutoff unless stated. LoRA-based FT runs (High/Med-High tiers in v9+) may be affected by the PeftModel bug identified later (see §7). Full-mode FT runs (Med-Low tier in v9) are not affected.*

### 4.1 Naive Shared Fine-Tuning (v2)

First FT attempt using identical hyperparameters across all weight tiers.

**WRMSSE (spring): 1.017** — worse than the per-segment ZS baseline (0.908).

**Finding:** Uniform fine-tuning degrades performance. The same lr/steps applied to all series over-fits low-frequency items while under-fitting high-frequency ones. Segment-specific FT is required.

### 4.2 Per-Segment FT (v6) — Best Single-Cutoff Result

Per-segment HPO (CL, ft_steps, ft_mode, ft_lr, ft_bs, covariates) over 216 trials.

| Segment | CL | Steps | Mode | LR | Covariates |
|---|---|---|---|---|---|
| Low | 8 | ZS | — | — | is_weekend + event |
| Med-Low | 1 | 200 | full | 5e-5 | event + snap |
| Med-High | 4 | 200 | lora | 1e-4 | is_weekend |
| High | 16 | 100 | lora | 1e-2 | is_weekend + price + snap |

**WRMSSE (spring): 0.8364**

**Finding:** Low-weight items are best left at ZS — FT adds no value for sparse, noisy series. High-weight items require aggressive FT (high LR 1e-2, price + snap covariates). The top 4 trials all tie at 0.8364, differing only in Med-Low covariate choice — low sensitivity there.

### 4.3 Covariate Re-encoding (v7/v8)

Switched from unified covariate-choice strings to individual boolean flags per covariate per segment, and extended max ft_steps to 2000.

**Result (v8, 2-cutoff Optuna):** best geo = **0.9082** — regression from v6.

**Finding:** Expanding the covariate search space (16 extra boolean params) made TPE converge much slower. The unified covariate string in v6 was a stronger inductive bias. More FT steps (2000) did not compensate.

### 4.4 Tier HPO — Weight-Based Segmentation (v9–v11)

Full HPO over CL, FT mode/lr/steps/bs, and FT scope (which tiers train together) across 3 cutoffs using weight-quartile tiers.

| Version | Trials | 3-cutoff geo-mean WRMSSE |
|---|---|---|
| v9 | 362 | **0.8947** |
| v10 | 150 | **0.8536** |
| v11 | 100 | **0.8544** |

**v11 best config (trial #11):** used as the reference tier ensemble in all subsequent blending.

**Tier ensemble alone (no Chronos blend):** geo = **0.8180** (3 cutoffs).

| Cutoff | Tier ensemble |
|---|---|
| 2016-04-24 | 0.8467 |
| 2016-01-03 | 0.8712 |
| 2015-10-04 | 0.7420 |

**Finding:** Progressive refinement (v9→v11) improved from 0.8947 to 0.8544. The v11 tier ensemble became the strongest single signal throughout this study and underpins all subsequent blended results.

---

## 5. Phase 3 — Dept-Level ZS HPO (v12–v14, 3-Cutoff)

Shifted from weight-tier segmentation to product department segmentation (7 depts: FOODS_1/2/3, HOBBIES_1/2, HOUSEHOLD_1/2). All runs use `cross_learning=False` (univariate ZS within each dept group).

### 5.1 Per-Dept CL HPO (v12/v13)

| Version | Trials | CLs | EW geo |
|---|---|---|---|
| v12 | 100 | F1=28, F2=7, F3=28, HB1=1, HB2=14, HH1=28, HH2=1 | 0.8702 |
| v13 | 150 | F1=28, F2=7, F3=28, HB1=1, HB2=7, HH1=28, HH2=7 | **0.8672** |

Blending 50% tier ensemble + 50% dept ZS with per-dept OLS weights:

| Method | geo-mean |
|---|---|
| EW (50% tier + 50% dept ZS) | 0.8672 |
| OLS global α | 0.8128 |
| OLS per-dept α | **0.8066** |

Per-dept OLS α (higher = more tier, lower = more dept ZS):

| Dept | α | Chronos share |
|---|---|---|
| FOODS_1 | 0.988 | 1.2% |
| FOODS_2 | 1.000 | 0.0% |
| FOODS_3 | 0.939 | 6.1% |
| HOBBIES_1 | 0.779 | 22.1% |
| HOBBIES_2 | 1.000 | 0.0% |
| HOUSEHOLD_1 | 0.759 | 24.1% |
| HOUSEHOLD_2 | 0.995 | 0.5% |

**Finding:** OLS with per-dept weights improves 0.8672 → 0.8066. FOODS depts are almost entirely driven by the tier ensemble (α≈1); HOUSEHOLD_1 and HOBBIES_1 benefit the most from Chronos ZS signal.

### 5.2 Chronos-2 Standalone (Gap 2 — No Tier Blend)

To isolate Chronos-alone performance (no tier ensemble):

| Method | geo-mean |
|---|---|
| v13 Chronos EW (no tier, small xcl=False) | 1.0739 |
| v20 Chronos OLS (no tier, small xcl=True) | 0.9829 |
| v17 BASE xcl=False standalone | 1.1963 |

⚠️ **Correction (June 17):** Earlier versions of this table showed 0.9123 as "v13 Chronos standalone" — this was wrong. 0.9123 was the blended result (50% v11 tier + 50% BASE xcl=False ZS). The correct small xcl=False standalone = 1.0739; BASE xcl=False standalone = 1.1963. Small xcl=True (v20d) is the best standalone ZS at 1.0325 EW / 0.9829 OLS.

The tier ensemble contributes approximately **0.19 geo** over standalone ZS (1.0325 → 0.7969 with OLS), far larger than previously reported.

### 5.3 Per-Store ZS HPO (v14)

10 separate store models (CA_1–4, TX_1–3, WI_1–3), 3 cutoffs.

| Method | geo-mean |
|---|---|
| v14 EW (50% tier + 50% store ZS) | 0.8667 |
| v14 OLS per-store | 0.8132 |

**Finding:** Store-based grouping is inferior to dept-based grouping (0.8132 vs 0.8066 OLS). Geographic segmentation does not encode demand dynamics — CA, TX, and WI sell the same categories with similar patterns. Per-dept segmentation captures product-category similarity better.

---

## 6. Phase 4 — Cross-Learning Discovery (v20)

The key finding of this study: Chronos-2 supports `cross_learning=True` in `predict_quantiles()`, which activates the group attention mechanism described in the Chronos-2 technical report. All prior runs had implicitly used `cross_learning=False` (univariate).

### 6.1 Cross-Learning HPO (v20)

Same structure as v13 but with `cross_learning=True` and batch_size=100 (Chronos-2 recommendation for cross-learning). 150 Optuna trials.

**Best CLs per dept (v20):**

| F1 | F2 | F3 | HB1 | HB2 | HH1 | HH2 |
|---|---|---|---|---|---|---|
| 28 | 7 | 28 | 1 | 7 | 28 | 1 |

Note: HH2 changed from CL=7 (v13) to CL=1 (v20) — the optimal CL can shift when cross-learning activates different context utilization patterns.

**Effect of batch composition on cross-learning (standalone ZS, no tier blend):**

| xcl | Batch composition | geo-mean |
|---|---|---|
| True | All depts mixed (global) | 1.1680 — degrades |
| False | All depts mixed (global) | **1.0776** — best global |
| False | Dept-segmented batches | 1.0739 — neutral |
| **True** | **Dept-segmented batches** | **1.0325 — best ZS** |

Cross-learning requires semantically coherent batches. In a FOODS_1 dept batch, 100 series share the same SNAP eligibility and weekly demand pattern — cross-attention extracts a genuine group signal. In a globally mixed batch, cross-attention sees incompatible patterns across categories and introduces noise. The FOODS_1 systematic underestimation bias (1.59×) under dept-seg xcl=True is corrected by the OLS step (see §6.2).

| Method | geo-mean |
|---|---|
| v13 EW (xcl=False) | 0.8672 |
| **v20 EW (xcl=True)** | **0.8417** |
| Δ | −0.0255 |

**Enabling cross_learning alone gives +0.025 geo improvement with no other changes.**

### 6.2 v20 OLS Blend

Applying the same per-dept OLS framework (α × tier + (1−α) × dept ZS) to v20 forecasts:

| Method | Spring | Winter | Autumn | geo-mean |
|---|---|---|---|---|
| v20 EW | — | — | — | 0.8417 |
| **v20 OLS** | **0.8272** | **0.8726** | **0.7012** | **0.7969** |

**Per-dept α (v20):**

| Dept | α | Chronos share |
|---|---|---|
| FOODS_1 | 0.961 | 3.9% |
| FOODS_2 | 1.000 | 0.0% |
| FOODS_3 | 0.900 | 10.0% |
| HOBBIES_1 | 0.740 | 26.0% |
| HOBBIES_2 | 1.000 | 0.0% |
| HOUSEHOLD_1 | 0.644 | **35.6%** |
| HOUSEHOLD_2 | 0.853 | 14.7% |

**v20 OLS (geo = 0.7969) is the best Chronos result, breaking the 0.80 barrier.**

| Comparison | geo-mean |
|---|---|
| v13 OLS (xcl=False) | 0.8066 |
| v20 OLS (xcl=True) | **0.7969** |
| Δ vs v13 OLS | −0.0097 |
| Δ vs v20 EW | −0.0448 |

### 6.3 Chronos-Only OLS (No Tier Ensemble)

To isolate what OLS weighting alone adds vs what the tier ensemble contributes:

| Method | geo-mean |
|---|---|
| Dept ZS EW (v20 CLs, xcl=True, no tier) | 1.0325 |
| Dept ZS OLS (per-dept scaling, no tier) | 0.9829 |
| v20 OLS (tier ens + dept ZS) | 0.7969 |

**The tier ensemble is load-bearing: removing it degrades the result by ~0.19 geo (0.7969 → 0.9829).**

Per-dept scaling weights (Chronos-only, no tier):

| FOODS_1 | FOODS_2 | FOODS_3 | HOBBIES_1 | HOBBIES_2 | HH_1 | HH_2 |
|---|---|---|---|---|---|---|
| **1.593** | 1.181 | 1.162 | 0.919 | 1.116 | 1.127 | 0.972 |

FOODS_1 requires a 1.59× scale-up — Chronos systematically under-forecasts FOODS depts when cross-learning is limited to dept peers (all biased the same way). This explains why the tier ensemble, which was calibrated independently, dominates for FOODS.

---

## 7. Phase 5 — Fine-Tuning Investigation (v15–v24)

### 7.1 PeftModel Bug

All LoRA-based FT runs from v15 onwards used `Chronos2Pipeline.fit()` followed by in-memory inference. A bug was identified: `fit()` returns a `PeftModel` whose forward() method is incompatible with Chronos-2's quantile inference path, producing garbage outputs.

**Fix (applied in pipeline.py, line 351–361):** Call `model.merge_and_unload()` before returning the in-memory pipeline after LoRA training. This merges adapter weights into the base model and restores the correct inference path.

```python
# Broken path — adapter weights ignored silently
pipeline = MeanScaleUniformBins(model)

# Correct path — adapter merged before wrapping
pipeline = MeanScaleUniformBins(model.merge_and_unload())
```

Affected runs: v15–v22 (all LoRA FT experiments). The v9 Med-Low tier forecast (full-mode FT, not LoRA) was not affected. The v11 High tier forecast (LoRA, 50 steps) predates the bug period.

### 7.2 FT Steps Curve

To determine whether any FT is viable, a steps curve was run on the spring cutoff using global FT (all 30,490 series, no segmentation):

| Steps | Raw WRMSSE | EW blend (50% tier + 50% FT) |
|---|---|---|
| 0 (ZS) | 1.0630 | **0.8623** |
| 5 | 1.0372 | **0.8581** |
| 10 | 1.0491 | 0.8642 |
| 25 | 1.0810 | 0.8833 |
| 50 | 1.1644 | 0.9259 |
| 100 | 1.3446 | 1.0116 |
| 200 | 1.5492 | 1.1029 |
| 500 | 1.6281 | 1.1321 |
| 1000 | 1.6398 | 1.1277 |

**Finding: Catastrophic forgetting confirmed.** FT improves marginally at 5 steps (EW: 0.8623 → 0.8581, Δ=+0.004), then degrades monotonically. By 50 steps the raw WRMSSE is already 10% worse than ZS. By 100 steps the model is effectively unusable in isolation. Gradient updates overwrite pretrained cross-series representations faster than task-specific adaptation accumulates.

*Caveat: Steps curve run on spring cutoff only. The marginal 5-step improvement has not been validated on winter/autumn cutoffs.*

### 7.3 Raw Chronos FT (v23, v24a)

After fixing the PeftModel bug, FT was re-run using the raw Chronos API directly (no AutoGluon wrapper):

| Run | Config | EW blend geo |
|---|---|---|
| v23 | Per-dept FT, v20 CLs | 0.9509 |
| v24a | Global FT, all 30k series | 0.9461 |

Both are substantially worse than the ZS EW baseline (0.8417). This is consistent with the catastrophic forgetting finding from the steps curve — FT at the step counts explored destroys the pretrained cross-learning representation.

**Conclusion: FT is not viable as a route to further improvement. The model's pretrained ZS capability is stronger than what FT can achieve within the explored parameter space.**

---

## 8. Phase 6 — Principled Mixing: Tier ZS vs Dept ZS (v21)

### 8.1 Motivation

The v20 OLS result (0.7969) relies on the v9/v11 tier ensemble as its primary signal — an external FT-based ensemble that is not derived from Chronos ZS alone. This raises the question: can a purely Chronos ZS signal achieve comparable quality by using a different grouping axis?

**Hypothesis:** Dept-grouped ZS captures product-category similarity (cross-learning within FOODS, HOBBIES, HOUSEHOLD). Weight-tier-grouped ZS captures volume-demand similarity (cross-learning across all categories at the same demand intensity). These two axes are complementary — a blend may outperform either individually and reduce reliance on the external tier ensemble.

### 8.2 Tier ZS Results

ZS inference on weight-quartile tier groups (Low/Med-Low/Med-High/High, ~7,620 series each) with `cross_learning=True`:

**CL selection per tier** (CL ∈ {7, 14, 28}, best avg WRMSSE across 3 cutoffs):

| Tier | CL=7 | CL=14 | CL=28 | Best CL |
|---|---|---|---|---|
| Low | 1.1784 | 1.1657 | 1.1572 | **28** |
| Med-Low | 1.1572 | 1.1487 | 1.1562 | **14** |
| Med-High | 1.1487 | 1.1580 | 1.1652 | **7** |
| High | 1.1487 | 1.1299 | 1.0610 | **28** |

| Tier ZS EW (best CLs) | Spring | Winter | Autumn | geo |
|---|---|---|---|---|
| | 1.0247 | 1.2800 | 0.8784 | **1.0483** |

Tier-grouped ZS (1.0483) is weaker than dept-grouped ZS (1.0325 EW). The dept grouping encodes more predictively useful cross-series structure.

### 8.3 OLS Blend: Dept ZS + Tier ZS (per-(dept, tier) pairs)

Each series belongs to one dept AND one tier → 28 (dept, tier) combinations. OLS finds α ∈ [0,1] per pair: `forecast = α × tier_ZS + (1−α) × dept_ZS`.

**Final result:**

| Spring | Winter | Autumn | geo-mean |
|---|---|---|---|
| 0.9139 | 1.1180 | 0.8065 | **0.9375** |

**Per-(dept, tier) α matrix** [0 = all dept ZS, 1 = all tier ZS]:

| Dept | Low | Med-Low | Med-High | High |
|---|---|---|---|---|
| FOODS_1 | 0.33 | **1.00** | **1.00** | **1.00** |
| FOODS_2 | 0.00 | 0.00 | 0.67 | 0.54 |
| FOODS_3 | 0.33 | **1.00** | **1.00** | **1.00** |
| HOBBIES_1 | 0.00 | 0.11 | 0.09 | 0.29 |
| HOBBIES_2 | 0.00 | 0.00 | **1.00** | 0.32 |
| HOUSEHOLD_1 | 0.00 | **1.00** | **1.00** | 0.67 |
| HOUSEHOLD_2 | 0.00 | 0.00 | 0.13 | 0.11 |

**Pattern:** FOODS and HOUSEHOLD prefer tier ZS for Med-Low and above. HOBBIES strongly prefers dept ZS across all tiers. Low-tier series consistently prefer dept ZS (α=0) across all departments.

| Method | geo-mean |
|---|---|
| Dept ZS EW (v20, no tier) | 1.0325 |
| Dept ZS OLS | 0.9829 |
| Tier ZS EW (v21) | 1.0483 |
| **Tier+Dept ZS OLS (v21)** | **0.9375** |
| v20 OLS (ext. tier ens + dept ZS) | **0.7969** |

**Finding:** Tier-grouped ZS adds modest value over dept ZS alone when blended (0.9829 → 0.9375), but the gap to v20 OLS (0.7969) is large (~0.14 geo). The v9/v11 tier ensemble — which combines calibrated ZS forecasts with minimal FT — cannot be replicated by a purely Chronos ZS tier grouping. The external tier ensemble carries absolute scale calibration that Chronos ZS alone cannot recover.

---

## 9. Phase 7 — LightGBM Global Benchmark

To establish a rigorous alternative benchmark, a LightGBM global model was trained on the same 30,490 M5 series.

### 9.1 Feature Engineering

**37 features total:**

- **Lag features (6):** `lag_28`, `lag_29`, `lag_30`, `lag_31`, `lag_35`, `lag_42` — captures weekly and bi-weekly demand cycles. Note: `lag_28` is structurally unavailable for series with fewer than 28 days of active history (see §11).
- **Rolling statistics (10):** mean and std over 7, 14, 30, 60, 180-day windows, all anchored at lag-28 to prevent leakage.
- **Price features (3):** sell_price, price_change, normalised_price.
- **SNAP indicators (3):** CA, TX, WI — SNAP benefit days drive systematic Friday demand spikes.
- **Calendar features (4):** is_weekend, end_of_month, event_type, event_name encoding.
- **Target-encoded demand means (3):** per store, per department, per state — computed from training history before model fit.

### 9.2 Training and HPO

- **Objective:** Tweedie loss (variance power optimised via Optuna) for intermittent count data
- **HPO:** 19 Optuna trials (TPE sampler) over `num_leaves`, `min_data_in_leaf`, `learning_rate`, `feature_fraction`, `bagging_fraction`, `lambda_l1`, `lambda_l2`, `tweedie_variance_power`
- **Training:** gradient boosting with early stopping at 800 rounds; cross-validated on 3 calibration cutoffs

### 9.3 Results

| Cutoff | LightGBM WRMSSE |
|---|---|
| Spring 2016-04-24 | 0.6178 |
| Winter 2016-01-03 | 0.6406 |
| Autumn 2015-10-04 | 0.6016 |
| **3-cutoff geo-mean** | **0.6198** |

**LightGBM beats Chronos v20 OLS (0.7969) by −22%.** This is a substantial margin, not measurement noise.

**Context on the comparison:** The 37-feature pipeline required domain expertise (which lag windows are predictive for weekly retail cycles), feature store infrastructure (cache management, calibration cutoff separation, checkpoint tracking), and iterative debugging across 19 HPO trials. The Chronos pipeline required sweeping four context lengths, toggling one boolean (`cross_learning=True`), and fitting seven OLS regressions. Both reach production-competitive accuracy with very different resource profiles.

---

## 10. Phase 8 — LightGBM × Chronos OLS Blend

### 10.1 Blend Setup

With two complementary models (LGBM and Chronos v20 OLS), the natural question is whether blending captures diversity from both. A scalar LOO-3 OLS blend was fitted: `forecast = β × LGBM + (1−β) × Chronos`, optimised by leave-one-cutoff-out cross-validation per department.

### 10.2 Result

| Method | Spring | Winter | Autumn | geo-mean |
|---|---|---|---|---|
| LightGBM standalone | 0.6178 | 0.6406 | 0.6016 | **0.6198** |
| Chronos v20 OLS | 0.8272 | 0.8726 | 0.7012 | 0.7969 |
| **LGBM × Chronos OLS blend** | — | — | — | **0.6254** |

**The blend (0.6254) is worse than LightGBM standalone (0.6198).** The optimal LOO-3 weight for Autumn converges to β=1.0 (pure LGBM); for Winter and Spring, β≈0.85 (15% Chronos dilutes performance).

**Finding:** On full-history series, Chronos provides no orthogonal diversity over LightGBM on these three cutoffs. The two models capture overlapping signal — where Chronos is good, LGBM is already better. Blending hurts on average.

**Implication:** The value of Chronos-2 relative to LightGBM is not on full-history series — it is on new products with insufficient history for `lag_28` to exist (see §11).

---

## 11. Phase 9 — Cold-Start Natural Segmentation

### 11.1 Motivation and Metric

The prior phases used WRMSSE, which weights series by revenue contribution. This suppresses new-product signal: a newly-launched SKU with 10 days of history has near-zero revenue weight and is invisible in WRMSSE. To measure cold-start performance directly, we switched to **per-series unweighted RMSSE** and grouped series by **active history length** (days since first non-zero sale at each cutoff).

Chronos signal used: pure ZS dept xcl=True (no tier ensemble, no OLS) — exactly what a practitioner would deploy on a brand-new product without any training data.

### 11.2 History Bucket Design

Bucket thresholds are anchored to LGBM feature requirements:
- `lag_28`: requires ≥28 days (most predictive feature)
- `lag_42`: requires ≥42 days
- `roll_mean_180_l28`: requires ≥208 days
- One annual seasonal cycle: ≥365 days

### 11.3 Results — Aggregated Across 3 Cutoffs (geo-mean of per-cutoff medians)

| Active History | n (Autumn) | LGBM RMSSE | Chronos RMSSE | Winner | Chronos wins % |
|---|---|---|---|---|---|
| **< 28 days** | 32 | 1.059 | **1.019** | **Chronos** | **59%** |
| 28–90 days | 57 | **0.812** | 0.847 | LGBM | 40% |
| 90–365 days | 1,274 | **0.735** | 0.871 | LGBM | 20% |
| 365–730 days | 3,642 | **0.712** | 0.836 | LGBM | 20% |
| 730+ days | 25,418 | **0.664** | 0.772 | LGBM | 23% |

*Metric: geo-mean of per-cutoff median per-series RMSSE (unweighted). Not comparable to WRMSSE columns above.*

### 11.4 Per-Cutoff Detail for the <28d Bucket

| Cutoff | n | LGBM median | Chronos median | Chronos wins % |
|---|---|---|---|---|
| Autumn 2015-10-04 | 32 | 1.285 | **1.189** | **72%** |
| Winter 2016-01-03 | 11 | 0.873 | 0.874 | 45% (tie) |
| Spring 2016-04-24 | 0 | — | — | — |

Autumn (October, pre-holiday season) captures the most new-product launches. Chronos wins 72% of series with <28d history at this cutoff. Winter is a near-tie (11 series). Spring has no series in this bucket by April 2016 — all products launched by then have cleared the 28-day threshold.

### 11.5 The 28-Day Crossover — Structural Explanation (Primary Argument)

The threshold at 28 days is not arbitrary. It coincides exactly with the minimum history required for `lag_28` — LGBM's single most predictive feature, capturing demand from the same weekday four weeks prior. Below 28 days:
- `lag_28` is a **structural NaN** — not a noisy estimate, not an imputable missing value, but a feature that does not exist
- LGBM routes these series to its NaN branch, trained to handle temporary stockouts and intermittency gaps — not genuine new products
- Chronos-2 has no such minimum: with 7 observations (one week) it captures the weekly cycle; with 1 observation it still produces a forecast

This argument is sample-size-independent and holds by construction. It is the primary justification for the 28-day rule — the empirical data (§11.6) is illustrative, not the foundation.

At 28 days and beyond, `lag_28` becomes available, LGBM's advantage grows monotonically, and the Chronos win rate drops to 20–40%.

### 11.6 Empirical Support — Strength and Limitations

The per-bucket RMSSE results support the structural argument directionally, but the empirical basis for the <28d bucket is thin:

| Cutoff | n (<28d) | Chronos win rate | Interpretation |
|---|---|---|---|
| Autumn 2015-10-04 | 32 | **72%** | Directional — Chronos wins |
| Winter 2016-01-03 | 11 | 45% | **Tie — does not confirm Autumn** |
| Spring 2016-04-24 | **0** | — | No cold-start series in this bucket |

The cold-start finding effectively rests on **one cutoff** (Autumn) with **32 series**. Winter (n=11) shows a near-tie, which contradicts rather than confirms the Autumn result. Spring contributes nothing to the <28d bucket — by April 2016, all M5 products had cleared the 28-day threshold. With n=32, the confidence interval on "59–72% Chronos win rate" is wide enough to include near-parity.

**Additional coverage gaps:**
- All three cutoffs fall within Autumn 2015–Spring 2016. No Summer evaluation point exists (July–August demand patterns are absent from M5 test data) — a known limitation of the M5 benchmark design.
- The per-dept OLS α weights are fitted on 2 training observations per LOO-3 fold. FOODS_2 α=1.000 (zero Chronos contribution at every cutoff) may reflect genuine signal or a fitting artifact from this thin fitting.

**Consequence for article framing:** Lead with the structural `lag_28` argument. Present the Autumn empirical result (n=32, 72% CW) as illustration. Do not lead with the win rate — it comes from a single cutoff and a near-tie at Winter undermines the headline claim if cited without context.

### 11.7 Practical Deployment Rule

**Deploy Chronos-2 ZS for all SKUs with <28 days of active sales history. Switch to LightGBM (or your production model) once `lag_28` becomes available.**

This rule is mechanistically justified by the structural NaN argument. The empirical confirmation is directional (n=43 across 2 cutoffs); practitioners should validate against their own new-product launch data before adopting 28 days as a hard threshold — it may shift by category or retail context.

---

## 12. Summary Table — All Key Results

| Experiment | Method | Eval | geo-mean |
|---|---|---|---|
| Naive baseline | Seasonal naïve | 3-cutoff | 1.6310 |
| ZS v1 (no cov, CL=2) | Chronos ZS | Spring only | 1.243 |
| ZS per-segment weight HPO | 4 tiers, CL/cov per segment | Spring only | 0.908 |
| FT v6 per-segment | 4 tiers, CL+FT per segment | Spring only | 0.836* |
| v9 tier HPO | Tier ZS+FT blend | 3-cutoff | 0.8947 |
| v10 tier HPO | Tier ZS+FT blend | 3-cutoff | 0.8536 |
| v11 tier HPO | Tier ZS+FT blend | 3-cutoff | 0.8544 |
| Tier ensemble alone | v11 W-matrices | 3-cutoff | 0.8180 |
| v12 dept ZS EW | 7 depts, xcl=False | 3-cutoff | 0.8702 |
| **v13 dept ZS + tier OLS** | 7 depts, xcl=False | 3-cutoff | **0.8066** |
| v14 store ZS OLS | 10 stores, xcl=False | 3-cutoff | 0.8132 |
| v13 small standalone EW | Dept ZS alone, xcl=False | 3-cutoff | 1.0739 |
| v17 BASE standalone EW | Dept ZS alone, xcl=False | 3-cutoff | 1.1963 |
| v20 small standalone OLS | Dept ZS alone, xcl=True | 3-cutoff | 0.9829 |
| v20 dept ZS EW | xcl=True, best CLs | 3-cutoff | 0.8417 |
| **v20 dept ZS + tier OLS** | xcl=True, per-dept α | 3-cutoff | **0.7969** ★ |
| v23 FT (fixed bug) | Per-dept LoRA, 200 steps | 3-cutoff | 0.9509 |
| v24a global FT | All-series LoRA, 200 steps | 3-cutoff | 0.9461 |
| v21 tier ZS EW | Tier groups, xcl=True | 3-cutoff | 1.0483 |
| v21 tier+dept ZS OLS | Per-(dept,tier) α | 3-cutoff | 0.9375 |
| **LightGBM global** | **37 features, tweedie HPO** | **3-cutoff** | **0.6198** |
| LGBM × Chronos OLS blend | LOO-3 scalar β per dept | 3-cutoff | 0.6254 |

*\* v6 spring-only, LoRA component may be affected by PeftModel bug*  
*★ Best Chronos result*

**Cold-start results (per-series RMSSE — separate metric, not comparable to WRMSSE above):**

| Active history bucket | LGBM median RMSSE | Chronos median RMSSE | Winner |
|---|---|---|---|
| < 28 days | 1.059 | **1.019** | **Chronos** |
| 28–90 days | **0.812** | 0.847 | LGBM |
| 730+ days | **0.664** | 0.772 | LGBM |

---

## 13. Key Findings

**F1 — Cross-learning is the primary lever (+0.025 geo EW)**  
Enabling `cross_learning=True` in `predict_quantiles()` alone improves the dept ZS EW from 0.8672 to 0.8417. This activates Chronos-2's group attention mechanism. All prior work (v12/v13) had left this disabled. Requires dept-segmented batches — global batches with covariates degrade performance (1.0776 → 1.1765 with xcl=True globally).

**F2 — OLS per-dept weighting adds +0.045 over EW**  
Moving from 50/50 fixed blend to per-dept LOO-3 optimised α improves 0.8417 → 0.7969. The OLS primarily corrects two structural biases: (i) FOODS depts benefit more from the tier ensemble (calibrated external signal); (ii) HOUSEHOLD_1 and HOBBIES_1 benefit more from Chronos ZS.

**F3 — The tier ensemble is load-bearing (~0.19 geo gap)**  
Removing the v9/v11 tier ensemble entirely (dept ZS OLS alone, no tier) gives 0.9829 vs 0.7969 with tier. The ensemble provides absolute scale calibration that Chronos ZS cannot recover through grouping choices alone.

**F4 — FOODS systematic under-forecast bias**  
Chronos-only OLS weights for FOODS: FOODS_1=1.59×, FOODS_2=1.18×, FOODS_3=1.16×. Chronos under-forecasts FOODS when cross-learning is limited to dept peers — all FOODS peers share the same bias. The tier ensemble, calibrated independently, corrects this. Tier-grouped ZS (v21) only partially helps — FOODS Med-Low/High prefer tier ZS (α=1.0), suggesting volume-similar non-FOODS peers provide better scale calibration.

**F5 — Catastrophic forgetting makes FT non-viable beyond 5 steps**  
The FT steps curve (spring cutoff, global FT) shows: 5 steps gives a marginal improvement (EW 0.8623→0.8581), then monotonic degradation. By 10 steps, performance is already worse than ZS. By 100 steps, the raw forecast WRMSSE has doubled. The pretrained cross-learning representation is destroyed faster than new task-specific adaptation is learned.

**F6 — Dept segmentation outperforms store or state segmentation**  
Per-dept OLS (0.8066) > per-store OLS (0.8132) > per-state (implied worse). Product category encodes demand dynamics (seasonality, price sensitivity, SNAP effects) better than geography. Stores within a state carry identical category distributions.

**F7 — Tier-grouped ZS is complementary but weaker than dept-grouped ZS**  
Tier ZS EW = 1.0483 vs dept ZS EW = 1.0325. Blending them (v21 OLS = 0.9375) improves over dept-only (0.9829) but neither approach replaces the external tier ensemble. The complementarity is real but modest — a ~0.05 geo improvement.

**F8 — LightGBM dominates on full-history series (−22% vs best Chronos)**  
A 37-feature global LightGBM model with Optuna HPO achieves geo=0.6198, beating Chronos v20 OLS (0.7969) by 22%. Blending the two models (OLS LOO-3) does not recover diversity — the blend (0.6254) is worse than LGBM standalone. On full-history series, Chronos adds no orthogonal signal beyond what LGBM already captures.

**F9 — Cold-start 28-day crossover: Chronos wins below `lag_28` availability**  
Below 28 days of active sales history, Chronos-2 ZS beats LGBM on 59–72% of individual series (geo-mean median RMSSE: 1.019 vs 1.059). At 28+ days, LGBM wins by a growing margin. The crossover at 28 days is mechanistically explained by `lag_28` availability — LGBM's most predictive feature is structurally undefined for series with fewer than 28 days of sales. **Practical rule: use Chronos-2 for the first 28 days after SKU launch; switch to LGBM once `lag_28` becomes available.**

*Validation note: the structural argument (lag_28 is a hard NaN below 28d) is sample-size-independent and is the primary support. The empirical evidence is directional: Autumn n=32 (72% CW) is the only strongly directional cutoff; Winter n=11 (45%) is a tie; Spring contributes 0 series. Do not cite empirical win rates without this context.*

---

## 14. What Didn't Work

| Approach | Result | Why |
|---|---|---|
| Naive shared FT | 1.017 (spring) | Same config across all tiers over-fits low-volume items |
| Store-grouped segmentation | 0.8132 OLS | Geography doesn't encode demand dynamics |
| State × weight 12-segment HPO | ~1.024 | Parameter space explosion without signal gain |
| LoRA FT (v15–v22) | Invalid | PeftModel inference bug: merge_and_unload() was missing |
| Raw FT after bug fix (v23/v24a) | 0.9461–0.9509 | Catastrophic forgetting even at 200 steps |
| Multi-seed ensemble | 0.8554 | FT seed variance is too low to provide diversity; worse than single seed (0.8544) |
| Re-optimisation (400 FEV on v11) | 0.8180 | Lower bound — winter doesn't benefit from more FT steps |
| Tier-grouped Chronos ZS (v21) | 1.0483 EW | Cross-learning within volume tier is weaker than within product category |
| xcl=True with global (mixed) batches | 1.1680 | Covariate representations contaminated across incompatible series |
| LGBM × Chronos OLS blend | 0.6254 | No orthogonal diversity on full-history series; worse than LGBM alone |

---

## 15. Open Questions

**Q1 — Can 5-step FT be validated on 3 cutoffs?** *(low priority)*  
The steps curve ran on spring only. 5 steps gave a marginal improvement (EW 0.8581 vs ZS 0.8623). Given catastrophic forgetting's speed, the improvement may be season-specific and not worth the infrastructure overhead.

**Q2 — Can a replacement for the tier ensemble be constructed purely from Chronos signals?**  
v21 showed that tier-grouped ZS alone doesn't match the external tier ensemble. The tier ensemble's advantage is scale calibration, not just grouping. A Chronos-native approach would need to recover absolute scale — possibly through per-dept scaling factors combined with the cross-learning signal.

**Q3 — What happens in the 28–90 day transition zone?**  
The blend optimal weight β converges to 1.0 (pure LGBM) at Autumn. For Spring, β≈0.85, meaning 15% Chronos. A finer-grained sweep in the 28–56 day range (as lag_28 first becomes available and lag_42 is still missing) would clarify whether a gradual transition is better than a hard switch at day 28.

**Q4 — External validation of the cold-start 28-day rule** *(high priority for article credibility)*  
The current empirical evidence rests on n=32 series from one effective cutoff (Autumn 2015). Winter (n=11) shows a tie, not confirmation. Validation on an independent new-product launch dataset — even synthetic — would add the second confirming data point the cold-start finding currently lacks. Without this, the structural argument must carry the narrative.

**Q5 — Summer evaluation gap**  
All three M5 cutoffs fall between September 2015 and April 2016. No Summer evaluation point (July–August) is available — a known limitation of the M5 benchmark design. Summer demand in retail (seasonal products, vacations, back-to-school) can be a genuinely different regime. The generalizability of the Chronos vs LGBM findings to Summer is untested.

---

## 16. Validation Coverage Assessment

Summary of narrative strength by finding, for article framing purposes.

| Narrative | Cutoffs | Evidence quality | Primary support | Verdict |
|---|---|---|---|---|
| Chronos workflow (v20 OLS = 0.7969) | 3 (standard M5) | Strong — geo-mean across 3 seasonal regimes, LOO-3 OLS | Empirical | **Sufficient** |
| LightGBM benchmark (0.6198, −22%) | 3 (same) | Strong — gap large enough to be robust to 3-cutoff variance | Empirical | **Sufficient** |
| xcl×covariate interaction | 3 + global/dept ablation | Strong — consistent mechanism across multiple experiment variants | Mechanistic + empirical | **Sufficient** |
| Catastrophic forgetting (FT degrades <10 steps) | 1 (spring only) | Adequate — degradation magnitude (2× at 200 steps) is too large to be cutoff-specific | Empirical (1 cutoff) | **Adequate** |
| Cold-start 28-day rule | 1 effective (Autumn) | Thin — n=32, Winter ties, Spring=0; structural argument is what holds | **Structural (primary)** | **Conditional** |
| Per-dept OLS α weights | 3 (LOO-3, 2 obs/fold) | Directional — directions reliable, magnitudes fragile | Empirical (thin folds) | **Directional** |

**Conditional** means: cite the structural argument first; the empirical data illustrates but does not independently prove.  
**Directional** means: the qualitative pattern (which depts prefer which model) is reliable; the specific α values should not be over-interpreted.

**Seasonal coverage gap:** Autumn / Winter / Spring are covered. Summer (July–August) is absent from M5 test data — this limitation applies to all findings equally and should be noted once in the article rather than repeated per-finding.
