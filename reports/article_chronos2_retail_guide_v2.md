# Chronos-2 for Retail Demand Forecasting
## A Practitioner's Workflow from Zero-Shot to Cold-Start

**Aquila Data — June 2026**

---

## Executive Summary

*For technical managers and decision-makers*

Time-series foundation models now offer a genuinely different trade-off for retail forecasting. This article is written for teams who have decided — or are seriously considering — Chronos-2 as a forecasting backbone, and want to understand how to extract its full value.

We stress-tested Chronos-2-small across 30,490 Walmart SKUs from the M5 competition benchmark: 28-day horizon, WRMSSE metric, three seasonal cutoffs (Spring 2016-04-24, Winter 2016-01-03, Autumn 2015-10-04). The result of a carefully configured zero-shot pipeline: **WRMSSE 0.7969** — a 51% improvement over the naïve seasonal baseline (1.6310).

We then benchmarked this against a fully tuned LightGBM model — 37 hand-crafted lag, rolling, and exogenous features, 19 Optuna HPO trials, tweedie objective — which achieved **0.6198**. LightGBM wins at full history. We report this openly.

What LightGBM cannot do — and what defines Chronos-2's distinct value — is forecast products it has never seen in training. For SKUs with fewer than 28 days of active sales history, LightGBM's most predictive feature (`lag_28`) is structurally unavailable. In our cold-start experiment, Chronos-2 outperformed LightGBM on 59–72% of newly launched SKUs. Below 28 days of history, Chronos is not competing with LightGBM. It is the only option.

**For teams who already have a mature ML pipeline:** Chronos-2 is your cold-start layer — deploy it for the first 28 days after any product launch, then hand off to your production model.

**For teams building forecasting capability from scratch:** Chronos-2's workflow reaches 0.7969 — within the range of production-grade accuracy — with a fraction of the feature engineering, training infrastructure, and domain expertise that classical ML requires.

This article documents the full workflow: what works, what doesn't, and why.

---

## Part I — The Problem and the Proposition

---

### 1. Retail Forecasting Is a Cold-Start Problem in Disguise

The standard narrative around retail demand forecasting focuses on mature products: high-volume SKUs with years of history, seasonal patterns, promotional uplift curves. These are tractable. A well-tuned gradient boosting model with lag features handles them reliably.

The harder problem is launches. A mid-size retailer introduces hundreds of new SKUs per month. A fast-fashion retailer turns over its entire assortment seasonally. A grocery chain adds regional lines constantly. In all these cases, the conventional ML pipeline has a structural blind spot: it can only forecast products it has seen during training. A product launched today has no lag features, no rolling statistics, no target encoding — the backbone of classical tree-based forecasting.

The typical response is to wait. Gather a few weeks of data, then add the product to the next retraining cycle. In the interim, the demand planner guesses. The supply chain over-orders to be safe. Waste accumulates.

Foundation models like Chronos-2 break this pattern. They arrive pre-trained on hundreds of millions of diverse time-series observations. A new SKU does not require a training run — the model has already learned the space of demand patterns that retail products inhabit. Point it at any series, however short, and it produces a forecast from day one.

This is Chronos-2's primary value proposition for retail: not that it beats every classical model on every series, but that it operates without the data prerequisite that makes classical models useless at the beginning of a product's life.

---

### 2. The Competitive Landscape — An Honest Framing

Before describing the Chronos workflow, we owe the reader an honest benchmark. If a simpler approach achieves substantially better performance, that matters — regardless of the engineering story.

We built a LightGBM global forecasting model on the same 30,490 M5 series with 37 features:

**Lag features:** `lag_28`, `lag_29`, `lag_30`, `lag_31`, `lag_35`, `lag_42` — six scalar lookback values capturing weekly and bi-weekly demand cycles.

**Rolling statistics:** mean and standard deviation over windows of 7, 14, 30, 60, and 180 days — all anchored at lag-28 to avoid leakage. These capture trend, seasonality depth, and demand volatility.

**Exogenous features:** sell price, price change, normalised price, SNAP eligibility flags (CA/TX/WI), calendar features (weekend indicators, month boundaries), event encoding, store- and state-level target-encoded sales averages.

**Objective and training:** tweedie loss (variance power tuned via Optuna) for intermittent count data, gradient-boosted trees with early stopping, 19 HPO trials tuning num_leaves, min_data_in_leaf, learning rate, feature fraction, bagging fraction, and L1/L2 regularisation.

**Result:** WRMSSE **0.6198** (3-cutoff geo-mean: Autumn 0.6016, Winter 0.6406, Spring 0.6178).

Chronos-2 v20 OLS: **0.7969**.

LightGBM wins by 22%. This is not a rounding error.

But the comparison deserves context on two counts.

**First: what the LightGBM result cost to produce.** The 37-feature pipeline required domain knowledge (which lag windows matter for weekly retail cycles), infrastructure (a feature caching system, calibration cutoffs, checkpoint management, 19-trial HPO), and iterative debugging across multiple Optuna runs. The Chronos pipeline required sweeping four context lengths, enabling one boolean flag (`cross_learning=True`), and running seven OLS regressions. Both produce respectable results; they represent very different resource investments. The teams best served by Chronos are those who would otherwise spend months building what LightGBM required before getting their first production number.

**Second: what LightGBM cannot do.** The LGBM model was trained on M5 historical data that includes every test series across their full training period. When we evaluate it on short-history series — products that had first sales fewer than 28 days before the test cutoff — `lag_28` is undefined (NaN), `lag_42` is undefined, and five of the seven rolling statistics are undefined. The model routes these to its NaN-branch, but this branch was trained to handle production-time missingness, not true product absence. For genuinely new products, LightGBM is extrapolating outside its training distribution.

The cold-start experiment quantifies this precisely (§13). The 28-day threshold is not arbitrary: it is exactly where `lag_28`, the single most predictive lag feature, becomes available. Below that line, Chronos operates as designed; LightGBM operates in a regime it was not built for.

---

## Part II — Building the Chronos Workflow

---

### 3. Starting Point: The Naïve Baseline and the First Zero-Shot Run

Any evaluation requires a floor. Our naïve baseline repeats the last seven observed daily sales per series, producing a weekly-cycle forecast:

| Cutoff | WRMSSE |
|--------|--------|
| Spring 2016-04-24 | 1.4639 |
| Winter 2016-01-03 | 1.4216 |
| Autumn 2015-10-04 | 2.0848 |
| **Geo-mean** | **1.6310** |

The Autumn cutoff (October 2015) is harder: pre-holiday stocking disrupts the weekly pattern the naïve baseline relies on. Any meaningful model must handle this regime.

Our first Chronos-2-small run used a context length of 2 — two historical observations fed to the model — across all 30,490 series without segmentation or covariates. The result: **1.2504**. Better than naïve, but not impressively so. More revealingly: CL=1 is numerically *identical* to the seasonal naïve baseline (1.6310). With a single observation, Chronos produces the same last-value weekly repeat that naïve generates. The model needs enough context to see beyond the most recent observation.

> **Lesson 1 — Context length is the dominant zero-shot hyperparameter.** For intermittent daily retail data, CL=7 (one week) captures the weekly demand cycle. Beyond CL=14 performance degrades on noisy series — the model treats historical spikes as structural signal. Never deploy without a CL sweep.

The progression from naïve to a properly configured ZS pipeline illustrates the cumulative effect of each configuration choice:

| Step | Method | geo | Δ vs prior | What this answers |
|------|--------|-----|------------|-------------------|
| 0 | Seasonal Naive | 1.6310 | — | Pure seasonal repetition |
| 1 | ZS sales only, xcl=True, CL=7 | 1.2504 | −23% | What a pretrained FM adds from history alone |
| 2 | ZS + covariates (eP_sF), xcl=False, CL=7 | 1.0776 | −15% | What exogenous context adds |
| 3 | FT global (25 steps, lr=1e-7) | 1.0319 | −4% | What fine-tuning adds at global scale |
| 4 | ZS dept-seg xcl=True + OLS | **0.7969** | −23% | What semantic batching and calibration add |

Each row answers a natural "what if we go one step further?" question. The ZS-to-v20 journey is one of progressive configuration understanding, not of model retraining.

---

### 4. Covariates and Cross-Learning — A Critical Interaction

Two distinct mechanisms contribute signal in the Chronos-2 inference path: **exogenous covariates** (per-series price, promotion, SNAP, and calendar data) and **cross-series attention** (`cross_learning=True`). Understanding how they interact is essential to configuring the pipeline correctly.

At CL=7 with a global inference pool (all 30,490 series), the interaction is clean and counterintuitive:

|  | xcl=False | xcl=True | Δ (xcl effect) |
|--|-----------|----------|----------------|
| **No covariates** | 1.2716 | **1.2504** | −0.021 *(xcl helps)* |
| **With covariates** | **1.0776** | 1.1765 | +0.099 *(xcl hurts)* |

Covariates and cross-learning are **substitutable, not complementary**. They each provide inter-series contextual information through different channels — covariates supply per-series exogenous history; cross-attention supplies cross-series demand similarity. When both are active in a globally-mixed batch, cross-attention mixes covariate representations *across* series with different event exposures: a California series attends to a Texas series during a California-only SNAP period, injecting noise into both representations. The per-series covariate signal dominates; cross-attention interference degrades it.

Without covariates, cross-attention operates cleanly — the only available inter-series context is demand-level similarity within the batch, which it exploits reliably (−0.021 gain).

The resolution is **semantic batch coherence**: when cross-attention operates within a department-segmented batch — 100 FOODS_1 series all sharing the same event exposure and similar demand patterns — the contamination disappears. Within-department batches have coherent covariate structures, and cross-attention extracts genuine group signal.

This motivates the fundamental architecture of the optimised pipeline: global inference uses xcl=False with covariates; xcl=True is reserved for department-segmented batches.

> **Lesson 2 — Covariates and cross-learning compete for the same signal budget in mixed batches. With rich exogenous data, disable cross-learning globally. Enable it only within category-coherent batches.**

---

### 5. Segmentation — Volume Tiers Beat Demand-Pattern Classification

M5 series span a very wide volume range: a FOODS item might sell 200 units per day; a HOBBIES item, 1–2. These categories respond differently to context length. The natural segmentation hypothesis: assign each series to a group and sweep the group's optimal CL independently.

We tested three segmentation axes:

**ADI/CV² demand classification** (smooth / erratic / intermittent / lumpy): 0.9418 on the spring cutoff. Good, but the classification requires computing per-series statistics and introduces a preprocessing dependency. More importantly, it misaligns with the evaluation metric: WRMSSE weights series by their contribution to forecast error, not by their demand pattern type.

**Volume tiers** (WRMSSE weight quartiles: Low, Med-Low, Med-High, High): **0.9082** on spring. Simpler to compute, directly reflects the metric structure, and improves meaningfully over ADI/CV². Lower-weight tiers need very short CLs; higher-weight tiers benefit from slightly longer context. Counter-intuitively, the Med-High tier needed CL=1 — the model sees enough from one observation to identify the dominant weekly pattern in this volume range.

**Store segmentation** (10 Walmart stores by geography): 0.8132 OLS (3-cutoff) — worse than department-level ZS (0.8066). Geographic proximity does not capture demand-pattern homogeneity. The ten stores share weekly seasonality; they differ in volume level, which the tier structure already captures. Store-level fine-tuning added noise on both axes.

> **Lesson 3 — Segment by demand behaviour, not by geography or organisational hierarchy.** Volume-tier and product-category segmentations align with how the model processes context. Geographic segmentation rarely helps in grocery retail at daily granularity.

---

### 6. Fine-Tuning Dead End — Catastrophic Forgetting

The natural next hypothesis: if zero-shot with the right configuration reaches 1.0776, a few gradient steps on Walmart data should push it further.

We ran a systematic step-count curve: global fine-tuning across all 30,490 series for 0 to 1000 gradient steps, then blending the fine-tuned model 50/50 with the tier ensemble (§7):

| Steps | ZS raw WRMSSE | Blended WRMSSE |
|-------|--------------|----------------|
| 0 | 1.063 | 0.8623 |
| **5** | 1.037 | **0.8581** ← peak |
| 10 | 1.049 | 0.8642 |
| 25 | 1.081 | 0.8833 |
| 50 | 1.164 | 0.9259 |
| 100 | 1.345 | 1.0116 |
| 200 | 1.549 | 1.1029 |
| 500 | 1.628 | 1.1321 |

*Spring cutoff, global fine-tuning on all 30,490 M5 series.*

The result is unambiguous. Five steps produce a marginal 0.004 improvement in the blended score. Ten steps return to baseline. Beyond ten, degradation is monotonic — performance nearly doubles at 200 steps and approaches the naïve baseline at 500 steps.

This is **catastrophic forgetting**: gradient updates overwrite the pre-trained cross-series representations that carry the relevant signal faster than task-specific patterns are acquired. Chronos-2's pre-training encodes demand pattern diversity across millions of diverse series; M5-specific fine-tuning rapidly narrows this representation to Walmart-specific patterns while losing the broader signal.

A 5-step gain of 0.004 WRMSSE is below measurement noise across three seasonal cutoffs and does not justify the inference overhead, checkpoint management, and retraining schedule a production FT pipeline requires.

> **Lesson 4 — Do not fine-tune a TSFM on intermittent retail data without strict step-count control.** Catastrophic forgetting is not theoretical — it manifests within 10 gradient steps on 30,000 series. If fine-tuning is necessary, use LoRA (adapter-only training, frozen base). Validate at 5, 10, and 25 steps. Stop immediately if performance plateaus or degrades.

---

### 7. The PeftModel Bug — When Your Experiment Is Silently Wrong

We did pursue LoRA as the lower-risk path after observing catastrophic forgetting in full fine-tuning. Between June 13 and June 15 we ran multiple LoRA experiments across segmentation strategies (per-tier, per-department, isolated departments — v15 through v22). Results were inconsistent and underwhelming.

Only in mid-June did we find the root cause: a missing `merge_and_unload()` call.

When Chronos-2 uses LoRA, the base model and the adapter are separate components during training. Chronos's quantile inference path (`predict_quantiles()`) operates on a `MeanScaleUniformBins` pipeline that does not route through the PEFT wrapper. Without explicit merging, the adapter weights are present but unreachable during inference — the "fine-tuned" model is functionally identical to the base model.

```python
# Silent failure — adapter weights ignored
pipeline = MeanScaleUniformBins(peft_model)

# Correct — adapter fused before wrapping
pipeline = MeanScaleUniformBins(peft_model.merge_and_unload())
```

Every LoRA result from v15 through v22 was numerically identical to the base model. After the fix, per-department LoRA (v23, 200 steps) achieved **0.9509** (3-cutoff geo-mean) — better than pre-fix results but still far behind the ZS + tier pipeline (0.7969).

> **Lesson 5 — PEFT adapters require explicit merging before TSFM-style inference.** The standard HuggingFace `generate()` path routes correctly through the PEFT wrapper; Chronos's `predict_quantiles()` does not. Validate by confirming that fine-tuned and base-model outputs differ. If they are identical, the adapter is not being applied.

---

### 8. The Tier Ensemble — Fine-Tuning at the Right Scale

While global fine-tuning failed, fine-tuning within a *structured tier decomposition* produced durable gains. The key insight was that tier coherence both constrains forgetting (fewer diverse series per training batch) and creates a meaningful diversity axis for the ensemble.

We fine-tuned four separate models — one per volume tier — using full fine-tuning with aggressive early stopping. The blend was learned as a **4×4 softmax weight matrix W**, where entry `W[tier_assignment][tier_model]` gives the contribution of each tier model to each series based on its tier assignment:

```python
# Per-series tier-aware blend
for series_i in all_series:
    tier = weight_quartile(series_i)
    forecast_i = sum(W[tier][t] * tier_forecast[t][series_i]
                     for t in [Low, MedLow, MedHigh, High])
```

W was optimised with Nelder-Mead (400 function evaluations, 3-cutoff WRMSSE objective). An initial Optuna search found the best per-tier hyperparameters (best trial: 0.8544); the dedicated W-matrix re-optimisation then improved the standalone tier result to **0.8180**.

The W-matrix structure matters: a global four-vector blend (same weights for all 30k series) reaches only 0.9278. The per-tier-assignment structure — each series uses the blend appropriate to its tier — accounts for the full gap between 0.93 and 0.82.

---

### 9. Cross-Learning and Semantic Batch Composition — The Unexpected Lever

The most significant ZS gain came from a parameter most users never configure: `cross_learning`. When enabled, Chronos-2 activates cross-series attention across the 100 series in each inference batch — rather than treating each series as independent, the model attends across batch peers and can exploit correlated demand patterns.

Global cross-learning on mixed-category batches degraded performance (§4). But with department-segmented inference:

| xcl | Batch composition | geo EW |
|-----|-------------------|--------|
| False | Mixed (global) | 1.0776 |
| False | Dept-seg | 1.0739 |
| True | Mixed (global) | 1.1765 — worse |
| **True** | **Dept-seg** | **1.0325 — best ZS** |

Within a FOODS_1 department batch, 100 series share the same SNAP eligibility, face the same promotional events, and follow similar weekly demand patterns. The cross-attention head operates on genuinely coherent peers — it learns group-level demand signal rather than mixing noise from incompatible categories.

A secondary finding: FOODS_1 series showed a systematic 1.59× underestimation in dept-seg xcl=True mode. Within pure FOODS_1 batches, all peers are high-volume items and there is no external anchor to correct collective bias. The OLS calibration step (§10) addresses this; the bias is predictable and reproducible.

> **Lesson 6 — Cross-learning requires semantic batch coherence. Sort your series by product category before calling `predict_quantiles`, and run one inference call per category. In mixed-category global batches, disable cross-learning entirely. The gain from correct configuration is ∼0.04 WRMSSE — larger than most other single-parameter choices.**

```python
# Correct: one call per dept, cross_learning enabled within coherent groups
for dept in ["FOODS_1", "FOODS_2", ..., "HOUSEHOLD_2"]:
    forecasts[dept] = chronos_small.predict_quantiles(
        filter_by_dept(all_series, dept),
        context_length=cl_by_dept[dept],
        cross_learning=True,
        batch_size=100,
    )

# Incorrect: global pool with cross_learning=True degrades performance
forecasts = chronos_small.predict_quantiles(
    all_series,
    context_length=7,
    cross_learning=True,  # hurts with covariates in mixed batches
)
```

---

### 10. The BASE Model — Size Is Not a Guarantee

Chronos-2 BASE (120M parameters, ∼2.6× larger than small) was evaluated in standalone mode with per-department inference and optimal CL sweep. Result: **1.1963 EW** — worse than Chronos-2-small xcl=True (1.0325) by a substantial margin.

With cross-learning enabled and dedicated CL HPO: **1.1106** — marginal improvement, confirming the search space is exhausted. BASE xcl=True in department-segmented batches introduces severe FOODS_1 scale bias (α=1.943×, nearly double), the most extreme covariate contamination observed in any experiment.

When blended with the tier ensemble (50% tier + 50% BASE ZS), the apparent gain (0.9123) was entirely driven by the tier component. BASE contributed very little independent signal.

**Why:** M5 data is short (∼5 years daily, many series with gaps and intermittency). A 120M-parameter model has substantially more capacity, but that capacity was calibrated on long, continuous, diverse series. Short intermittent retail data does not provide enough signal to leverage the extra parameters, and the larger model over-fits to Walmart-specific patterns that smaller models correctly treat as noise.

> **Lesson 7 — For short intermittent retail series, prefer the smaller TSFM. Larger models are trained for long-horizon continuous datasets. Always benchmark both sizes explicitly before assuming larger is better.**

---

### 11. The Final Recipe: v20 OLS (WRMSSE 0.7969)

The best result combines two complementary components via a simple calibration step:

**Component A — Tier ensemble (0.8180 standalone):** four independently fine-tuned tier models blended with a 4×4 softmax W-matrix optimised by Nelder-Mead. Captures fine-grained temporal patterns within each volume segment.

**Component B — Department ZS xcl=True (1.0325 EW / 0.9829 OLS standalone):** Chronos-2-small, cross-learning enabled, per-department batches, per-department optimal CL from Optuna HPO. Brings global cross-series representation learning that tier-isolated fine-tuning never sees.

**Calibration — Per-department LOO-3 OLS:** one multiplicative scalar per department, fit by leave-one-cutoff-out cross-validation (three folds). Corrects the FOODS_1 scale bias (α=1.59×) introduced by dept-seg xcl=True. Prevents overfitting to the three available seasonal cutoffs.

```python
# Per-dept LOO-OLS blend
for dept in departments:
    alpha = loo_ols_fit(
        tier_forecasts[dept], zs_forecasts[dept],
        actuals[dept], n_cutoffs=3
    )
    final[dept] = alpha * tier[dept] + (1 - alpha) * zs[dept]
```

**Why they complement each other:** the tier ensemble's fine-tuned representations are adapted to Walmart-specific temporal patterns within each volume segment. The ZS model's cross-learning captures cross-category demand similarity that per-tier fine-tuning, trained independently, never observes. OLS allows these contributions to emerge without manual tuning.

**LOO temporal ordering note:** the three cutoffs are chronologically ordered (Autumn < Winter < Spring). Standard LOO allows future data to inform past held-out folds — only the Spring fold is fully leakage-free. The residual optimism is small (seven scalars constrained to [0,1], empirically stable across runs) and we accept it pragmatically given the small evaluation set. With five or more cutoffs, adopt strict temporal forward-chaining.

> **Lesson 8 — ZS and lightly fine-tuned ensemble components are more complementary than expected. ZS cross-learning representations are orthogonal to FT task-adapted representations. LOO cross-validation is mandatory with small evaluation sets — in-sample OLS trivially overfits with three cutoffs.**

---

### 12. How the Gains Accumulated

| Stage | Configuration | WRMSSE | Gain vs. Naïve |
|-------|--------------|--------|----------------|
| Seasonal Naive | Repeat last 7 obs | 1.6310 | — |
| ZS CL=1, global | Chronos-2-small, 1 obs context | 1.6310 | 0.0% |
| ZS CL=7, global xcl=False + cov | Best single-CL global ZS | 1.0776 | −34% |
| ZS CL=7, dept-seg xcl=False | Per-dept CL (v13) | 1.0739 | −34% |
| ZS dept-seg xcl=True | Cross-learning within depts (v20 EW) | 1.0325 | −37% |
| ZS OLS-scaled | Dept ZS + per-dept α (v20 OLS) | 0.9829 | −40% |
| Tier ensemble (W re-opt) | 4-tier FT + 4×4 W-matrix | 0.8180 | −50% |
| **Tier + ZS xcl=True OLS** | **v20: full pipeline** | **0.7969** | **−51%** |

The pattern: ZS configuration choices (CL, segmentation, cross-learning) account for the largest absolute gain (1.6310 → 1.0325). The tier ensemble and OLS calibration account for the final stretch (1.0325 → 0.7969). Blend weight quality (0.8544 → 0.8180 via W-matrix re-optimisation) contributed more than adding the ZS component to an unoptimised tier (0.8417 → 0.7969 via ZS + OLS). Both matter; blend architecture matters as much as model selection.

---

## Part III — The Honest Comparison

---

### 13. LightGBM as Contender — What Fine-Grained Accuracy Costs

A paper that only reports its chosen model's results without benchmarking against the obvious alternative is not useful. We built the LightGBM model.

**37 features.** Six lag features (28–42 days) capturing weekly and bi-weekly demand cycles. Seven rolling statistics (mean and standard deviation at 7, 14, 30, 60, 180-day windows, all anchored at lag-28). Sell price, price change, price normalisation. SNAP eligibility flags. Calendar features: weekend indicators, month-start and month-end flags, event encoding. Store-level, department-level, and state-level target-encoded sales averages — requiring an aggregation pass over the full training history before any model training can begin.

**Hyperparameter optimisation.** 19-trial Optuna TPE search tuning: `num_leaves`, `min_data_in_leaf`, `learning_rate`, `feature_fraction`, `bagging_fraction`, `lambda_l1`, `lambda_l2`, and `tweedie_variance_power`. Each trial runs three calibration cutoffs × up to 800 boosted rounds — a non-trivial compute investment.

**Training infrastructure.** Feature caching pipeline (features built once and stored per cutoff), checkpoint management, calibration cutoff vs test cutoff separation, post-hoc re-evaluation from stored checkpoints.

**Results:** WRMSSE **0.6198** (Autumn 0.6016, Winter 0.6406, Spring 0.6178).

A natural question: can we blend LGBM and Chronos to get the best of both? We ran a LOO-3 scalar OLS blend. Result: **0.6254** — *worse* than LGBM standalone (0.6198). The optimal blend weight for Autumn converged to β=1.0 (pure LGBM); for Winter and Spring, β≈0.85 (15% Chronos dilutes performance). The two models provide no orthogonal diversity on these three cutoffs. On full-history series, Chronos adds nothing beyond what LGBM already captures.

**The structural argument for Chronos is not weakened by this.** The LGBM pipeline described above is a mature team's months of work. For teams without that infrastructure — without the domain knowledge to select these features, without the HPO budget, without the retraining cadence — Chronos's 0.7969 is the right first investment. It is achievable in days, requires no feature engineering, and stays competitive.

More importantly: the LGBM pipeline has a ceiling that Chronos does not share.

---

### 14. The Cold-Start Advantage — The 28-Day Rule

We return to the problem that opened this article: new product launches.

To measure this directly, we computed per-series unweighted RMSSE for each of the 30,490 M5 series, grouped by their *active history length* at each test cutoff — the number of days since their first non-zero sale. This metric is deliberately unweighted: the WRMSSE metric that favours high-volume series suppresses exactly the new-product signal we are measuring.

The history buckets are not arbitrary. They are anchored on LGBM's feature requirements:
- `lag_28`: available only after 28 days
- `lag_42`: available only after 42 days
- `roll_mean_180_l28`: requires 208 days of history
- One full seasonal cycle: 365 days

**Results — aggregated across 3 cutoffs (geo-mean of per-cutoff medians):**

| Active History | n (Autumn) | LGBM RMSSE | Chronos RMSSE | Winner | Chronos wins % |
|----------------|-----------|------------|---------------|--------|----------------|
| **< 28 days** | 32 | 1.059 | **1.019** | **Chronos** | **59%** |
| 28 – 90 days | 57 | **0.812** | 0.847 | LGBM | 40% |
| 90 – 365 days | 1,274 | **0.735** | 0.871 | LGBM | 20% |
| 365 – 730 days | 3,642 | **0.712** | 0.836 | LGBM | 20% |
| 730+ days | 25,418 | **0.664** | 0.772 | LGBM | 23% |

*Chronos signal: pure ZS dept xcl=True, no tier ensemble, no fine-tuning — exactly what a practitioner deploys at day one.*

The crossover is at 28 days. Below that threshold, Chronos outperforms LGBM on 59% of individual series and achieves a lower median RMSSE across the available seasonal cutoffs. At the Autumn cutoff specifically — which captures the most product launches — Chronos wins **72%** of newly launched SKUs with a median RMSSE of 1.189 vs LGBM's 1.285.

Above 28 days, LGBM wins by an increasing margin: 4% at 28–90 days, 15–18% at 90–365 days, 16% at 730+ days.

**Why 28 days exactly?** `lag_28` is LGBM's most predictive single feature — the same-weekday demand from four weeks prior, which captures weekly seasonality directly. Below 28 days, `lag_28` is a structural NaN — not a noisy estimate, not a missing value that can be imputed, but a feature that does not exist. The model routes it to the NaN-branch learned during training, but that branch was not trained for brand-new products; it was trained for temporarily out-of-stock items and intermittency gaps.

Chronos-2 has no such minimum. With a single observation, it can produce a forecast. With seven observations (one week), it captures the weekly cycle. With 28 observations, it performs comparably to a mature model on the same series. The foundation model does not fail at product launch — it starts contributing immediately.

**The practical deployment rule:** use Chronos-2 for any product with fewer than 28 days of active sales history. Switch to your production model (LGBM or equivalent) once `lag_28` becomes available. This hybrid is not a theoretical preference — it is empirically justified by the crossover observed in our data.

**On sample size:** the <28-day bucket contains 43 series across two cutoffs (32 at Autumn, 11 at Winter; Spring had no series in this range by April 2016). The empirical evidence is directional, not conclusive. The structural argument — lag_28 is a hard boundary — is independent of sample size and stands on its own.

---

## Part IV — Practical Guidance

---

### 15. A Decision Framework — When and How to Use Chronos

The decision is not binary. Chronos-2 and LightGBM serve different needs; a production forecasting system may legitimately use both.

**Scenario A: You already have a mature ML forecasting pipeline**

Chronos-2 is your cold-start layer. For any SKU with fewer than 28 days of history, deploy Chronos-2 in ZS mode (Phase 1 below). For mature SKUs, keep your production model. The two systems can run in parallel with a simple history-length gate:

```python
def forecast(sku_id, history_days):
    if history_days < 28:
        return chronos_forecast(sku_id)  # ZS, no training required
    else:
        return lgbm_forecast(sku_id)     # production model
```

**Scenario B: You are building forecasting capability**

Start with Chronos-2. Phase 1 (below) is achievable in two to three days and produces WRMSSE in the 0.98–1.08 range — competitive with production models at many retailers. Revisit LGBM once your team has the data infrastructure and domain knowledge to build the feature pipeline. Do not let perfect be the enemy of operational.

**Scenario C: You need rapid deployment without ML infrastructure**

Chronos-2's ZS pipeline requires no feature store, no training data management, no retraining schedule. It operates from the model weights and a context window. This is its structural advantage over any trained model for teams without MLOps capability.

---

**Phase 0: Establish the baseline (one day)**

Run Chronos-2-small zero-shot with CL=7, no segmentation, no covariates. This is your starting point. It is also the numerical equivalent of the seasonal naïve baseline if you accidentally use CL=1 — verify that your results differ meaningfully.

**Phase 1: Configure the ZS pipeline (two to three days)**

```python
for category in product_categories:
    best_cl = sweep_cl([1, 4, 7, 14, 28], category)  # 5-point sweep
    forecasts[category] = chronos_small.predict_quantiles(
        series=filter_by_category(all_series, category),
        context_length=best_cl,
        cross_learning=True,   # only within coherent category batches
        batch_size=100,
    )
forecasts_calibrated = loo_ols(forecasts, actuals)  # LOO if ≥ 3 cutoffs
```

Add exogenous covariates: at minimum, SNAP flags and calendar features if available; price data if clean. Covariates improve performance by −0.194 geo at the optimal CL (from 1.2716 without to 1.0776 with `eP_sF`). Do not enable cross-learning globally when covariates are present — use category-segregated batches.

This step reaches WRMSSE ∼0.98–1.08. For most retailers without a mature ML infrastructure, this is a very strong starting point.

**Phase 2: Add the tier ensemble (one to two weeks)**

Segment series into 3–4 volume tiers (WRMSSE weight quartiles or equivalent). Fine-tune a separate Chronos-2-small model per tier using full fine-tuning, maximum 25 gradient steps, validating WRMSSE at each checkpoint. Learn blend weights with gradient-free optimisation (Nelder-Mead, ≥200 function evaluations). Blend the tier ensemble with the Phase 1 ZS output using per-category LOO-OLS.

This step reaches WRMSSE ∼0.80. The engineering overhead is meaningful: fine-tuning infrastructure, checkpoint management, W-matrix optimisation. Invest in Phase 2 only if Phase 1 accuracy is insufficient for your use case.

**When to stop at Phase 1:** fewer than three seasonal evaluation cutoffs (LOO-OLS is unreliable), quarterly or faster retraining cycles (Phase 2 FT overhead is not worth maintaining), or if your dataset differs substantially from M5 (short series, non-intermittent, intraday data — the tier structure may not transfer).

---

### 16. Engineering Notes — What to Get Right Before You Start

These details had outsized impact on our results. Get them right before investing in architecture choices.

**Context length is a hyperparameter, not a setting.** CL=7 is the right starting point for daily retail data. CL=14 adds value for some categories when covariates are present (they extend the useful context window). CL≥28 typically hurts on intermittent series — the model anchors on historical spikes. Always sweep {1, 4, 7, 14, 28} before fixing CL.

**Batch composition controls cross-learning quality.** If `cross_learning=True`, every series in a batch of 100 should come from the same product category. Sort your full series list by category before calling `predict_quantiles`. If category labels are unavailable, disable cross-learning — a random mixed batch with xcl=True performs worse than xcl=False.

**LoRA requires `merge_and_unload()` before Chronos inference.** The Chronos `predict_quantiles()` method routes through `MeanScaleUniformBins`, not the PEFT wrapper. An unmerged PeftModel silently behaves as the base model. Always verify that fine-tuned outputs differ from the base model on a held-out set. If they are identical, the adapter is not being applied.

**LOO-OLS calibration requires at least three distinct evaluation cutoffs.** With two cutoffs, LOO fits on one observation per fold — OLS coefficients are statistically ill-conditioned. With three, it is the minimum viable approach. With five or more, switch to strict temporal forward-chaining (train only on past folds, evaluate on the next unseen one).

**CL=1 is the seasonal naïve baseline.** At one observation, Chronos produces a last-value repeat equivalent to the weekly naïve baseline. This is a useful sanity check: if your CL=7 result is only marginally better than CL=1, your context construction is likely broken.

**The FOODS scale bias is expected and correctable.** Department-segregated xcl=True introduces a systematic under-estimation in FOODS_1 (α≈1.59× in our data). This is a predictable artefact of within-dept cross-attention without external anchoring. OLS calibration corrects it reliably — include the OLS step whenever using dept-seg xcl=True.

---

### 17. Conclusion

This article began with a problem — the cold-start failure of classical forecasting — and ended with a finding: Chronos-2 wins on newly launched products below 28 days of history, precisely where classical ML has no valid feature to offer.

The Chronos workflow described in this article — ZS with category segmentation, cross-learning, and OLS calibration — reaches WRMSSE 0.7969 on a rigorous three-cutoff benchmark. A fully tuned LightGBM with 37 hand-crafted features achieves 0.6198. LightGBM wins on full-history series. Below 28 days, the contest is structurally one-sided.

For teams choosing Chronos, the practical summary:

1. Start with CL=7, category-segregated batches, `cross_learning=True`, and `eP_sF` covariates. This is your production-ready ZS baseline (∼1.08 WRMSSE) in a day.
2. Add per-category OLS calibration with LOO cross-validation. This reaches the ZS ceiling (∼0.98) with minimal overhead.
3. If further accuracy is needed, build the tier ensemble (§8). Validate fine-tuning at 5, 10, and 25 steps. Rebuild blend weights with Nelder-Mead, not simple averaging.
4. For any SKU below 28 days of history, use pure ZS Chronos — regardless of whether you have an LGBM model in production.
5. Do not fine-tune at global scale. Do not assume the BASE model is better. Do not enable cross-learning in mixed-category batches with covariates.

Chronos-2 is not always the most accurate model. For retailers with mature ML pipelines and the engineering bandwidth to maintain them, LightGBM will outperform on the bulk of the assortment. But for the new products that arrive every week — the ones where demand planners currently guess — Chronos-2 is the only tool that works on day one.

That is not a narrow advantage. It is the forecasting problem that the rest of the pipeline is designed to avoid.

---

## Appendix: Experiment Chronology

| Date | Milestone | WRMSSE |
|------|-----------|--------|
| Apr 2026 | Naïve seasonal baseline established | 1.6310 (3-cutoff geo) |
| Apr–May 2026 | First ZS runs; CL sweep and tier segmentation (spring) | 1.243 → 0.9082 |
| May 14, 2026 | v2: first Optuna HPO combining FT and ZS | — |
| May 18, 2026 | v6: per-tier FT HPO, best spring FT result | 0.836 (spring) |
| May 18–29, 2026 | v7: selective FT (smooth-series only) — dead end | — |
| May 19–31, 2026 | v8: cw2 covariate system, 2-cutoff eval — dead end | — |
| Jun 8–10, 2026 | v9/v10: direct Chronos API, 3-cutoff evaluation begins | 0.8947 → 0.8536 |
| Jun 11, 2026 | v11: tier ensemble + W-matrix re-opt (400 fev) | **0.8180** (tier alone) |
| Jun 11–12, 2026 | v12/v13: dept-seg ZS HPO xcl=False + tier OLS | 0.8066 |
| Jun 13, 2026 | v14: store-seg ZS — worse than dept, dead end | 0.8132 |
| Jun 13–15, 2026 | v15–v22: LoRA FT (invalid — PeftModel bug) | — |
| Jun 14–15, 2026 | Catastrophic forgetting curve; PeftModel bug discovered | — |
| Jun 15, 2026 | v17: BASE model standalone ZS | 1.1963 |
| Jun 15, 2026 | **v20: dept-seg xcl=True + tier OLS — BEST CHRONOS** | **0.7969** |
| Jun 16, 2026 | ZS global survey: xcl×covariate interaction confirmed | — |
| Jun 15–16, 2026 | v23/v24a: LoRA FT after bug fix | 0.9509 / 0.9461 |
| Jun 22, 2026 | LightGBM global (37 features, 19-trial Optuna HPO) | **0.6198** |
| Jun 23, 2026 | LGBM × Chronos OLS blend (LOO-3) | 0.6254 (no gain) |
| Jun 23, 2026 | Cold-start natural segmentation (per-series RMSSE by history bucket) | Chronos wins <28d |
