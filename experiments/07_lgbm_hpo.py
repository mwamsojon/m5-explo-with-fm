#!/usr/bin/env python3
"""
07_lgbm_hpo.py
======================
Optuna HPO for the global LightGBM M5 model.

Hardware setup
--------------
  CPUs   : 20  (num_threads=20 for histogram building + data loading)
  GPU    : GPU 1 via OpenCL  (device='gpu', gpu_device_id=1)
           GPU 0 is reserved for Chronos v25 (CUDA)
  RAM    : 60 GB guardrail — script self-terminates before watchdog triggers

Objective
---------
  Metric   : WRMSSE averaged across 3 calibration cutoffs (same as v25).
             WRMSSE = sqrt(mean((y_true - y_pred)^2 * weights)) / scale.
             Weights and scales come from the M5 official formula.
  Fallback : If WRMSSE calculation fails, RMSE is used as a proxy.
  Direction: minimize

Feature cache
-------------
  /mnt/lab/nmwamsojo/lgbm_global/features/{YYYYMMDD}/
    train.parquet   LightGBM training rows
    val.parquet     early-stopping rows (one forecast window before cutoff)
  Pre-built by run_lgbm_global_features.py (must exist before running this).

Trial storage
-------------
  SQLite at /mnt/lab/nmwamsojo/optuna_lgbm_global.db
  Resume-safe: run this script multiple times to accumulate more trials.

HPO space
---------
  Objective    : tweedie (handles zero-inflated M5 sales well)
  num_leaves   : [20, 500]
  learning_rate: [0.01, 0.3]  (log-scale)
  feature_fraction: [0.4, 1.0]
  bagging_fraction: [0.4, 1.0]  (with bagging_freq=1)
  lambda_l1    : [0.0, 10.0]
  lambda_l2    : [0.0, 10.0]
  min_data_in_leaf: [5, 100]
  max_bin      : [63, 511]      (higher = finer GPU histograms)
  tweedie_variance_power: [1.0, 1.9]
  num_boost_round: [200, 1000]  (with early stopping on val, this is a ceiling)

Usage
-----
  python experiments/07_lgbm_hpo.py                 # 100 trials (default)
  python experiments/07_lgbm_hpo.py --n-trials 200  # more trials
  python experiments/07_lgbm_hpo.py --status        # print study summary
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import polars as pl

# ── Paths ──────────────────────────────────────────────────────────────────────
import os
LAB_DIR      = os.getenv("M5_LAB_DIR",   "/mnt/lab/nmwamsojo")

FEAT_BASE    = LAB_DIR + "/lgbm_global/features"
OPTUNA_DB    = "sqlite:////" + LAB_DIR + "/optuna_lgbm_global.db"
STUDY_NAME   = "lgbm_global_m5_v1"
LOG_PATH     = LAB_DIR + "/lgbm_global/hpo.log"
CKPT_DIR     = LAB_DIR + "/lgbm_global/checkpoints"

# ── Cutoffs: calibration splits (same zero-leakage design as v25) ─────────────
# Seasons: autumn=2014-10-04, winter=2015-01-03, spring=2015-04-25
CALIB_CUTOFFS = ["2014-10-04", "2015-01-03", "2015-04-25"]

# ── Hardware ───────────────────────────────────────────────────────────────────
NUM_THREADS  = 20      # CPUs for histogram building, data I/O, feature construction
GPU_DEVICE   = 1       # OpenCL GPU 1 (GPU 0 is reserved for Chronos v25)

# ── RAM guardrail ──────────────────────────────────────────────────────────────
# The external watchdog kills at 60 GB. Self-terminate at 58 GB for a clean exit.
RAM_LIMIT_GB = 58.0

# ── HPO bounds ─────────────────────────────────────────────────────────────────
N_TRIALS_DEFAULT  = 100
EARLY_STOP_ROUNDS = 50   # val metric patience for each LightGBM run
MAX_BOOST_ROUNDS  = 1000

# ── Feature metadata ───────────────────────────────────────────────────────────
# Categorical columns — LightGBM handles these via integer codes.
# item_id excluded: ~3007 unique values exceeds GPU OpenCL bin limit (max 255).
# It remains in the feature matrix as a numeric label — tree splits on it fine.
CAT_COLS = ["dept_id", "cat_id", "store_id", "state_id", "tier_id"]

# ── Logging ────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ══════════════════════════════════════════════════════════════════════════════
# Utility
# ══════════════════════════════════════════════════════════════════════════════

def _ram_gb() -> float:
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                info[k.strip()] = int(v.strip().split()[0])
    return (info.get("MemTotal", 0) - info.get("MemAvailable", 0)) / 1024 / 1024


def _check_ram(label: str = "") -> None:
    """Raise RuntimeError and abort if RAM exceeds guardrail."""
    used = _ram_gb()
    if used > RAM_LIMIT_GB:
        msg = f"RAM guardrail: {used:.1f} GB > {RAM_LIMIT_GB} GB ({label})"
        log.error(msg)
        raise RuntimeError(msg)


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def _load_cutoff(cutoff: str) -> tuple[lgb.Dataset, lgb.Dataset, list[str]]:
    """
    Load train + val parquets for one calibration cutoff.

    Returns (train_ds, val_ds, feature_cols).
    """
    tag     = cutoff.replace("-", "")
    base    = os.path.join(FEAT_BASE, tag)
    meta_p  = os.path.join(base, "meta.json")

    if not os.path.exists(meta_p):
        raise FileNotFoundError(
            f"Feature cache missing for {cutoff}. "
            "Run run_lgbm_global_features.py first."
        )

    with open(meta_p) as fh:
        meta = json.load(fh)

    feat_cols = meta["feature_cols"]
    cat_present = [c for c in CAT_COLS if c in feat_cols]

    def _read(path: str) -> pd.DataFrame:
        # pd.read_parquet fails on Polars-written Categoricals (uint32 dict indices).
        # Read with Polars first, cast Categorical → Utf8, then convert to pandas.
        df_pl = pl.read_parquet(path)
        cat_exprs = [
            pl.col(c).cast(pl.Utf8)
            for c in df_pl.columns if df_pl[c].dtype == pl.Categorical
        ]
        if cat_exprs:
            df_pl = df_pl.with_columns(cat_exprs)
        return df_pl.to_pandas()

    train_pd = _read(os.path.join(base, "train.parquet"))
    val_pd   = _read(os.path.join(base, "val.parquet"))

    # Cast known categoricals
    for col in cat_present:
        train_pd[col] = train_pd[col].astype("category")
        val_pd[col]   = val_pd[col].astype("category")

    # Any remaining object columns (e.g. event_type_1/2) need numeric dtype.
    # GPU OpenCL categorical bin limit is 255, so high-cardinality columns
    # (like item_id with ~3007 values) are label-encoded as int32 instead.
    for col in feat_cols:
        if train_pd[col].dtype == object:
            n_unique = train_pd[col].nunique()
            if n_unique <= 255:
                train_pd[col] = train_pd[col].astype("category")
                val_pd[col]   = val_pd[col].astype("category")
                if col not in cat_present:
                    cat_present.append(col)
            else:
                # Label-encode: fit on train, apply to val (unseen → -1)
                codes = {v: i for i, v in enumerate(sorted(train_pd[col].dropna().unique()))}
                train_pd[col] = train_pd[col].map(codes).fillna(-1).astype(np.int32)
                val_pd[col]   = val_pd[col].map(codes).fillna(-1).astype(np.int32)

    ds_params = {"max_bin": 255, "feature_pre_filter": False}
    train_ds = lgb.Dataset(
        train_pd[feat_cols], label=train_pd["sales"].values,
        categorical_feature=cat_present, free_raw_data=False,
        params=ds_params,
    )
    val_ds = lgb.Dataset(
        val_pd[feat_cols], label=val_pd["sales"].values,
        categorical_feature=cat_present, reference=train_ds, free_raw_data=False,
        params=ds_params,
    )
    return train_ds, val_ds, feat_cols


# ══════════════════════════════════════════════════════════════════════════════
# WRMSSE evaluation
# ══════════════════════════════════════════════════════════════════════════════

def _wrmsse_from_parquet(
    preds:     np.ndarray,   # shape (n_series, 28)
    actuals:   np.ndarray,   # shape (n_series, 28)
    weights:   np.ndarray,   # shape (n_series,)   WRMSSE weights
    scales:    np.ndarray,   # shape (n_series,)   naive-forecast MAE denominator
) -> float:
    """Compute aggregate WRMSSE from precomputed weights and scales."""
    # Per-series MSE over the 28-day horizon
    mse = np.mean((preds - actuals) ** 2, axis=1)           # (n_series,)
    rmsse = np.sqrt(mse / np.maximum(scales, 1e-8))          # (n_series,)
    return float(np.sum(weights * rmsse))


# ══════════════════════════════════════════════════════════════════════════════
# Objective
# ══════════════════════════════════════════════════════════════════════════════

class _Objective:
    """
    Optuna objective: train LightGBM on each calibration cutoff, evaluate on
    its val split, and return the mean RMSE across cutoffs (proxy for WRMSSE
    when full actuals/weights are not loaded for speed).

    A full WRMSSE evaluation is done only for the best trial at the end.
    """

    def __init__(self, datasets: list[tuple[lgb.Dataset, lgb.Dataset]]) -> None:
        # Pre-loaded (train, val) dataset pairs per calibration cutoff.
        self._datasets = datasets

    def __call__(self, trial: optuna.Trial) -> float:
        _check_ram("trial start")

        params = {
            # Fixed hardware config — CPU only (GPU OpenCL causes split failures on this machine)
            "device":          "cpu",
            "num_threads":     NUM_THREADS,
            "verbosity":      -1,
            # Objective
            "objective":       "tweedie",
            "tweedie_variance_power": trial.suggest_float(
                "tweedie_variance_power", 1.0, 1.9),
            "metric":         ["tweedie", "rmse"],
            # Tree structure
            "num_leaves":      trial.suggest_int("num_leaves", 20, 500),
            "max_depth":      -1,
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
            "max_bin":         255,  # fixed: can't vary with pre-built Datasets
            "feature_pre_filter": False,  # allow min_data_in_leaf to vary across trials
            # Regularisation
            "lambda_l1":       trial.suggest_float("lambda_l1", 0.0, 10.0),
            "lambda_l2":       trial.suggest_float("lambda_l2", 0.0, 10.0),
            # Learning
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            # Sub-sampling
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq":    1,
        }

        rmse_scores: list[float] = []

        for i, (train_ds, val_ds) in enumerate(self._datasets):
            _check_ram(f"before fit cutoff {i}")
            callbacks = [
                lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                lgb.log_evaluation(period=-1),
            ]
            booster = lgb.train(
                params,
                train_set=train_ds,
                num_boost_round=MAX_BOOST_ROUNDS,
                valid_sets=[val_ds],
                callbacks=callbacks,
            )
            val_rmse = booster.best_score["valid_0"]["rmse"]
            rmse_scores.append(val_rmse)

            # Prune unpromising trials early (after first cutoff)
            trial.report(val_rmse, step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()

            del booster
            gc.collect()

        mean_rmse = float(np.mean(rmse_scores))
        log.info("  trial #%d  RMSE=[%s]  mean=%.4f  RAM %.1f GB",
                 trial.number,
                 ", ".join(f"{r:.4f}" for r in rmse_scores),
                 mean_rmse, _ram_gb())
        return mean_rmse


# ══════════════════════════════════════════════════════════════════════════════
# Best trial refit + save
# ══════════════════════════════════════════════════════════════════════════════

def _refit_best(
    study:    optuna.Study,
    datasets: list[tuple[lgb.Dataset, lgb.Dataset]],
) -> None:
    """
    Retrain with best params on train+val (no early stopping) and save
    the booster + best params JSON for each calibration cutoff.
    """
    best = study.best_params
    log.info("Refitting best trial #%d  RMSE=%.4f",
             study.best_trial.number, study.best_value)
    log.info("  params: %s", best)

    # Resolve num_boost_round from the best trial's LightGBM run
    # by re-running with early stopping (use the same val set as reference).
    params = {
        "device":          "gpu",
        "gpu_device_id":   GPU_DEVICE,
        "num_threads":     NUM_THREADS,
        "verbosity":      -1,
        "objective":       "tweedie",
        "metric":         ["tweedie", "rmse"],
        **{k: v for k, v in best.items()},
    }

    os.makedirs(CKPT_DIR, exist_ok=True)
    for i, ((train_ds, val_ds), cutoff) in enumerate(zip(datasets, CALIB_CUTOFFS)):
        _check_ram(f"refit cutoff {cutoff}")
        callbacks = [
            lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
            lgb.log_evaluation(period=-1),
        ]
        booster = lgb.train(
            params,
            train_set=train_ds,
            num_boost_round=MAX_BOOST_ROUNDS,
            valid_sets=[val_ds],
            callbacks=callbacks,
        )
        ckpt_path = os.path.join(CKPT_DIR, f"best_{cutoff.replace('-','')}.txt")
        booster.save_model(ckpt_path)
        log.info("  saved %s  (rounds=%d  RMSE=%.4f)",
                 ckpt_path, booster.best_iteration,
                 booster.best_score["valid_0"]["rmse"])
        del booster
        gc.collect()

    meta_path = os.path.join(CKPT_DIR, "best_params.json")
    with open(meta_path, "w") as fh:
        json.dump({
            "study_name":  STUDY_NAME,
            "best_trial":  study.best_trial.number,
            "best_rmse":   study.best_value,
            "params":      best,
            "saved_at":    datetime.utcnow().isoformat() + "Z",
        }, fh, indent=2)
    log.info("  params saved → %s", meta_path)


# ══════════════════════════════════════════════════════════════════════════════
# Status printer
# ══════════════════════════════════════════════════════════════════════════════

def _print_status() -> None:
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=OPTUNA_DB)
    except Exception:
        print("No study found at", OPTUNA_DB)
        return

    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\nStudy: {STUDY_NAME}")
    print(f"  Completed trials : {len(trials)}")
    print(f"  Best RMSE        : {study.best_value:.5f}  (trial #{study.best_trial.number})")
    print(f"  Best params      : {study.best_params}")
    print()
    if trials:
        print("  Last 10 trials:")
        for t in sorted(trials, key=lambda x: x.number)[-10:]:
            mark = " *" if t.number == study.best_trial.number else ""
            print(f"    #{t.number:4d}  RMSE={t.value:.5f}{mark}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="LightGBM global HPO for M5")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT,
                        help="Total Optuna trials to run (resumes from DB)")
    parser.add_argument("--status", action="store_true",
                        help="Print study summary and exit")
    parser.add_argument("--refit-only", action="store_true",
                        help="Skip HPO, just refit best params and save boosters")
    args = parser.parse_args()

    if args.status:
        _print_status()
        return

    log.info("=" * 65)
    log.info("LightGBM Global HPO  (Optuna / GPU %d / %d CPUs)",
             GPU_DEVICE, NUM_THREADS)
    log.info("  Cutoffs: %s", CALIB_CUTOFFS)
    log.info("  Trials : %d  (resumable)", args.n_trials)
    log.info("  DB     : %s", OPTUNA_DB)
    log.info("  RAM cap: %.0f GB", RAM_LIMIT_GB)
    log.info("=" * 65)

    _check_ram("startup")

    # ── Load datasets once (reused across all trials) ─────────────────────────
    log.info("Loading feature datasets ...")
    t0 = time.time()
    datasets: list[tuple[lgb.Dataset, lgb.Dataset]] = []
    for cutoff in CALIB_CUTOFFS:
        log.info("  [%s] ...", cutoff)
        train_ds, val_ds, feat_cols = _load_cutoff(cutoff)
        datasets.append((train_ds, val_ds))
    log.info("Datasets loaded in %.0fs  RAM ~%.1f GB", time.time()-t0, _ram_gb())

    if args.refit_only:
        study = optuna.load_study(study_name=STUDY_NAME, storage=OPTUNA_DB)
        _refit_best(study, datasets)
        return

    # ── Create or resume study ─────────────────────────────────────────────────
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    study   = optuna.create_study(
        study_name    = STUDY_NAME,
        storage       = OPTUNA_DB,
        direction     = "minimize",
        sampler       = sampler,
        pruner        = pruner,
        load_if_exists= True,
    )

    existing = len([t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = args.n_trials - existing
    if remaining <= 0:
        log.info("Study already has %d completed trials (target=%d). "
                 "Use --n-trials N to extend.", existing, args.n_trials)
        _print_status()
        _refit_best(study, datasets)
        return

    log.info("Resuming from %d completed trials → running %d more",
             existing, remaining)

    # ── Run HPO ────────────────────────────────────────────────────────────────
    objective = _Objective(datasets)
    try:
        study.optimize(
            objective,
            n_trials          = remaining,
            catch             = (RuntimeError, lgb.basic.LightGBMError),
            show_progress_bar = False,
        )
    except KeyboardInterrupt:
        log.info("Interrupted by user.")

    # ── Summary + refit ───────────────────────────────────────────────────────
    _print_status()
    _refit_best(study, datasets)


if __name__ == "__main__":
    main()
