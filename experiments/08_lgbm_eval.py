#!/usr/bin/env python3
"""
08_lgbm_eval.py
=======================
Evaluate best LightGBM global HPO params on the 3 test cutoffs.

Uses the exact same WRMSSE computation as v20 Chronos HPO (all 12 hierarchy
levels, M5Evaluator, jointed_M5.parquet actuals) for comparable results.

For each test cutoff:
  1. Train a fresh LightGBM on train.parquet (full historical data to that cutoff)
  2. Predict on test.parquet (30,490 series × 28 days, no actuals column)
  3. Load actuals + hierarchy from prepared_data cache
  4. Compute WRMSSE across all 12 levels (same formula as v20)

Run:
    cd /home/nmwamsojo/tsfm-explo
    source .venv/bin/activate
    nohup python experiments/08_lgbm_eval.py \
        >> /mnt/lab/nmwamsojo/lgbm_global_eval.log 2>&1 &
    tail -f /mnt/lab/nmwamsojo/lgbm_global_eval.log
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse

from tsfm_m5 import M5DataPipeline
from tsfm_m5 import M5Evaluator

# ── Paths ──────────────────────────────────────────────────────────────────────
import os
DATA_PATH  = os.getenv("M5_DATA_PATH", "/mnt/lab/datasets/M5/jointed_M5.parquet")
LAB_DIR    = os.getenv("M5_LAB_DIR",   "/mnt/lab/nmwamsojo")

DATA_TAG   = "sales_only"
FEAT_BASE  = LAB_DIR + "/lgbm_global/features"
CKPT_DIR   = LAB_DIR + "/lgbm_global/checkpoints"
RESULT_OUT = LAB_DIR + "/lgbm_global_eval_result.json"
OPTUNA_DB  = "sqlite:////" + LAB_DIR + "/optuna_lgbm_global.db"
STUDY_NAME = "lgbm_global_m5_v1"
LEVEL      = 12
HORIZON    = 28

TEST_CUTOFFS = [
    {"date": "2015-10-04", "label": "Autumn"},
    {"date": "2016-01-03", "label": "Winter"},
    {"date": "2016-04-24", "label": "Spring"},
]

NUM_THREADS = 20
EARLY_STOP  = 50
MAX_ROUNDS  = 2000
CAT_COLS    = ["dept_id", "cat_id", "store_id", "state_id", "tier_id"]
DS_PARAMS   = {"max_bin": 255, "feature_pre_filter": False}

_LEVEL_COLS = {
    1: [], 2: ["state_id"], 3: ["store_id"], 4: ["cat_id"], 5: ["dept_id"],
    6: ["state_id", "cat_id"], 7: ["state_id", "dept_id"],
    8: ["store_id", "cat_id"], 9: ["store_id", "dept_id"],
    10: ["item_id"], 11: ["state_id", "item_id"], 12: ["id"],
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)
os.makedirs(CKPT_DIR, exist_ok=True)


# ── Best params from Optuna ────────────────────────────────────────────────────
def _best_params() -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=STUDY_NAME, storage=OPTUNA_DB)
    p = study.best_params
    log.info("Best trial #%d  val RMSE=%.4f", study.best_trial.number, study.best_value)
    log.info("  %s", p)
    return {
        "device":                  "cpu",
        "num_threads":             NUM_THREADS,
        "verbosity":              -1,
        "objective":               "tweedie",
        "metric":                 ["tweedie", "rmse"],
        "tweedie_variance_power":  p["tweedie_variance_power"],
        "num_leaves":              p["num_leaves"],
        "max_depth":              -1,
        "min_data_in_leaf":        p["min_data_in_leaf"],
        "max_bin":                 255,
        "feature_pre_filter":      False,
        "lambda_l1":               p["lambda_l1"],
        "lambda_l2":               p["lambda_l2"],
        "learning_rate":           p["learning_rate"],
        "feature_fraction":        p["feature_fraction"],
        "bagging_fraction":        p["bagging_fraction"],
        "bagging_freq":            1,
    }


# ── Data helpers ──────────────────────────────────────────────────────────────
def _read(path: str) -> pd.DataFrame:
    df = pl.read_parquet(path)
    cats = [pl.col(c).cast(pl.Utf8) for c in df.columns if df[c].dtype == pl.Categorical]
    if cats:
        df = df.with_columns(cats)
    return df.to_pandas()


def _prep(df: pd.DataFrame, feat_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    cat_present = []
    for col in feat_cols:
        if df[col].dtype == object:
            n = df[col].nunique()
            if n <= 255:
                df[col] = df[col].astype("category")
                cat_present.append(col)
            else:
                codes = {v: i for i, v in enumerate(sorted(df[col].dropna().unique()))}
                df[col] = df[col].map(codes).fillna(-1).astype(np.int32)
        elif col in CAT_COLS:
            cat_present.append(col)
    return df, cat_present


# ── WRMSSE (same as v20 Chronos HPO) ──────────────────────────────────────────
def _build_wrmsse_fn(cutoff_date: str, all_ids: list[str], evaluator: M5Evaluator,
                     actual_wide: np.ndarray) -> callable:
    """
    Build a WRMSSE closure identical to run_hpo_dept_v20._build_eval_states().
    actual_wide: (n_items, 28) float64, rows ordered as all_ids.
    Returns a function  wrmsse(fcst: np.ndarray) -> float.
    """
    static_loc = evaluator._static.reindex(all_ids).copy()
    static_loc["id"] = list(all_ids)
    n_items = len(all_ids)

    levels = {}
    for L in range(1, 13):
        cols = _LEVEL_COLS[L]
        info = evaluator.level_info[L]
        sc   = info["scales"].astype(np.float64)
        wt   = info["weights"].astype(np.float64)
        if L == 1:
            levels[L] = {"t": "total",    "s": sc, "w": wt, "a": actual_wide.sum(0, keepdims=True)}
        elif L == 12:
            levels[L] = {"t": "identity", "s": sc, "w": wt, "a": actual_wide}
        else:
            can  = list(info["index"])
            u2i  = {u: i for i, u in enumerate(can)}
            gk   = (static_loc[cols[0]].astype(str).values if len(cols) == 1
                    else static_loc[cols].astype(str).agg("_".join, axis=1).values)
            i2u  = np.array([u2i.get(k, -1) for k in gk], dtype=np.int32)
            v    = i2u >= 0
            A    = scipy.sparse.csr_matrix(
                (np.ones(v.sum()), (i2u[v], np.where(v)[0])),
                shape=(len(can), n_items))
            levels[L] = {"t": "sparse", "A": A, "s": sc, "w": wt, "a": A @ actual_wide}

    def _wrmsse(fcst, _lv=levels):
        s = []
        for L in range(1, 13):
            d = _lv[L]
            f = (fcst.sum(0, keepdims=True) if d["t"] == "total"
                 else fcst if d["t"] == "identity"
                 else d["A"] @ fcst)
            mse = np.mean((d["a"] - f) ** 2, axis=1)
            s.append(float(np.dot(np.sqrt(np.maximum(mse / d["s"], 0.0)), d["w"])))
        return float(np.mean(s))

    return _wrmsse


# ── Per-cutoff evaluation ──────────────────────────────────────────────────────
def _run_cutoff(cutoff: dict, params: dict, pipeline: M5DataPipeline) -> dict:
    date  = cutoff["date"]
    label = cutoff["label"]
    tag   = date.replace("-", "")
    base  = os.path.join(FEAT_BASE, tag)

    with open(os.path.join(base, "meta.json")) as f:
        meta = json.load(f)
    feat_cols = meta["feature_cols"]

    # ── 1. Load training data from feature cache ───────────────────────────────
    log.info("[%s] Loading train (%s rows) + val ...", label, f"{meta['n_train']:,}")
    t0 = time.time()
    train_pd = _read(os.path.join(base, "train.parquet"))
    val_pd   = _read(os.path.join(base, "val.parquet"))

    train_pd, cat_present = _prep(train_pd, feat_cols)
    val_pd,   _           = _prep(val_pd,   feat_cols)

    train_ds = lgb.Dataset(train_pd[feat_cols], label=train_pd["sales"].values,
                           categorical_feature=cat_present, free_raw_data=False,
                           params=DS_PARAMS)
    val_ds   = lgb.Dataset(val_pd[feat_cols],   label=val_pd["sales"].values,
                           categorical_feature=cat_present, reference=train_ds,
                           free_raw_data=False, params=DS_PARAMS)
    del train_pd, val_pd
    gc.collect()

    # ── 2. Train ───────────────────────────────────────────────────────────────
    log.info("[%s] Training ...", label)
    bst = lgb.train(
        params, train_ds, num_boost_round=MAX_ROUNDS,
        valid_sets=[val_ds],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                   lgb.log_evaluation(period=50)],
    )
    best_round = bst.best_iteration
    val_rmse   = bst.best_score["valid_0"]["rmse"]
    elapsed    = time.time() - t0
    log.info("[%s] Best round=%d  val RMSE=%.4f  (%.0fs)", label, best_round, val_rmse, elapsed)

    ckpt = os.path.join(CKPT_DIR, f"test_{tag}.txt")
    bst.save_model(ckpt)
    log.info("[%s] Booster saved → %s", label, ckpt)

    del train_ds, val_ds
    gc.collect()

    # ── 3. Predict test window ─────────────────────────────────────────────────
    log.info("[%s] Predicting test window ...", label)
    test_pd = _read(os.path.join(base, "test.parquet"))
    test_pd, _ = _prep(test_pd, feat_cols)
    preds = bst.predict(test_pd[feat_cols])
    del bst
    gc.collect()

    test_pd["pred"] = np.clip(preds, 0.0, None)

    # ── 4. Load actuals + hierarchy from prepared_data cache ───────────────────
    log.info("[%s] Loading prepared data cache ...", label)
    hist_df, hist_trimmed, future_df, static_df, weights_scales = pipeline.get_prepared_data(
        DATA_PATH, date, level=LEVEL, force_reprepare=False)

    all_ids = list(static_df["id"].unique())
    n_items = len(all_ids)
    log.info("[%s] %d series", label, n_items)

    # Horizon dates: cutoff+1 … cutoff+28
    horizon_dates = list(pd.date_range(
        pd.Timestamp(date) + pd.Timedelta(days=1), periods=HORIZON))

    # Actual sales from jointed_M5.parquet for horizon dates
    horizon_set = set(d.date() for d in horizon_dates)
    act_df = (pd.read_parquet(DATA_PATH, columns=["id", "date", "sold"])
              .rename(columns={"sold": "sales_quantity"})
              .assign(id=lambda df: df["id"].astype(str)
                      .str.replace("_evaluation", "", regex=False)
                      .str.replace("_validation", "", regex=False)))
    act_df["date"] = pd.to_datetime(act_df["date"])
    act_df = act_df[act_df["date"].dt.date.isin(horizon_set)]
    actual_wide = (act_df
                   .pivot(index="id", columns="date", values="sales_quantity")
                   .reindex(index=all_ids, columns=horizon_dates)
                   .fillna(0.0).values.astype(np.float64))
    del act_df, hist_trimmed, future_df

    # ── 5. Build evaluator ─────────────────────────────────────────────────────
    evaluator = M5Evaluator(
        raw_train_df=hist_df, trimmed_train_df=hist_df,
        static_df=static_df, weights_df=weights_scales,
        target_col="sales_quantity", price_col="sell_price")
    del hist_df

    wrmsse_fn = _build_wrmsse_fn(date, all_ids, evaluator, actual_wide)
    del evaluator, actual_wide
    gc.collect()

    # ── 6. Pivot predictions to (n_items, 28) ─────────────────────────────────
    test_pd["date"] = pd.to_datetime(test_pd["date"])
    test_pd["id"] = (test_pd["id"].astype(str)
                     .str.replace("_evaluation", "", regex=False)
                     .str.replace("_validation", "", regex=False))
    pred_wide = (test_pd[["id", "date", "pred"]]
                 .pivot(index="id", columns="date", values="pred")
                 .reindex(index=all_ids, columns=horizon_dates)
                 .fillna(0.0).values.astype(np.float64))
    del test_pd
    gc.collect()

    # ── 7. WRMSSE ─────────────────────────────────────────────────────────────
    log.info("[%s] Computing WRMSSE ...", label)
    wrmsse = wrmsse_fn(pred_wide)
    log.info("[%s] WRMSSE = %.4f", label, wrmsse)

    return {
        "cutoff":     date,
        "label":      label,
        "wrmsse":     wrmsse,
        "best_round": best_round,
        "val_rmse":   val_rmse,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    log.info("=" * 65)
    log.info("LightGBM Global — Test Cutoff Evaluation (v20-comparable WRMSSE)")
    log.info("=" * 65)

    params = _best_params()

    pipeline = M5DataPipeline(config={"tag": DATA_TAG})

    results = []
    for cutoff in TEST_CUTOFFS:
        res = _run_cutoff(cutoff, params, pipeline)
        results.append(res)
        gc.collect()

    wrmsse_vals = [r["wrmsse"] for r in results]
    geo = float(np.exp(np.mean(np.log(wrmsse_vals))))

    log.info("")
    log.info("=" * 65)
    log.info("RESULTS")
    log.info("=" * 65)
    for r in results:
        log.info("  %-8s  WRMSSE=%.4f  (round=%d  val_rmse=%.4f)",
                 r["label"], r["wrmsse"], r["best_round"], r["val_rmse"])
    log.info("  geo-mean  WRMSSE = %.4f", geo)
    log.info("")
    log.info("Baselines:")
    log.info("  Seasonal naive      : geo = 1.6310")
    log.info("  v20 OLS (Chronos-2) : geo = 0.7969")
    log.info("=" * 65)

    out = {
        "results": results,
        "geo":     geo,
        "params":  params,
        "note":    "WRMSSE computed over all 12 M5 hierarchy levels, same as v20",
    }
    with open(RESULT_OUT, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Saved → %s", RESULT_OUT)


if __name__ == "__main__":
    main()
