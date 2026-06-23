#!/usr/bin/env python3
"""
09_lgbm_chronos_blend.py
=========================
OLS blend of LGBM global (geo=0.6198) + Chronos v20 OLS (geo=0.7969).
Two fundamentally different signal families — tree-based vs foundation model.

Signals:
  A = LGBM global      : from saved test-cutoff checkpoints
  B = Chronos v20 OLS  : tier_v11 × α_v20_dept + chronos_v20_ZS × (1-α_v20_dept)
                         (reconstructed from stored forecast parquets + fixed v20 OLS weights)

Blend formula:
  F = β × A + (1-β) × B

Fit: LOO-3 cross-validation (same as v20's own OLS methodology) — scalar β
     optimised per fold via scipy L-BFGS-B, evaluated with full 12-level WRMSSE.

Run:
    cd /home/nmwamsojo/tsfm-explo
    source .venv/bin/activate
    nohup python experiments/09_lgbm_chronos_blend.py \
        >> /mnt/lab/nmwamsojo/blend_lgbm_chronos.log 2>&1 &
    tail -f /mnt/lab/nmwamsojo/blend_lgbm_chronos.log
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
import scipy.optimize
import scipy.sparse

from tsfm_m5 import M5DataPipeline
from tsfm_m5 import M5Evaluator

# ── Config ────────────────────────────────────────────────────────────────────
import os
DATA_PATH  = os.getenv("M5_DATA_PATH", "/mnt/lab/datasets/M5/jointed_M5.parquet")
LAB_DIR    = os.getenv("M5_LAB_DIR",   "/mnt/lab/nmwamsojo")

DATA_TAG   = "sales_only"
FEAT_BASE  = LAB_DIR + "/lgbm_global/features"
CKPT_DIR   = LAB_DIR + "/lgbm_global/checkpoints"
FCST_BASE  = LAB_DIR + "/prepared_data/sales_only/level_12"
RESULT_OUT = LAB_DIR + "/blend_lgbm_chronos_result.json"
LEVEL      = 12
HORIZON    = 28

CUTOFFS = [
    {"date": "2015-10-04", "label": "Autumn", "tag": "20151004"},
    {"date": "2016-01-03", "label": "Winter", "tag": "20160103"},
    {"date": "2016-04-24", "label": "Spring", "tag": "20160424"},
]

# ── v20 constants (from experimental_log.md) ─────────────────────────────────
# Tier weights applied per-series based on weight quartile
_V11_W = {
    "2016-04-24": np.array([[0.6402,0.1199,0.1200,0.1199],
                             [0.0104,0.9689,0.0104,0.0104],
                             [0.3073,0.3075,0.0776,0.3076],
                             [0.1520,0.1521,0.1521,0.5438]]),
    "2016-01-03": np.array([[0.9314,0.0229,0.0229,0.0229],
                             [0.0592,0.8224,0.0592,0.0592],
                             [0.0805,0.0805,0.7585,0.0805],
                             [0.3317,0.3319,0.3317,0.0048]]),
    "2015-10-04": np.array([[0.6917,0.1028,0.1028,0.1028],
                             [0.2981,0.1051,0.2980,0.2987],
                             [0.2209,0.2214,0.3362,0.2215],
                             [0.0012,0.0012,0.0012,0.9965]]),
}
_TIER_TAGS = [
    "v9d_zs_low_m0_cl7_eP_weight",
    "v9d_ft_med_low_m0_cl1_scope4_steps500_full_lr1e-5_bs32_base_weight",
    "v9d_zs_med_high_m0_cl7_eP_sF_pF_weight",
    "v11d_ft_high_m0_cl14_scope1_steps50_lora_lr1e-5_bs256_eP_sF_weight",
]
_TIER_ORDER = ["Low", "Med-Low", "Med-High", "High"]

# v20 best trial #140 per-dept CL and tag map
_V20_DEPT_CL = {
    "FOODS_1": 28, "FOODS_2": 7, "FOODS_3": 28,
    "HOBBIES_1": 1, "HOBBIES_2": 7, "HOUSEHOLD_1": 28, "HOUSEHOLD_2": 1,
}
# Dept name → tag component
_DEPT_TAG = {
    "FOODS_1": "foods1", "FOODS_2": "foods2", "FOODS_3": "foods3",
    "HOBBIES_1": "hobbies1", "HOBBIES_2": "hobbies2",
    "HOUSEHOLD_1": "household1", "HOUSEHOLD_2": "household2",
}
# v20 per-dept OLS alpha weights (from experimental_log.md)
# alpha = weight on tier; (1-alpha) = weight on Chronos
_V20_DEPT_ALPHA = {
    "FOODS_1":     0.961,
    "FOODS_2":     1.000,
    "FOODS_3":     0.900,
    "HOBBIES_1":   0.740,
    "HOBBIES_2":   1.000,
    "HOUSEHOLD_1": 0.644,
    "HOUSEHOLD_2": 0.853,
}
DEPTS = list(_V20_DEPT_CL.keys())

_LEVEL_COLS = {
    1:[], 2:["state_id"], 3:["store_id"], 4:["cat_id"], 5:["dept_id"],
    6:["state_id","cat_id"], 7:["state_id","dept_id"],
    8:["store_id","cat_id"], 9:["store_id","dept_id"],
    10:["item_id"], 11:["state_id","item_id"], 12:["id"],
}
CAT_COLS = ["dept_id", "cat_id", "store_id", "state_id", "tier_id"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


# ── Data helpers ──────────────────────────────────────────────────────────────
def _read_full(path):
    df = pl.read_parquet(path)
    cats = [pl.col(c).cast(pl.Utf8) for c in df.columns if df[c].dtype == pl.Categorical]
    if cats:
        df = df.with_columns(cats)
    return df.to_pandas()


def _prep(df, feat_cols):
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


def _fcst_path(cutoff_tag, model_tag):
    return os.path.join(FCST_BASE, cutoff_tag, "models", model_tag, "forecasts.parquet")


# ── WRMSSE (identical to v20 / global eval) ───────────────────────────────────
def _build_wrmsse_fn(all_ids, evaluator, actual_wide):
    static_loc = evaluator._static.reindex(all_ids).copy()
    static_loc["id"] = list(all_ids)
    n_items = len(all_ids)
    levels = {}
    for L in range(1, 13):
        cols = _LEVEL_COLS[L]
        info = evaluator.level_info[L]
        sc, wt = info["scales"].astype(np.float64), info["weights"].astype(np.float64)
        if L == 1:
            levels[L] = {"t": "total",    "s": sc, "w": wt, "a": actual_wide.sum(0, keepdims=True)}
        elif L == 12:
            levels[L] = {"t": "identity", "s": sc, "w": wt, "a": actual_wide}
        else:
            can = list(info["index"]); u2i = {u: i for i, u in enumerate(can)}
            gk  = (static_loc[cols[0]].astype(str).values if len(cols) == 1
                   else static_loc[cols].astype(str).agg("_".join, axis=1).values)
            i2u = np.array([u2i.get(k, -1) for k in gk], dtype=np.int32); v = i2u >= 0
            A   = scipy.sparse.csr_matrix((np.ones(v.sum()), (i2u[v], np.where(v)[0])),
                                          shape=(len(can), n_items))
            levels[L] = {"t": "sparse", "A": A, "s": sc, "w": wt, "a": A @ actual_wide}

    def _wrmsse(fcst, _lv=levels):
        s = []
        for L in range(1, 13):
            d = _lv[L]
            f = (fcst.sum(0, keepdims=True) if d["t"] == "total"
                 else fcst if d["t"] == "identity" else d["A"] @ fcst)
            mse = np.mean((d["a"] - f) ** 2, axis=1)
            s.append(float(np.dot(np.sqrt(np.maximum(mse / d["s"], 0.0)), d["w"])))
        return float(np.mean(s))
    return _wrmsse


# ── Per-cutoff signal building ────────────────────────────────────────────────
def _build_cutoff_state(co: dict, pipeline: M5DataPipeline) -> dict:
    """
    Returns dict with:
      lgbm_pred   : (n_items, 28) — LGBM global predictions
      chronos_v20 : (n_items, 28) — Chronos v20 OLS assembled prediction
      wrmsse_fn   : callable
      all_ids     : list[str]
      horizon_dates: list[Timestamp]
    """
    date, label, tag = co["date"], co["label"], co["tag"]
    base = os.path.join(FEAT_BASE, tag)

    with open(os.path.join(base, "meta.json")) as f:
        meta = json.load(f)
    feat_cols = meta["feature_cols"]

    # ── LGBM signal ───────────────────────────────────────────────────────────
    log.info("[%s] Building LGBM signal ...", label)
    bst = lgb.Booster(model_file=os.path.join(CKPT_DIR, f"test_{tag}.txt"))
    test_pd = _read_full(os.path.join(base, "test.parquet"))
    test_pd, _ = _prep(test_pd, feat_cols)
    test_pd["pred"] = np.clip(bst.predict(test_pd[feat_cols]), 0.0, None)
    test_pd["id"] = (test_pd["id"].astype(str)
                     .str.replace("_evaluation", "", regex=False)
                     .str.replace("_validation", "", regex=False))
    test_pd["date"] = pd.to_datetime(test_pd["date"])
    del bst; gc.collect()

    # ── Hierarchy setup ───────────────────────────────────────────────────────
    log.info("[%s] Loading prepared data ...", label)
    hist_df, _, _, static_df, weights_scales = pipeline.get_prepared_data(
        DATA_PATH, date, level=LEVEL, force_reprepare=False)
    all_ids = list(static_df["id"].unique())
    n_items = len(all_ids)
    id_to_idx = {aid: i for i, aid in enumerate(all_ids)}
    horizon_dates = list(pd.date_range(
        pd.Timestamp(date) + pd.Timedelta(days=1), periods=HORIZON))

    # Tier membership (weight quartile)
    _, _, _, _, wts = pipeline.get_prepared_data(
        DATA_PATH, date, level=LEVEL, force_reprepare=False)
    w_lv = wts[wts["level"] == LEVEL].copy()
    w_lv["tier"] = pd.qcut(w_lv["weight"], q=4, labels=_TIER_ORDER)
    tier_ids = {t: set(w_lv[w_lv["tier"] == t]["id"]) for t in _TIER_ORDER}
    item_tier = np.array([
        next((i for i, t in enumerate(_TIER_ORDER) if aid in tier_ids[t]), 0)
        for aid in all_ids], dtype=np.int32)

    # Actuals
    horizon_set = set(d.date() for d in horizon_dates)
    act_df = (pd.read_parquet(DATA_PATH, columns=["id", "date", "sold"])
              .rename(columns={"sold": "sales_quantity"})
              .assign(id=lambda df: df["id"].astype(str)
                      .str.replace("_evaluation", "", regex=False)
                      .str.replace("_validation", "", regex=False)))
    act_df["date"] = pd.to_datetime(act_df["date"])
    act_df = act_df[act_df["date"].dt.date.isin(horizon_set)]
    actual_wide = (act_df.pivot(index="id", columns="date", values="sales_quantity")
                   .reindex(index=all_ids, columns=horizon_dates)
                   .fillna(0.0).values.astype(np.float64))
    del act_df

    # WRMSSE evaluator
    evaluator = M5Evaluator(
        raw_train_df=hist_df, trimmed_train_df=hist_df,
        static_df=static_df, weights_df=weights_scales,
        target_col="sales_quantity", price_col="sell_price")
    del hist_df
    wrmsse_fn = _build_wrmsse_fn(all_ids, evaluator, actual_wide)
    del evaluator, actual_wide; gc.collect()

    # ── Pivot LGBM to (n_items, 28) ───────────────────────────────────────────
    lgbm_pred = (test_pd[["id", "date", "pred"]]
                 .pivot(index="id", columns="date", values="pred")
                 .reindex(index=all_ids, columns=horizon_dates)
                 .fillna(0.0).values.astype(np.float64))
    del test_pd; gc.collect()

    # ── Chronos v20 OLS signal ────────────────────────────────────────────────
    log.info("[%s] Building Chronos v20 OLS signal ...", label)
    W = _V11_W[date]   # (4, 4) tier blend weights

    def _load_wide(model_tag, ids=None):
        path = _fcst_path(tag, model_tag)
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        piv = (df.pivot(index="id", columns="date", values="sales_quantity")
               .reindex(index=ids if ids is not None else all_ids, columns=horizon_dates)
               .fillna(0.0).values.astype(np.float64))
        return piv

    # 1. Tier blend: each series uses its own tier row of W
    tier_stack = np.stack([_load_wide(t) for t in _TIER_TAGS], axis=0)  # (4, n, 28)
    tier_blend  = np.clip(np.einsum("ij,jid->id", W[item_tier], tier_stack), 0.0, None)
    del tier_stack; gc.collect()

    # 2. Per-dept Chronos blend: α × tier + (1-α) × chronos_dept_zs
    chronos_v20 = np.zeros((n_items, HORIZON), dtype=np.float64)
    for dept in DEPTS:
        dept_tag   = f"v20d_zs_{_DEPT_TAG[dept]}_dept_m0_cl{_V20_DEPT_CL[dept]}_xcl100_eP_sF"
        dept_map   = static_df.set_index("id")["dept_id"].to_dict()
        dept_mask  = np.array([dept_map.get(aid, "") == dept for aid in all_ids])
        dept_row_ids = [aid for aid, m in zip(all_ids, dept_mask) if m]
        chronos_dept = _load_wide(dept_tag, ids=dept_row_ids)  # (n_dept, 28)
        tier_dept    = tier_blend[dept_mask]                   # (n_dept, 28)
        alpha        = _V20_DEPT_ALPHA[dept]
        blended      = np.clip(alpha * tier_dept + (1 - alpha) * chronos_dept, 0.0, None)
        chronos_v20[dept_mask] = blended

    log.info("[%s] LGBM standalone WRMSSE = %.4f", label, wrmsse_fn(lgbm_pred))
    log.info("[%s] v20 OLS standalone WRMSSE = %.4f", label, wrmsse_fn(chronos_v20))

    return {
        "date":          date,
        "label":         label,
        "lgbm":          lgbm_pred,
        "chronos_v20":   chronos_v20,
        "wrmsse_fn":     wrmsse_fn,
        "all_ids":       all_ids,
        "horizon_dates": horizon_dates,
    }


# ── LOO-3 OLS blend ───────────────────────────────────────────────────────────
def _loo3_blend(states: list[dict]) -> tuple[list[dict], list[float]]:
    """
    LOO-3 cross-validation of scalar β: F = β × LGBM + (1-β) × Chronos_v20
    For each fold, fit β on the 2 in-sample cutoffs, evaluate on held-out.
    Returns list of per-fold results and fitted β values.
    """
    results = []
    betas   = []
    n = len(states)

    for k, held in enumerate(states):
        in_states = [s for i, s in enumerate(states) if i != k]

        def _geo_obj(params):
            beta = float(np.clip(params[0], 0.0, 1.0))
            scores = []
            for s in in_states:
                blended = np.clip(beta * s["lgbm"] + (1 - beta) * s["chronos_v20"], 0.0, None)
                scores.append(s["wrmsse_fn"](blended))
            return float(np.exp(np.mean(np.log(scores))))

        res = scipy.optimize.minimize(
            _geo_obj, x0=[0.5],
            bounds=[(0.0, 1.0)],
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-10},
        )
        beta = float(np.clip(res.x[0], 0.0, 1.0))
        betas.append(beta)

        # Evaluate on held-out with fitted β
        blended = np.clip(beta * held["lgbm"] + (1 - beta) * held["chronos_v20"], 0.0, None)
        wrmsse  = held["wrmsse_fn"](blended)
        log.info("LOO fold %-8s  β=%.3f  held-out WRMSSE=%.4f  (in-sample geo=%.4f)",
                 held["label"], beta, wrmsse, res.fun)
        results.append({"cutoff": held["date"], "label": held["label"],
                        "wrmsse": wrmsse, "beta": beta})

    return results, betas


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 65)
    log.info("OLS Blend: LGBM Global + Chronos v20")
    log.info("=" * 65)

    pipeline = M5DataPipeline(config={"tag": DATA_TAG})
    states   = []
    for co in CUTOFFS:
        s = _build_cutoff_state(co, pipeline)
        states.append(s)
        gc.collect()

    log.info("")
    log.info("Running LOO-3 OLS ...")
    results, betas = _loo3_blend(states)

    geo = float(np.exp(np.mean(np.log([r["wrmsse"] for r in results]))))

    log.info("")
    log.info("=" * 65)
    log.info("RESULTS — LOO-3 OLS Blend")
    log.info("=" * 65)
    for r in results:
        log.info("  %-8s  WRMSSE=%.4f  β(LGBM)=%.3f", r["label"], r["wrmsse"], r["beta"])
    log.info("  geo-mean  = %.4f", geo)
    log.info("")
    log.info("Standalones:")
    log.info("  LGBM global   : 0.6198")
    log.info("  Chronos v20   : 0.7969")
    log.info("  Blend LOO-3   : %.4f  (Δ vs LGBM: %+.4f)", geo, geo - 0.6198)
    log.info("=" * 65)

    out = {
        "results":         results,
        "geo":             geo,
        "betas":           betas,
        "lgbm_standalone": 0.6198,
        "chronos_v20":     0.7969,
        "note":            "LOO-3 scalar beta, F=beta*LGBM+(1-beta)*Chronos_v20_OLS",
    }
    with open(RESULT_OUT, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Saved → %s", RESULT_OUT)


if __name__ == "__main__":
    main()
