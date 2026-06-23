"""
Gap 2 — Standalone Chronos-2 ZS baseline (no tier ensemble)
=============================================================
Assemble v13 per-dept ZS forecasts (already cached) into a full 30,490-series
forecast matrix and compute WRMSSE directly — WITHOUT any tier ensemble blend.

This answers: what does Chronos-2 alone achieve on M5?

Three benchmarks reported side-by-side:
  A) Pure tier ensemble (v11 W-matrices, no Chronos-2)     → geo ≈ 0.8180
  B) Chronos-2 standalone (this script)                    → geo = ?
  C) OLS blend of tier + dept (v13 best result)            → geo = 0.8066

Segmentation: v13 best CLs {F1:28, F2:7, F3:28, H1:1, H2:7, HH1:28, HH2:7}

Run:
    cd /home/nmwamsojo/tsfm-explo
    source .venv/bin/activate
    python experiments/12_chronos_zs_standalone.py | tee /mnt/lab/nmwamsojo/gap2_standalone.log
"""

import gc, json, os, sys
import numpy as np
import pandas as pd
import scipy.sparse

import os
DATA_PATH  = os.getenv("M5_DATA_PATH", "/mnt/lab/datasets/M5/jointed_M5.parquet")
LAB_DIR    = os.getenv("M5_LAB_DIR",   "/mnt/lab/nmwamsojo")
DATA_TAG   = "sales_only"
CUTOFFS    = ["2016-04-24", "2016-01-03", "2015-10-04"]
CUTOFF_LABELS = {"2016-04-24": "spring", "2016-01-03": "winter", "2015-10-04": "autumn"}
LEVEL      = 12
HORIZON    = 28
FCST_BASE  = LAB_DIR + "/prepared_data"

DEPT_IDS   = ["FOODS_1","FOODS_2","FOODS_3","HOBBIES_1","HOBBIES_2","HOUSEHOLD_1","HOUSEHOLD_2"]
BEST_CLS   = {
    "FOODS_1": 28, "FOODS_2": 7,  "FOODS_3": 28,
    "HOBBIES_1": 1, "HOBBIES_2": 7,
    "HOUSEHOLD_1": 28, "HOUSEHOLD_2": 7,
}

_TIER_ORDER = ["Low", "Med-Low", "Med-High", "High"]
_V11_W = {
    "2016-04-24": np.array([
        [0.6402, 0.1199, 0.1200, 0.1199],
        [0.0104, 0.9689, 0.0104, 0.0104],
        [0.3073, 0.3075, 0.0776, 0.3076],
        [0.1520, 0.1521, 0.1521, 0.5438],
    ]),
    "2016-01-03": np.array([
        [0.9314, 0.0229, 0.0229, 0.0229],
        [0.0592, 0.8224, 0.0592, 0.0592],
        [0.0805, 0.0805, 0.7585, 0.0805],
        [0.3317, 0.3319, 0.3317, 0.0048],
    ]),
    "2015-10-04": np.array([
        [0.6917, 0.1028, 0.1028, 0.1028],
        [0.2981, 0.1051, 0.2980, 0.2987],
        [0.2209, 0.2214, 0.3362, 0.2215],
        [0.0012, 0.0012, 0.0012, 0.9965],
    ]),
}
_TIER_TAGS = {
    "Low":      "v9d_zs_low_m0_cl7_eP_weight",
    "Med-Low":  "v9d_ft_med_low_m0_cl1_scope4_steps500_full_lr1e-5_bs32_base_weight",
    "Med-High": "v9d_zs_med_high_m0_cl7_eP_sF_pF_weight",
    "High":     "v11d_ft_high_m0_cl14_scope1_steps50_lora_lr1e-5_bs256_eP_sF_weight",
}
_LEVEL_COLS = {
    1:[], 2:["state_id"], 3:["store_id"], 4:["cat_id"], 5:["dept_id"],
    6:["state_id","cat_id"], 7:["state_id","dept_id"],
    8:["store_id","cat_id"], 9:["store_id","dept_id"],
    10:["item_id"], 11:["state_id","item_id"], 12:["id"],
}
RESULT_PATH = LAB_DIR + "/gap2_standalone_result.json"

def _log(msg): print(msg, flush=True)

def _fcst_path(tag, cutoff):
    return os.path.join(
        FCST_BASE, DATA_TAG, f"level_{LEVEL}",
        cutoff.replace("-",""), "models", tag, "forecasts.parquet")

def _seg_tag(dept_id, cl):
    clean = dept_id.lower().replace("_","")
    return f"v13d_zs_{clean}_dept_m0_cl{cl}_eP_sF"

def _geo(scores):
    return float(np.prod(scores) ** (1.0 / len(scores)))


if __name__ == "__main__":
    from tsfm_m5 import M5DataPipeline
    from tsfm_m5 import M5Evaluator

    _log("Loading static data...")
    pipeline = M5DataPipeline(config={"tag": DATA_TAG})
    _, _, _, static_df, _ = pipeline.get_prepared_data(
        DATA_PATH, CUTOFFS[0], level=LEVEL, force_reprepare=False)

    all_ids   = list(static_df["id"].unique())
    n_items   = len(all_ids)
    id_to_idx = {aid: i for i, aid in enumerate(all_ids)}
    _log(f"  {n_items} total series")

    # Tier membership
    _, _, _, _, wt0 = pipeline.get_prepared_data(
        DATA_PATH, CUTOFFS[0], level=LEVEL, force_reprepare=False)
    w_lv = wt0[wt0["level"] == LEVEL].copy()
    w_lv["tier"] = pd.qcut(w_lv["weight"], q=4, labels=_TIER_ORDER)
    tier_ids = {t: set(w_lv[w_lv["tier"] == t]["id"]) for t in _TIER_ORDER}
    del wt0, w_lv
    item_tier = np.array(
        [next((i for i, t in enumerate(_TIER_ORDER) if aid in tier_ids[t]), 0)
         for aid in all_ids], dtype=np.int32)

    results = {}

    for cutoff in CUTOFFS:
        lbl = CUTOFF_LABELS[cutoff]
        _log(f"\n[{lbl}] building eval state...")

        hist_df, hist_trimmed, future_df, _, wts = pipeline.get_prepared_data(
            DATA_PATH, cutoff, level=LEVEL, force_reprepare=False)
        evaluator = M5Evaluator(
            raw_train_df=hist_df, trimmed_train_df=hist_trimmed,
            static_df=static_df, weights_df=wts,
            target_col="sales_quantity", price_col="sell_price")
        del hist_df, hist_trimmed, future_df, wts

        dates = sorted(pd.read_parquet(
            _fcst_path(_TIER_TAGS["Low"], cutoff))["date"].unique())

        # ── Tier ensemble ────────────────────────────────────────────────────
        tier_stack = np.stack([
            pd.read_parquet(_fcst_path(_TIER_TAGS[seg], cutoff))
            .pivot(index="id", columns="date", values="sales_quantity")
            .reindex(index=all_ids, columns=dates).fillna(0.0).values.astype(np.float64)
            for seg in _TIER_ORDER], axis=0)
        W = _V11_W[cutoff]
        tier_fcst = np.clip(np.einsum("ij,jid->id", W[item_tier], tier_stack), 0.0, None)
        del tier_stack

        # ── Chronos-2 standalone (v13 per-dept, assembled) ──────────────────
        chron_fcst = np.zeros((n_items, HORIZON), dtype=np.float64)
        for dept_id in DEPT_IDS:
            cl  = BEST_CLS[dept_id]
            tag = _seg_tag(dept_id, cl)
            df  = pd.read_parquet(_fcst_path(tag, cutoff))
            piv = (df.pivot(index="id", columns="date", values="sales_quantity")
                     .reindex(columns=dates).fillna(0.0).values.astype(np.float64))
            for row_i, aid in enumerate(sorted(df["id"].unique())):
                if aid in id_to_idx:
                    chron_fcst[id_to_idx[aid]] = piv[row_i]

        # ── Actuals ──────────────────────────────────────────────────────────
        actual_wide = (
            pd.read_parquet(DATA_PATH, columns=["id","date","sold"])
            .rename(columns={"sold":"sales_quantity"})
            .assign(id=lambda df: df["id"].astype(str)
                    .str.replace("_evaluation","",regex=False)
                    .str.replace("_validation","",regex=False))
            .pipe(lambda df: df[df["date"].isin(set(dates))])
            .pivot(index="id", columns="date", values="sales_quantity")
            .reindex(index=all_ids, columns=dates).fillna(0.0).values.astype(np.float64))

        # ── Sparse hierarchy ─────────────────────────────────────────────────
        static_loc = evaluator._static.reindex(all_ids).copy()
        static_loc["id"] = list(all_ids)
        levels = {}
        for L in range(1, 13):
            cols = _LEVEL_COLS[L]
            info = evaluator.level_info[L]
            sc = info["scales"].astype(np.float64)
            wt = info["weights"].astype(np.float64)
            if L == 1:
                levels[L] = {"t":"total","s":sc,"w":wt,"a":actual_wide.sum(0,keepdims=True)}
            elif L == 12:
                levels[L] = {"t":"identity","s":sc,"w":wt,"a":actual_wide}
            else:
                can = list(info["index"])
                u2i = {u:i for i,u in enumerate(can)}
                gk  = (static_loc[cols[0]].astype(str).values if len(cols)==1
                       else static_loc[cols].astype(str).agg("_".join,axis=1).values)
                i2u = np.array([u2i.get(k,-1) for k in gk], dtype=np.int32)
                v   = i2u >= 0
                A   = scipy.sparse.csr_matrix(
                    (np.ones(v.sum()), (i2u[v], np.where(v)[0])),
                    shape=(len(can), n_items))
                levels[L] = {"t":"sparse","A":A,"s":sc,"w":wt,"a":A@actual_wide}
        del actual_wide, evaluator, static_loc

        def _wrmsse(fcst, _lv=levels):
            s = []
            for L in range(1, 13):
                d = _lv[L]
                f = (fcst.sum(0,keepdims=True) if d["t"]=="total"
                     else fcst if d["t"]=="identity"
                     else d["A"]@fcst)
                mse = np.mean((d["a"]-f)**2, axis=1)
                s.append(float(np.dot(np.sqrt(np.maximum(mse/d["s"],0.0)),d["w"])))
            return float(np.mean(s))

        score_tier  = _wrmsse(tier_fcst)
        score_chron = _wrmsse(chron_fcst)
        score_ew    = _wrmsse(np.clip(0.5*tier_fcst + 0.5*chron_fcst, 0.0, None))

        _log(f"  [{lbl}]  tier={score_tier:.4f}  chronos={score_chron:.4f}  "
             f"equal-weight={score_ew:.4f}")

        results[cutoff] = {
            "tier": score_tier,
            "chronos_standalone": score_chron,
            "equal_weight": score_ew,
        }

        del tier_fcst, chron_fcst, levels
        gc.collect()

    # Summary
    tier_geo  = _geo([results[c]["tier"]              for c in CUTOFFS])
    chron_geo = _geo([results[c]["chronos_standalone"] for c in CUTOFFS])
    ew_geo    = _geo([results[c]["equal_weight"]       for c in CUTOFFS])
    ols_geo   = 0.8066   # v13 OLS per_dept

    _log("\n" + "="*60)
    _log("SUMMARY — geo-mean WRMSSE across 3 cutoffs")
    _log(f"  Tier ensemble standalone (v11 W)  : {tier_geo:.4f}")
    _log(f"  Chronos-2 standalone (v13 CLs)    : {chron_geo:.4f}")
    _log(f"  Equal-weight blend (0.5+0.5)       : {ew_geo:.4f}")
    _log(f"  OLS blend (v13 per_dept, ref)      : {ols_geo:.4f}")
    _log("")
    if chron_geo < tier_geo:
        _log(f"  → Chronos-2 BEATS tier ensemble standalone by Δ={tier_geo-chron_geo:.4f}")
    else:
        _log(f"  → Chronos-2 is WORSE than tier ensemble by Δ={chron_geo-tier_geo:.4f}")
    _log("="*60)

    results["geo"] = {
        "tier": tier_geo,
        "chronos_standalone": chron_geo,
        "equal_weight": ew_geo,
        "ols_reference": ols_geo,
    }
    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    _log(f"\nResults saved → {RESULT_PATH}")
