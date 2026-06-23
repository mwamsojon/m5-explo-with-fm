"""
OLS blend — v20 cross_learning=True forecasts
=============================================
Finds per-dept blend weight α ∈ [0,1]:
    combined[dept] = α × tier_ensemble + (1-α) × v20_chronos

Selection: LOO-3 (each cutoff held out, α chosen on the other 2).
Final score: apply per-dept α on all 3 cutoffs → geo-mean WRMSSE.

v20 best CLs: F1=28, F2=7, F3=28, H1=1, H2=7, HH1=28, HH2=1
Baselines:
  v13 EW  geo=0.8672  (cross_learning=False)
  v13 OLS geo=0.8066  (cross_learning=False)
  v20 EW  geo=0.8417  (cross_learning=True)

Run:
    cd /home/nmwamsojo/tsfm-explo && source .venv/bin/activate
    python experiments/05_chronos_ols_v20.py
"""
import gc, os, sys
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.optimize

import os
DATA_PATH  = os.getenv("M5_DATA_PATH", "/mnt/lab/datasets/M5/jointed_M5.parquet")
LAB_DIR    = os.getenv("M5_LAB_DIR",   "/mnt/lab/nmwamsojo")

DATA_TAG   = "sales_only"
CUTOFFS    = ["2016-04-24", "2016-01-03", "2015-10-04"]
LEVEL      = 12
HORIZON    = 28
FCST_BASE  = LAB_DIR + "/prepared_data"

DEPT_IDS   = ["FOODS_1","FOODS_2","FOODS_3","HOBBIES_1","HOBBIES_2","HOUSEHOLD_1","HOUSEHOLD_2"]
V20_CLS    = {"FOODS_1":28,"FOODS_2":7,"FOODS_3":28,"HOBBIES_1":1,"HOBBIES_2":7,"HOUSEHOLD_1":28,"HOUSEHOLD_2":1}

_TIER_ORDER = ["Low","Med-Low","Med-High","High"]
_V11_W = {
    "2016-04-24": np.array([[0.6402,0.1199,0.1200,0.1199],[0.0104,0.9689,0.0104,0.0104],
                             [0.3073,0.3075,0.0776,0.3076],[0.1520,0.1521,0.1521,0.5438]]),
    "2016-01-03": np.array([[0.9314,0.0229,0.0229,0.0229],[0.0592,0.8224,0.0592,0.0592],
                             [0.0805,0.0805,0.7585,0.0805],[0.3317,0.3319,0.3317,0.0048]]),
    "2015-10-04": np.array([[0.6917,0.1028,0.1028,0.1028],[0.2981,0.1051,0.2980,0.2987],
                             [0.2209,0.2214,0.3362,0.2215],[0.0012,0.0012,0.0012,0.9965]]),
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

def _log(msg): print(msg, flush=True)

def _fcst_path(tag, cutoff):
    return os.path.join(FCST_BASE, DATA_TAG, f"level_{LEVEL}",
                        cutoff.replace("-",""), "models", tag, "forecasts.parquet")

def _v20_tag(dept_id, cl):
    clean = dept_id.lower().replace("_","")
    return f"v20d_zs_{clean}_dept_m0_cl{cl}_xcl100_eP_sF"

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

    # Dept membership
    seg_ids = {d: set(static_df[static_df["dept_id"] == d]["id"].tolist()) for d in DEPT_IDS}

    # Tier quartile membership
    _, _, _, _, wt0 = pipeline.get_prepared_data(
        DATA_PATH, CUTOFFS[0], level=LEVEL, force_reprepare=False)
    w_lv = wt0[wt0["level"] == LEVEL].copy()
    w_lv["tier"] = pd.qcut(w_lv["weight"], q=4, labels=_TIER_ORDER)
    tier_ids = {t: set(w_lv[w_lv["tier"] == t]["id"]) for t in _TIER_ORDER}
    item_tier = np.array(
        [next((i for i,t in enumerate(_TIER_ORDER) if aid in tier_ids[t]), 0)
         for aid in all_ids], dtype=np.int32)
    del wt0, w_lv

    # Per-item dept index
    dept_of = np.array(
        [next((i for i,d in enumerate(DEPT_IDS) if aid in seg_ids[d]), -1)
         for aid in all_ids], dtype=np.int32)

    _log(f"  {n_items} series, {len(DEPT_IDS)} depts")

    # ── Build per-cutoff arrays ───────────────────────────────────────────────
    _log("Building tier, chronos and actuals arrays per cutoff...")
    cutoff_data = {}

    for cutoff in CUTOFFS:
        hist_df, hist_trimmed, future_df, _, wts = pipeline.get_prepared_data(
            DATA_PATH, cutoff, level=LEVEL, force_reprepare=False)
        evaluator = M5Evaluator(
            raw_train_df=hist_df, trimmed_train_df=hist_trimmed,
            static_df=static_df, weights_df=wts,
            target_col="sales_quantity", price_col="sell_price")
        del hist_df, hist_trimmed, future_df, wts

        dates = sorted(pd.read_parquet(
            _fcst_path(_TIER_TAGS["Low"], cutoff))["date"].unique())

        # Tier ensemble
        tier_stack = np.stack([
            pd.read_parquet(_fcst_path(_TIER_TAGS[seg], cutoff))
            .pivot(index="id", columns="date", values="sales_quantity")
            .reindex(index=all_ids, columns=dates).fillna(0.0).values.astype(np.float64)
            for seg in _TIER_ORDER], axis=0)
        W = _V11_W[cutoff]
        tier = np.clip(np.einsum("ij,jid->id", W[item_tier], tier_stack), 0.0, None)
        del tier_stack

        # v20 Chronos (best CLs)
        chron = np.zeros((n_items, HORIZON), dtype=np.float64)
        for dept_id in DEPT_IDS:
            cl  = V20_CLS[dept_id]
            tag = _v20_tag(dept_id, cl)
            df  = pd.read_parquet(_fcst_path(tag, cutoff))
            piv = (df.pivot(index="id", columns="date", values="sales_quantity")
                     .reindex(columns=dates).fillna(0.0).values.astype(np.float64))
            for row_i, aid in enumerate(sorted(df["id"].unique())):
                if aid in id_to_idx:
                    chron[id_to_idx[aid]] = piv[row_i]

        # Actuals
        actual = (
            pd.read_parquet(DATA_PATH, columns=["id","date","sold"])
            .rename(columns={"sold":"sales_quantity"})
            .assign(id=lambda d: d["id"].astype(str)
                    .str.replace("_evaluation","",regex=False)
                    .str.replace("_validation","",regex=False))
            .pipe(lambda d: d[d["date"].isin(set(dates))])
            .pivot(index="id", columns="date", values="sales_quantity")
            .reindex(index=all_ids, columns=dates).fillna(0.0).values.astype(np.float64))

        # WRMSSE closure
        static_loc = evaluator._static.reindex(all_ids).copy()
        static_loc["id"] = list(all_ids)
        levels = {}
        for L in range(1, 13):
            cols = _LEVEL_COLS[L]
            info = evaluator.level_info[L]
            sc = info["scales"].astype(np.float64)
            wt = info["weights"].astype(np.float64)
            if L == 1:
                levels[L] = {"t":"total","s":sc,"w":wt,"a":actual.sum(0,keepdims=True)}
            elif L == 12:
                levels[L] = {"t":"identity","s":sc,"w":wt,"a":actual}
            else:
                can = list(info["index"])
                u2i = {u:i for i,u in enumerate(can)}
                gk  = (static_loc[cols[0]].astype(str).values if len(cols)==1
                       else static_loc[cols].astype(str).agg("_".join,axis=1).values)
                i2u = np.array([u2i.get(k,-1) for k in gk], dtype=np.int32)
                v   = i2u >= 0
                A   = scipy.sparse.csr_matrix(
                    (np.ones(v.sum()),(i2u[v],np.where(v)[0])), shape=(len(can),n_items))
                levels[L] = {"t":"sparse","A":A,"s":sc,"w":wt,"a":A@actual}
        del actual, evaluator, static_loc

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

        cutoff_data[cutoff] = {"tier": tier, "chron": chron, "wrmsse": _wrmsse}
        gc.collect()
        _log(f"  [{cutoff}] arrays ready.")

    # ── Per-dept LOO-3 alpha optimisation ─────────────────────────────────────
    _log("\nOptimising per-dept blend α via LOO-3 ...")

    ALPHA_GRID = np.linspace(0.0, 1.0, 201)  # 0, 0.005, 0.010, ..., 1.0

    def _combined_wrmsse(alphas, cutoff):
        """Build combined forecast with per-dept alphas and compute WRMSSE."""
        tier  = cutoff_data[cutoff]["tier"]
        chron = cutoff_data[cutoff]["chron"]
        fcst  = np.empty_like(tier)
        for di, dept_id in enumerate(DEPT_IDS):
            mask = (dept_of == di)
            a    = alphas[di]
            fcst[mask] = a * tier[mask] + (1.0 - a) * chron[mask]
        fcst = np.clip(fcst, 0.0, None)
        return cutoff_data[cutoff]["wrmsse"](fcst)

    dept_alphas = {}
    dept_loo_scores = {}

    for di, dept_id in enumerate(DEPT_IDS):
        best_loo_alpha = 0.5
        best_loo_score = float("inf")

        # LOO-3: for each held-out cutoff, find alpha on remaining 2
        loo_results = []
        for hold_idx, hold_cutoff in enumerate(CUTOFFS):
            train_cutoffs = [c for c in CUTOFFS if c != hold_cutoff]

            def _train_score(alpha):
                scores = []
                # Build per-dept alphas with all other depts at their current best
                # (use neutral 0.5 for others during search — we're optimising one at a time)
                alphas = np.full(len(DEPT_IDS), 0.5)
                alphas[di] = alpha
                for tc in train_cutoffs:
                    scores.append(_combined_wrmsse(alphas, tc))
                return np.mean(scores)

            grid_scores = [_train_score(a) for a in ALPHA_GRID]
            best_alpha  = ALPHA_GRID[np.argmin(grid_scores)]

            # Fine search around best
            lo  = max(0.0, best_alpha - 0.01)
            hi  = min(1.0, best_alpha + 0.01)
            fine = np.linspace(lo, hi, 41)
            fine_scores  = [_train_score(a) for a in fine]
            best_alpha   = fine[np.argmin(fine_scores)]

            # Eval on hold-out
            alphas = np.full(len(DEPT_IDS), 0.5)
            alphas[di] = best_alpha
            hold_score = _combined_wrmsse(alphas, hold_cutoff)
            loo_results.append((best_alpha, hold_score))

        # Pick alpha by averaging LOO-optimal alphas
        avg_alpha = float(np.mean([r[0] for r in loo_results]))
        avg_score = float(np.mean([r[1] for r in loo_results]))
        dept_alphas[dept_id]     = avg_alpha
        dept_loo_scores[dept_id] = avg_score
        _log(f"  {dept_id:<14}  α={avg_alpha:.4f}  LOO avg score={avg_score:.4f}  "
             f"(per-holdout α: {[f'{r[0]:.3f}' for r in loo_results]})")

    # ── Final evaluation with per-dept alphas ─────────────────────────────────
    _log("\nFinal evaluation with per-dept α ...")
    alphas_vec = np.array([dept_alphas[d] for d in DEPT_IDS])

    final_scores = []
    for cutoff in CUTOFFS:
        score = _combined_wrmsse(alphas_vec, cutoff)
        final_scores.append(score)
        _log(f"  @{cutoff}  WRMSSE={score:.4f}")

    geo = _geo(final_scores)

    _log(f"\n{'='*60}")
    _log(f"V20 OLS geo-mean: {geo:.4f}")
    _log(f"  v13 EW  baseline: 0.8672  (cross_learning=False)")
    _log(f"  v13 OLS baseline: 0.8066  (cross_learning=False)")
    _log(f"  v20 EW  baseline: 0.8417  (cross_learning=True)")
    _log(f"  v20 OLS result:   {geo:.4f}  (cross_learning=True)")
    delta_ew  = 0.8417 - geo
    delta_ols = 0.8066 - geo
    _log(f"  Δ vs v20 EW:  {delta_ew:+.4f}")
    _log(f"  Δ vs v13 OLS: {delta_ols:+.4f}")
    _log(f"\nPer-dept α (higher = more tier, lower = more Chronos):")
    for dept_id in DEPT_IDS:
        pct_chron = (1.0 - dept_alphas[dept_id]) * 100
        _log(f"  {dept_id:<14}  α={dept_alphas[dept_id]:.4f}  ({pct_chron:.1f}% Chronos)")
    _log(f"{'='*60}")
