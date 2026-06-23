"""
Dept HPO v20 — cross_learning=True, GPU inference
===================================================
All prior HPO (v12/v13) used cross_learning=False (default) — pure univariate.
Group attention was NEVER activated. This run enables it and re-tunes CLs.

Key changes vs v13:
  - cross_learning=True in predict_quantiles → group attention ON
  - batch_size=100 (Chronos-2 technical report recommendation for cross-learning)
  - GPU inference (single device) — no CPU multiprocessing
  - Wider CL grid for HOBBIES_1 and HOUSEHOLD_1 (may prefer longer context with ICL)
  - Tag: v20d_zs_{dept}_dept_m0_cl{cl}_xcl100_eP_sF

Search space (same asymmetry philosophy as v13):
  FOODS_1     : [21, 28]
  FOODS_2     : [7, 14]
  FOODS_3     : [14, 21, 28]
  HOBBIES_1   : [1, 7, 14, 21, 28]       ← HIGHLY sensitive in v13; test all
  HOBBIES_2   : [7, 14]
  HOUSEHOLD_1 : [14, 21, 28, 35, 42, 56]  ← moderately sensitive; may want longer
  HOUSEHOLD_2 : [1, 7]

Total combos: 2×2×3×5×2×6×2 = 1,440  (150 TPE trials ≈ good coverage)

Baseline comparisons:
  v13 equal-weight geo: 0.8672  (cross_learning=False, v13 CLs)
  v13 OLS geo:          0.8066  (cross_learning=False, v13 CLs)

Run:
    cd /home/nmwamsojo/tsfm-explo
    source .venv/bin/activate
    nohup python experiments/04_chronos_hpo_v20.py >> /mnt/lab/nmwamsojo/hpo_dept_v20.log 2>&1 &
    tail -f /mnt/lab/nmwamsojo/hpo_dept_v20.log
"""

import gc, logging, os, sys, time
import numpy as np
import pandas as pd
import scipy.sparse
import optuna
import torch

from chronos import Chronos2Pipeline

# ── Config ─────────────────────────────────────────────────────────────────────
import os
DATA_PATH  = os.getenv("M5_DATA_PATH", "/mnt/lab/datasets/M5/jointed_M5.parquet")
LAB_DIR    = os.getenv("M5_LAB_DIR",   "/mnt/lab/nmwamsojo")

DATA_TAG   = "sales_only"
CUTOFFS    = ["2016-04-24", "2016-01-03", "2015-10-04"]
LEVEL      = 12
HORIZON    = 28
FCST_BASE  = LAB_DIR + "/prepared_data"
MODEL_ID   = "autogluon/chronos-2-small"
DEVICE     = "cuda:0" if torch.cuda.is_available() else "cpu"

N_TRIALS   = 150
CL_BS      = 100    # cross_learning batch size (recommended by Chronos-2 report)
DEFAULT_COV = "eP_sF"

DEPT_IDS = ["FOODS_1","FOODS_2","FOODS_3","HOBBIES_1","HOBBIES_2","HOUSEHOLD_1","HOUSEHOLD_2"]

DEPT_CL_OPTIONS = {
    "FOODS_1":     [21, 28],
    "FOODS_2":     [7, 14],
    "FOODS_3":     [14, 21, 28],
    "HOBBIES_1":   [1, 7, 14, 21, 28],
    "HOBBIES_2":   [7, 14],
    "HOUSEHOLD_1": [14, 21, 28, 35, 42, 56],
    "HOUSEHOLD_2": [1, 7],
}
KEEP = max(cl for cls in DEPT_CL_OPTIONS.values() for cl in cls)   # 56

OPTUNA_DB  = LAB_DIR + "/optuna_hpo_dept_v20.db"
STUDY_NAME = "hpo_dept_v20"

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
_TIER_ORDER = ["Low", "Med-Low", "Med-High", "High"]

_CALENDAR_COLS = ["is_friday", "is_saturday", "is_sunday", "is_month_end", "is_month_start"]
_EVENT_COLS    = ["event_name_1", "event_type_1"]
_PRICE_COLS    = ["sell_price", "price_change", "price_norm"]
_SNAP_COLS     = ["snap_CA", "snap_TX", "snap_WI"]
_FUTURE_COV_OK = set(_CALENDAR_COLS + _SNAP_COLS + _PRICE_COLS)
_LEVEL_COLS    = {
    1:[], 2:["state_id"], 3:["store_id"], 4:["cat_id"], 5:["dept_id"],
    6:["state_id","cat_id"], 7:["state_id","dept_id"],
    8:["store_id","cat_id"], 9:["store_id","dept_id"],
    10:["item_id"], 11:["state_id","item_id"], 12:["id"],
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def _seg_tag(dept_id, cl):
    clean = dept_id.lower().replace("_","")
    return f"v20d_zs_{clean}_dept_m0_cl{cl}_xcl100_eP_sF"

def _fcst_path(tag, cutoff):
    return os.path.join(
        FCST_BASE, DATA_TAG, f"level_{LEVEL}",
        cutoff.replace("-",""), "models", tag, "forecasts.parquet")

def _cached(dept_id, cl, cutoff):
    return os.path.exists(_fcst_path(_seg_tag(dept_id, cl), cutoff))

def _to_f32(arr):
    if hasattr(arr, "dtype") and (arr.dtype.kind in ("O","U","S")
                                   or str(arr.dtype) == "category"):
        return (pd.Categorical(arr).codes >= 0).astype(np.float32)
    return np.asarray(arr, dtype=np.float32)


# ── Feature engineering + input building ──────────────────────────────────────
def _build_inputs(hist_seg, future_seg, seg_price):
    for df in (hist_seg, future_seg):
        dates = pd.to_datetime(df["date"])
        day   = dates.dt.day
        df["is_month_end"]   = (dates.dt.days_in_month - day < 3).astype("int8")
        df["is_month_start"] = (day <= 3).astype("int8")
        mean_p = df["id"].map(seg_price)
        df["price_norm"] = (df["sell_price"] / mean_p.replace(0, float("nan"))).fillna(1.0)

    hist_seg["price_change"] = (
        hist_seg.groupby("id", observed=True)["sell_price"]
        .transform(lambda s: (s - s.shift(7)).fillna(0.0)))

    hist_tail = hist_seg.groupby("id", sort=False, observed=True).tail(7)[
        ["id", "date", "sell_price"]]
    combined = pd.concat([
        hist_tail.assign(_is_fut=False),
        future_seg[["id", "date", "sell_price"]].assign(_is_fut=True),
    ], ignore_index=True).sort_values(["id", "date"])
    pc = combined.groupby("id", observed=True)["sell_price"].transform(
        lambda s: (s - s.shift(7)).fillna(0.0))
    future_seg["price_change"] = pc[combined["_is_fut"]].values
    del combined, pc, hist_tail

    # Truncate to KEEP for prediction inputs
    hist_seg = hist_seg.groupby("id", sort=False, observed=True).tail(KEEP)

    known_cols = list(_CALENDAR_COLS) + list(_SNAP_COLS)
    past_cols  = list(_EVENT_COLS)

    item_ids = sorted(hist_seg["id"].unique())
    h_grp = {k: v for k, v in hist_seg.groupby("id", observed=True)}
    f_grp = {k: v for k, v in future_seg.groupby("id", observed=True)}

    inputs = []
    for iid in item_ids:
        h = h_grp[iid]
        f = f_grp.get(iid)
        entry = {"target": _to_f32(h["sales_quantity"].values)}
        past_dict = {}
        for col in past_cols + known_cols:
            if col in h.columns:
                past_dict[col] = _to_f32(h[col].values)
        if past_dict:
            entry["past_covariates"] = past_dict
        if f is not None:
            safe = [c for c in known_cols if c in f.columns and c in _FUTURE_COV_OK]
            if safe:
                fut_dict = {c: _to_f32(f[c].values[:HORIZON]) for c in safe}
                if all(len(v) == HORIZON for v in fut_dict.values()):
                    entry["future_covariates"] = fut_dict
        inputs.append(entry)

    return item_ids, inputs


# ── Inference: one dept × one cutoff × one CL ─────────────────────────────────
def _run_inference(model, dept_id, cl, cutoff):
    """Run cross_learning=True inference for one (dept, cl, cutoff). Cache result."""
    tag      = _seg_tag(dept_id, cl)
    out_path = _fcst_path(tag, cutoff)
    if os.path.exists(out_path):
        return

    job_key = (dept_id, cutoff)
    eval_dates, item_ids, inputs_list = _job_data[job_key]

    t0 = time.time()
    with torch.no_grad():
        _, pred_means = model.predict_quantiles(
            inputs_list,
            prediction_length = HORIZON,
            batch_size        = CL_BS,
            context_length    = cl,
            quantile_levels   = [0.5],
            cross_learning    = True,
        )
    preds = torch.stack(pred_means)
    point = np.clip(preds.squeeze(1).cpu().numpy(), 0.0, None)

    fcst_df = pd.DataFrame({
        "id":             np.repeat(item_ids, HORIZON),
        "date":           np.tile(eval_dates, len(item_ids)),
        "sales_quantity": point.flatten(),
    })
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fcst_df.to_parquet(out_path, index=False)
    _log(f"    cached {tag} @{cutoff} in {time.time()-t0:.0f}s")


# ── Fill cache for all jobs a trial needs ─────────────────────────────────────
def _fill_cache(trial_cls):
    needed = [
        (dept_id, trial_cls[dept_id], cutoff)
        for dept_id in DEPT_IDS
        for cutoff in CUTOFFS
        if not _cached(dept_id, trial_cls[dept_id], cutoff)
    ]
    # Fill ALL pending combos (66 total max) — front-load inference so
    # subsequent trials are evaluation-only (fast).
    all_pending = [
        (dept_id, cl, cutoff)
        for dept_id in DEPT_IDS
        for cl in DEPT_CL_OPTIONS[dept_id]
        for cutoff in CUTOFFS
        if not _cached(dept_id, cl, cutoff)
    ]
    if not all_pending:
        return
    _log(f"  [cache-fill] {len(all_pending)} combos pending (cross_learning=True bs={CL_BS})")
    for dept_id, cl, cutoff in all_pending:
        _run_inference(_model, dept_id, cl, cutoff)


# ── WRMSSE evaluation ──────────────────────────────────────────────────────────
def _evaluate_trial(trial_cls):
    scores = []
    for cutoff in CUTOFFS:
        es = _eval_states[cutoff]
        n_items    = es["tier"].shape[0]
        dept_blend = np.zeros((n_items, HORIZON), dtype=np.float64)
        dates      = es["dates"]

        for dept_id in DEPT_IDS:
            cl  = trial_cls[dept_id]
            tag = _seg_tag(dept_id, cl)
            df  = pd.read_parquet(_fcst_path(tag, cutoff))
            piv = (df.pivot(index="id", columns="date", values="sales_quantity")
                     .reindex(columns=dates).fillna(0.0).values.astype(np.float64))
            for row_i, aid in enumerate(sorted(df["id"].unique())):
                if aid in _id_to_idx:
                    dept_blend[_id_to_idx[aid]] = piv[row_i]

        combined = np.clip(0.5 * es["tier"] + 0.5 * dept_blend, 0.0, None)
        scores.append(es["wrmsse"](combined))

    return scores


def _build_eval_states():
    from tsfm_m5 import M5Evaluator
    n_items   = len(_all_ids)
    item_tier = np.array(
        [next((i for i, t in enumerate(_TIER_ORDER) if aid in _tier_ids[t]), 0)
         for aid in _all_ids], dtype=np.int32)
    states = {}
    for cutoff in CUTOFFS:
        hist_df, hist_trimmed, future_df, _, wts = _pipeline.get_prepared_data(
            DATA_PATH, cutoff, level=LEVEL, force_reprepare=False)
        evaluator = M5Evaluator(
            raw_train_df=hist_df, trimmed_train_df=hist_trimmed,
            static_df=_static_df, weights_df=wts,
            target_col="sales_quantity", price_col="sell_price")
        del hist_df, hist_trimmed, future_df, wts

        dates = sorted(pd.read_parquet(
            _fcst_path(_TIER_TAGS["Low"], cutoff))["date"].unique())
        tier_stack = np.stack([
            pd.read_parquet(_fcst_path(_TIER_TAGS[seg], cutoff))
            .pivot(index="id", columns="date", values="sales_quantity")
            .reindex(index=_all_ids, columns=dates).fillna(0.0).values.astype(np.float64)
            for seg in _TIER_ORDER], axis=0)
        W = _V11_W[cutoff]
        tier = np.clip(np.einsum("ij,jid->id", W[item_tier], tier_stack), 0.0, None)
        del tier_stack

        actual_wide = (
            pd.read_parquet(DATA_PATH, columns=["id","date","sold"])
            .rename(columns={"sold":"sales_quantity"})
            .assign(id=lambda df: df["id"].astype(str)
                    .str.replace("_evaluation","",regex=False)
                    .str.replace("_validation","",regex=False))
            .pipe(lambda df: df[df["date"].isin(set(dates))])
            .pivot(index="id", columns="date", values="sales_quantity")
            .reindex(index=_all_ids, columns=dates).fillna(0.0).values.astype(np.float64))

        static_loc = evaluator._static.reindex(_all_ids).copy()
        static_loc["id"] = list(_all_ids)
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

        states[cutoff] = {"tier": tier, "wrmsse": _wrmsse, "dates": dates}
        gc.collect()

    return states


# ── Optuna objective ───────────────────────────────────────────────────────────
def _objective(trial):
    trial_cls = {
        dept_id: trial.suggest_categorical(dept_id, DEPT_CL_OPTIONS[dept_id])
        for dept_id in DEPT_IDS
    }

    _fill_cache(trial_cls)

    t0     = time.time()
    scores = _evaluate_trial(trial_cls)
    geo    = float(np.prod(scores) ** (1.0 / len(scores)))

    for cutoff, score in zip(CUTOFFS, scores):
        trial.set_user_attr(f"wrmsse_{cutoff}", round(score, 6))

    cls_str    = " ".join(f"{d.replace('_','')}={trial_cls[d]}" for d in DEPT_IDS)
    scores_str = "  ".join(f"@{c[:10]}={s:.4f}" for c, s in zip(CUTOFFS, scores))
    _log(f"  Trial {trial.number:>3}  geo={geo:.4f}  [{time.time()-t0:.1f}s]  "
         f"{scores_str}  {cls_str}")
    return geo


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    _log = logging.info

    from tsfm_m5 import M5DataPipeline

    _log(f"Device: {DEVICE}")
    _log(f"cross_learning=True  batch_size={CL_BS}")
    _log(f"Model: {MODEL_ID}")
    _log(f"Baselines — v13 EW geo=0.8672  OLS geo=0.8066  (cross_learning=False)")
    _log("")

    # ── Load model (stays alive for the full HPO) ─────────────────────────────
    _log(f"Loading model {MODEL_ID} ...")
    _model = Chronos2Pipeline.from_pretrained(
        MODEL_ID, device_map=DEVICE, dtype=torch.bfloat16)
    _log("  Model loaded.")

    # ── Static data ───────────────────────────────────────────────────────────
    _log("Loading static data...")
    _pipeline = M5DataPipeline(config={"tag": DATA_TAG})
    _, _, _, _static_df, _ = _pipeline.get_prepared_data(
        DATA_PATH, CUTOFFS[0], level=LEVEL, force_reprepare=False)

    _seg_ids = {
        d: set(_static_df[_static_df["dept_id"] == d]["id"].tolist())
        for d in DEPT_IDS
    }
    _all_ids   = list(_static_df["id"].unique())
    _id_to_idx = {aid: i for i, aid in enumerate(_all_ids)}
    _log(f"  {len(_all_ids)} total series")

    # Price means
    _pr = pd.read_parquet(DATA_PATH, columns=["id","sell_price"])
    _pr["id"] = (_pr["id"].astype(str)
                 .str.replace("_evaluation","",regex=False)
                 .str.replace("_validation","",regex=False))
    _item_price_mean = _pr.groupby("id")["sell_price"].mean().to_dict()
    del _pr

    # ── Build inputs (all dept × cutoff pairs) ────────────────────────────────
    _log("Pre-building inputs for all dept × cutoff pairs...")
    _job_data = {}
    for cutoff in CUTOFFS:
        _, hist_trimmed, future_df, _, _ = _pipeline.get_prepared_data(
            DATA_PATH, cutoff, level=LEVEL, force_reprepare=False)
        eval_dates     = list(pd.date_range(
            pd.Timestamp(cutoff) + pd.Timedelta(days=1), periods=HORIZON))
        eval_dates_set = set(eval_dates)

        for dept_id in DEPT_IDS:
            seg_ids   = _seg_ids[dept_id]
            seg_price = {k: v for k, v in _item_price_mean.items() if k in seg_ids}
            hist_seg  = hist_trimmed[hist_trimmed["id"].isin(seg_ids)].copy()
            fut_seg   = future_df[
                (future_df["id"].isin(seg_ids)) &
                (future_df["date"].isin(eval_dates_set))
            ].copy()
            item_ids, inputs_list = _build_inputs(hist_seg, fut_seg, seg_price)
            _job_data[(dept_id, cutoff)] = (eval_dates, item_ids, inputs_list)
            del hist_seg, fut_seg

        del hist_trimmed, future_df
        gc.collect()
        _log(f"  [{cutoff}] inputs ready for {len(DEPT_IDS)} depts.")

    # ── Tier IDs ──────────────────────────────────────────────────────────────
    _, _, _, _, wt0 = _pipeline.get_prepared_data(
        DATA_PATH, CUTOFFS[0], level=LEVEL, force_reprepare=False)
    w_lv = wt0[wt0["level"] == LEVEL].copy()
    w_lv["tier"] = pd.qcut(w_lv["weight"], q=4, labels=_TIER_ORDER)
    _tier_ids = {t: set(w_lv[w_lv["tier"] == t]["id"]) for t in _TIER_ORDER}
    del wt0, w_lv

    # ── Cache status ──────────────────────────────────────────────────────────
    all_jobs = [(d, cl, c) for d in DEPT_IDS
                for cl in DEPT_CL_OPTIONS[d] for c in CUTOFFS]
    n_cached = sum(1 for j in all_jobs if _cached(*j))
    _log(f"Cache: {n_cached}/{len(all_jobs)} jobs already cached.")

    # ── Eval states ───────────────────────────────────────────────────────────
    _log("Building WRMSSE evaluation states...")
    _eval_states = _build_eval_states()
    del _static_df
    gc.collect()
    _log("Eval states ready.")

    # ── Optuna ────────────────────────────────────────────────────────────────
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name     = STUDY_NAME,
        storage        = f"sqlite:///{OPTUNA_DB}",
        direction      = "minimize",
        load_if_exists = True,
        sampler        = optuna.samplers.TPESampler(seed=42),
    )
    existing = len(study.trials)
    n_combos = 1
    for d in DEPT_IDS:
        n_combos *= len(DEPT_CL_OPTIONS[d])
    _log(f"Study '{STUDY_NAME}': {existing} existing trials, targeting {N_TRIALS} total.")
    _log(f"Search space: {n_combos} combos  |  cross_learning=True  bs={CL_BS}")

    study.optimize(_objective, n_trials=max(0, N_TRIALS - existing),
                   show_progress_bar=False)

    best = study.best_trial
    _log(f"\n{'='*60}")
    _log(f"BEST TRIAL #{best.number}  geo={best.value:.4f}")
    _log(f"  v13 baseline (cross_learning=False): EW=0.8672  OLS=0.8066")
    _log(f"  v20 best EW (cross_learning=True):   {best.value:.4f}")
    for cutoff in CUTOFFS:
        score = best.user_attrs.get(f"wrmsse_{cutoff}", float("nan"))
        _log(f"  @{cutoff}  WRMSSE={score:.4f}")
    _log("")
    for dept_id in DEPT_IDS:
        _log(f"  {dept_id:<14}  cl={best.params[dept_id]}")
    _log(f"{'='*60}")
