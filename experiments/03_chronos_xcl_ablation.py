"""
Gap 1 — Cross-learning ablation (group attention enabled vs disabled)
======================================================================
All prior experiments (v12–v17) used cross_learning=False (default) — pure
univariate inference. Group attention was NEVER activated.

This script tests whether enabling cross_learning=True improves M5 results
for HOBBIES_1 and HOUSEHOLD_1 (the two active depts per OLS α weights).

Three conditions:
  A) cross_learning=False (baseline, univariate) — reuse v13 cached parquets
  B) cross_learning=True, semantic batching (all same-dept series together)
     batch_size=100 as recommended by Chronos-2 technical report
  C) cross_learning=True, random batching (dept series mixed with other depts)
     batch_size=100 — tests whether semantic coherence matters for group attn

Tags:
  B: v19d_zs_{dept}_dept_m0_cl{cl}_cl_bs100_eP_sF   (cl = cross-learning)
  C: v19d_zs_{dept}_rnd_m0_cl{cl}_cl_bs100_eP_sF    (rnd = random batching)

Run:
    cd /home/nmwamsojo/tsfm-explo
    source .venv/bin/activate
    nohup python experiments/03_chronos_xcl_ablation.py >> /mnt/lab/nmwamsojo/gap1_cl.log 2>&1 &
    tail -f /mnt/lab/nmwamsojo/gap1_cl.log
"""

import gc, json, os, sys, time
import numpy as np
import pandas as pd
import scipy.sparse
import torch

from chronos import Chronos2Pipeline

# ── Config ─────────────────────────────────────────────────────────────────────
import os
DATA_PATH  = os.getenv("M5_DATA_PATH", "/mnt/lab/datasets/M5/jointed_M5.parquet")
LAB_DIR    = os.getenv("M5_LAB_DIR",   "/mnt/lab/nmwamsojo")

DATA_TAG   = "sales_only"
CUTOFFS    = ["2016-04-24", "2016-01-03", "2015-10-04"]
CUTOFF_LABELS = {"2016-04-24": "spring", "2016-01-03": "winter", "2015-10-04": "autumn"}
LEVEL      = 12
HORIZON    = 28
FCST_BASE  = LAB_DIR + "/prepared_data"
MODEL_ID   = "autogluon/chronos-2-small"

# Focus on the two active depts (OLS α < 1.0)
TARGET_DEPTS = {
    "HOBBIES_1":   1,   # CL=1 (v13 optimal)
    "HOUSEHOLD_1": 28,  # CL=28 (v13 optimal)
}
ALL_DEPT_IDS = ["FOODS_1","FOODS_2","FOODS_3","HOBBIES_1","HOBBIES_2","HOUSEHOLD_1","HOUSEHOLD_2"]

KEEP    = 28
CL_BS   = 100   # recommended batch size for cross-learning (Chronos-2 technical report)
PRED_BS = 32    # for non-cross-learning baseline check

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

_CALENDAR_COLS = ["is_friday", "is_saturday", "is_sunday", "is_month_end", "is_month_start"]
_EVENT_COLS    = ["event_name_1", "event_type_1"]
_SNAP_COLS     = ["snap_CA", "snap_TX", "snap_WI"]
_FUTURE_COV_OK = set(_CALENDAR_COLS + _SNAP_COLS + ["sell_price", "price_change", "price_norm"])

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
RESULT_PATH = LAB_DIR + "/gap1_cl_result.json"

def _log(msg): print(msg, flush=True)

def _fcst_path(tag, cutoff):
    return os.path.join(
        FCST_BASE, DATA_TAG, f"level_{LEVEL}",
        cutoff.replace("-",""), "models", tag, "forecasts.parquet")

def _tag_sem(dept_id, cl):
    clean = dept_id.lower().replace("_","")
    return f"v19d_zs_{clean}_dept_m0_cl{cl}_cl_bs100_eP_sF"

def _tag_rnd(dept_id, cl):
    clean = dept_id.lower().replace("_","")
    return f"v19d_zs_{clean}_rnd_m0_cl{cl}_cl_bs100_eP_sF"

def _tag_baseline(dept_id, cl):
    clean = dept_id.lower().replace("_","")
    return f"v13d_zs_{clean}_dept_m0_cl{cl}_eP_sF"

def _geo(scores):
    return float(np.prod(scores) ** (1.0 / len(scores)))

def _to_f32(arr):
    if hasattr(arr, "dtype") and (arr.dtype.kind in ("O","U","S")
                                   or str(arr.dtype) == "category"):
        return (pd.Categorical(arr).codes >= 0).astype(np.float32)
    return np.asarray(arr, dtype=np.float32)


def _add_features(hist_seg, future_seg, seg_price):
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


def _inputs_from_df(hist_seg, future_seg, truncate_to=None):
    known_cols = list(_CALENDAR_COLS) + list(_SNAP_COLS)
    past_cols  = list(_EVENT_COLS)
    if truncate_to is not None:
        hist_seg = hist_seg.groupby("id", sort=False, observed=True).tail(truncate_to)
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


def _run_inference(model, item_ids, inputs, cl, cross_learning, eval_dates):
    """Run inference and return fcst_df."""
    t0 = time.time()
    with torch.no_grad():
        _, pred_means = model.predict_quantiles(
            inputs,
            prediction_length = HORIZON,
            batch_size        = CL_BS,
            context_length    = cl,
            quantile_levels   = [0.5],
            cross_learning    = cross_learning,
        )
    preds = torch.stack(pred_means)
    point = np.clip(preds.squeeze(1).cpu().numpy(), 0.0, None)
    _log(f"      done in {time.time()-t0:.0f}s")
    return pd.DataFrame({
        "id":             np.repeat(item_ids, HORIZON),
        "date":           np.tile(eval_dates, len(item_ids)),
        "sales_quantity": point.flatten(),
    })


def _build_wrmsse_state(pipeline, static_df, all_ids, item_tier, cutoff):
    from tsfm_m5 import M5Evaluator
    n_items   = len(all_ids)
    id_to_idx = {aid: i for i, aid in enumerate(all_ids)}

    hist_df, hist_trimmed, future_df, _, wts = pipeline.get_prepared_data(
        DATA_PATH, cutoff, level=LEVEL, force_reprepare=False)
    evaluator = M5Evaluator(
        raw_train_df=hist_df, trimmed_train_df=hist_trimmed,
        static_df=static_df, weights_df=wts,
        target_col="sales_quantity", price_col="sell_price")
    del hist_df, hist_trimmed, future_df, wts

    dates = sorted(pd.read_parquet(_fcst_path(_TIER_TAGS["Low"], cutoff))["date"].unique())

    tier_stack = np.stack([
        pd.read_parquet(_fcst_path(_TIER_TAGS[seg], cutoff))
        .pivot(index="id", columns="date", values="sales_quantity")
        .reindex(index=all_ids, columns=dates).fillna(0.0).values.astype(np.float64)
        for seg in _TIER_ORDER], axis=0)
    W = _V11_W[cutoff]
    tier_fcst = np.clip(np.einsum("ij,jid->id", W[item_tier], tier_stack), 0.0, None)
    del tier_stack

    actual_wide = (
        pd.read_parquet(DATA_PATH, columns=["id","date","sold"])
        .rename(columns={"sold":"sales_quantity"})
        .assign(id=lambda df: df["id"].astype(str)
                .str.replace("_evaluation","",regex=False)
                .str.replace("_validation","",regex=False))
        .pipe(lambda df: df[df["date"].isin(set(dates))])
        .pivot(index="id", columns="date", values="sales_quantity")
        .reindex(index=all_ids, columns=dates).fillna(0.0).values.astype(np.float64))

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

    return {"tier": tier_fcst, "wrmsse": _wrmsse, "dates": dates, "id_to_idx": id_to_idx}


def _score_dept_tag(tag, cutoff_state, all_ids):
    """Score a single-dept tag blended with tier ensemble."""
    es       = cutoff_state
    dates    = es["dates"]
    id_to_idx = es["id_to_idx"]
    n_items  = len(all_ids)
    dept_blend = np.zeros((n_items, HORIZON), dtype=np.float64)
    path = _fcst_path(tag, list(CUTOFFS)[0])   # placeholder — path passed directly
    return dept_blend, es


def _blend_wrmsse(tag, cutoff, state, all_ids):
    es = state
    dates = es["dates"]
    id_to_idx = es["id_to_idx"]
    n_items = len(all_ids)
    dept_blend = np.zeros((n_items, HORIZON), dtype=np.float64)
    path = _fcst_path(tag, cutoff)
    if not os.path.exists(path):
        return float("nan")
    df = pd.read_parquet(path)
    piv = (df.pivot(index="id", columns="date", values="sales_quantity")
             .reindex(columns=dates).fillna(0.0).values.astype(np.float64))
    for row_i, aid in enumerate(sorted(df["id"].unique())):
        if aid in id_to_idx:
            dept_blend[id_to_idx[aid]] = piv[row_i]
    combined = np.clip(0.5 * es["tier"] + 0.5 * dept_blend, 0.0, None)
    return es["wrmsse"](combined)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from tsfm_m5 import M5DataPipeline
    from tsfm_m5 import M5Evaluator

    _log(f"Device: {DEVICE}")
    _log(f"Target depts: {list(TARGET_DEPTS.keys())}")
    _log(f"cross_learning batch_size: {CL_BS}")
    _log("")

    _log("Loading static data...")
    pipeline = M5DataPipeline(config={"tag": DATA_TAG})
    _, _, _, static_df, _ = pipeline.get_prepared_data(
        DATA_PATH, CUTOFFS[0], level=LEVEL, force_reprepare=False)

    all_ids   = list(static_df["id"].unique())
    id_to_idx = {aid: i for i, aid in enumerate(all_ids)}
    dept_series = {
        d: set(static_df[static_df["dept_id"] == d]["id"].tolist())
        for d in ALL_DEPT_IDS
    }
    _log(f"  {len(all_ids)} total series")

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

    # Price means
    _pr = pd.read_parquet(DATA_PATH, columns=["id","sell_price"])
    _pr["id"] = (_pr["id"].astype(str)
                 .str.replace("_evaluation","",regex=False)
                 .str.replace("_validation","",regex=False))
    item_price_mean = _pr.groupby("id")["sell_price"].mean().to_dict()
    del _pr

    # ── Phase 1: Inference ────────────────────────────────────────────────────
    _log("\n" + "="*60)
    _log("PHASE 1: INFERENCE")
    _log("="*60)

    # Build random-dept pool for condition C
    # For each target dept, interleave its series with random series from ALL other depts
    rng = np.random.default_rng(42)

    _log(f"\nLoading model {MODEL_ID} ...")
    model = Chronos2Pipeline.from_pretrained(
        MODEL_ID, device_map=DEVICE, dtype=torch.bfloat16)

    for dept_id, cl in TARGET_DEPTS.items():
        seg_ids   = dept_series[dept_id]
        seg_price = {k: v for k, v in item_price_mean.items() if k in seg_ids}
        other_ids = [aid for d, ids in dept_series.items()
                     if d != dept_id for aid in ids]
        _log(f"\n[{dept_id}]  cl={cl}  n_series={len(seg_ids)}")

        for cutoff in CUTOFFS:
            lbl = CUTOFF_LABELS[cutoff]

            tag_sem = _tag_sem(dept_id, cl)
            tag_rnd = _tag_rnd(dept_id, cl)
            need_sem = not os.path.exists(_fcst_path(tag_sem, cutoff))
            need_rnd = not os.path.exists(_fcst_path(tag_rnd, cutoff))

            if not need_sem and not need_rnd:
                _log(f"  [{lbl}] both conditions cached — skip")
                continue

            _log(f"  [{lbl}] loading data...")
            _, hist_trimmed, future_df, _, _ = pipeline.get_prepared_data(
                DATA_PATH, cutoff, level=LEVEL, force_reprepare=False)
            eval_dates     = list(pd.date_range(
                pd.Timestamp(cutoff) + pd.Timedelta(days=1), periods=HORIZON))
            eval_dates_set = set(eval_dates)

            hist_seg = hist_trimmed[hist_trimmed["id"].isin(seg_ids)].copy()
            fut_seg  = future_df[
                (future_df["id"].isin(seg_ids)) &
                (future_df["date"].isin(eval_dates_set))
            ].copy()
            _add_features(hist_seg, fut_seg, seg_price)
            target_item_ids, target_inputs = _inputs_from_df(
                hist_seg, fut_seg, truncate_to=KEEP)

            # ── Condition B: semantic (same-dept batch, cross_learning=True) ──
            if need_sem:
                _log(f"  [{lbl}] Condition B: semantic cross_learning=True, bs={CL_BS} ...")
                fcst_df = _run_inference(
                    model, target_item_ids, target_inputs, cl,
                    cross_learning=True, eval_dates=eval_dates)
                path = _fcst_path(tag_sem, cutoff)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                fcst_df.to_parquet(path, index=False)
                _log(f"  [{lbl}] saved → {path}")
                del fcst_df

            # ── Condition C: random (mixed-dept batch, cross_learning=True) ──
            if need_rnd:
                _log(f"  [{lbl}] building random mixed inputs (n_other={len(seg_ids)})...")
                # Sample same number of "context" series from other depts
                rnd_ids_sample = rng.choice(other_ids, size=len(seg_ids), replace=False).tolist()
                rnd_ids_set = set(rnd_ids_sample)
                other_price = {k: v for k, v in item_price_mean.items() if k in rnd_ids_set}

                hist_other = hist_trimmed[hist_trimmed["id"].isin(rnd_ids_set)].copy()
                fut_other  = future_df[
                    (future_df["id"].isin(rnd_ids_set)) &
                    (future_df["date"].isin(eval_dates_set))
                ].copy()
                _add_features(hist_other, fut_other, other_price)
                other_item_ids, other_inputs = _inputs_from_df(
                    hist_other, fut_other, truncate_to=KEEP)

                # Interleave target + other inputs and shuffle
                combined_ids    = target_item_ids + other_item_ids
                combined_inputs = target_inputs + other_inputs
                shuffle_idx     = rng.permutation(len(combined_ids)).tolist()
                combined_ids    = [combined_ids[i] for i in shuffle_idx]
                combined_inputs = [combined_inputs[i] for i in shuffle_idx]

                _log(f"  [{lbl}] Condition C: random cross_learning=True, "
                     f"n={len(combined_ids)}, bs={CL_BS} ...")
                t0 = time.time()
                with torch.no_grad():
                    _, pred_means = model.predict_quantiles(
                        combined_inputs,
                        prediction_length = HORIZON,
                        batch_size        = CL_BS,
                        context_length    = cl,
                        quantile_levels   = [0.5],
                        cross_learning    = True,
                    )
                preds = torch.stack(pred_means)
                point = np.clip(preds.squeeze(1).cpu().numpy(), 0.0, None)
                _log(f"  [{lbl}] done in {time.time()-t0:.0f}s")

                # Extract only the target dept predictions
                target_id_set  = set(target_item_ids)
                target_rows    = [i for i, iid in enumerate(combined_ids) if iid in target_id_set]
                target_ids_rnd = [combined_ids[i] for i in target_rows]
                target_preds   = point[target_rows]

                fcst_df = pd.DataFrame({
                    "id":             np.repeat(target_ids_rnd, HORIZON),
                    "date":           np.tile(eval_dates, len(target_ids_rnd)),
                    "sales_quantity": target_preds.flatten(),
                })
                path = _fcst_path(tag_rnd, cutoff)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                fcst_df.to_parquet(path, index=False)
                _log(f"  [{lbl}] saved → {path}")
                del hist_other, fut_other, other_inputs, combined_inputs, fcst_df

            del hist_seg, fut_seg, target_inputs
            del hist_trimmed, future_df
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    del model
    gc.collect()

    # ── Phase 2: Evaluation ───────────────────────────────────────────────────
    _log("\n" + "="*60)
    _log("PHASE 2: EVALUATION — per-dept blended WRMSSE (0.5×tier + 0.5×dept)")
    _log("="*60)

    _log("Building eval states...")
    eval_states = {}
    for cutoff in CUTOFFS:
        _log(f"  [{CUTOFF_LABELS[cutoff]}] ...")
        eval_states[cutoff] = _build_wrmsse_state(
            pipeline, static_df, all_ids, item_tier, cutoff)
    gc.collect()

    _log(f"\n{'Dept':<14}  {'Condition':<22}  "
         f"{'spring':>8}  {'winter':>8}  {'autumn':>8}  {'geo':>8}")
    _log("-" * 80)

    results = {}
    for dept_id, cl in TARGET_DEPTS.items():
        results[dept_id] = {}
        tag_base = _tag_baseline(dept_id, cl)
        tag_sem  = _tag_sem(dept_id, cl)
        tag_rnd  = _tag_rnd(dept_id, cl)

        for cond_label, tag in [
            ("A: ZS univariate", tag_base),
            ("B: CL semantic", tag_sem),
            ("C: CL random", tag_rnd),
        ]:
            scores = [_blend_wrmsse(tag, c, eval_states[c], all_ids) for c in CUTOFFS]
            geo    = _geo([s for s in scores if not np.isnan(s)])
            _log(f"  {dept_id:<14}  {cond_label:<22}  "
                 f"{scores[0]:>8.4f}  {scores[1]:>8.4f}  {scores[2]:>8.4f}  {geo:>8.4f}")
            results[dept_id][cond_label] = {
                "scores": dict(zip(CUTOFFS, scores)), "geo": geo
            }
        _log("")

    _log("="*60)
    _log("KEY QUESTION: does cross_learning=True improve over univariate ZS?")
    for dept_id in TARGET_DEPTS:
        base_geo = results[dept_id]["A: ZS univariate"]["geo"]
        sem_geo  = results[dept_id]["B: CL semantic"]["geo"]
        rnd_geo  = results[dept_id]["C: CL random"]["geo"]
        _log(f"  {dept_id}:")
        _log(f"    Semantic CL vs univariate: Δ={sem_geo-base_geo:+.4f} "
             f"({'BETTER' if sem_geo < base_geo else 'WORSE'})")
        _log(f"    Random CL vs univariate:   Δ={rnd_geo-base_geo:+.4f} "
             f"({'BETTER' if rnd_geo < base_geo else 'WORSE'})")
        _log(f"    Semantic vs Random:        Δ={sem_geo-rnd_geo:+.4f} "
             f"({'semantic wins' if sem_geo < rnd_geo else 'random wins or equal'})")
    _log("="*60)

    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    _log(f"\nResults saved → {RESULT_PATH}")
