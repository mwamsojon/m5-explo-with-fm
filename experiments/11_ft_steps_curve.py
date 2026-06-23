"""
V25c — WRMSSE vs FT steps curve (spring cutoff, global FT)
===========================================================
steps=0 = per-dept ZS (base model, no FT).
steps>0 = global LoRA FT on all 30k series (TRAIN_CL=28), infer per-dept V20 CLs.

Reference lines on plot:
  v20 OLS  = 0.7969   (all-series ZS + OLS weights)
  v20 EW   = 0.8417   (all-series ZS + equal weights)

Results saved to /mnt/lab/nmwamsojo/steps_curve_results.json after each point.
Plot saved to /mnt/lab/nmwamsojo/steps_curve.png at the end.

Run:
    cd /home/nmwamsojo/tsfm-explo && source .venv/bin/activate
    nohup python experiments/11_ft_steps_curve.py > /mnt/lab/nmwamsojo/ft_steps_curve.log 2>&1 &
    tail -f /mnt/lab/nmwamsojo/ft_steps_curve.log
"""
import gc, json, os, sys, time
import numpy as np
import pandas as pd
import scipy.sparse
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chronos import Chronos2Pipeline

from tsfm_m5 import M5DataPipeline
from tsfm_m5 import M5Evaluator

# ── Config ─────────────────────────────────────────────────────────────────────
import os
DATA_PATH    = os.getenv("M5_DATA_PATH", "/mnt/lab/datasets/M5/jointed_M5.parquet")
LAB_DIR      = os.getenv("M5_LAB_DIR",   "/mnt/lab/nmwamsojo")
DATA_TAG     = "sales_only"
CUTOFF       = "2016-04-24"   # spring only — sufficient for the curve shape
LEVEL        = 12
HORIZON      = 28
FCST_BASE    = LAB_DIR + "/prepared_data"
DEVICE       = "cuda:0" if torch.cuda.is_available() else "cpu"
RESULTS_PATH = LAB_DIR + "/steps_curve_results.json"
PLOT_PATH    = LAB_DIR + "/steps_curve.png"

STEPS_LIST = [0, 5, 10, 25, 50, 100, 200, 500, 1000]

ALL_DEPTS = ["FOODS_1","FOODS_2","FOODS_3","HOBBIES_1","HOBBIES_2","HOUSEHOLD_1","HOUSEHOLD_2"]
V20_CLS   = {
    "FOODS_1":28, "FOODS_2":7, "FOODS_3":28,
    "HOBBIES_1":1, "HOBBIES_2":7,
    "HOUSEHOLD_1":28, "HOUSEHOLD_2":1,
}

TRAIN_CL  = 28
FT_LR     = 2e-5
FT_MODE   = "lora"
FT_BS     = 256
LORA_CFG  = {"r": 16, "lora_alpha": 32}
INFER_BS  = 100
MODEL_ID  = "autogluon/chronos-2-small"

SNAP_COLS      = ["snap_CA", "snap_TX", "snap_WI"]
CALENDAR_COLS  = ["is_friday", "is_saturday", "is_sunday"]
KNOWN_COV_COLS = CALENDAR_COLS + SNAP_COLS
EVENT_COLS     = ["event_name_1", "event_type_1"]

_TIER_ORDER = ["Low","Med-Low","Med-High","High"]
_V11_W = {
    "2016-04-24": np.array([[0.6402,0.1199,0.1200,0.1199],[0.0104,0.9689,0.0104,0.0104],
                             [0.3073,0.3075,0.0776,0.3076],[0.1520,0.1521,0.1521,0.5438]]),
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


# ── Chronos helpers ────────────────────────────────────────────────────────────
def _to_f32(arr):
    if hasattr(arr, "dtype") and (arr.dtype.kind in ("O","U","S") or str(arr.dtype) == "category"):
        return (pd.Categorical(arr).codes >= 0).astype(np.float32)
    return np.asarray(arr, dtype=np.float32)

def _encode_events(df):
    for col in EVENT_COLS:
        if col in df.columns:
            df[col] = (df[col].astype(str).str.strip()
                       .ne("nan").astype("float32")
                       * df[col].astype(str).str.strip().ne("").astype("float32"))
    return df

def _build_inputs(hist_seg, fut_seg, truncate_to=None):
    if truncate_to is not None:
        hist_seg = hist_seg.groupby("id", sort=False, observed=True).tail(truncate_to)
    item_ids = sorted(hist_seg["id"].unique())
    h_grp = {k: v for k, v in hist_seg.groupby("id", observed=True)}
    f_grp = {k: v for k, v in fut_seg.groupby("id", observed=True)}
    inputs = []
    for iid in item_ids:
        h = h_grp[iid]; f = f_grp.get(iid)
        entry = {"target": _to_f32(h["sales_quantity"].values)}
        past_dict = {}
        for col in EVENT_COLS:
            if col in h.columns: past_dict[col] = _to_f32(h[col].values)
        for col in KNOWN_COV_COLS:
            if col in h.columns: past_dict[col] = _to_f32(h[col].values)
        if past_dict: entry["past_covariates"] = past_dict
        if f is not None:
            fut_dict = {col: _to_f32(f[col].values[:HORIZON])
                        for col in KNOWN_COV_COLS
                        if col in f.columns and len(f[col].values) >= HORIZON}
            if fut_dict: entry["future_covariates"] = fut_dict
        inputs.append(entry)
    return item_ids, inputs


# ── WRMSSE setup ───────────────────────────────────────────────────────────────
def _build_wrmsse_state(pipeline_obj, static_df, all_ids):
    n_items   = len(all_ids)
    id_to_idx = {aid: i for i, aid in enumerate(all_ids)}

    _, _, _, _, wt0 = pipeline_obj.get_prepared_data(
        DATA_PATH, CUTOFF, level=LEVEL, force_reprepare=False)
    w_lv = wt0[wt0["level"] == LEVEL].copy()
    w_lv["tier"] = pd.qcut(w_lv["weight"], q=4, labels=_TIER_ORDER)
    tier_ids  = {t: set(w_lv[w_lv["tier"] == t]["id"]) for t in _TIER_ORDER}
    item_tier = np.array(
        [next((i for i, t in enumerate(_TIER_ORDER) if aid in tier_ids[t]), 0)
         for aid in all_ids], dtype=np.int32)

    hist_df, hist_trimmed, future_df, _, wts = pipeline_obj.get_prepared_data(
        DATA_PATH, CUTOFF, level=LEVEL, force_reprepare=False)
    evaluator = M5Evaluator(
        raw_train_df=hist_df, trimmed_train_df=hist_trimmed,
        static_df=static_df, weights_df=wts,
        target_col="sales_quantity", price_col="sell_price")

    dates = sorted(pd.read_parquet(_fcst_path(_TIER_TAGS["Low"], CUTOFF))["date"].unique())

    tier_stack = np.stack([
        pd.read_parquet(_fcst_path(_TIER_TAGS[seg], CUTOFF))
        .pivot(index="id", columns="date", values="sales_quantity")
        .reindex(index=all_ids, columns=dates).fillna(0.0).values.astype(np.float64)
        for seg in _TIER_ORDER], axis=0)
    W    = _V11_W[CUTOFF]
    tier = np.clip(np.einsum("ij,jid->id", W[item_tier], tier_stack), 0.0, None)
    del tier_stack

    actual = (
        pd.read_parquet(DATA_PATH, columns=["id","date","sold"])
        .rename(columns={"sold":"sales_quantity"})
        .assign(id=lambda d: d["id"].astype(str)
                .str.replace("_evaluation","",regex=False)
                .str.replace("_validation","",regex=False))
        .pipe(lambda d: d[d["date"].isin(set(dates))])
        .pivot(index="id", columns="date", values="sales_quantity")
        .reindex(index=all_ids, columns=dates).fillna(0.0).values.astype(np.float64))

    static_loc = evaluator._static.reindex(all_ids).copy()
    static_loc["id"] = list(all_ids)
    levels = {}
    for L in range(1, 13):
        cols = _LEVEL_COLS[L]
        info = evaluator.level_info[L]
        sc   = info["scales"].astype(np.float64)
        wt   = info["weights"].astype(np.float64)
        if L == 1:
            levels[L] = {"t":"total","s":sc,"w":wt,"a":actual.sum(0,keepdims=True)}
        elif L == 12:
            levels[L] = {"t":"identity","s":sc,"w":wt,"a":actual}
        else:
            can = list(info["index"]); u2i = {u:i for i,u in enumerate(can)}
            gk  = (static_loc[cols[0]].astype(str).values if len(cols)==1
                   else static_loc[cols].astype(str).agg("_".join,axis=1).values)
            i2u = np.array([u2i.get(k,-1) for k in gk], dtype=np.int32); v = i2u >= 0
            A   = scipy.sparse.csr_matrix(
                (np.ones(v.sum()),(i2u[v],np.where(v)[0])), shape=(len(can),n_items))
            levels[L] = {"t":"sparse","A":A,"s":sc,"w":wt,"a":A@actual}

    def _wrmsse(fcst, _lv=levels):
        s = []
        for L in range(1, 13):
            d = _lv[L]
            f = (fcst.sum(0,keepdims=True) if d["t"]=="total"
                 else fcst if d["t"]=="identity" else d["A"]@fcst)
            mse = np.mean((d["a"]-f)**2, axis=1)
            s.append(float(np.dot(np.sqrt(np.maximum(mse/d["s"],0.0)),d["w"])))
        return float(np.mean(s))

    return {"wrmsse": _wrmsse, "tier": tier, "dates": dates,
            "id_to_idx": id_to_idx, "hist_trimmed": hist_trimmed, "future_df": future_df}


def _infer_per_dept(pipeline, hist_trimmed, future_df, static_df, state, all_ids):
    """Run per-dept inference and return (raw_wrmsse, ew_wrmsse)."""
    eval_dates     = state["dates"]
    eval_dates_set = set(eval_dates)
    n              = len(all_ids)
    chron          = np.zeros((n, HORIZON), dtype=np.float64)

    for dept_id in ALL_DEPTS:
        cl       = V20_CLS[dept_id]
        dept_ids = set(static_df[static_df["dept_id"] == dept_id]["id"])

        hd = hist_trimmed[hist_trimmed["id"].isin(dept_ids)].copy()
        fd = future_df[future_df["id"].isin(dept_ids) & future_df["date"].isin(eval_dates_set)].copy()
        for _df in (hd, fd):
            if hasattr(_df["id"], "cat"):
                _df["id"] = _df["id"].cat.remove_unused_categories()
        hd = _encode_events(hd); fd = _encode_events(fd)

        pred_item_ids, pred_inputs = _build_inputs(hd, fd, truncate_to=cl)
        with torch.no_grad():
            _, pred_means = pipeline.predict_quantiles(
                pred_inputs, prediction_length=HORIZON,
                batch_size=INFER_BS, context_length=cl,
                quantile_levels=[0.5], cross_learning=True,
            )
        preds = torch.stack(pred_means)
        point = np.clip(preds.squeeze(1).cpu().numpy(), 0.0, None)

        for row_i, aid in enumerate(pred_item_ids):
            if aid in state["id_to_idx"]:
                chron[state["id_to_idx"][aid]] = point[row_i]

        del pred_inputs, preds, point, hd, fd
        gc.collect()

    raw = state["wrmsse"](chron)
    ew  = state["wrmsse"](np.clip(0.5*state["tier"] + 0.5*chron, 0.0, None))
    return raw, ew


# ── Plot ───────────────────────────────────────────────────────────────────────
def _plot(results):
    steps = sorted(int(k) for k in results)
    raw   = [results[str(s)]["raw"] for s in steps]
    ew    = [results[str(s)]["ew"]  for s in steps]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, raw, "o-", color="steelblue", label="Raw FT WRMSSE (per-dept infer)")
    ax.plot(steps, ew,  "s-", color="darkorange", label="EW blend (50% tier + 50% FT)")
    ax.axhline(0.7969, color="red",    linestyle="--", linewidth=1.2, label="v20 OLS = 0.7969")
    ax.axhline(0.8417, color="purple", linestyle="--", linewidth=1.2, label="v20 ZS EW = 0.8417")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("FT steps  (0 = ZS, no fine-tuning)", fontsize=11)
    ax.set_ylabel("WRMSSE (spring 2016-04-24)", fontsize=11)
    ax.set_title("WRMSSE vs LoRA FT steps — global train (30k series), per-dept inference", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    _log(f"Plot saved → {PLOT_PATH}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _log("=" * 65)
    _log("V25c: WRMSSE vs FT steps curve")
    _log(f"  Cutoff : {CUTOFF}  (spring)")
    _log(f"  Steps  : {STEPS_LIST}")
    _log(f"  TRAIN_CL={TRAIN_CL}  lr={FT_LR}  {FT_MODE}  r={LORA_CFG['r']}  bs={FT_BS}")
    _log("=" * 65)

    # Load saved results if resuming
    results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)
        _log(f"Resuming — already have steps: {sorted(int(k) for k in results)}")

    pipeline_obj = M5DataPipeline(config={"tag": DATA_TAG})

    _log("\nLoading static data...")
    _, _, _, static_df, _ = pipeline_obj.get_prepared_data(
        DATA_PATH, CUTOFF, level=LEVEL, force_reprepare=False)
    all_ids = list(static_df["id"].unique())
    _log(f"  {len(all_ids)} series")

    _log("\nBuilding WRMSSE state + tier ensemble...")
    state = _build_wrmsse_state(pipeline_obj, static_df, all_ids)
    hist_trimmed = state.pop("hist_trimmed")
    future_df    = state.pop("future_df")
    _log("  Done.")

    _log("\nBuilding global FT inputs (all 30k series)...")
    eval_dates_set = set(state["dates"])
    h_all = hist_trimmed.copy()
    f_all = future_df[future_df["date"].isin(eval_dates_set)].copy()
    for _df in (h_all, f_all):
        if hasattr(_df["id"], "cat"):
            _df["id"] = _df["id"].cat.remove_unused_categories()
    h_all = _encode_events(h_all); f_all = _encode_events(f_all)
    ft_item_ids, ft_inputs = _build_inputs(h_all, f_all, truncate_to=None)
    _log(f"  {len(ft_item_ids)} series in FT inputs.")

    _log("\n" + "="*65)

    for n_steps in STEPS_LIST:
        key = str(n_steps)
        if key in results:
            _log(f"  steps={n_steps:4d}  [cached]  raw={results[key]['raw']:.4f}  ew={results[key]['ew']:.4f}")
            continue

        t0 = time.time()
        if n_steps == 0:
            _log(f"  steps=   0  loading base model (ZS)...")
            pipeline = Chronos2Pipeline.from_pretrained(MODEL_ID, device_map=DEVICE, dtype=torch.bfloat16)
        else:
            _log(f"  steps={n_steps:4d}  FT from base...")
            import tempfile
            base = Chronos2Pipeline.from_pretrained(MODEL_ID, device_map=DEVICE, dtype=torch.bfloat16)
            with tempfile.TemporaryDirectory() as tmp:
                pipeline = base.fit(
                    inputs=ft_inputs, prediction_length=HORIZON,
                    finetune_mode=FT_MODE, lora_config=LORA_CFG,
                    context_length=TRAIN_CL, learning_rate=FT_LR, num_steps=n_steps,
                    batch_size=FT_BS, output_dir=tmp, finetuned_ckpt_name="ckpt",
                    remove_printer_callback=True, report_to="none",
                    logging_steps=999999, save_steps=999999,
                )
            del base
            gc.collect()

        _log(f"         inferring per dept...")
        raw, ew = _infer_per_dept(pipeline, hist_trimmed, future_df, static_df, state, all_ids)
        del pipeline
        gc.collect()

        results[key] = {"raw": raw, "ew": ew}
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

        elapsed = time.time() - t0
        _log(f"  steps={n_steps:4d}  raw={raw:.4f}  ew={ew:.4f}  [{elapsed:.0f}s]")

    _log("\n" + "="*65)
    _log("RESULTS SUMMARY")
    _log(f"  {'steps':>6}  {'raw WRMSSE':>12}  {'EW blend':>10}")
    _log(f"  {'-'*34}")
    for n in STEPS_LIST:
        r = results[str(n)]
        marker = " ← ZS" if n == 0 else ""
        _log(f"  {n:6d}  {r['raw']:12.4f}  {r['ew']:10.4f}{marker}")
    _log(f"  {'---':>6}  {'---':>12}  {'---':>10}")
    _log(f"  {'v20 OLS':>6}  {'':>12}  {'0.7969':>10}  ← reference")
    _log(f"  {'v20 EW':>6}  {'':>12}  {'0.8417':>10}  ← reference")
    _log("="*65)

    _plot(results)
