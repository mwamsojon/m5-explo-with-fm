"""
Phase 1: CL Sweep — Dept / State / Store ZS segments
======================================================
Runs Chronos-2-small ZS inference for all 20 geo/dept segments
across CL ∈ {1, 7, 14, 28} and 3 cutoffs = 240 independent jobs.
Parallelized across N_WORKERS CPU processes.

Cache-first: any existing parquet is skipped automatically.
All parquets are reusable in Round 2 when FT columns are added.

Tag convention:
  v12d_zs_{seg_id}_{seg_type}_m0_cl{cl}_{cov}
  e.g.  v12d_zs_foods3_dept_m0_cl7_eP_sF
        v12d_zs_ca1_store_m0_cl14_eP_sF
        v12d_zs_ca_state_m0_cl7_eP_sF

Memory design (Option B):
  Main process loads each cutoff ONCE, pre-builds feature-engineered Chronos
  inputs (numpy arrays only, history trimmed to last 28 rows) per segment,
  deletes the full prepared data, then submits 20 group-jobs per cutoff.
  Workers receive ~3–11 MB of pre-built arrays, load Chronos once, run all
  4 CLs for that (segment, cutoff) group, and save parquets.
  Workers never read from the source parquet or call get_prepared_data().

Run:
    cd /home/nmwamsojo/tsfm-explo
    source .venv/bin/activate
    nohup python experiments/02_chronos_zs_cl_sweep.py >> /mnt/lab/nmwamsojo/cl_sweep_zs.log 2>&1 &
    tail -f /mnt/lab/nmwamsojo/cl_sweep_zs.log
"""

import gc
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

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
N_WORKERS  = 20
BATCH_SIZE = 32   # CPU-friendly; each worker uses 1 thread

CL_OPTIONS  = [1, 7, 14, 28]
DEFAULT_COV = "eP_sF"

# Which segment types to run. Subset to keep sweep targeted.
# "dept" only → 84 jobs (needed for dept HPO)
# ["dept", "state", "store"] → full 240 jobs
SEG_TYPES = ["dept"]

DEPT_IDS  = [
    "FOODS_1", "FOODS_2", "FOODS_3",
    "HOBBIES_1", "HOBBIES_2",
    "HOUSEHOLD_1", "HOUSEHOLD_2",
]
STATE_IDS = ["CA", "TX", "WI"]
STORE_IDS = [
    "CA_1", "CA_2", "CA_3", "CA_4",
    "TX_1", "TX_2", "TX_3",
    "WI_1", "WI_2", "WI_3",
]

_CALENDAR_COLS = ["is_friday", "is_saturday", "is_sunday", "is_month_end", "is_month_start"]
_EVENT_COLS    = ["event_name_1", "event_type_1"]
_PRICE_COLS    = ["sell_price", "price_change", "price_norm"]
_SNAP_COLS     = ["snap_CA", "snap_TX", "snap_WI"]
_FUTURE_COV_OK = set(_CALENDAR_COLS + _SNAP_COLS + _PRICE_COLS)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _to_f32(arr) -> np.ndarray:
    if hasattr(arr, "dtype") and (arr.dtype.kind in ("O", "U", "S")
                                   or str(arr.dtype) == "category"):
        codes = pd.Categorical(arr).codes
        return (codes >= 0).astype(np.float32)
    return np.asarray(arr, dtype=np.float32)


def _seg_tag(seg_id: str, seg_type: str, cl: int, cov: str) -> str:
    clean = seg_id.lower().replace("_", "")
    return f"v12d_zs_{clean}_{seg_type}_m0_cl{cl}_{cov}"


def _fcst_path(tag: str, cutoff: str) -> str:
    return os.path.join(
        FCST_BASE, DATA_TAG, f"level_{LEVEL}",
        cutoff.replace("-", ""), "models", tag, "forecasts.parquet",
    )


def _build_inputs(
    hist_seg: pd.DataFrame,
    future_seg: pd.DataFrame,
    cov: str,
    seg_price: dict,
) -> tuple[list, list]:
    """
    Feature-engineer and convert to Chronos input dicts (numpy arrays only).
    History is trimmed to last max(CL_OPTIONS)=28 rows after feature computation,
    so pickled job args are tiny (~3-11 MB per segment).
    Returns (item_ids, inputs_list).
    """
    KEEP = max(CL_OPTIONS)  # 28 — maximum context length Chronos will use

    # ── Calendar and price features ──────────────────────────────────────────
    for df in (hist_seg, future_seg):
        dates = pd.to_datetime(df["date"])
        day   = dates.dt.day
        df["is_month_end"]   = (dates.dt.days_in_month - day < 3).astype("int8")
        df["is_month_start"] = (day <= 3).astype("int8")
        mean_p = df["id"].map(seg_price)
        df["price_norm"] = (df["sell_price"] / mean_p.replace(0, float("nan"))).fillna(1.0)

    # price_change on full history (needs shift(7), so compute before trimming)
    hist_seg["price_change"] = (
        hist_seg.groupby("id", observed=True)["sell_price"]
        .transform(lambda s: (s - s.shift(7)).fillna(0.0)))

    # price_change for future (uses last 7 hist rows for continuity)
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

    # ── Trim history to last KEEP rows (features already computed above) ─────
    hist_seg = hist_seg.groupby("id", sort=False, observed=True).tail(KEEP)

    # ── Covariate column selection ──────────────────────────────────────────
    known_cols, past_cols = [], []
    if "eF" in cov:   known_cols.extend(_EVENT_COLS)
    elif "eP" in cov: past_cols.extend(_EVENT_COLS)
    if "sF" in cov:   known_cols.extend(_SNAP_COLS)
    elif "sP" in cov: past_cols.extend(_SNAP_COLS)
    if "pF" in cov:   known_cols.extend(_PRICE_COLS)
    elif "pP" in cov: past_cols.extend(_PRICE_COLS)
    known_cols = list(_CALENDAR_COLS) + known_cols

    # ── Build Chronos input dicts (numpy copies — DataFrames can be freed) ──
    item_ids = sorted(hist_seg["id"].unique())
    h_grp = {k: v for k, v in hist_seg.groupby("id", observed=True)}
    f_grp = {k: v for k, v in future_seg.groupby("id", observed=True)}

    inputs = []
    for iid in item_ids:
        h = h_grp[iid]
        f = f_grp.get(iid)
        entry: dict = {"target": _to_f32(h["sales_quantity"].values)}
        past_dict: dict = {}
        for col in past_cols:
            if col in h.columns:
                past_dict[col] = _to_f32(h[col].values)
        for col in known_cols:
            if col in h.columns:
                past_dict[col] = _to_f32(h[col].values)
        if past_dict:
            entry["past_covariates"] = past_dict
        if known_cols and f is not None:
            safe = [c for c in known_cols if c in f.columns and c in _FUTURE_COV_OK]
            if safe:
                fut_dict = {c: _to_f32(f[c].values[:HORIZON]) for c in safe}
                if all(len(v) == HORIZON for v in fut_dict.values()):
                    entry["future_covariates"] = fut_dict
        inputs.append(entry)

    return item_ids, inputs


# ── Worker ─────────────────────────────────────────────────────────────────────
def _run_job_group(args: tuple) -> list[tuple]:
    """
    Runs all pending CLs for one (seg_id, cutoff) group.
    Loads Chronos once for the group, predicts for each pending CL, saves parquets.
    Returns list of (tag, cutoff, status, elapsed) — one entry per completed CL.
    """
    seg_type, seg_id, cutoff, eval_dates, item_ids, inputs_list, cov = args

    os.environ["OMP_NUM_THREADS"]       = "1"
    os.environ["MKL_NUM_THREADS"]       = "1"
    os.environ["OPENBLAS_NUM_THREADS"]  = "1"
    os.environ["NUMEXPR_NUM_THREADS"]   = "1"
    import torch
    torch.set_num_threads(1)

    from chronos import Chronos2Pipeline

    pending_cls = [cl for cl in CL_OPTIONS
                   if not os.path.exists(
                       _fcst_path(_seg_tag(seg_id, seg_type, cl, cov), cutoff))]

    if not pending_cls:
        return [(_seg_tag(seg_id, seg_type, cl, cov), cutoff, "cached", 0.0)
                for cl in CL_OPTIONS]

    model = Chronos2Pipeline.from_pretrained(
        MODEL_ID, device_map="cpu", dtype=torch.bfloat16)

    n_items  = len(item_ids)
    results  = []
    for cl in pending_cls:
        tag      = _seg_tag(seg_id, seg_type, cl, cov)
        out_path = _fcst_path(tag, cutoff)
        t0 = time.time()

        with torch.no_grad():
            _, pred_means = model.predict_quantiles(
                inputs_list,
                prediction_length = HORIZON,
                batch_size        = BATCH_SIZE,
                context_length    = cl,
                quantile_levels   = [0.5],
            )
        preds = torch.stack(pred_means)
        point = np.clip(preds.squeeze(1).cpu().numpy(), 0.0, None)

        fcst_df = pd.DataFrame({
            "id":             np.repeat(item_ids, HORIZON),
            "date":           np.tile(eval_dates, n_items),
            "sales_quantity": point.flatten(),
        })
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fcst_df.to_parquet(out_path, index=False)
        results.append((tag, cutoff, "computed", time.time() - t0))

    del model
    gc.collect()
    return results


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(message)s",
        datefmt = "%H:%M:%S",
    )
    log = logging.info

    from tsfm_m5 import M5DataPipeline

    # ── Price means (one-time read of source parquet) ─────────────────────────
    log("Loading item price means...")
    _pr = pd.read_parquet(DATA_PATH, columns=["id", "sell_price"])
    _pr["id"] = (_pr["id"].astype(str)
                 .str.replace("_evaluation", "", regex=False)
                 .str.replace("_validation",  "", regex=False))
    item_price_mean = _pr.groupby("id")["sell_price"].mean().to_dict()
    del _pr
    log(f"  {len(item_price_mean)} item price means loaded.")

    # ── Segment ID sets ───────────────────────────────────────────────────────
    log("Building segment ID sets...")
    _pipeline = M5DataPipeline(config={"tag": DATA_TAG})
    _, _, _, _static_df, _ = _pipeline.get_prepared_data(
        DATA_PATH, CUTOFFS[0], level=LEVEL, force_reprepare=False)
    seg_ids_map: dict = {}
    for seg_id in DEPT_IDS:
        seg_ids_map[seg_id] = set(_static_df[_static_df["dept_id"] == seg_id]["id"].tolist())
    for seg_id in STATE_IDS:
        seg_ids_map[seg_id] = set(_static_df[_static_df["state_id"] == seg_id]["id"].tolist())
    for seg_id in STORE_IDS:
        seg_ids_map[seg_id] = set(_static_df[_static_df["store_id"] == seg_id]["id"].tolist())
    del _static_df
    log(f"  {len(seg_ids_map)} segments ready.")

    _seg_map = {"dept": DEPT_IDS, "state": STATE_IDS, "store": STORE_IDS}
    N_SEGS  = sum(len(_seg_map[t]) for t in SEG_TYPES)
    N_TOTAL = len(CUTOFFS) * len(CL_OPTIONS) * N_SEGS
    log(f"Segment types: {SEG_TYPES} → {N_SEGS} segments, {N_TOTAL} total jobs.")
    done    = 0
    errors  = []
    t_start = time.time()

    for cutoff in CUTOFFS:
        log(f"\n── Cutoff {cutoff} ─────────────────────────────────")

        # ── Load full prepared data for this cutoff (once) ────────────────
        log("  Loading prepared data...")
        _, hist_trimmed, future_df, _, _ = _pipeline.get_prepared_data(
            DATA_PATH, cutoff, level=LEVEL, force_reprepare=False)

        eval_dates     = list(pd.date_range(
            pd.Timestamp(cutoff) + pd.Timedelta(days=1), periods=HORIZON))
        eval_dates_set = set(eval_dates)

        # ── Build one group-job per segment (main process) ────────────────
        log("  Building segment inputs (feature eng + trim to 28 rows)...")
        cutoff_groups = []
        seg_map = {"dept": DEPT_IDS, "state": STATE_IDS, "store": STORE_IDS}
        for seg_type, seg_list in [(t, seg_map[t]) for t in SEG_TYPES]:
            for seg_id in seg_list:
                seg_ids   = seg_ids_map[seg_id]
                seg_price = {k: v for k, v in item_price_mean.items() if k in seg_ids}

                hist_seg   = hist_trimmed[hist_trimmed["id"].isin(seg_ids)].copy()
                future_seg = future_df[
                    (future_df["id"].isin(seg_ids)) &
                    (future_df["date"].isin(eval_dates_set))
                ].copy()

                item_ids_seg, inputs_list = _build_inputs(
                    hist_seg, future_seg, DEFAULT_COV, seg_price)
                del hist_seg, future_seg

                cutoff_groups.append((
                    seg_type, seg_id, cutoff,
                    eval_dates, item_ids_seg, inputs_list, DEFAULT_COV,
                ))

        # Free the full prepared data — workers only need the sliced inputs
        del hist_trimmed, future_df
        gc.collect()
        log(f"  {len(cutoff_groups)} group-jobs built; full data freed.")

        # ── Filter to groups with at least one pending CL ─────────────────
        pending_groups = [
            g for g in cutoff_groups
            if any(
                not os.path.exists(_fcst_path(_seg_tag(g[1], g[0], cl, g[6]), g[2]))
                for cl in CL_OPTIONS
            )
        ]
        n_cached = len(cutoff_groups) - len(pending_groups)
        if n_cached:
            log(f"  {n_cached} group(s) fully cached, skipping.")
            done += n_cached * len(CL_OPTIONS)
        if not pending_groups:
            log(f"  All cached for {cutoff}.")
            continue

        log(f"  Submitting {len(pending_groups)} groups to {N_WORKERS} workers...")
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {pool.submit(_run_job_group, g): g for g in pending_groups}
            for fut in as_completed(futures):
                g = futures[fut]
                try:
                    for tag, co, status, elapsed in fut.result():
                        done += 1
                        e_str = f"{elapsed:.0f}s" if status == "computed" else "—"
                        log(f"  [{done:>3}/{N_TOTAL}]  {status:8s}  {e_str:>6}  {tag}  @{co}")
                except Exception as exc:
                    errors.append((g, str(exc)))
                    log(f"  ERROR  {g[1]}@{g[2]} → {exc}")

    wall = time.time() - t_start
    log(f"\nDone. {done}/{N_TOTAL} jobs in {wall/60:.1f} min.")
    if errors:
        log(f"ERRORS ({len(errors)}):")
        for g, e in errors:
            log(f"  {g[1]}@{g[2]} → {e}")
    else:
        log("No errors.")
