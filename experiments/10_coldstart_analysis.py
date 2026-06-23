#!/usr/bin/env python3
"""
10_coldstart_analysis.py
============================
Natural segmentation: compare LightGBM vs Chronos-2 ZS per-series RMSSE
grouped by active history length at each test cutoff.

Key design choices (for article honesty + Chronos-favorable framing):
  - Metric: UNWEIGHTED per-series RMSSE (not WRMSSE — revenue weights suppress
    new-product signal, hiding exactly where Chronos wins)
  - Chronos signal: pure ZS dept xcl=True assembled (NO tier OLS, NO FT blend)
    — this is what a practitioner deploys on a new product, zero training
  - LGBM: natural failure on short-history series (NaN features are NOT imputed)
  - History = days since first non-zero sale before the cutoff (active history)

Outputs:
  /mnt/lab/nmwamsojo/coldstart_natural_seg_result.json
  (table: per bucket × cutoff × model — mean RMSSE, median RMSSE, % Chronos wins,
   n_series; aggregated across cutoffs as primary reporting metric)

Run:
    cd /home/nmwamsojo/tsfm-explo
    source .venv/bin/activate
    nohup python experiments/10_coldstart_analysis.py \
        >> /mnt/lab/nmwamsojo/coldstart_natural_seg.log 2>&1 &
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

from tsfm_m5 import M5DataPipeline

# ── Config ────────────────────────────────────────────────────────────────────
import os
DATA_PATH  = os.getenv("M5_DATA_PATH", "/mnt/lab/datasets/M5/jointed_M5.parquet")
LAB_DIR    = os.getenv("M5_LAB_DIR",   "/mnt/lab/nmwamsojo")

DATA_TAG   = "sales_only"
FEAT_BASE  = LAB_DIR + "/lgbm_global/features"
CKPT_DIR   = LAB_DIR + "/lgbm_global/checkpoints"
FCST_BASE  = LAB_DIR + "/prepared_data/sales_only/level_12"
RESULT_OUT = LAB_DIR + "/coldstart_natural_seg_result.json"
LEVEL      = 12
HORIZON    = 28

TEST_CUTOFFS = [
    {"date": "2015-10-04", "label": "Autumn", "tag": "20151004"},
    {"date": "2016-01-03", "label": "Winter", "tag": "20160103"},
    {"date": "2016-04-24", "label": "Spring", "tag": "20160424"},
]

# History buckets — boundaries chosen around LGBM feature requirements:
#   lag_28=28d, lag_42=42d, roll_mean_180_l28=208d, full_seasonal=365d
HIST_BINS   = [0, 28, 90, 365, 730, 99999]
HIST_LABELS = ["<28d", "28–90d", "90–365d", "365–730d", "730d+"]

# v20 best trial #140 per-dept CL and tag map
V20_DEPT_CL  = {"FOODS_1":28,"FOODS_2":7,"FOODS_3":28,
                "HOBBIES_1":1,"HOBBIES_2":7,"HOUSEHOLD_1":28,"HOUSEHOLD_2":1}
DEPT_TAG_MAP = {"FOODS_1":"foods1","FOODS_2":"foods2","FOODS_3":"foods3",
                "HOBBIES_1":"hobbies1","HOBBIES_2":"hobbies2",
                "HOUSEHOLD_1":"household1","HOUSEHOLD_2":"household2"}
DEPTS = list(V20_DEPT_CL.keys())

CAT_COLS = ["dept_id","cat_id","store_id","state_id","tier_id"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _read_full(path: str) -> pd.DataFrame:
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


def _strip_suffix(s: pd.Series) -> pd.Series:
    return (s.astype(str)
             .str.replace("_evaluation", "", regex=False)
             .str.replace("_validation", "", regex=False))


# ── Active history computation ─────────────────────────────────────────────────
def _active_history(cutoff_date: str) -> pd.Series:
    """Days since first non-zero sale, per series, as of cutoff_date."""
    log.info("Computing active history lengths ...")
    cut = pd.Timestamp(cutoff_date)
    sales = pd.read_parquet(DATA_PATH, columns=["id", "date", "sold"])
    sales["id"]   = _strip_suffix(sales["id"])
    sales["date"] = pd.to_datetime(sales["date"])
    sales = sales[sales["date"] <= cut]
    first_nz = (sales[sales["sold"] > 0]
                .groupby("id")["date"].min())
    active = ((cut - first_nz).dt.days).clip(lower=0)
    return active   # Series indexed by id


# ── Per-series RMSSE ──────────────────────────────────────────────────────────
def _rmsse_per_series(pred_wide: np.ndarray, actual_wide: np.ndarray,
                      scales: np.ndarray) -> np.ndarray:
    """
    pred_wide, actual_wide: (n_series, 28)
    scales: (n_series,) — naive MSE denominators from weights_scales
    Returns RMSSE per series (n_series,).
    """
    mse = np.mean((pred_wide - actual_wide) ** 2, axis=1)
    return np.sqrt(mse / np.maximum(scales, 1e-8))


# ── Load predictions ──────────────────────────────────────────────────────────
def _lgbm_preds(co: dict, all_ids: list, horizon_dates: list) -> np.ndarray:
    """Load LGBM checkpoint and predict test window → (n, 28)."""
    tag, base = co["tag"], os.path.join(FEAT_BASE, co["tag"])
    with open(os.path.join(base, "meta.json")) as f:
        feat_cols = json.load(f)["feature_cols"]

    bst = lgb.Booster(model_file=os.path.join(CKPT_DIR, f"test_{tag}.txt"))
    test = _read_full(os.path.join(base, "test.parquet"))
    test, _ = _prep(test, feat_cols)
    test["pred"] = np.clip(bst.predict(test[feat_cols]), 0.0, None)
    test["id"]   = _strip_suffix(test["id"])
    test["date"] = pd.to_datetime(test["date"])
    del bst; gc.collect()

    preds = (test[["id", "date", "pred"]]
             .pivot(index="id", columns="date", values="pred")
             .reindex(index=all_ids, columns=horizon_dates)
             .fillna(0.0).values.astype(np.float64))
    del test; gc.collect()
    return preds


def _chronos_preds(co: dict, all_ids: list, horizon_dates: list) -> np.ndarray:
    """Assemble v20 dept ZS xcl=True predictions (NO tier, NO OLS) → (n, 28)."""
    tag   = co["tag"]
    fcst  = os.path.join(FCST_BASE, tag, "models")
    n     = len(all_ids)
    out   = np.zeros((n, HORIZON), dtype=np.float64)
    id_to_idx = {aid: i for i, aid in enumerate(all_ids)}

    for dept in DEPTS:
        dtag  = f"v20d_zs_{DEPT_TAG_MAP[dept]}_dept_m0_cl{V20_DEPT_CL[dept]}_xcl100_eP_sF"
        path  = os.path.join(fcst, dtag, "forecasts.parquet")
        df    = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        piv = (df.pivot(index="id", columns="date", values="sales_quantity")
               .reindex(columns=horizon_dates).fillna(0.0))
        for aid, row in piv.iterrows():
            if aid in id_to_idx:
                out[id_to_idx[aid]] = np.clip(row.values, 0.0, None)

    return out


# ── Per-cutoff analysis ───────────────────────────────────────────────────────
def _run_cutoff(co: dict, pipeline: M5DataPipeline,
                hist_full: pd.Series) -> dict:
    date, label = co["date"], co["label"]
    log.info("[%s] Loading prepared data ...", label)

    hist_df, _, _, static_df, weights_scales = pipeline.get_prepared_data(
        DATA_PATH, date, level=LEVEL, force_reprepare=False)

    all_ids = list(static_df["id"].unique())
    horizon_dates = list(pd.date_range(
        pd.Timestamp(date) + pd.Timedelta(days=1), periods=HORIZON))

    # Scales at level 12 (individual series)
    scales_df = weights_scales[weights_scales["level"] == LEVEL].set_index("id")
    scales    = np.array([scales_df.loc[aid, "scale"] if aid in scales_df.index
                          else 1e-8 for aid in all_ids], dtype=np.float64)

    # Actuals
    horizon_set = set(d.date() for d in horizon_dates)
    act = pd.read_parquet(DATA_PATH, columns=["id", "date", "sold"])
    act["id"]   = _strip_suffix(act["id"])
    act["date"] = pd.to_datetime(act["date"])
    act = act[act["date"].dt.date.isin(horizon_set)]
    actual_wide = (act.pivot(index="id", columns="date", values="sold")
                   .reindex(index=all_ids, columns=horizon_dates)
                   .fillna(0.0).values.astype(np.float64))
    del act, hist_df; gc.collect()

    # Predictions
    log.info("[%s] Loading LGBM predictions ...", label)
    lgbm_wide = _lgbm_preds(co, all_ids, horizon_dates)
    log.info("[%s] Loading Chronos ZS predictions ...", label)
    chron_wide = _chronos_preds(co, all_ids, horizon_dates)

    # Per-series RMSSE
    rmsse_lgbm  = _rmsse_per_series(lgbm_wide,  actual_wide, scales)
    rmsse_chron = _rmsse_per_series(chron_wide, actual_wide, scales)
    del lgbm_wide, chron_wide, actual_wide; gc.collect()

    # Active history at this cutoff
    active = hist_full.reindex(all_ids)    # NaN → never sold → treat as 0
    active = active.fillna(0).astype(int)

    # Bucket
    buckets = pd.cut(active, bins=HIST_BINS, labels=HIST_LABELS, right=True)

    SCALE_THRESHOLD = 1.0   # exclude near-zero-scale series from mean (scale=0 inflates RMSSE)

    rows = []
    for bkt in HIST_LABELS:
        mask = (buckets == bkt).values
        if mask.sum() == 0:
            continue
        rl = rmsse_lgbm[mask]
        rc = rmsse_chron[mask]
        sc = scales[mask]
        chronos_wins = float((rc < rl).mean())

        # Scale-filtered view: exclude near-zero-scale series (they have no meaningful training signal)
        scale_ok = sc >= SCALE_THRESHOLD
        n_scaled = int(scale_ok.sum())
        lgbm_mean_sf   = float(np.mean(rl[scale_ok])) if n_scaled > 0 else float("nan")
        chron_mean_sf  = float(np.mean(rc[scale_ok])) if n_scaled > 0 else float("nan")
        chron_wins_sf  = float((rc[scale_ok] < rl[scale_ok]).mean()) if n_scaled > 0 else float("nan")

        rows.append({
            "cutoff":       date,
            "label":        label,
            "bucket":       bkt,
            "n_series":     int(mask.sum()),
            "n_scale_ok":   n_scaled,
            # Raw mean (scale=0 inflates; unreliable for <28d)
            "lgbm_mean":    float(np.mean(rl)),
            "lgbm_median":  float(np.median(rl)),
            "chronos_mean": float(np.mean(rc)),
            "chronos_median": float(np.median(rc)),
            "chronos_wins_pct": round(chronos_wins * 100, 1),
            "delta_mean":   float(np.mean(rl) - np.mean(rc)),   # >0 means Chronos better
            "delta_median": float(np.median(rl) - np.median(rc)),
            # Scale-filtered mean (primary for <28d bucket)
            "lgbm_mean_sf":    lgbm_mean_sf,
            "chronos_mean_sf": chron_mean_sf,
            "chronos_wins_sf": round(chron_wins_sf * 100, 1) if not np.isnan(chron_wins_sf) else None,
        })
        log.info("[%s] %-10s  n=%-5d (n_ok=%d)  LGBM_med=%.3f  Chronos_med=%.3f  "
                 "LGBM_sf=%.3f  Chronos_sf=%.3f  CW=%.0f%%",
                 label, bkt, mask.sum(), n_scaled,
                 np.median(rl), np.median(rc),
                 lgbm_mean_sf if not np.isnan(lgbm_mean_sf) else -1,
                 chron_mean_sf if not np.isnan(chron_mean_sf) else -1,
                 chronos_wins * 100)

    return rows


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    log.info("=" * 65)
    log.info("Cold-Start Natural Segmentation: LightGBM vs Chronos ZS")
    log.info("Metric: per-series RMSSE (unweighted) grouped by active history")
    log.info("=" * 65)

    pipeline = M5DataPipeline(config={"tag": DATA_TAG})

    # Compute active history once across all series/cutoffs
    # (use earliest test cutoff to get a superset of all history)
    log.info("Computing active history lengths across all series ...")
    hist_by_cutoff = {}
    for co in TEST_CUTOFFS:
        hist_by_cutoff[co["date"]] = _active_history(co["date"])

    all_rows = []
    for co in TEST_CUTOFFS:
        rows = _run_cutoff(co, pipeline, hist_by_cutoff[co["date"]])
        all_rows.extend(rows)
        gc.collect()

    # Aggregate across 3 cutoffs (geo-mean of mean RMSSE per bucket)
    log.info("")
    log.info("=" * 65)
    log.info("AGGREGATED RESULTS (across 3 cutoffs)")
    log.info("Bucket       | n(Aut)  | LGBM   | Chronos | Δ     | Chronos wins%%")
    log.info("-" * 65)

    agg = {}
    for bkt in HIST_LABELS:
        bkt_rows = [r for r in all_rows if r["bucket"] == bkt]
        if not bkt_rows:
            continue
        # Primary: geo-mean of per-cutoff medians (robust to scale=0 outliers)
        lgbm_geo_med   = float(np.exp(np.mean(np.log([r["lgbm_median"]    for r in bkt_rows]))))
        chron_geo_med  = float(np.exp(np.mean(np.log([r["chronos_median"] for r in bkt_rows]))))
        # Secondary: geo-mean of scale-filtered means
        sf_lgbm  = [r["lgbm_mean_sf"]    for r in bkt_rows if not np.isnan(r.get("lgbm_mean_sf", float("nan")))]
        sf_chron = [r["chronos_mean_sf"] for r in bkt_rows if not np.isnan(r.get("chronos_mean_sf", float("nan")))]
        lgbm_geo_sf   = float(np.exp(np.mean(np.log(sf_lgbm))))   if sf_lgbm  else float("nan")
        chron_geo_sf  = float(np.exp(np.mean(np.log(sf_chron))))  if sf_chron else float("nan")
        wins_mean  = float(np.mean([r["chronos_wins_pct"] for r in bkt_rows]))
        n_autumn   = next((r["n_series"] for r in bkt_rows if r["label"]=="Autumn"), 0)
        agg[bkt]   = {
            "lgbm_geo_median": lgbm_geo_med, "chronos_geo_median": chron_geo_med,
            "lgbm_geo_sf": lgbm_geo_sf,      "chronos_geo_sf": chron_geo_sf,
            "chronos_wins_pct": wins_mean,    "n_autumn": n_autumn,
        }
        log.info("%-12s | %-7d | %.3f  | %.3f   | %+.3f | %.0f%%  (sf: %.3f vs %.3f)",
                 bkt, n_autumn, lgbm_geo_med, chron_geo_med, lgbm_geo_med - chron_geo_med, wins_mean,
                 lgbm_geo_sf if not np.isnan(lgbm_geo_sf) else -1,
                 chron_geo_sf if not np.isnan(chron_geo_sf) else -1)

    log.info("=" * 65)

    out = {"per_cutoff": all_rows, "aggregated": agg,
           "note": ("Chronos = pure ZS dept xcl=True, no tier OLS, no FT. "
                    "LGBM = global model from 19-trial HPO. "
                    "RMSSE unweighted per series.")}
    with open(RESULT_OUT, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Saved → %s", RESULT_OUT)


if __name__ == "__main__":
    main()
