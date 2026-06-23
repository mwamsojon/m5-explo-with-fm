#!/usr/bin/env python3
"""
06_lgbm_build_features.py
============================
Build the tabular feature matrix for the global LightGBM M5 model.

Compute strategy
----------------
All heavy computation (lag, rolling, price, group aggregates) is done in
Polars, which parallelises across all available CPUs via its Rust/Rayon
thread-pool. On 20 CPUs this is ~15-20× faster than single-threaded pandas
and uses ~40% less RAM (Arrow columnar format vs NumPy object arrays).

The result is converted to pandas only for the per-cutoff split / mean-
encoding step, which is lightweight (a few groupby aggregations on a
fraction of the rows).

Feature design — zero-leakage for direct 28-day forecast
----------------------------------------------------------
M5 horizon = 28 and every lag/rolling feature is anchored at lag ≥ 28, so
all features are observable at inference time without any recursion.

"roll_mean_{W}_l28 at day d" = mean(sales[d-28-W : d-28])
Implemented as: shift sales by 28 first, then rolling(W) on that shifted col.
At d = cutoff+h (h ≤ 28) the window ends at cutoff+h-28 ≤ cutoff. ✓

Feature groups
--------------
  Lag              lag_28..35, lag_42   (all ≥ 28 → safe for direct forecast)
  Rolling mean     W ∈ {7,14,30,60,180} (l28-anchored)
  Rolling std      W ∈ {7,28}           (l28-anchored)
  Price            log_price, price_norm, price_mom_7d/28d  (ffill-corrected)
  Calendar         wday, month, year, week_of_year, snap_CA/TX/WI
  Events           event_type_1/2 as int codes
  IDs              item_id, dept_id, cat_id, store_id, state_id (category)
  Tier             Low/Med-Low/Med-High/High by avg-sales quartile (like v25)
  Group agg        tier_l28_mean, dept_store_l28_mean (cross-series signal)
  Mean encoding    enc_store_item_mean, enc_store_dept_mean,
                   enc_store_dept_wday_mean (fit on train split, zero-leakage)

Cache layout
------------
  FEAT_BASE/{cutoff_yyyymmdd}/
    train.parquet   rows d ∈ [FIRST_DAY, cutoff_dint - VAL_DAYS]
    val.parquet     rows d ∈ (cutoff_dint - VAL_DAYS, cutoff_dint]
    test.parquet    rows d ∈ [cutoff_dint+1, cutoff_dint+HORIZON] (no target)
    meta.json       feature list, row counts, build timestamp, git_hash

LightGBM target: "sales" (raw counts, use tweedie objective).

Quickstart
----------
  python experiments/06_lgbm_build_features.py           # build all 6 cutoffs
  python experiments/06_lgbm_build_features.py --cutoffs 2014-10-04 --force
  python experiments/06_lgbm_build_features.py --status  # cache summary (no data load)
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq

# ── Paths ──────────────────────────────────────────────────────────────────────
import os
DATA_PATH = os.getenv("M5_DATA_PATH", "/mnt/lab/datasets/M5/jointed_M5.parquet")
LAB_DIR   = os.getenv("M5_LAB_DIR",   "/mnt/lab/nmwamsojo")

FEAT_BASE = LAB_DIR + "/lgbm_global/features"
LOG_PATH  = LAB_DIR + "/lgbm_global/feature_build.log"

# ── Cutoffs: same zero-leakage split used for Chronos v25 ─────────────────────
CALIB_CUTOFFS = ["2014-10-04", "2015-01-03", "2015-04-25"]
TEST_CUTOFFS  = ["2015-10-04", "2016-01-03", "2016-04-24"]
ALL_CUTOFFS   = CALIB_CUTOFFS + TEST_CUTOFFS

# ── Feature constants ──────────────────────────────────────────────────────────
HORIZON       = 28
LAG_COLS      = [28, 29, 30, 31, 35, 42]   # all ≥ 28 → safe for direct forecast
ROLL_WINS     = [7, 14, 30, 60, 180]        # rolling mean window sizes (l28-anchored)
ROLL_STD_WINS = [7, 28]                     # rolling std window sizes  (l28-anchored)

# First valid training day: deepest feature needs 28 (anchor) + 180 (rolling) + 42 (lag) = 250.
# Use 300 for a comfortable buffer; only removes ~1.5% of training rows.
FIRST_DAY = 300

# Days before cutoff reserved for validation (one forecast window).
VAL_DAYS = 28

# Tier labels — quartile bins on avg daily sales, matching v25 WRMSSE-weight tiers.
TIER_LABELS = ["Low", "Med-Low", "Med-High", "High"]

# ID columns that LightGBM treats as categoricals (enables histogram splits on cats).
CAT_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "tier_id"]

# Ordered list of feature columns (order is reproducible across runs).
FEATURE_COLS: list[str] = (
    [f"lag_{l}"            for l in LAG_COLS]
  + [f"roll_mean_{w}_l28"  for w in ROLL_WINS]
  + [f"roll_std_{w}_l28"   for w in ROLL_STD_WINS]
  + ["log_price", "price_norm", "price_mom_7", "price_mom_28"]
  + ["wday", "month", "year", "week_of_year",
     "snap_CA", "snap_TX", "snap_WI",
     "event_type_1", "event_type_2"]
  + CAT_COLS
  + ["tier_l28_mean", "dept_store_l28_mean"]
  + ["enc_store_item_mean", "enc_store_dept_mean", "enc_store_dept_wday_mean"]
)

# ── Logging ────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
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


# ══════════════════════════════════════════════════════════════════════════════
# Utility: RAM usage
# ══════════════════════════════════════════════════════════════════════════════

def _ram_gb() -> float:
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                info[k.strip()] = int(v.strip().split()[0])
    return (info.get("MemTotal", 0) - info.get("MemAvailable", 0)) / 1024 / 1024


# ══════════════════════════════════════════════════════════════════════════════
# Core: Polars feature computation (parallel, multi-CPU)
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_grid() -> pl.DataFrame:
    """
    Load the full M5 parquet and compute all time-independent features.

    Uses Polars throughout so the Rust/Rayon thread-pool saturates all 20 CPUs.
    Returns a Polars DataFrame — caller converts to pandas only for the
    lightweight per-cutoff split step.

    Memory estimate: ~5-7 GB for the final feature grid (Arrow format).
    Peak during computation: ~12-14 GB.
    """
    t0 = time.time()
    log.info("=" * 65)
    log.info("Step 1/5: Loading %s ...", DATA_PATH)

    # ── Load ─────────────────────────────────────────────────────────────────
    df = pl.read_parquet(DATA_PATH)
    df = df.with_columns([
        pl.col("d").str.slice(2).cast(pl.Int32).alias("d_int"),
        pl.col("date").cast(pl.Date),
        pl.col("sold").cast(pl.Float32).alias("sales"),
    ])
    log.info("  %s rows  %d series  %.0fs  RAM ~%.1f GB",
             f"{len(df):,}", df["id"].n_unique(), time.time()-t0, _ram_gb())

    # ── Forward-fill sell_price within each series ────────────────────────────
    # ~21% of rows have null price (items not yet listed or discontinued).
    # Sort once by (id, d_int); all subsequent ops rely on this order.
    log.info("Step 2/5: Price fill + lag/rolling features (all CPUs) ...")
    t1 = time.time()

    df = df.sort(["id", "d_int"]).with_columns(
        pl.col("sell_price").forward_fill().backward_fill().over("id")
    )

    # ── Intermediate: sales shifted by 28 for the l28-anchor trick ───────────
    # shift_28[d] = sales[d-28]  →  rolling over shift_28 gives l28-anchored stats.
    # Computed once, reused for all rolling features, then dropped.
    df = df.with_columns(
        pl.col("sales").shift(28).over("id").alias("_s28")
    )

    # ── Lag features ──────────────────────────────────────────────────────────
    lag_exprs = [
        pl.col("sales").shift(lag).over("id").alias(f"lag_{lag}")
        for lag in LAG_COLS
    ]

    # ── Rolling mean (l28-anchored): rolling(W) applied to _s28 ──────────────
    # At row d: mean(_s28[d-W+1 : d+1]) = mean(sales[d-28-W+1 : d-28+1])
    roll_mean_exprs = [
        pl.col("_s28").rolling_mean(window_size=w, min_samples=1).over("id")
          .alias(f"roll_mean_{w}_l28")
        for w in ROLL_WINS
    ]

    # ── Rolling std (l28-anchored) ────────────────────────────────────────────
    roll_std_exprs = [
        pl.col("_s28").rolling_std(window_size=w, min_samples=1, ddof=0).over("id")
          .alias(f"roll_std_{w}_l28")
        for w in ROLL_STD_WINS
    ]

    # Single with_columns call → Polars executes all in one parallel pass
    df = df.with_columns(lag_exprs + roll_mean_exprs + roll_std_exprs).drop("_s28")
    log.info("  Lag/rolling done in %.0fs  RAM ~%.1f GB", time.time()-t1, _ram_gb())

    # ── Price features ────────────────────────────────────────────────────────
    log.info("Step 3/5: Price + calendar + tier features ...")
    t2 = time.time()

    item_max = df.group_by("item_id").agg(
        pl.col("sell_price").max().alias("item_price_max")
    )
    df = df.join(item_max, on="item_id", how="left")

    p7  = pl.col("sell_price").shift(7).over("id")
    p28 = pl.col("sell_price").shift(28).over("id")

    df = df.with_columns([
        pl.col("sell_price").log1p().alias("log_price"),
        (pl.col("sell_price") / pl.col("item_price_max").replace(0.0, 1.0)).alias("price_norm"),
        ((pl.col("sell_price") / pl.when(p7  == 0.0).then(pl.lit(None)).otherwise(p7)  - 1.0).fill_null(0.0)).alias("price_mom_7"),
        ((pl.col("sell_price") / pl.when(p28 == 0.0).then(pl.lit(None)).otherwise(p28) - 1.0).fill_null(0.0)).alias("price_mom_28"),
    ]).drop("item_price_max")

    # ── Calendar features ─────────────────────────────────────────────────────
    df = df.with_columns(
        pl.col("date").dt.week().cast(pl.Int8).alias("week_of_year"),
    )

    df = df.with_columns([
        # Keep event types as Categorical — pandas .astype("category") preserves this
        # for LightGBM. Physical u32 codes can't be narrowed to Int8 (global pool IDs).
        pl.col("event_type_1").fill_null("none").cast(pl.Categorical),
        pl.col("event_type_2").fill_null("none").cast(pl.Categorical),
        pl.col("snap_CA").cast(pl.Int8),
        pl.col("snap_TX").cast(pl.Int8),
        pl.col("snap_WI").cast(pl.Int8),
        pl.col("wday").cast(pl.Int8),
        pl.col("month").cast(pl.Int8),
        pl.col("year").cast(pl.Int16),
    ])

    # ── Tier assignment ───────────────────────────────────────────────────────
    # Quartile bin by average daily sales — proxy for WRMSSE weight, same as v25.
    avg_s = (
        df.group_by("id")
          .agg(pl.col("sales").mean().alias("avg_sales"))
    )
    q25, q75 = (
        avg_s["avg_sales"].quantile(0.25),
        avg_s["avg_sales"].quantile(0.75),
    )
    q50 = avg_s["avg_sales"].quantile(0.50)

    avg_s = avg_s.with_columns(
        pl.when(pl.col("avg_sales") <= q25).then(pl.lit("Low"))
          .when(pl.col("avg_sales") <= q50).then(pl.lit("Med-Low"))
          .when(pl.col("avg_sales") <= q75).then(pl.lit("Med-High"))
          .otherwise(pl.lit("High"))
          .alias("tier_id")
    ).drop("avg_sales")

    df = df.join(avg_s, on="id", how="left")
    log.info("  Price/calendar/tier done in %.0fs  RAM ~%.1f GB",
             time.time()-t2, _ram_gb())

    # ── Group aggregate features (l28-anchored, parallel groupby) ─────────────
    # tier_l28_mean:       mean of lag_28 across all series in the same tier on each day.
    # dept_store_l28_mean: mean of lag_28 within each (dept, store) pair on each day.
    # Both serve as the tabular cross-learning signal: how is the group trending
    # 28 days ago? Fully safe (lag_28 ≤ cutoff for any test row).
    log.info("Step 4/5: Group aggregate features ...")
    t3 = time.time()

    df = df.with_columns([
        pl.col("lag_28").mean().over(["tier_id", "d_int"]).alias("tier_l28_mean"),
        pl.col("lag_28").mean().over(["dept_id", "store_id", "d_int"])
          .alias("dept_store_l28_mean"),
    ])
    log.info("  Group agg done in %.0fs  RAM ~%.1f GB", time.time()-t3, _ram_gb())

    # ── Cast ID columns to Categorical ───────────────────────────────────────
    id_cast = {c: pl.Categorical for c in ["item_id","dept_id","cat_id",
                                            "store_id","state_id","tier_id"]
               if c in df.columns}
    if id_cast:
        df = df.with_columns([pl.col(c).cast(dt) for c, dt in id_cast.items()])

    log.info("Full feature grid: %s  RAM ~%.1f GB  total %.0fs",
             str(df.shape), _ram_gb(), time.time()-t0)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Per-cutoff: split + mean encodings + save  (all in Polars — no pandas copy)
# ══════════════════════════════════════════════════════════════════════════════

def build_and_save_cutoff(
    df_pl:  pl.DataFrame,
    cutoff: str,
    force:  bool = False,
) -> None:
    """
    Build and cache train / val / test parquets for one cutoff.

    Everything stays in Polars (Arrow memory): mean encodings via group_by+join,
    parquet written with write_parquet(). This avoids the ~25 GB RAM spike that
    to_pandas() causes when converting 30-50M rows × 45 cols simultaneously.

    Splits
    ------
    train   d_int ∈ [FIRST_DAY,  cutoff_dint - VAL_DAYS]   → LightGBM training
    val     d_int ∈ (cutoff_dint - VAL_DAYS, cutoff_dint]  → early-stopping
    test    d_int ∈ (cutoff_dint, cutoff_dint + HORIZON]   → inference (no target)
    """
    tag     = cutoff.replace("-", "")
    out_dir = os.path.join(FEAT_BASE, tag)
    os.makedirs(out_dir, exist_ok=True)

    paths     = {s: os.path.join(out_dir, f"{s}.parquet")
                 for s in ("train", "val", "test")}
    meta_path = os.path.join(out_dir, "meta.json")

    if not force and all(os.path.exists(p) for p in paths.values()):
        log.info("  [%s] CACHED — skipping (--force to rebuild)", cutoff)
        with open(meta_path) as fh:
            m = json.load(fh)
        log.info("    train=%s  val=%s  test=%s  feats=%d",
                 f"{m['n_train']:,}", f"{m['n_val']:,}",
                 f"{m['n_test']:,}", m["n_features"])
        return

    log.info("  [%s] Splitting + mean encodings ...", cutoff)
    t0 = time.time()

    # ── Resolve cutoff day index ──────────────────────────────────────────────
    c_dint = int(
        df_pl.filter(pl.col("date") == pl.lit(cutoff).str.to_date())["d_int"][0]
    )

    # ── Narrow the Polars frame to the relevant window (Arrow view, near-zero cost)
    sub_pl = df_pl.filter(
        pl.col("d_int").is_between(FIRST_DAY, c_dint + HORIZON)
    )

    # ── Split masks (as Polars boolean series) ────────────────────────────────
    train_pl = sub_pl.filter(pl.col("d_int").is_between(FIRST_DAY, c_dint - VAL_DAYS))
    val_pl   = sub_pl.filter(pl.col("d_int").is_between(c_dint - VAL_DAYS + 1, c_dint))
    test_pl  = sub_pl.filter(pl.col("d_int").is_between(c_dint + 1, c_dint + HORIZON))

    # ── Mean encodings (Polars group_by+join; fit on train only) ─────────────
    # Joining back to the full sub_pl ensures val and test rows get train-period means.
    enc_specs = [
        (["store_id", "item_id"],         "enc_store_item_mean"),
        (["store_id", "dept_id"],         "enc_store_dept_mean"),
        (["store_id", "dept_id", "wday"], "enc_store_dept_wday_mean"),
    ]
    for keys, col_name in enc_specs:
        enc = (
            train_pl
            .group_by(keys)
            .agg(pl.col("sales").mean().cast(pl.Float32).alias(col_name))
        )
        sub_pl   = sub_pl.join(enc,   on=keys, how="left")
        train_pl = train_pl.join(enc, on=keys, how="left")
        val_pl   = val_pl.join(enc,   on=keys, how="left")
        test_pl  = test_pl.join(enc,  on=keys, how="left")

    # ── Feature column list ───────────────────────────────────────────────────
    feat_present = [c for c in FEATURE_COLS if c in sub_pl.columns]
    keep_base    = ["id", "date", "d_int", "sales"]
    keep_all     = keep_base + feat_present
    keep_test    = [c for c in keep_all if c != "sales"]

    # ── Drop rows where all lag features are null (insufficient series history)
    lag_cols = [c for c in feat_present if c.startswith("lag_")]
    null_expr = pl.all_horizontal([pl.col(c).is_null() for c in lag_cols])
    train_pl = train_pl.select(keep_all).filter(~null_expr)
    val_pl   = val_pl.select(keep_all).filter(~null_expr)
    test_pl  = test_pl.select(keep_test)

    # ── Write directly from Polars (no to_pandas — saves ~25 GB peak RAM) ────
    train_pl.write_parquet(paths["train"])
    val_pl.write_parquet(  paths["val"])
    test_pl.write_parquet( paths["test"])

    n_train, n_val, n_test = len(train_pl), len(val_pl), len(test_pl)

    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_hash = "unknown"

    meta = {
        "cutoff":       cutoff,
        "cutoff_dint":  c_dint,
        "feature_cols": feat_present,
        "n_features":   len(feat_present),
        "n_train":      n_train,
        "n_val":        n_val,
        "n_test":       n_test,
        "first_day":    FIRST_DAY,
        "val_days":     VAL_DAYS,
        "horizon":      HORIZON,
        "lag_cols":     LAG_COLS,
        "roll_wins":    ROLL_WINS,
        "built_at":     datetime.utcnow().isoformat() + "Z",
        "git_hash":     git_hash,
    }
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    log.info("  [%s] train=%s  val=%s  test=%s  (%.0fs)",
             cutoff, f"{n_train:,}", f"{n_val:,}", f"{n_test:,}", time.time()-t0)

    del sub_pl, train_pl, val_pl, test_pl
    gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# Status report (no data loading)
# ══════════════════════════════════════════════════════════════════════════════

def print_status() -> None:
    """Summarise cache state for all cutoffs without loading any data."""
    hdr = f"\n{'Type':<7} {'Cutoff':<13} {'Train':>12} {'Val':>10} {'Test':>10}  {'Built':<20}  Feats"
    print(hdr)
    print("-" * 80)
    for cutoff in ALL_CUTOFFS:
        tag     = cutoff.replace("-", "")
        label   = "CALIB" if cutoff in CALIB_CUTOFFS else "TEST "
        mp      = os.path.join(FEAT_BASE, tag, "meta.json")
        if os.path.exists(mp):
            with open(mp) as fh:
                m = json.load(fh)
            print(f"[{label}] {cutoff}  {m['n_train']:>12,}  {m['n_val']:>10,}  "
                  f"{m['n_test']:>10,}  {m['built_at'][:19]}  {m['n_features']}")
        else:
            print(f"[{label}] {cutoff}  {'—':>12}  {'—':>10}  {'—':>10}  NOT BUILT")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(cutoffs: list[str] | None = None, force: bool = False) -> None:
    if cutoffs is None:
        cutoffs = ALL_CUTOFFS

    t0 = time.time()
    log.info("=" * 65)
    log.info("LightGBM Global — Feature Builder  (Polars %s)", pl.__version__)
    log.info("  Cutoffs : %s", cutoffs)
    log.info("  Output  : %s", FEAT_BASE)
    log.info("  CPUs    : %d  (Polars thread-pool)", os.cpu_count() or 1)
    log.info("=" * 65)

    # ── Phase 1: build the full parallel feature grid once ────────────────────
    df_pl = build_feature_grid()

    # ── Phase 2: per-cutoff split, mean-encode, and save ─────────────────────
    log.info("Step 5/5: Per-cutoff splits ...")
    for cutoff in cutoffs:
        build_and_save_cutoff(df_pl, cutoff, force=force)

    del df_pl
    gc.collect()

    log.info("")
    log.info("All done in %.1f min  RAM ~%.1f GB", (time.time()-t0)/60, _ram_gb())
    print_status()


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Build LightGBM M5 feature matrix (Polars, all CPUs)."
    )
    p.add_argument("--cutoffs", nargs="+", default=None, metavar="YYYY-MM-DD",
                   help="Specific cutoffs to build (default: all 6).")
    p.add_argument("--force",   action="store_true",
                   help="Rebuild even if cache already exists.")
    p.add_argument("--status",  action="store_true",
                   help="Print cache status and exit (no data load).")
    args = p.parse_args()

    if args.status:
        print_status()
    else:
        main(cutoffs=args.cutoffs, force=args.force)
