#!/usr/bin/env python3
"""
00_run_benchmarks.py
====================
Run all M5 competition baselines (Naive, Seasonal Naive, SES, MA,
AutoGluon ETS-BU, AutoGluon ARIMA-BU) across the three evaluation cutoffs
and write per-method per-cutoff WRMSSE results to disk.

These are the comparison benchmarks referenced in Section 13 of the article.
They exactly reproduce Point_Forecasts_-_Benchmarks.R from the M5 competition.

Output
------
  $M5_LAB_DIR/m5_benchmarks_result.json
    {
      "naive":   {"autumn": 2.08, "winter": 1.42, "spring": 1.46, "geo": 1.63},
      "snaive":  {...},
      "ses":     {...},
      "ma":      {...},
      "ets_bu":  {...},
      "arima_bu":{...},
    }

Run
---
  # Quick (pure-NumPy methods only, no AutoGluon):
  python experiments/00_run_benchmarks.py --fast

  # Full (all methods including AutoGluon ETS / ARIMA):
  python experiments/00_run_benchmarks.py

  # Single cutoff:
  python experiments/00_run_benchmarks.py --cutoff 2015-10-04
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

from tsfm_m5 import M5BenchmarkSuite, M5DataPipeline, M5Evaluator

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = os.getenv("M5_DATA_PATH", "/mnt/lab/datasets/M5/jointed_M5.parquet")
LAB_DIR   = os.getenv("M5_LAB_DIR",   "/mnt/lab/nmwamsojo")

LEVEL     = 12
HORIZON   = 28
DATA_TAG  = "sales_only"
RESULT_PATH = os.path.join(LAB_DIR, "m5_benchmarks_result.json")
LOG_PATH    = os.path.join(LAB_DIR, "m5_benchmarks.log")

CUTOFFS = [
    {"label": "autumn", "date": "2015-10-04"},
    {"label": "winter", "date": "2016-01-03"},
    {"label": "spring", "date": "2016-04-24"},
]

# Methods included in --fast mode (no AutoGluon dependency)
FAST_METHODS = ["naive", "snaive", "ses", "ma"]
ALL_METHODS  = FAST_METHODS + ["ets_bu", "arima_bu"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def run_cutoff(date: str, label: str, methods: list[str]) -> dict:
    pipeline = M5DataPipeline(config={"tag": DATA_TAG})
    hist_df, hist_trimmed, future_df, static_df, weights_df = (
        pipeline.get_prepared_data(DATA_PATH, date, level=LEVEL)
    )

    evaluator = M5Evaluator(
        raw_train_df     = hist_df,
        trimmed_train_df = hist_trimmed,
        static_df        = static_df,
        weights_df       = weights_df,
    )

    suite = M5BenchmarkSuite(
        hist_df          = hist_trimmed,
        future_df        = future_df,
        static_df        = static_df,
        evaluator        = evaluator,
        horizon          = HORIZON,
        target_col       = "sales_quantity",
    )

    results = {}
    for method in methods:
        log.info(f"  [{label}] running {method} …")
        t0 = time.perf_counter()
        score = suite.run(method)
        elapsed = time.perf_counter() - t0
        log.info(f"  [{label}] {method}: WRMSSE={score:.4f}  ({elapsed:.1f}s)")
        results[method] = round(score, 6)

    del hist_df, hist_trimmed, future_df, static_df, weights_df, evaluator, suite
    gc.collect()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast",   action="store_true", help="Skip AutoGluon methods")
    parser.add_argument("--cutoff", default=None,        help="Single cutoff date (YYYY-MM-DD)")
    args = parser.parse_args()

    methods = FAST_METHODS if args.fast else ALL_METHODS
    cutoffs = [c for c in CUTOFFS if args.cutoff is None or c["date"] == args.cutoff]

    log.info(f"Methods : {methods}")
    log.info(f"Cutoffs : {[c['label'] for c in cutoffs]}")

    all_results: dict[str, dict] = {m: {} for m in methods}

    for cutoff in cutoffs:
        log.info(f"\n{'='*55}\nCutoff: {cutoff['label']} ({cutoff['date']})\n{'='*55}")
        cutoff_results = run_cutoff(cutoff["date"], cutoff["label"], methods)
        for method, score in cutoff_results.items():
            all_results[method][cutoff["label"]] = score

    # Add geo-mean across cutoffs
    for method in methods:
        scores = list(all_results[method].values())
        if scores:
            all_results[method]["geo"] = round(float(np.prod(scores) ** (1 / len(scores))), 6)

    log.info("\n── Final Results ──────────────────────────────────")
    for method, res in all_results.items():
        log.info(f"  {method:10s}: {res}")

    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nSaved → {RESULT_PATH}")


if __name__ == "__main__":
    main()
