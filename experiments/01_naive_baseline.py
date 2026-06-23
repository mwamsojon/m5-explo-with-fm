#!/usr/bin/env python3
"""
01_naive_baseline.py
=====================
Compute the M5 seasonal-naive WRMSSE for all three evaluation cutoffs
(autumn=2015-10-04, winter=2016-01-03, spring=2016-04-24) sequentially.

Seasonal naive: repeat the last 7 observed values for each series
(weekly cycle), matching R's Point_Forecasts_-_Benchmarks.R implementation.

Runs sequentially (not parallel) to avoid OOM — each cutoff loads the full
59M-row M5 parquet (~20 GB); three simultaneous loads would exceed 60 GB.

Results written to /mnt/lab/nmwamsojo/naive_baseline_result.json
"""

import gc
import json
import logging
import math
import os
import sys
import time

import numpy as np
import pandas as pd

from tsfm_m5 import M5BenchmarkSuite
from tsfm_m5 import M5DataPipeline
from tsfm_m5 import M5Evaluator

import os
DATA_PATH   = os.getenv("M5_DATA_PATH", "/mnt/lab/datasets/M5/jointed_M5.parquet")
LAB_DIR     = os.getenv("M5_LAB_DIR",   "/mnt/lab/nmwamsojo")

DATA_TAG    = "sales_only"
HORIZON     = 28
LEVEL       = 12
RESULT_PATH = LAB_DIR + "/naive_baseline_result.json"
LOG_PATH    = LAB_DIR + "/naive_baseline.log"

CUTOFFS = [
    {"label": "spring", "date": "2016-04-24"},
    {"label": "winter", "date": "2016-01-03"},
    {"label": "autumn", "date": "2015-10-04"},
]

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


def main() -> None:
    log.info("=" * 60)
    log.info("Seasonal Naive Baseline — 3 cutoffs (sequential)")
    log.info("  Cutoffs: %s", [c["date"] for c in CUTOFFS])
    log.info("=" * 60)

    # Load full actuals once — evaluator filters internally to the forecast window
    log.info("Loading full actuals from jointed_M5.parquet ...")
    df_actual = (
        pd.read_parquet(DATA_PATH, columns=["id", "date", "sold"])
        .rename(columns={"sold": "sales_quantity"})
    )
    df_actual["id"] = (
        df_actual["id"]
        .str.replace("_evaluation", "", regex=False)
        .str.replace("_validation", "", regex=False)
    )
    log.info("  Actuals loaded: %d rows", len(df_actual))

    pipeline = M5DataPipeline(config={"tag": DATA_TAG})
    results: dict[str, dict] = {}

    for c in CUTOFFS:
        label, date = c["label"], c["date"]
        log.info("[%s] %s starting ...", label, date)
        t0 = time.time()

        hist_df, hist_trimmed, _, static_df, wts = pipeline.get_prepared_data(
            DATA_PATH, date, level=LEVEL, force_reprepare=False,
        )

        evaluator = M5Evaluator(
            raw_train_df=hist_df, trimmed_train_df=hist_trimmed,
            static_df=static_df, weights_df=wts,
            target_col="sales_quantity", price_col="sell_price",
        )
        del hist_df

        # Seasonal naive: tile last-7-day pattern (R benchmark)
        suite   = M5BenchmarkSuite(horizon=HORIZON)
        fcst_df = suite.run(train_df=hist_trimmed, methods=["Naive"])["Naive"]
        del hist_trimmed

        metrics = evaluator.evaluate_all(fcst_df, df_actual)
        del evaluator, fcst_df
        gc.collect()

        wrmsse  = float(metrics["WRMSSE"])
        elapsed = time.time() - t0

        log.info("  [%s]  WRMSSE=%.4f  (%.0fs)", label, wrmsse, elapsed)
        results[label] = {"label": label, "date": date, "wrmsse": wrmsse}

    scores = [results[c["label"]]["wrmsse"] for c in CUTOFFS]
    geo    = math.exp(sum(math.log(v) for v in scores) / len(scores))

    log.info("")
    log.info("=" * 60)
    log.info("RESULTS — Seasonal Naive")
    log.info("  Spring (2016-04-24): %.4f", results["spring"]["wrmsse"])
    log.info("  Winter (2016-01-03): %.4f", results["winter"]["wrmsse"])
    log.info("  Autumn (2015-10-04): %.4f", results["autumn"]["wrmsse"])
    log.info("  Geo-mean           : %.4f", geo)
    log.info("=" * 60)

    out = {
        "method":      "seasonal_naive",
        "description": "Repeat last 7 observed values per series (M5 R benchmark)",
        "spring":      results["spring"]["wrmsse"],
        "winter":      results["winter"]["wrmsse"],
        "autumn":      results["autumn"]["wrmsse"],
        "geo":         geo,
    }
    with open(RESULT_PATH, "w") as fh:
        json.dump(out, fh, indent=2)
    log.info("Saved → %s", RESULT_PATH)


if __name__ == "__main__":
    main()
