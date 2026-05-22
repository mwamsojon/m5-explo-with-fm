"""
Reproduce the original IndexError.
Run two stores on the SAME suite WITHOUT clearing the TSDF cache between them,
mimicking the pre-fix behavior where store_future was unfiltered.
"""
import sys
sys.path.insert(0, "/home/nmwamsojo/tsfm-explo/src/jobs/")

import gc, traceback
import torch
import pandas as pd

from m5_exploration import M5ExplorationSuite, DEFAULT_CHRONOS_CONFIG

DATA_ROOT  = "/mnt/lab/nmwamsojo/prepared_data/sales_only/level_12/20160424"
DATA_TAG   = "sales_only"
CUTOFF_DAY = "2016-04-24"

print("Loading cached parquets...")
hist_df_trimmed = pd.read_parquet(f"{DATA_ROOT}/hist_trimmed.parquet")
future_df       = pd.read_parquet(f"{DATA_ROOT}/future.parquet")
static_df       = pd.read_parquet(f"{DATA_ROOT}/static.parquet")

STORES = ["CA_1", "CA_2"]
STORE_IDS = {
    store: set(static_df[static_df["store_id"] == store]["id"].tolist())
    for store in STORES
}

device = "cuda:0" if torch.cuda.is_available() else "cpu"
suite = M5ExplorationSuite(
    horizon  = 28,
    ag_path  = "/mnt/lab/nmwamsojo/autogluon_models/debug_crash2",
    base_dir = "/mnt/lab/nmwamsojo/prepared_data",
)

WRAPPER = {
    "eval_metric": "RMSSE", "enable_ensemble": False,
    "skip_model_selection": True, "verbosity": 1,
}

cov_type = "price"
COV_COLS = ["sell_price"]

print("\n=== SCENARIO A: pre-fix (unfiltered future, no cache eviction) ===\n")
for store_id in STORES:
    store_items  = STORE_IDS[store_id]
    store_hist   = hist_df_trimmed[hist_df_trimmed["id"].isin(store_items)]
    # BUG: use FULL future_df (all 30490 items) — pre-fix behaviour
    store_future = future_df   # <-- unfiltered!

    exp_cfg = {
        **DEFAULT_CHRONOS_CONFIG,
        "use_static": False, "known_cov_cols": COV_COLS,
        "device": device, "batch_size": 64,
        "context_length": 16, "fine_tune_steps": 0,
    }
    store_tag = f"dbg2_A_{store_id.lower()}_cl16_{cov_type}"
    print(f"[{store_id}] hist={store_hist.shape}  future={store_future.shape}")
    # NO cache eviction
    try:
        fcst = suite.run(
            hist_df=store_hist, future_df=store_future, static_df=None,
            model="Chronos2", exp_config=exp_cfg, exp_tag=store_tag,
            data_tag=DATA_TAG, cutoff_day=CUTOFF_DAY,
            wrapper_dict=WRAPPER, force_run=True,
        )
        print(f"  SUCCESS: {fcst.shape}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
    gc.collect()
    torch.cuda.empty_cache()

print("\n=== SCENARIO B: filtered future but NO cache eviction ===\n")
suite2 = M5ExplorationSuite(
    horizon=28,
    ag_path="/mnt/lab/nmwamsojo/autogluon_models/debug_crash2b",
    base_dir="/mnt/lab/nmwamsojo/prepared_data",
)
for store_id in STORES:
    store_items  = STORE_IDS[store_id]
    store_hist   = hist_df_trimmed[hist_df_trimmed["id"].isin(store_items)]
    store_future = future_df[future_df["id"].isin(store_items)]  # filtered
    # BUT no cache eviction between stores

    exp_cfg = {
        **DEFAULT_CHRONOS_CONFIG,
        "use_static": False, "known_cov_cols": COV_COLS,
        "device": device, "batch_size": 64,
        "context_length": 16, "fine_tune_steps": 0,
    }
    store_tag = f"dbg2_B_{store_id.lower()}_cl16_{cov_type}"
    print(f"[{store_id}] hist={store_hist.shape}  future={store_future.shape}")
    try:
        fcst = suite2.run(
            hist_df=store_hist, future_df=store_future, static_df=None,
            model="Chronos2", exp_config=exp_cfg, exp_tag=store_tag,
            data_tag=DATA_TAG, cutoff_day=CUTOFF_DAY,
            wrapper_dict=WRAPPER, force_run=True,
        )
        print(f"  SUCCESS: {fcst.shape}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
    gc.collect()
    torch.cuda.empty_cache()

print("\nDone.")
