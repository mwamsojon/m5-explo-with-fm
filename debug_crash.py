"""Minimal reproducer for IndexError in df_utils.py — tests parallel execution."""
import sys, os, contextlib, threading, traceback
sys.path.insert(0, "/home/nmwamsojo/tsfm-explo/src/jobs/")

import gc
import torch
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from m5_exploration import M5ExplorationSuite, DEFAULT_CHRONOS_CONFIG

DATA_ROOT  = "/mnt/lab/nmwamsojo/prepared_data/sales_only/level_12/20160424"
DATA_TAG   = "sales_only"
CUTOFF_DAY = "2016-04-24"

print("Loading cached parquets...")
hist_df_trimmed = pd.read_parquet(f"{DATA_ROOT}/hist_trimmed.parquet")
future_df       = pd.read_parquet(f"{DATA_ROOT}/future.parquet")
static_df       = pd.read_parquet(f"{DATA_ROOT}/static.parquet")
print(f"  hist_trimmed : {hist_df_trimmed.shape}")
print(f"  future_df    : {future_df.shape}")

STORES = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]
STORE_IDS = {
    store: set(static_df[static_df["store_id"] == store]["id"].tolist())
    for store in STORES
}

import torch
DEVICES = [f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"]
N_GPUS = len(DEVICES)
print(f"Devices: {DEVICES}")

STORE_DEVICE_MAP = {store: DEVICES[i % N_GPUS] for i, store in enumerate(STORES)}

suites = {
    device: M5ExplorationSuite(
        horizon  = 28,
        ag_path  = "/mnt/lab/nmwamsojo/autogluon_models/debug_crash_parallel",
        base_dir = "/mnt/lab/nmwamsojo/prepared_data",
    )
    for device in DEVICES
}

WRAPPER = {
    "eval_metric":          "RMSSE",
    "enable_ensemble":      False,
    "skip_model_selection": True,
    "verbosity":            1,
}

_model_load_lock = threading.Lock()


def _run_one_store(store_id, device):
    device_idx = int(device.split(":")[-1]) if ":" in device else 0
    torch.cuda.set_device(device_idx)

    store_items  = STORE_IDS[store_id]
    store_hist   = hist_df_trimmed[hist_df_trimmed["id"].isin(store_items)]
    store_future = future_df[future_df["id"].isin(store_items)]

    exp_cfg = {
        **DEFAULT_CHRONOS_CONFIG,
        "use_static":     False,
        "known_cov_cols": ["event_name_1", "event_type_1", "sell_price"],
        "device":         device,
        "batch_size":     64,
        "context_length": 16,
        "fine_tune_steps": 0,
    }

    store_tag = f"dbgp_{store_id.lower()}_zs_cl16_event_price"

    suites[device]._cached_tsdf.clear()
    suites[device]._cached_future.clear()

    with _model_load_lock:
        fcst = suites[device].run(
            hist_df      = store_hist,
            future_df    = store_future,
            static_df    = None,
            model        = "Chronos2",
            exp_config   = exp_cfg,
            exp_tag      = store_tag,
            data_tag     = DATA_TAG,
            cutoff_day   = CUTOFF_DAY,
            wrapper_dict = WRAPPER,
            force_run    = True,
        )
    filtered = fcst[fcst["id"].isin(store_items)].copy()
    return store_id, filtered


print(f"\nRunning {len(STORES)} stores in parallel ({N_GPUS} at a time)...\n")

results = {}
with ThreadPoolExecutor(max_workers=N_GPUS) as executor:
    future_map = {
        executor.submit(_run_one_store, store, STORE_DEVICE_MAP[store]): store
        for store in STORES
    }
    for fut in as_completed(future_map):
        store = future_map[fut]
        try:
            name, fcst = fut.result()
            print(f"  {name}: SUCCESS {fcst.shape}")
            results[name] = fcst
        except Exception as e:
            print(f"  {store}: FAILED — {e}")
            traceback.print_exc()

print(f"\nCompleted: {len(results)}/{len(STORES)} stores")

