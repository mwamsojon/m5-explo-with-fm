"""
Run TWO back-to-back trials to reproduce crash in the second trial.
Trial 1: all cache misses
Trial 2: some cache hits, some misses (different covariate combos)
"""
import sys, os, contextlib, threading, traceback
sys.path.insert(0, "/home/nmwamsojo/tsfm-explo/src/jobs/")

import gc, torch, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from m5_exploration import M5ExplorationSuite, DEFAULT_CHRONOS_CONFIG

DATA_ROOT  = "/mnt/lab/nmwamsojo/prepared_data/sales_only/level_12/20160424"
DATA_TAG   = "sales_only"
CUTOFF_DAY = "2016-04-24"

print("Loading cached parquets...")
hist_df_trimmed = pd.read_parquet(f"{DATA_ROOT}/hist_trimmed.parquet")
future_df       = pd.read_parquet(f"{DATA_ROOT}/future.parquet")
static_df       = pd.read_parquet(f"{DATA_ROOT}/static.parquet")

STORES  = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]
STORE_IDS = {s: set(static_df[static_df["store_id"] == s]["id"].tolist()) for s in STORES}

DEVICES = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
N_GPUS  = len(DEVICES)
STORE_DEVICE_MAP = {s: DEVICES[i % N_GPUS] for i, s in enumerate(STORES)}

suites = {
    d: M5ExplorationSuite(
        horizon=28,
        ag_path="/mnt/lab/nmwamsojo/autogluon_models/debug_crash3",
        base_dir="/mnt/lab/nmwamsojo/prepared_data",
    )
    for d in DEVICES
}

WRAPPER = {"eval_metric": "RMSSE", "enable_ensemble": False, "skip_model_selection": True, "verbosity": 0}
_lock = threading.Lock()

ALL_COV_OPTIONS = {
    "none":  [],
    "price": ["sell_price"],
    "event_price": ["event_name_1", "event_type_1", "sell_price"],
}


def _run_one_store(store_id, device, cov_type, trial_id):
    device_idx = int(device.split(":")[-1]) if ":" in device else 0
    torch.cuda.set_device(device_idx)

    store_items  = STORE_IDS[store_id]
    store_hist   = hist_df_trimmed[hist_df_trimmed["id"].isin(store_items)]
    store_future = future_df[future_df["id"].isin(store_items)]

    cov_cols = ALL_COV_OPTIONS[cov_type]
    exp_cfg = {
        **DEFAULT_CHRONOS_CONFIG,
        "use_static": False, "known_cov_cols": cov_cols,
        "device": device, "batch_size": 64,
        "context_length": 16, "fine_tune_steps": 0,
    }

    store_tag = f"dc3_t{trial_id}_{store_id.lower()}_cl16_{cov_type}"

    forecast_path = os.path.join(
        suites[device].base_dir, DATA_TAG, "level_12",
        CUTOFF_DAY.replace("-", ""), "models", store_tag, "forecasts.parquet"
    )
    is_cache_hit = os.path.exists(forecast_path)
    lock_ctx = contextlib.nullcontext() if is_cache_hit else _lock

    suites[device]._cached_tsdf.clear()
    suites[device]._cached_future.clear()

    with lock_ctx:
        fcst = suites[device].run(
            hist_df=store_hist, future_df=store_future, static_df=None,
            model="Chronos2", exp_config=exp_cfg, exp_tag=store_tag,
            data_tag=DATA_TAG, cutoff_day=CUTOFF_DAY,
            wrapper_dict=WRAPPER, force_run=False,
        )
    filtered = fcst[fcst["id"].isin(store_items)].copy()
    return store_id, filtered


def run_trial(trial_id, store_cov_map):
    """Run all stores in parallel and return results."""
    print(f"\n=== Trial {trial_id} ===")
    for store in STORES:
        _s = suites[STORE_DEVICE_MAP[store]]
        _s._cached_tsdf.clear()
        _s._cached_future.clear()

    results = {}
    with ThreadPoolExecutor(max_workers=N_GPUS) as executor:
        fmap = {
            executor.submit(_run_one_store, s, STORE_DEVICE_MAP[s], store_cov_map[s], trial_id): s
            for s in STORES
        }
        for fut in as_completed(fmap):
            store = fmap[fut]
            try:
                name, fcst = fut.result()
                print(f"  {name}: OK {fcst.shape}")
                results[name] = fcst
            except Exception as e:
                print(f"  {store}: FAILED — {type(e).__name__}: {e}")
                traceback.print_exc()
    return results


# Trial 1: all stores use "event_price" covariates
t1_map = {s: "event_price" for s in STORES}
r1 = run_trial(1, t1_map)

# Trial 2: same settings as trial 1 → all stores are cache HITS (fast reads)
r2 = run_trial(2, t1_map)

# Trial 3: different cov_type → all cache misses again (new tags)
t3_map = {s: ("price" if i % 2 == 0 else "event_price") for i, s in enumerate(STORES)}
r3 = run_trial(3, t3_map)

# Trial 4: back to event_price → half cache hits, half misses
t4_map = {s: "event_price" for s in STORES}
r4 = run_trial(4, t4_map)

print("\nAll trials done.")
