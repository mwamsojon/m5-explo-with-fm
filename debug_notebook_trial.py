"""
Replicates the actual notebook objective() execution for 1 trial.
Uses EXACT same code as notebook cells to reproduce the IndexError.
"""
import os, sys, gc, ctypes, contextlib, threading
sys.path.insert(0, "/home/nmwamsojo/tsfm-explo/src/jobs/")
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import optuna
import numpy as np
import pandas as pd

from m5_dataprep import M5DataPipeline
from m5_exploration import M5ExplorationSuite, DEFAULT_CHRONOS_CONFIG
from m5_evaluator import M5Evaluator

# ── Config ─────────────────────────────────────────────────────────────────
DATA_PATH     = "/mnt/lab/datasets/M5/jointed_M5.parquet"
CALENDAR_PATH = "/mnt/lab/nmwamsojo/m5_data/calendar.csv"
ACTUALS_PATH  = "/mnt/lab/nmwamsojo/m5_data/sales_test_evaluation.csv"
DATA_TAG   = "sales_only"
CUTOFF_DAY = (pd.to_datetime("2016-05-22") - pd.Timedelta(days=28)).strftime("%Y-%m-%d")

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
DEVICES = [f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"]
N_GPUS  = len(DEVICES)

STORES = ["CA_1","CA_2","CA_3","CA_4","TX_1","TX_2","TX_3","WI_1","WI_2","WI_3"]

COVARIATE_OPTIONS = {
    "none": [],
    "is_weekend": ["is_friday","is_saturday","is_sunday"],
    "event": ["event_name_1","event_type_1"],
    "price": ["sell_price"],
    "snap": ["snap_CA","snap_TX","snap_WI"],
    "is_weekend_event": ["is_friday","is_saturday","is_sunday","event_name_1","event_type_1"],
    "is_weekend_price": ["is_friday","is_saturday","is_sunday","sell_price"],
    "is_weekend_snap": ["is_friday","is_saturday","is_sunday","snap_CA","snap_TX","snap_WI"],
    "event_price": ["event_name_1","event_type_1","sell_price"],
    "event_snap": ["event_name_1","event_type_1","snap_CA","snap_TX","snap_WI"],
    "price_snap": ["sell_price","snap_CA","snap_TX","snap_WI"],
    "is_weekend_event_price": ["is_friday","is_saturday","is_sunday","event_name_1","event_type_1","sell_price"],
    "is_weekend_event_snap": ["is_friday","is_saturday","is_sunday","event_name_1","event_type_1","snap_CA","snap_TX","snap_WI"],
    "is_weekend_price_snap": ["is_friday","is_saturday","is_sunday","sell_price","snap_CA","snap_TX","snap_WI"],
    "event_price_snap": ["event_name_1","event_type_1","sell_price","snap_CA","snap_TX","snap_WI"],
    "is_weekend_event_price_snap": ["is_friday","is_saturday","is_sunday","event_name_1","event_type_1","sell_price","snap_CA","snap_TX","snap_WI"],
}

WRAPPER = {"eval_metric": "RMSSE", "enable_ensemble": False, "skip_model_selection": True, "verbosity": 1}
CFG_CHRONOS_BASE = {**DEFAULT_CHRONOS_CONFIG, "use_static": False}
BATCH_SIZE = 64

# ── Load Data ────────────────────────────────────────────────────────────────
print("Loading data...")
pipeline = M5DataPipeline(config={"tag": DATA_TAG})
hist_df, hist_df_trimmed, future_df, static_df, weights_scales = (
    pipeline.get_prepared_data(DATA_PATH, CUTOFF_DAY, level=12, force_reprepare=False)
)
del pipeline; gc.collect()

# ── Actuals ──────────────────────────────────────────────────────────────────
df_actual = (
    pd.read_parquet(DATA_PATH, columns=["id", "date", "sold"])
    .rename(columns={"sold": "sales_quantity"})
)
df_actual["id"] = (
    df_actual["id"].astype(str)
    .str.replace("_evaluation", "", regex=False)
    .str.replace("_validation",  "", regex=False)
)

# ── Evaluator + Suites ───────────────────────────────────────────────────────
evaluator = M5Evaluator(
    raw_train_df=hist_df, trimmed_train_df=hist_df_trimmed,
    static_df=static_df, weights_df=weights_scales,
    target_col="sales_quantity", price_col="sell_price",
)
suites = {
    device: M5ExplorationSuite(
        horizon=28, ag_path="/mnt/lab/nmwamsojo/autogluon_models/debug_notebook_trial",
        base_dir="/mnt/lab/nmwamsojo/prepared_data",
    )
    for device in DEVICES
}

# ── Store IDs ────────────────────────────────────────────────────────────────
STORE_IDS = {
    store: set(static_df[static_df["store_id"] == store]["id"].tolist())
    for store in STORES
}
STORE_DEVICE_MAP = {store: DEVICES[i % N_GPUS] for i, store in enumerate(STORES)}

# ── Helper functions (EXACT copies from notebook) ────────────────────────────
_model_load_lock = threading.Lock()

def _make_seg_tag(seg_scheme, cl, ft_steps, ft_mode, ft_lr, ft_bs, cov_type):
    if ft_steps == 0:
        return f"hpo_zs_{seg_scheme}_cl{cl}_{cov_type}"
    lr_str = f"{ft_lr:.0e}".replace("-0", "-")
    return f"hpo_ft_{seg_scheme}_cl{cl}_steps{ft_steps}_{ft_mode}_lr{lr_str}_bs{ft_bs}_{cov_type}"

def _bools_to_cov_type(use_is_weekend, use_event, use_price, use_snap):
    parts = []
    if use_is_weekend: parts.append("is_weekend")
    if use_event:      parts.append("event")
    if use_price:      parts.append("price")
    if use_snap:       parts.append("snap")
    return "_".join(parts) if parts else "none"

def _make_store_tag(store_id, cl, ft_steps, ft_mode, ft_lr, ft_bs, cov_type):
    base = _make_seg_tag("all", cl, ft_steps, ft_mode, ft_lr, ft_bs, cov_type)
    return f"store_{store_id.lower()}_{base}"

def _get_forecast_path(store_tag):
    return os.path.join(
        suites[DEVICES[0]].base_dir, DATA_TAG, "level_12",
        CUTOFF_DAY.replace("-", ""), "models", store_tag, "forecasts.parquet",
    )

# EXACT _run_one_store from notebook (with our two fixes applied)
def _run_one_store(store_id, store_params, device):
    device_idx = int(device.split(":")[-1]) if ":" in device else 0
    torch.cuda.set_device(device_idx)

    ft_steps = store_params["ft_steps"]
    ft_mode  = store_params["ft_mode"]  if ft_steps > 0 else "lora"
    ft_lr    = store_params["ft_lr"]    if ft_steps > 0 else 1e-4
    ft_bs    = store_params["ft_bs"]    if ft_steps > 0 else 128
    cl       = store_params["cl"]
    cov_type = store_params["cov_type"]
    store_items = STORE_IDS[store_id]

    store_tag     = _make_store_tag(store_id, cl, ft_steps, ft_mode, ft_lr, ft_bs, cov_type)
    forecast_path = _get_forecast_path(store_tag)
    is_cache_hit  = os.path.exists(forecast_path)
    _lock_ctx     = contextlib.nullcontext() if is_cache_hit else _model_load_lock

    # FIX 1: filter both hist and future to store items
    # FIX 3: remove unused categories — id columns are Categorical; value_counts(sort=False)
    # returns counts for ALL categories including zero-count ones, causing iloc OOB crash.
    store_hist   = hist_df_trimmed[hist_df_trimmed["id"].isin(store_items)].copy()
    store_future = future_df[future_df["id"].isin(store_items)].copy()
    if hasattr(store_hist["id"], "cat"):
        store_hist["id"]   = store_hist["id"].cat.remove_unused_categories()
    if hasattr(store_future["id"], "cat"):
        store_future["id"] = store_future["id"].cat.remove_unused_categories()
    store_static = (
        static_df[static_df["id"].isin(store_items)].copy()
        if static_df is not None else None
    )

    exp_cfg = {
        **CFG_CHRONOS_BASE,
        "context_length": cl,
        "fine_tune_steps": ft_steps,
        "fine_tune_mode": ft_mode,
        "fine_tune_lr": ft_lr,
        "fine_tune_batch_size": ft_bs,
        "batch_size": BATCH_SIZE,
        "known_cov_cols": COVARIATE_OPTIONS[cov_type],
        "device": device,
    }

    # FIX 2: evict TSDF cache before building new TSDFs for this store
    suites[device]._cached_tsdf.clear()
    suites[device]._cached_future.clear()

    with _lock_ctx:
        fcst = suites[device].run(
            hist_df=store_hist, future_df=store_future, static_df=store_static,
            model="Chronos2", exp_config=exp_cfg, exp_tag=store_tag,
            data_tag=DATA_TAG, cutoff_day=CUTOFF_DAY, wrapper_dict=WRAPPER, force_run=False,
        )

    filtered = fcst[fcst["id"].isin(store_items)].copy()
    del fcst, store_hist, store_static
    gc.collect()
    with torch.cuda.device(device_idx):
        torch.cuda.empty_cache()
    return store_id, filtered

# ── Run 3 simulated trials ───────────────────────────────────────────────────
# Use a FIXED set of params that represents a realistic HPO trial
TRIAL_PARAMS = [
    # Trial 1: all zero-shot, various cov types
    {s: {"cl": cl, "ft_steps": 0, "ft_mode": "lora", "ft_lr": 1e-4, "ft_bs": 128,
         "use_is_weekend": uw, "use_event": ue, "use_price": up, "use_snap": us,
         "cov_type": _bools_to_cov_type(uw, ue, up, us)}
     for s, cl, uw, ue, up, us in [
        ("CA_1", 128, False, True,  True, False),
        ("CA_2",  64, False, False, True, True),
        ("CA_3",  16, True,  False, False, False),
        ("CA_4", 256, False, True,  False, True),
        ("TX_1",   8, True,  True,  True, False),
        ("TX_2",  32, False, False, False, True),
        ("TX_3", 128, True,  False, True, False),
        ("WI_1",  64, False, True,  False, False),
        ("WI_2",  16, True,  True,  False, True),
        ("WI_3", 256, False, False, True, True),
     ]
    },
    # Trial 2: same cov types (cache hits) but different CL → same tags if ft_steps=0 and CL same
    # Actually different CL → different tags → cache misses
    {s: {"cl": cl, "ft_steps": 0, "ft_mode": "lora", "ft_lr": 1e-4, "ft_bs": 128,
         "use_is_weekend": uw, "use_event": ue, "use_price": up, "use_snap": us,
         "cov_type": _bools_to_cov_type(uw, ue, up, us)}
     for s, cl, uw, ue, up, us in [
        ("CA_1",  16, True,  True,  True, True),   # all_covs
        ("CA_2",  64, False, False, True, False),
        ("CA_3", 128, False, True,  True, False),
        ("CA_4",  32, True,  False, True, True),
        ("TX_1",   8, False, False, False, True),
        ("TX_2", 256, True,  True,  False, False),
        ("TX_3",  64, False, True,  True, True),
        ("WI_1",  16, True,  False, True, False),
        ("WI_2",  32, False, True,  False, True),
        ("WI_3", 128, True,  True,  True, False),
     ]
    },
]

for i, store_params in enumerate(TRIAL_PARAMS):
    trial_num = i + 1
    print(f"\n{'='*60}")
    print(f"Trial {trial_num}")

    for _s in suites.values():
        _s._cached_tsdf.clear()
        _s._cached_future.clear()
    gc.collect(2)
    for _d in DEVICES:
        with torch.cuda.device(_d):
            torch.cuda.empty_cache()

    results = {}
    try:
        with ThreadPoolExecutor(max_workers=N_GPUS) as executor:
            future_map = {
                executor.submit(_run_one_store, store, store_params[store], STORE_DEVICE_MAP[store]): store
                for store in STORES
            }
            for fut in as_completed(future_map):
                name, fcst = fut.result()
                print(f"  {name}: OK rows={len(fcst)}")
                results[name] = fcst

        ensemble = pd.concat(list(results.values()), ignore_index=True)
        metrics = evaluator.evaluate_all(ensemble, df_actual)
        print(f"  WRMSSE = {metrics['WRMSSE']:.4f}")
    except Exception as e:
        import traceback
        print(f"  Trial {trial_num} FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        results.clear()
        gc.collect(2)

print("\nDone.")
