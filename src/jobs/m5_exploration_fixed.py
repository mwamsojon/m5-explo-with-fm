"""
m5_exploration.py
=================
AutoGluon-backed exploration runner for Chronos-2 and TFT on M5.

Design contracts (mirrors m5_benchmarks.py exactly):
  - Same pipeline save/load logic as M5DataPipeline.get_forecasts()
  - Same output schema: [id (str), date, sales_quantity] at L12
  - Same aggregate_bu() — copy from benchmark suite, no changes
  - All outputs safe for M5Evaluator.evaluate_all() without transformation

Covariate philosophy
--------------------
AutoGluon TimeSeries has two covariate channels:

  known_covariates   — columns whose future values are KNOWN at forecast time.
                       Examples: calendar flags, SNAP, events, price (if planned).
                       Passed to both fit() and predict() via known_covariates_names
                       and a future covariate DataFrame at predict time.

  past_covariates    — columns known only up to the forecast origin (past only).
                       Examples: lagged price, actual sell volume proxies.
                       AutoGluon reads these from the training frame automatically
                       when they appear as extra columns alongside the target.
                       They are NOT passed at predict time.

  static_features    — time-invariant per-series attributes.
                       Examples: item_id, store_id, cat_id, dept_id, state_id.
                       Assigned to ag_df.static_features before fit().
                       TFT embeds them via learnable embeddings.
                       Chronos ignores them in zero-shot mode.

To add/remove covariates from the notebook:
  - Edit `known_cov_cols` and/or `past_cov_cols` in the experiment config dict.
  - Set either to [] to run without that covariate channel.
  - Static features are controlled by passing static_df (or None).

Experiment save paths follow M5DataPipeline conventions:
  <base_dir>/<data_tag>/level_<level>/<cutoff_yyyymmdd>/models/<exp_tag>/forecasts.parquet
  <base_dir>/<data_tag>/level_<level>/<cutoff_yyyymmdd>/models/<exp_tag>/metrics.json
"""

from __future__ import annotations

import gc
import json
import os
import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
from autogluon.timeseries import TimeSeriesDataFrame


warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  Covariate column catalogue
#  Edit these lists to control what is available for experiments.
#  The notebook selects subsets via known_cov_cols / past_cov_cols.
# ─────────────────────────────────────────────────────────────────────────────

# All columns that are KNOWN in the future (calendar-derived, announced prices)
ALL_KNOWN_COV_COLS = [
    "wday",          # day of week (1-7)
    "month",         # month of year (1-12)
    "year",          # calendar year
    "snap_CA",       # SNAP benefit day — California
    "snap_TX",       # SNAP benefit day — Texas
    "snap_WI",       # SNAP benefit day — Wisconsin
    "event_name_1",  # primary event name (encoded as int or category)
    "event_type_1",  # primary event type
    "sell_price",    # treat as future-known if promotions are pre-planned
]

# All columns that are only known in the PAST (cannot be forecasted ahead)
ALL_PAST_COV_COLS = [
    # None by default in standard M5 setup.
    # Add lagged features here if you engineer them before calling fit().
]

# Static feature columns (time-invariant per series)
ALL_STATIC_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


# ─────────────────────────────────────────────────────────────────────────────
#  Default experiment configs
#  These are the starting points — override any key in the notebook.
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CHRONOS_CONFIG = {
    # ── Model identity ──────────────────────────────────────────────────────
    "model_path":       "autogluon/chronos-2",   # HuggingFace hub ID
    # Options: chronos-2-tiny / chronos-2-small / chronos-2-base / chronos-2-large
    # Amazon originals: amazon/chronos-t5-tiny .. amazon/chronos-t5-large
    #"model_path":       "amazon/chronos-t5-base",

    # ── Context window ──────────────────────────────────────────────────────
    # How many past timesteps the encoder sees.
    # M5 suggestions: 91 (13 weeks), 365 (1 year), 512 (standard), 1913 (full)
    "context_length":   2,

    # ── Zero-shot vs fine-tuning ─────────────────────────────────────────────
    "fine_tune_steps":  0,      # 0 = pure zero-shot; 100-1000 = fine-tune
    # fine_tune_steps > 0 requires GPU and significantly more time.
    # For the M5 intermittent demand pattern, try 200-500 as a first step.

    # ── Fine-tuning mode (only matters when fine_tune_steps > 0) ────────────
    # "lora"  — parameter-efficient, fast, recommended starting point
    # "full"  — all weights updated, best results, needs large GPU memory
    "fine_tune_mode":   "lora",

    # ── Compute ──────────────────────────────────────────────────────────────
    # AutoGluon passes this value directly to HuggingFace's `device_map`
    # inside Chronos2Pipeline.from_pretrained(). HuggingFace only accepts
    # PyTorch device strings — "cuda", "cuda:0", "cpu" — NOT "gpu" (AutoGluon
    # internal) and NOT a torch.device object.
    # Use "cuda:1" to pin to a specific card (the idle second Quadro RTX 5000).
    "device":  "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size":       32,     # reduce to 8-16 if GPU OOM on large model

    # ── Probabilistic output ─────────────────────────────────────────────────
    # Chronos is natively probabilistic. num_samples controls the MC estimate.
    "num_samples":      20,

    # ── Covariates ───────────────────────────────────────────────────────────
    # Chronos zero-shot IGNORES known_cov_cols (no covariate encoder).
    # Covariates only take effect when fine_tune_steps > 0 AND the model
    # supports them (chronos-2 does via its regression head).
    # Set to [] to run covariate-free (fastest, recommended for zeroshot).
    "known_cov_cols":   [],     # subset of ALL_KNOWN_COV_COLS
    "past_cov_cols":    [],     # subset of ALL_PAST_COV_COLS
    "use_static":       False,  # Chronos zeroshot ignores static features
}

DEFAULT_TFT_CONFIG = {
    # ── Architecture ─────────────────────────────────────────────────────────
    "context_length":   56,     # look-back window. Try 28, 56, 84, 112, 182
    "hidden_size":      64,     # TFT attention width. Try 32, 64, 128, 256
    "num_heads":        4,      # must divide hidden_size evenly
    "dropout_rate":     0.1,    # regularisation. Try 0.05-0.3

    # ── Training ─────────────────────────────────────────────────────────────
    "lr":               1e-3,
    "max_epochs":       100,
    "patience":         10,     # early stopping patience

    # ── Covariates ───────────────────────────────────────────────────────────
    # TFT natively handles both covariate channels. Adding them is high-value.
    # Start with known_cov_cols = ["wday", "month", "snap_CA", "snap_TX", "snap_WI"]
    # then add events and price to see marginal contribution.
    "known_cov_cols":   ["wday", "month", "snap_CA", "snap_TX", "snap_WI",
                         "event_type_1"],
    "past_cov_cols":    [],
    "use_static":       True,   # TFT embeds static cols as learnable vectors
}


# ─────────────────────────────────────────────────────────────────────────────
#  Core runner
# ─────────────────────────────────────────────────────────────────────────────

class M5ExplorationSuite:
    """
    AutoGluon exploration runner for Chronos-2 and TFT on M5.

    Structural contract: identical interface to M5BenchmarkSuite so that
    outputs plug directly into M5Evaluator.evaluate_all() and aggregate_bu().

    Save/load contract: identical path logic to M5DataPipeline.get_forecasts()
    so that experiments are cached under the same folder tree.

    Parameters
    ----------
    horizon    : forecast horizon in days (28 for M5 evaluation phase)
    ag_path    : base path for AutoGluon model artefacts (weights, checkpoints)
    base_dir   : base path for saved forecasts (same as M5DataPipeline.base_dir)
    """

    SUPPORTED_MODELS = ["Chronos2", "TFT"]

    def __init__(
        self,
        horizon:  int = 28,
        ag_path:  str = "/mnt/lab/nmwamsojo/autogluon_models/explorations",
        base_dir: str = "/mnt/lab/nmwamsojo/prepared_data",
    ):
        self.horizon  = horizon
        self.ag_path  = ag_path
        self.base_dir = base_dir

    # ──────────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────────

    def run(
        self,
        *,
        # ── Data ──────────────────────────────────────────────────────────────
        hist_df:    pd.DataFrame,
        future_df:  pd.DataFrame,         # future window with covariate cols
        static_df:  pd.DataFrame | None,

        # ── Experiment identity ───────────────────────────────────────────────
        model:      str,                  # "Chronos2" or "TFT"
        exp_config: dict,                 # model config dict (see defaults above)
        exp_tag:    str,                  # unique tag for this experiment run
        data_tag:   str  = "default",     # data prep tag (matches pipeline)
        cutoff_day: str  = "2016-05-22",  # forecast origin date
        level:      int  = 12,            # hierarchy level (always 12 for L12 BU)

        # ── Run control ───────────────────────────────────────────────────────
        wrapper_dict: dict | None = None, # AutoGluon predictor settings
        force_run:    bool = False,        # rerun even if forecast exists
    ) -> pd.DataFrame:
        """
        Fit and forecast, with caching identical to M5DataPipeline.

        Returns long-format [id (str), date, sales_quantity] at L12.
        Saves forecasts.parquet to the standard path automatically.

        Parameters
        ----------
        hist_df     : training history [id, date, sales_quantity, <cov cols>]
        future_df   : future window [id, date, <known cov cols>] — NO target col
        static_df   : [id, item_id, dept_id, cat_id, store_id, state_id]
        model       : "Chronos2" or "TFT"
        exp_config  : config dict — start from DEFAULT_CHRONOS_CONFIG or
                      DEFAULT_TFT_CONFIG and override as needed in notebook
        exp_tag     : unique string identifying this exact experiment, e.g.
                      "chronos2_base_zeroshot_ctx512" — used as the folder name
        data_tag    : matches the M5DataPipeline tag used to prepare hist_df
        cutoff_day  : training cutoff date string "YYYY-MM-DD"
        wrapper_dict: AutoGluon predictor settings (eval_metric, verbosity, …)
        force_run   : set True to rerun and overwrite existing forecast cache
        """
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(f"model must be one of {self.SUPPORTED_MODELS}")

        # ── Check forecast cache ───────────────────────────────────────────────
        paths = self._get_paths(data_tag, level, cutoff_day, exp_tag)
        if os.path.exists(paths["forecast"]) and not force_run:
            print(f"[{exp_tag}] Cache hit — loading forecasts from {paths['forecast']}")
            return pd.read_parquet(paths["forecast"])


        print(f"[{exp_tag}] Forecasts not found at {paths['forecast']}…")
        print(f"[{exp_tag}] Running {model} experiment…")

        # ── Normalise ID dtypes ───────────────────────────────────────────────
        #hist_df   = hist_df.copy()
        #future_df = future_df.copy()
        #hist_df["id"]   = hist_df["id"].astype(str)
        #future_df["id"] = future_df["id"].astype(str)
        #hist_df["date"]   = pd.to_datetime(hist_df["date"])
        #future_df["date"] = pd.to_datetime(future_df["date"])

        # ── Resolve covariate column lists ────────────────────────────────────
        known_cov_cols = [
            c for c in exp_config.get("known_cov_cols", [])
            if c in hist_df.columns and c in future_df.columns
        ]
        past_cov_cols = [
            c for c in exp_config.get("past_cov_cols", [])
            if c in hist_df.columns
        ]
        use_static = exp_config.get("use_static", False) and static_df is not None

        if known_cov_cols:
            print(f"  known covariates : {known_cov_cols}")
        if past_cov_cols:
            print(f"  past  covariates : {past_cov_cols}")
        if use_static:
            print(f"  static features  : {list(static_df.columns[1:])}")

        # ── Build TimeSeriesDataFrame ─────────────────────────────────────────
        ag_train, ag_future = self._build_tsdfs(
            hist_df, future_df, static_df,
            known_cov_cols, past_cov_cols, use_static,
        )

        del hist_df, future_df, static_df
        gc.collect()

        # ── Build hyperparameters ─────────────────────────────────────────────
        if model == "Chronos2":
            hp = self._chronos_hp(exp_config)
        else:
            hp = self._tft_hp(exp_config)

        print(f"fit known_cov_cols : {known_cov_cols}")

        # ── Fit and predict ────────────────────────────────────────────────────
        f_df = self._fit_predict(
            ag_train    = ag_train,
            ag_future   = ag_future,
            hp          = hp,
            known_cov_cols = known_cov_cols,
            exp_tag     = exp_tag,
            wrapper_dict = wrapper_dict or {},
        )

        # List of high-footprint dataframes/objects
        del ag_train, ag_future
        gc.collect()


        # ── Save forecast ──────────────────────────────────────────────────────
        os.makedirs(paths["folder"], exist_ok=True)
        f_df.to_parquet(paths["forecast"], index=False)
        print(f"[{exp_tag}] Forecast saved → {paths['forecast']}")

        return f_df
    
    def optimize_dtypes(self, df):
        # Sales are always small integers in M5
        if "sales_quantity" in df.columns:
            df["sales_quantity"] = pd.to_numeric(df["sales_quantity"], downcast="float")
        # Prices and SNAP flags
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast="float")
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")
        return df
    
    def save_metrics(
        self,
        metrics:    dict,
        exp_tag:    str,
        data_tag:   str  = "default",
        cutoff_day: str  = "2016-05-22",
        level:      int  = 12,
    ) -> None:
        """
        Persist M5Evaluator metrics dict alongside the forecast parquet.
        Call this after evaluator.evaluate_all() to keep results traceable.
        """
        paths = self._get_paths(data_tag, level, cutoff_day, exp_tag)
        os.makedirs(paths["folder"], exist_ok=True)
        # Convert numpy floats to Python floats for JSON serialisation
        clean = {k: (float(v) if hasattr(v, "item") else v) for k, v in metrics.items()}
        with open(paths["metrics"], "w") as f:
            json.dump(clean, f, indent=2)
        print(f"[{exp_tag}] Metrics saved → {paths['metrics']}")

    def load_all_metrics(
        self,
        data_tag:   str = "default",
        cutoff_day: str = "2016-05-22",
        level:      int = 12,
    ) -> pd.DataFrame:
        """
        Scan the experiment folder tree and return a DataFrame of all saved
        metrics — one row per experiment, columns = metric names + exp_tag.
        Useful for building comparison tables in the notebook.
        """
        base = self._get_base_folder(data_tag, level, cutoff_day)
        models_root = os.path.join(base, "models")
        rows = []
        if not os.path.exists(models_root):
            return pd.DataFrame()
        for tag in os.listdir(models_root):
            metrics_path = os.path.join(models_root, tag, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    m = json.load(f)
                m["exp_tag"] = tag
                rows.append(m)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).set_index("exp_tag")
        # Sort by WRMSSE ascending for easy reading
        if "WRMSSE" in df.columns:
            df = df.sort_values("WRMSSE")
        return df

    # ──────────────────────────────────────────────────────────────────────────
    #  Bottom-up aggregation — verbatim from M5BenchmarkSuite
    # ──────────────────────────────────────────────────────────────────────────

    def aggregate_bu(
        self,
        f_df:      pd.DataFrame,
        static_df: pd.DataFrame,
        level_map: dict,
    ) -> pd.DataFrame:
        """Aggregate L12 forecasts to all hierarchy levels. No changes vs benchmark."""
        f_df["id"] = f_df["id"].astype(str)
        f_df = f_df.merge(
            static_df.assign(id=static_df["id"].astype(str)), on="id", how="left"
        )
        assert f_df["id"].nunique() == 30490, "Bottom-level count mismatch"

        all_group_cols = [c for cols in level_map.values() if cols for c in cols]
        for col in set(all_group_cols):
            if col in f_df.columns:
                f_df[col] = f_df[col].astype("category")

        all_levels = []
        for lvl, group_cols in level_map.items():
            if lvl == 12:
                all_levels.append(f_df[["id", "date", "sales_quantity"]])
                continue
            if not group_cols:
                agg = (
                    f_df.groupby("date", sort=False)["sales_quantity"]
                        .sum().reset_index()
                )
                agg["id"] = "Total"
            else:
                agg = (
                    f_df.groupby(group_cols + ["date"], observed=True, sort=False)
                        ["sales_quantity"].sum().reset_index()
                )
                agg["id"] = agg[group_cols].astype(str).agg("_".join, axis=1)
            all_levels.append(agg[["id", "date", "sales_quantity"]])

        return pd.concat(all_levels, axis=0, ignore_index=True)

    # ──────────────────────────────────────────────────────────────────────────
    #  Path helpers  (mirrors M5DataPipeline path logic exactly)
    # ──────────────────────────────────────────────────────────────────────────

    def _get_base_folder(self, data_tag: str, level: int, cutoff_day: str) -> str:
        return os.path.join(
            self.base_dir,
            data_tag,
            f"level_{level}",
            cutoff_day.replace("-", ""),
        )

    def _get_paths(self, data_tag: str, level: int, cutoff_day: str, exp_tag: str) -> dict:
        folder = os.path.join(
            self._get_base_folder(data_tag, level, cutoff_day), "models", exp_tag
        )
        return {
            "folder":   folder,
            "forecast": os.path.join(folder, "forecasts.parquet"),
            "metrics":  os.path.join(folder, "metrics.json"),
            "ag_model": os.path.join(self.ag_path, exp_tag),
        }

    # ──────────────────────────────────────────────────────────────────────────
    #  TimeSeriesDataFrame construction
    # ──────────────────────────────────────────────────────────────────────────

    def _build_tsdfs(
        self,
        hist_df:         pd.DataFrame,
        future_df:       pd.DataFrame,
        static_df:       pd.DataFrame | None,
        known_cov_cols:  list[str],
        past_cov_cols:   list[str],
        use_static:      bool,
    ):

        # Training frame: target + all covariate columns present in hist
        train_cols = ["id", "date", "sales_quantity"] + known_cov_cols + past_cov_cols
        train_cols = [c for c in train_cols if c in hist_df.columns]


        ag_train = TimeSeriesDataFrame.from_data_frame(
            hist_df[train_cols],
            id_column        = "id",
            timestamp_column = "date",
        )

        # Attach static features (TFT embeds these; Chronos ignores in zeroshot)
        if use_static and static_df is not None:
            tmp = static_df.set_index("id").copy()
            tmp = tmp.loc[:, ~tmp.columns.duplicated()]
            tmp.index = tmp.index.astype(str)
            tmp.index.name = "item_id"
            # Drop cols that AutoGluon will rename to avoid collision warnings
            if "item_id" in tmp.columns:
                tmp = tmp.drop(columns=["item_id"])
            ag_train.static_features = tmp.reindex(ag_train.item_ids)

        # Future frame: known covariates for the forecast horizon only
        if known_cov_cols:
            fut_cols = ["id", "date"] + known_cov_cols
            fut_cols = [c for c in fut_cols if c in future_df.columns]

            ag_future = TimeSeriesDataFrame.from_data_frame(
                future_df[fut_cols],
                id_column        = "id",
                timestamp_column = "date",
            )
        else:
            ag_future = None
        

        return ag_train, ag_future

    # ──────────────────────────────────────────────────────────────────────────
    #  Hyperparameter builders
    # ──────────────────────────────────────────────────────────────────────────

    def _chronos_hp(self, cfg: dict) -> dict:
        """
        Map the user-facing config dict to AutoGluon Chronos hyperparameters.

        Device translation — full call chain:
          Our config  →  AutoGluon Chronos2._fit()
                       →  Chronos2Pipeline.from_pretrained(device_map=<value>)
                         →  HuggingFace modeling_utils.from_pretrained()

          HuggingFace device_map accepts ONLY:
            - "cpu"
            - "cuda"  or  "cuda:N"   (N = GPU index)
            - "auto", "balanced", "balanced_low_0", "sequential"
          It does NOT accept "gpu" (AutoGluon internal) or torch.device objects.
          Passing either raises:
            ValueError: the value needs to be a device name (e.g. cpu, cuda:0)
              or 'auto'... but found gpu
        """
        raw_device = cfg.get("device", DEFAULT_CHRONOS_CONFIG["device"])

        # Normalise to a HuggingFace-compatible string
        if isinstance(raw_device, torch.device):
            # torch.device("cuda") → "cuda", torch.device("cuda:1") → "cuda:1"
            hf_device = str(raw_device)
        elif isinstance(raw_device, str):
            if raw_device == "gpu":
                # AutoGluon internal alias — translate to HuggingFace name
                hf_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                hf_device = raw_device  # "cuda", "cuda:0", "cuda:1", "cpu" — pass through
        else:
            hf_device = "cuda" if torch.cuda.is_available() else "cpu"

        hp: dict[str, Any] = {
            "model_path":     cfg.get("model_path",     DEFAULT_CHRONOS_CONFIG["model_path"]),
            "context_length": cfg.get("context_length", DEFAULT_CHRONOS_CONFIG["context_length"]),
            "device":         hf_device,
            "batch_size":     cfg.get("batch_size",      DEFAULT_CHRONOS_CONFIG["batch_size"]),
            #"num_samples":    cfg.get("num_samples",     DEFAULT_CHRONOS_CONFIG["num_samples"]),
        }

        fine_tune_steps = cfg.get("fine_tune_steps", 0)
        if fine_tune_steps > 0:
            hp["fine_tune_steps"]        = fine_tune_steps
            hp["fine_tune_mode"]  = cfg.get("fine_tune_mode", "lora")
            hp["fine_tune_lr"]          = cfg.get("fine_tune_lr", 1e-4)
            # Only pass num_val_windows when fine-tuning (uses a held-out window)
            #hp["num_val_windows"]        = 1

        return {"Chronos2": hp}

    def _tft_hp(self, cfg: dict) -> dict:
        """
        Map the user-facing config dict to AutoGluon TFT hyperparameters.

        context_length here is the TFT encoder look-back, NOT Chronos token budget.
        Typical M5 values: 2× to 4× the forecast horizon (56–112).
        """
        return {
            "TemporalFusionTransformer": {
                "context_length": cfg.get("context_length", 56),
                "hidden_size":    cfg.get("hidden_size",    64),
                "num_heads":      cfg.get("num_heads",      4),
                "dropout_rate":   cfg.get("dropout_rate",   0.1),
                "lr":             cfg.get("lr",             1e-3),
                "max_epochs":     cfg.get("max_epochs",     100),
                "patience":       cfg.get("patience",       10),
            }
        }

    # ──────────────────────────────────────────────────────────────────────────
    #  Fit + predict
    # ──────────────────────────────────────────────────────────────────────────

    def _fit_predict(
        self,
        ag_train:       Any,
        ag_future:      Any | None,
        hp:             dict,
        known_cov_cols: list[str],
        exp_tag:        str,
        wrapper_dict:   dict,
    ) -> pd.DataFrame:
        from autogluon.timeseries import TimeSeriesPredictor

        ag_model_path = os.path.join(self.ag_path, exp_tag)

        print(f"Fitting model with hyperparameters: ")
        for k, v in hp.items():
            print(f"\n\t{k}: {v}")
        
        predictor = TimeSeriesPredictor(
            prediction_length      = self.horizon,
            target                 = "sales_quantity",
            eval_metric            = wrapper_dict.get("eval_metric", "RMSSE"),
            quantile_levels        = wrapper_dict.get("quantile_levels", None),
            path                   = ag_model_path,
            # MOVE IT HERE:
            known_covariates_names = known_cov_cols if known_cov_cols else None,
        ).fit(
            ag_train,
            hyperparameters      = hp,
            enable_ensemble      = wrapper_dict.get("enable_ensemble", False),
            skip_model_selection = wrapper_dict.get("skip_model_selection", True),
            verbosity            = wrapper_dict.get("verbosity", 2),
        )

        # Predict: pass future covariates if we have them
        predict_kwargs: dict[str, Any] = {}
        if ag_future is not None and known_cov_cols:
            predict_kwargs["known_covariates"] = ag_future

        predictions = predictor.predict(ag_train, **predict_kwargs)

        # DELETE 
        del ag_train
        del predictor
        if ag_future is not None:
            del ag_future
        gc.collect()


        f_df = predictions["mean"].reset_index()

        del predictions
        gc.collect()

        f_df.columns = ["id", "date", "sales_quantity"]
        f_df["sales_quantity"] = f_df["sales_quantity"].clip(lower=0)
        f_df["id"] = f_df["id"].astype(str)
        return f_df.reset_index(drop=True)
