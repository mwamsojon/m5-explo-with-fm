"""
M5 Benchmark Suite — faithful Python translation of Point_Forecasts_-_Benchmarks.R
====================================================================================
[See previous session for full change log]

BUG FIX IN THIS VERSION vs. previous:

[BUG3-FIX] CONTEXT_LEN truncation in _run_autogluon was corrupting ETS state.
  Previous code:
      CONTEXT_LEN = max(14, self.horizon)   # 28 rows
      last_rows = ag_df.groupby(...).tail(CONTEXT_LEN)
      predictions = predictor.predict(last_rows)

  This meant AutoGluon's AutoETS received only the last 28 observations to
  initialise its state. R's es() fits on ALL trimmed observations (up to 1913
  days). The ETS smoothed level, trend, and seasonal components at the forecast
  origin are completely wrong when computed from only 28 points.

  Fix: pass the full ag_df to predictor.predict() for ES_bu and ARIMA_bu.
  Context truncation is only safe for Naive/SES where the forecast depends
  only on the last value(s), NOT for stateful models like ETS or ARIMA.
"""

from __future__ import annotations

import os
import shutil
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
#  Pure-NumPy implementations — exact R equivalents
# ═══════════════════════════════════════════════════════════════════════════════

def _trim_leading_zeros(x: np.ndarray) -> np.ndarray:
    """R lines 286-288 / 403-404: trim to first nonzero observation."""
    nz = np.flatnonzero(x)
    return x[nz[0]:] if len(nz) else x


def _naive(x: np.ndarray, h: int, seasonal: bool = False) -> np.ndarray:
    """R lines 119-125."""
    if seasonal:
        tail7 = x[-7:]
        return np.tile(tail7, int(np.ceil(h / 7)))[:h]
    return np.full(h, x[-1])


def _ses_level(x: np.ndarray, alpha: float) -> float:
    level = x[0]
    for val in x[1:]:
        level = alpha * val + (1 - alpha) * level
    return level


def _ses_mse(alpha: float, x: np.ndarray) -> float:
    fitted = [x[0]]
    for val in x[:-1]:
        fitted.append(alpha * val + (1 - alpha) * fitted[-1])
    return float(np.mean((np.array(fitted) - x) ** 2))


def _optimise_alpha(x: np.ndarray, bounds=(0.1, 0.3)) -> float:
    res = minimize_scalar(_ses_mse, bounds=bounds, args=(x,), method="bounded")
    return float(np.clip(res.x, *bounds))


def _ses_forecast(x: np.ndarray, h: int) -> np.ndarray:
    alpha = _optimise_alpha(x)
    return np.full(h, _ses_level(x, alpha))


def _ma_forecast(x: np.ndarray, h: int) -> np.ndarray:
    best_k, best_mse = 2, np.inf
    for k in range(2, 15):
        if k >= len(x):
            break
        fitted = np.array([np.mean(x[i - k:i]) for i in range(k, len(x))])
        mse = np.mean((fitted - x[k:]) ** 2)
        if mse < best_mse:
            best_mse, best_k = mse, k
    return np.full(h, float(np.mean(x[-best_k:])))


def _demand(x: np.ndarray) -> np.ndarray:
    return x[x != 0]


def _intervals(x: np.ndarray) -> np.ndarray:
    intervals, counter = [], 1
    for val in x:
        if val == 0:
            counter += 1
        else:
            intervals.append(counter)
            counter = 1
    arr = np.array(intervals, dtype=float)
    arr[arr == 0] = 1
    return arr if len(arr) else np.array([1.0])


def _croston_forecast(x: np.ndarray, h: int, variant: str = "classic") -> np.ndarray:
    d  = _demand(x)
    iv = _intervals(x)
    if len(d) == 0:
        return np.zeros(h)
    if variant == "optimized":
        a1, a2, mult = _optimise_alpha(d), _optimise_alpha(iv), 1.0
    elif variant == "sba":
        a1 = a2 = 0.1; mult = 0.95
    else:
        a1 = a2 = 0.1; mult = 1.0
    yd = _ses_level(d, a1)
    yi = _ses_level(iv, a2)
    return np.full(h, mult * yd / max(yi, 1e-8))


def _tsb_forecast(x: np.ndarray, h: int) -> np.ndarray:
    alphas = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.8]
    betas  = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3]
    n = len(x)
    p = (x != 0).astype(float)
    z = x[x != 0]
    if len(z) == 0:
        return np.zeros(h)
    best_mse, best_frc = np.inf, np.zeros(h)
    for a in alphas:
        for b in betas:
            zfit = np.zeros(n); pfit = np.zeros(n)
            zfit[0] = z[0]; pfit[0] = p[0]
            for i in range(1, n):
                pfit[i] = pfit[i-1] + a * (p[i] - pfit[i-1])
                zfit[i] = zfit[i-1] if p[i] == 0 else zfit[i-1] + b * (x[i] - zfit[i-1])
            yfit = pfit * zfit
            mse = np.nanmean((np.concatenate([[np.nan], yfit[:-1]]) - x) ** 2)
            if mse < best_mse:
                best_mse = mse; best_frc = np.full(h, yfit[-1])
    return best_frc


def _adida_forecast(x: np.ndarray, h: int) -> np.ndarray:
    al = max(1, int(round(float(np.mean(_intervals(x))))))
    n_full = (len(x) // al) * al
    if n_full == 0:
        return np.full(h, float(np.mean(x)))
    AS = x[-n_full:].reshape(-1, al).sum(axis=1).astype(float)
    return np.full(h, _ses_forecast(AS, 1)[0] / al)


def _imapa_forecast(x: np.ndarray, h: int) -> np.ndarray:
    mal = max(1, int(round(float(np.mean(_intervals(x))))))
    frcs = []
    for al in range(1, mal + 1):
        n_full = (len(x) // al) * al
        if n_full == 0:
            continue
        AS = x[-n_full:].reshape(-1, al).sum(axis=1).astype(float)
        frcs.append(_ses_forecast(AS, 1)[0] / al)
    return np.full(h, float(np.mean(frcs))) if frcs else np.full(h, float(np.mean(x)))


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-series dispatcher
# ═══════════════════════════════════════════════════════════════════════════════

_LOCAL_METHODS = [
    "Naive", "sNaive", "SES", "MA",
    "Croston", "optCroston", "SBA", "TSB",
    "ADIDA", "iMAPA",
]


def _forecast_one_series(
    series_id: str,
    x_raw: np.ndarray,
    h: int,
    methods: list[str],
) -> pd.DataFrame:
    x = _trim_leading_zeros(x_raw)
    rows = []
    for method in methods:
        if method == "Naive":        frc = _naive(x, h, False)
        elif method == "sNaive":     frc = _naive(x, h, True)
        elif method == "SES":        frc = _ses_forecast(x, h)
        elif method == "MA":         frc = _ma_forecast(x, h)
        elif method == "Croston":    frc = _croston_forecast(x, h, "classic")
        elif method == "optCroston": frc = _croston_forecast(x, h, "optimized")
        elif method == "SBA":        frc = _croston_forecast(x, h, "sba")
        elif method == "TSB":        frc = _tsb_forecast(x, h)
        elif method == "ADIDA":      frc = _adida_forecast(x, h)
        elif method == "iMAPA":      frc = _imapa_forecast(x, h)
        else: raise ValueError(f"Unknown: {method}")
        rows.append(np.clip(frc, 0, None))
    df = pd.DataFrame(np.stack(rows, axis=1), columns=methods)
    df.insert(0, "id", series_id)
    df.insert(1, "horizon", np.arange(1, h + 1))
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  Main class
# ═══════════════════════════════════════════════════════════════════════════════

class M5BenchmarkSuite:
    """
    Faithful Python reproduction of Point_Forecasts_-_Benchmarks.R.

    Supported methods (R's b_names, lines 426-430):
        Local : Naive, sNaive, SES, MA, Croston, optCroston, SBA, TSB, ADIDA, iMAPA
        BU    : ES_bu, ARIMA_bu
    """

    R_BENCHMARK_NAMES = [
        "Naive", "sNaive", "SES", "MA",
        "Croston", "optCroston", "SBA", "TSB",
        "ADIDA", "iMAPA",
        "ES_bu", "ARIMA_bu",
    ]

    def __init__(
        self,
        horizon: int = 28,
        model_path: str = "/mnt/lab/nmwamsojo/autogluon_models/",
        n_jobs: int = -1,
    ):
        self.horizon    = horizon
        self.model_path = model_path
        self.n_jobs     = n_jobs


    def run(
        self,
        train_df: pd.DataFrame,
        methods: list[str] | None = None,
        static_df: pd.DataFrame | None = None,
        model_dict: dict | None = None,
        wrapper_dict: dict | None = None,
        force_refit: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        Parameters
        ----------
        train_df : long-format [id, date, sales_quantity].
                   Pass the RAW untrimmed full-history data.
                   Trimming is applied internally per series for each method.
        """
        if methods is None:
            methods = self.R_BENCHMARK_NAMES

        train_df = train_df.copy()
        train_df["id"]   = train_df["id"].astype(str)
        train_df["date"] = pd.to_datetime(train_df["date"])
        train_df = train_df.sort_values(["id", "date"])

        local_methods = [m for m in methods if m in _LOCAL_METHODS]
        ag_methods    = [m for m in methods if m in ("ES_bu", "ARIMA_bu", "Chronos2")]
        unknown       = [m for m in methods if m not in _LOCAL_METHODS + ["ES_bu", "ARIMA_bu"]]
        if unknown:
            raise ValueError(f"Unknown methods: {unknown}. Valid: {self.R_BENCHMARK_NAMES}")

        results: dict[str, pd.DataFrame] = {}
        if local_methods:
            results.update(self._run_local(train_df, local_methods))
        

        for method in ag_methods:
            paths = self.get_forecast_paths(wrapper_dict["cutoff_day"],wrapper_dict["level"], model_tag=wrapper_dict["cutoff_day"], data_tag=wrapper_dict["data_tag"])
            
            use_cache = os.path.exists(paths["forecast"]) and not force_refit

            if use_cache:
                print(f"--- Cache Hit: Forecasts found in {paths['folder']} ---")
                results[method] =  pd.read_parquet(paths["forecast"])
            else:            
                results[method] = self._run_autogluon(train_df, method, static_df, model_dict, wrapper_dict)

                # save forecasts to cache for future runs
                os.makedirs(paths["folder"], exist_ok=True)
                results[method].to_parquet(paths["forecast"], index=False)


        return results

    def _run_local(self, train_df: pd.DataFrame, methods: list[str]) -> dict[str, pd.DataFrame]:
        print(f"--- Running local benchmarks {methods} (joblib n_jobs={self.n_jobs}) ---")

        last_date    = train_df["date"].max()
        freq         = pd.infer_freq(train_df["date"].drop_duplicates().sort_values().tail(14))
        future_dates = pd.date_range(
            start   = last_date + pd.tseries.frequencies.to_offset(freq or "D"),
            periods = self.horizon,
            freq    = freq or "D",
        )

        wide = (
            train_df.pivot(index="id", columns="date", values="sales_quantity")
                    .fillna(0).sort_index(axis=1)
        )
        series_ids    = wide.index.tolist()
        series_arrays = [wide.loc[sid].values for sid in series_ids]

        dfs = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_forecast_one_series)(sid, arr, self.horizon, methods)
            for sid, arr in zip(series_ids, series_arrays)
        )

        combined = pd.concat(dfs, axis=0, ignore_index=True)
        date_map = {i + 1: d for i, d in enumerate(future_dates)}
        combined["date"] = combined["horizon"].map(date_map)
        combined.drop(columns="horizon", inplace=True)

        out: dict[str, pd.DataFrame] = {}
        for method in methods:
            m_df = combined[["id", "date", method]].rename(columns={method: "sales_quantity"})
            m_df["id"] = m_df["id"].astype(str)
            out[method] = m_df.reset_index(drop=True)
        return out


    def _get_base_folder(self, cutoff_day, level, data_tag):
        """Returns the specific experiment leaf folder."""
        return os.path.join(
            self.base_dir, 
            data_tag, 
            f"level_{level}", 
            cutoff_day.replace("-", "")
        )

    def get_forecast_paths(self, cutoff_day, level=12, model_tag=None, data_tag=None):
        """Paths for model-specific outputs within the same experiment folder."""
        m_tag = model_tag or self.model_tag
        d_tag = data_tag or self.tag
        folder = os.path.join(self._get_base_folder(cutoff_day, level, d_tag), "models", m_tag)
        return {
            "folder": folder,
            "forecast": os.path.join(folder, "forecasts.parquet"),
            "metrics": os.path.join(folder, "metrics.json")
        }

    def _run_autogluon(
        self,
        train_df: pd.DataFrame,
        method: str,
        static_df: pd.DataFrame | None,
        model_dict: dict | None = None,
        wrapper_dict: dict | None = None,
    ) -> pd.DataFrame:
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

        print(f"--- Running {method} via AutoGluon (statsforecast backend) ---")

        # Trim leading zeros per series (R lines 286-288 / 403-404)
        trimmed_parts = []
        for sid, grp in train_df.groupby("id", sort=False):
            arr = grp.sort_values("date")["sales_quantity"].values
            nz  = np.flatnonzero(arr)
            trimmed_parts.append(
                grp.sort_values("date").iloc[nz[0]:] if len(nz) else grp
            )
        trimmed_df = pd.concat(trimmed_parts, ignore_index=True)

        ag_df = TimeSeriesDataFrame.from_data_frame(
            trimmed_df, id_column="id", timestamp_column="date",
        )

        if static_df is not None:
            temp_static = static_df.set_index("id").copy()
            temp_static = temp_static.loc[:, ~temp_static.columns.duplicated()]
            temp_static.index = temp_static.index.astype(str)
            temp_static.index.name = "item_id"
            ag_df.static_features = temp_static.reindex(ag_df.item_ids)

        hp_map = {
            # R line 302: es(ts(input, frequency=7), h=28) from smooth package
            # AutoETS ZZZ s=7 is the closest Python equivalent.
            "ES_bu":    {"AutoETS":   {"model": "ZZZ", "season_length": 7}},
            # R line 303: auto.arima(ts(input, frequency=7))
            "ARIMA_bu": {"AutoARIMA": {"season_length": 7}},
        }

        if model_dict is not None:
            hp_map.update(model_dict)

        specific_path = f"{self.model_path}_{method}"
        if os.path.exists(specific_path):
            shutil.rmtree(specific_path)

        predictor = TimeSeriesPredictor(
            prediction_length = self.horizon,
            target            = wrapper_dict["target"] if wrapper_dict and "target" in wrapper_dict else "sales_quantity",
            eval_metric       = wrapper_dict["eval_metric"] if wrapper_dict and "eval_metric" in wrapper_dict else "RMSSE",
            path              = specific_path,
        ).fit(
            ag_df,
            hyperparameters      = hp_map[method],
            enable_ensemble      = wrapper_dict["enable_ensemble"], # False,
            skip_model_selection = wrapper_dict["skip_model_selection"], #True,
            verbosity            = wrapper_dict["verbosity"], #0,
        )

        # [BUG3-FIX] Pass the FULL ag_df to predict — do NOT truncate context.
        # ETS and ARIMA are stateful: the forecast origin state depends on all
        # past observations. Truncating to 28 rows gives a completely wrong
        # smoothed state. R's es() fits on the full trimmed series.
        predictions = predictor.predict(ag_df)

        f_df = predictions["mean"].reset_index()
        f_df.columns = ["id", "date", "sales_quantity"]
        f_df["sales_quantity"] = f_df["sales_quantity"].clip(lower=0)
        f_df["id"] = f_df["id"].astype(str)

        return f_df.reset_index(drop=True)

    def aggregate_bu(
        self,
        f_df: pd.DataFrame,
        static_df: pd.DataFrame,
        level_map: dict,
    ) -> pd.DataFrame:
        f_df = f_df.copy()
        f_df["id"] = f_df["id"].astype(str)
        f_df = f_df.merge(static_df.assign(id=static_df["id"].astype(str)), on="id", how="left")
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
                agg = f_df.groupby("date", sort=False)["sales_quantity"].sum().reset_index()
                agg["id"] = "Total"
            else:
                agg = (
                    f_df.groupby(group_cols + ["date"], observed=True, sort=False)["sales_quantity"]
                        .sum().reset_index()
                )
                agg["id"] = agg[group_cols].astype(str).agg("_".join, axis=1)
            all_levels.append(agg[["id", "date", "sales_quantity"]])

        return pd.concat(all_levels, axis=0, ignore_index=True)
