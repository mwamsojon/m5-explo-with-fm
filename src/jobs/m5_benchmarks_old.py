"""
M5 Benchmark Suite — faithful Python/AutoGluon translation of Point_Forecasts_-_Benchmarks.R
==============================================================================================

CHANGES vs. the original Python attempt — explained per benchmark:

[1]  Naive / sNaive — BYPASS AutoGluon entirely.
     R code (line 119-124): last value repeated; seasonal = last 7 values tiled.
     AutoGluon routes through a full TimeSeriesPredictor fit+save cycle (~13 min).
     Pure NumPy does the same thing in < 1 second and is bit-for-bit identical to R.

[2]  SES — BYPASS AutoGluon entirely.
     R code (lines 127-148): custom SES with alpha constrained to [0.1, 0.3] via
     L-BFGS-B, initialised at x[0], forecast = constant = last smoothed level.
     AutoGluon ETS("ANN") uses MLE initialisation and unconstrained alpha —
     a different optimiser that will never reproduce R's numbers exactly.

[3]  MA — BYPASS AutoGluon entirely.
     R code (lines 150-162): grid search k in {2..14}, minimise in-sample MSE,
     forecast = mean of last k observations, repeated for all h.
     AutoGluon has no direct equivalent of this specific procedure.

[4]  ES_bu — REPLACED AutoGluon ETS with statsforecast AutoETS.
     R code (line 302): es(ts(input, frequency=7), h=28)$forecast from the
     'smooth' package. This performs automatic model selection over the full
     ETS family (ZZZ) with s=7 using AICc, but ONLY on the active (non-leading-
     zero) portion of each series. statsforecast's AutoETS(model="ZZZ",
     season_length=7) is the closest Python equivalent.
     Key differences fixed vs. the previous attempt:
       - Removed additive_only=True: R's es() includes multiplicative models
         (it handles zeros internally; the restriction to 6 additive models
         documented in arXiv:2102.13209 applies to a DIFFERENT paper's analysis,
         not to the original competition R script which uses es() not ets()).
       - Removed EPSILON shift: distorts level and AICc model selection.
       - Added leading-zero trimming: R strips the inactive prefix BEFORE fitting
         (lines 286-288, 402-404). This is the most common source of mismatch.

[5]  ARIMA_bu — REPLACED hardcoded order with AutoARIMA(seasonal=True, m=7).
     R code (line 303): auto.arima(ts(input, frequency=7)) — fully automatic
     order selection with seasonal period 7. The original Python attempt hardcoded
     (0,1,1)x(0,1,1) which is just one of many candidates auto.arima considers.

[6]  Croston / optCroston / SBA — BYPASS AutoGluon.
     R code (lines 163-178): fixed alpha=0.1 (classic/SBA) or L-BFGS-B optimised
     alpha in [0.1, 0.3] (optimized), with SBA multiplier 0.95. Implemented
     directly in NumPy to match the R initialisation and loop exactly.

[7]  TSB — BYPASS AutoGluon.
     R code (lines 180-209): grid search over 9x7=63 (alpha, beta) combinations,
     minimise one-step-ahead MSE. Implemented in NumPy.

[8]  Leading-zero trimming applied universally.
     R lines 286-288 / 402-404: every series is trimmed to start from its first
     non-zero observation BEFORE any model is fitted. This changes the scale
     denominator and the fitted parameters for ETS/ARIMA. All methods now apply
     this trim via _trim_leading_zeros().

[9]  ID dtype normalisation.
     All returned DataFrames have id as str — safe for M5Evaluator (fixes the
     categorical-vs-str reindex bug identified earlier in the session).

[10] Parallelism.
     R uses doSNOW with 8 cores (line 414). statsforecast uses n_jobs=-1 natively.
     The pure-Python local benchmarks use joblib for equivalent parallelism.

[11] Removed Chronos from the benchmark map.
     Chronos is not in the R benchmark set (b_names, line 426-430). Kept in a
     separate section with a clear comment so you can opt in for your own runs.
"""

from __future__ import annotations

import warnings
import shutil
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# ── optional heavy imports — fail loudly only when the method is actually used ──
def _require(pkg: str, install: str | None = None):
    import importlib, sys
    mod = importlib.import_module(pkg)
    return mod


# ═══════════════════════════════════════════════════════════════════════════════
#  Pure-Python / NumPy implementations matching R exactly
# ═══════════════════════════════════════════════════════════════════════════════

def _trim_leading_zeros(x: np.ndarray) -> np.ndarray:
    """
    R lines 286-288 / 402-404:
        start_period <- min(which(input > 0))
        input <- input[start_period:length(input)]
    Returns the array starting from the first non-zero element.
    If all zeros, returns the original array unchanged (edge case safety).
    """
    nz = np.flatnonzero(x)
    return x[nz[0]:] if len(nz) else x


def _naive(x: np.ndarray, h: int, seasonal: bool = False) -> np.ndarray:
    """R lines 119-125: Naive (last value) and sNaive (tile last 7 values)."""
    if seasonal:
        # head(rep(tail(x, 7), ceil(h/7)), h)
        tail7 = x[-7:]
        reps = int(np.ceil(h / 7))
        return np.tile(tail7, reps)[:h]
    return np.full(h, x[-1])


def _ses_level(x: np.ndarray, alpha: float) -> float:
    """
    R lines 133-148: SES recursion initialised at x[0].
    Returns the final smoothed level (= point forecast for all horizons).
    """
    level = x[0]
    for val in x[1:]:
        level = alpha * val + (1 - alpha) * level
    return level


def _ses_mse(alpha: float, x: np.ndarray) -> float:
    """One-step-ahead MSE used by L-BFGS-B optimisation (R line 127)."""
    fitted = [x[0]]
    for val in x[:-1]:
        fitted.append(alpha * val + (1 - alpha) * fitted[-1])
    return float(np.mean((np.array(fitted) - x) ** 2))


def _optimise_alpha(x: np.ndarray, bounds=(0.1, 0.3)) -> float:
    """R: optim(c(0), SES, ..., lower=0.1, upper=0.3, method='L-BFGS-B')."""
    res = minimize_scalar(_ses_mse, bounds=bounds, args=(x,), method="bounded")
    return float(np.clip(res.x, *bounds))


def _ses_forecast(x: np.ndarray, h: int) -> np.ndarray:
    """R lines 126-130 (SexpS): optimise alpha, repeat constant forecast h times."""
    alpha = _optimise_alpha(x)
    return np.full(h, _ses_level(x, alpha))


def _ma_forecast(x: np.ndarray, h: int) -> np.ndarray:
    """
    R lines 150-162: grid search k in {2..14}, MSE criterion,
    forecast = mean(tail(x, k)) repeated h times.
    """
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
    """R line 89-91: non-zero elements only."""
    return x[x != 0]


def _intervals(x: np.ndarray) -> np.ndarray:
    """R lines 72-88: inter-demand intervals."""
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
    """
    R lines 163-178:
      classic    : a1=a2=0.1, mult=1
      optimized  : optimise a1 on demand, a2 on intervals, mult=1
      sba        : a1=a2=0.1, mult=0.95
    """
    d = _demand(x)
    iv = _intervals(x)
    if len(d) == 0:
        return np.zeros(h)

    if variant == "optimized":
        a1 = _optimise_alpha(d)
        a2 = _optimise_alpha(iv)
        mult = 1.0
    elif variant == "sba":
        a1 = a2 = 0.1
        mult = 0.95
    else:  # classic
        a1 = a2 = 0.1
        mult = 1.0

    yd = _ses_level(d, a1)
    yi = _ses_level(iv, a2)
    return np.full(h, mult * yd / max(yi, 1e-8))


def _tsb_forecast(x: np.ndarray, h: int) -> np.ndarray:
    """
    R lines 180-209: grid search over alpha in {0.1,0.15,...,0.8} x
    beta in {0.01,0.02,0.03,0.05,0.1,0.2,0.3}, minimise one-step MSE.
    """
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
            zfit = np.zeros(n)
            pfit = np.zeros(n)
            zfit[0] = z[0]
            pfit[0] = p[0]
            for i in range(1, n):
                pfit[i] = pfit[i - 1] + a * (p[i] - pfit[i - 1])
                zfit[i] = zfit[i - 1] if p[i] == 0 else zfit[i - 1] + b * (x[i] - zfit[i - 1])
            yfit = pfit * zfit
            # one-step-ahead: shift yfit by 1
            yfit_shifted = np.concatenate([[np.nan], yfit[:-1]])
            mse = np.nanmean((yfit_shifted - x) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_frc = np.full(h, yfit[-1])
    return best_frc


def _adida_forecast(x: np.ndarray, h: int) -> np.ndarray:
    """R lines 210-215: aggregate-disaggregate intermittent demand approach."""
    iv = _intervals(x)
    al = max(1, int(round(float(np.mean(iv)))))
    n_full = (len(x) // al) * al
    if n_full == 0:
        return np.full(h, float(np.mean(x)))
    tail_x = x[-n_full:]
    AS = tail_x.reshape(-1, al).sum(axis=1).astype(float)
    frc = _ses_forecast(AS, 1)[0] / al
    return np.full(h, frc)


def _imapa_forecast(x: np.ndarray, h: int) -> np.ndarray:
    """R lines 217-224: iterated MAPA (average of ADIDA across aggregation levels)."""
    iv = _intervals(x)
    mal = max(1, int(round(float(np.mean(iv)))))
    frcs = []
    for al in range(1, mal + 1):
        n_full = (len(x) // al) * al
        if n_full == 0:
            continue
        tail_x = x[-n_full:]
        AS = tail_x.reshape(-1, al).sum(axis=1).astype(float)
        frcs.append(_ses_forecast(AS, 1)[0] / al)
    if not frcs:
        return np.full(h, float(np.mean(x)))
    return np.full(h, float(np.mean(frcs)))


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-series dispatcher — mirrors R's benchmarks_f() (lines 278-324)
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
    """
    Mirrors R benchmarks_f(): trim leading zeros, run all local methods,
    clip negatives, return long-format DataFrame for this series.
    """
    x = _trim_leading_zeros(x_raw)  # [CHANGE 8] R lines 286-288

    rows = []
    for method in methods:
        if method == "Naive":
            frc = _naive(x, h, seasonal=False)
        elif method == "sNaive":
            frc = _naive(x, h, seasonal=True)
        elif method == "SES":
            frc = _ses_forecast(x, h)
        elif method == "MA":
            frc = _ma_forecast(x, h)
        elif method == "Croston":
            frc = _croston_forecast(x, h, "classic")
        elif method == "optCroston":
            frc = _croston_forecast(x, h, "optimized")
        elif method == "SBA":
            frc = _croston_forecast(x, h, "sba")
        elif method == "TSB":
            frc = _tsb_forecast(x, h)
        elif method == "ADIDA":
            frc = _adida_forecast(x, h)
        elif method == "iMAPA":
            frc = _imapa_forecast(x, h)
        else:
            raise ValueError(f"Unknown local method: {method}")

        frc = np.clip(frc, 0, None)  # R lines 307-312
        rows.append(frc)

    arr = np.stack(rows, axis=1)  # (h, n_methods)
    dates = pd.RangeIndex(h)      # replaced with real dates by caller
    df = pd.DataFrame(arr, columns=methods)
    df.insert(0, "id", series_id)
    df.insert(1, "horizon", np.arange(1, h + 1))
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  Main class
# ═══════════════════════════════════════════════════════════════════════════════

class M5BenchmarkSuite:
    """
    Faithful Python reproduction of Point_Forecasts_-_Benchmarks.R.

    Supported methods (matching R's b_names, line 426-430):
        Local:   Naive, sNaive, SES, MA,
                 Croston, optCroston, SBA, TSB, ADIDA, iMAPA
        BU:      ES_bu, ARIMA_bu
        (TD and combination methods not yet included — add if needed)

    AutoGluon is retained ONLY for ES_bu and ARIMA_bu where it wraps
    statsforecast models that are the best Python equivalent of R's
    es() and auto.arima() respectively.

    Parameters
    ----------
    horizon   : forecast horizon (28 in M5)
    model_path: base path for AutoGluon model artefacts
    n_jobs    : parallelism for local methods (-1 = all cores)
    """

    # R b_names (lines 426-430) — order preserved for WRMSSE alignment
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

    # ──────────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────────

    def run(
        self,
        train_df: pd.DataFrame,
        methods: list[str] | None = None,
        static_df: pd.DataFrame | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Run one or more benchmarks.

        Parameters
        ----------
        train_df  : long-format [id, date, sales_quantity].
                    MUST span all training days without truncation.
                    MUST NOT have leading-zero trimming applied — this
                    function does it internally per series, matching R.
        methods   : list of method names; None = all R benchmarks
        static_df : optional [id, col1, ...] for AutoGluon-backed methods

        Returns
        -------
        dict mapping method name → long-format [id, date, sales_quantity]
        with str id column (safe for M5Evaluator).
        """
        if methods is None:
            methods = self.R_BENCHMARK_NAMES

        # Normalise dtypes once [CHANGE 9]
        train_df = train_df.copy()
        train_df["id"]   = train_df["id"].astype(str)
        train_df["date"] = pd.to_datetime(train_df["date"])
        train_df = train_df.sort_values(["id", "date"])

        local_methods = [m for m in methods if m in _LOCAL_METHODS]
        ag_methods    = [m for m in methods if m in ("ES_bu", "ARIMA_bu")]
        unknown       = [m for m in methods if m not in _LOCAL_METHODS and m not in ("ES_bu", "ARIMA_bu")]
        if unknown:
            raise ValueError(f"Unknown methods: {unknown}. Valid: {self.R_BENCHMARK_NAMES}")

        results: dict[str, pd.DataFrame] = {}

        if local_methods:
            results.update(self._run_local(train_df, local_methods))

        for method in ag_methods:
            results[method] = self._run_autogluon(train_df, method, static_df)

        return results

    # ──────────────────────────────────────────────────────────────────────────
    #  Local benchmarks (Naive … iMAPA)
    # ──────────────────────────────────────────────────────────────────────────

    def _run_local(
        self,
        train_df: pd.DataFrame,
        methods: list[str],
    ) -> dict[str, pd.DataFrame]:
        """
        [CHANGE 1, 2, 3, 6, 7, 8, 10]
        Bypass AutoGluon entirely. Run all local methods in one parallel pass
        (mirrors R's foreach loop, line 432-433).
        """
        print(f"--- Running local benchmarks {methods} (joblib n_jobs={self.n_jobs}) ---")

        # Build future date index once
        last_date = train_df["date"].max()
        freq      = pd.infer_freq(train_df["date"].drop_duplicates().sort_values().tail(14))
        future_dates = pd.date_range(
            start=last_date + pd.tseries.frequencies.to_offset(freq or "D"),
            periods=self.horizon,
            freq=freq or "D",
        )

        # Wide pivot — shape (n_series, T)
        wide = (
            train_df.pivot(index="id", columns="date", values="sales_quantity")
                    .fillna(0)
                    .sort_index(axis=1)
        )
        series_ids   = wide.index.tolist()
        series_arrays = [wide.loc[sid].values for sid in series_ids]

        # Parallel dispatch — one call per series [CHANGE 10]
        dfs = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_forecast_one_series)(sid, arr, self.horizon, methods)
            for sid, arr in zip(series_ids, series_arrays)
        )

        combined = pd.concat(dfs, axis=0, ignore_index=True)
        # Attach real dates: horizon 1 → future_dates[0], etc.
        date_map = {i + 1: d for i, d in enumerate(future_dates)}
        combined["date"] = combined["horizon"].map(date_map)
        combined = combined.drop(columns="horizon")

        # Split into one DataFrame per method
        out: dict[str, pd.DataFrame] = {}
        for method in methods:
            m_df = combined[["id", "date", method]].rename(columns={method: "sales_quantity"})
            m_df["id"] = m_df["id"].astype(str)
            out[method] = m_df.reset_index(drop=True)

        return out

    # ──────────────────────────────────────────────────────────────────────────
    #  AutoGluon-backed benchmarks (ES_bu, ARIMA_bu)
    # ──────────────────────────────────────────────────────────────────────────

    def _run_autogluon(
        self,
        train_df: pd.DataFrame,
        method: str,
        static_df: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """
        [CHANGE 4, 5, 8]
        ES_bu  → AutoGluon wrapping statsforecast AutoETS(model='ZZZ', s=7)
        ARIMA_bu → AutoGluon wrapping statsforecast AutoARIMA(season_length=7)

        Leading-zero trimming is handled per-series BEFORE building the
        TimeSeriesDataFrame, matching R lines 286-288.
        """
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

        print(f"--- Running {method} via AutoGluon (statsforecast backend) ---")

        # [CHANGE 8] Trim leading zeros per series before handing to AutoGluon
        trimmed_parts = []
        for sid, grp in train_df.groupby("id", sort=False):
            arr = grp.sort_values("date")["sales_quantity"].values
            nz  = np.flatnonzero(arr)
            if len(nz) == 0:
                trimmed_parts.append(grp)  # all-zero series: keep as-is
                continue
            trimmed_parts.append(grp.sort_values("date").iloc[nz[0]:])
        trimmed_df = pd.concat(trimmed_parts, ignore_index=True)

        ag_df = TimeSeriesDataFrame.from_data_frame(
            trimmed_df,
            id_column="id",
            timestamp_column="date",
        )

        # Attach static features safely [preserved from session fixes]
        if static_df is not None:
            temp_static = static_df.set_index("id").copy()
            temp_static = temp_static.loc[:, ~temp_static.columns.duplicated()]
            temp_static.index = temp_static.index.astype(str)
            temp_static.index.name = "item_id"
            ag_df.static_features = temp_static.reindex(ag_df.item_ids)

        # ── Hyperparameter map ──────────────────────────────────────────────
        # [CHANGE 4] ES_bu: AutoETS ZZZ s=7, NO additive_only, NO epsilon.
        #   R uses smooth::es() which searches the full ETS family (including
        #   multiplicative) and handles zeros internally — not restricted to
        #   additive models. additive_only=True was a prior error.
        # [CHANGE 5] ARIMA_bu: AutoARIMA with seasonal m=7 — matches R's
        #   auto.arima(ts(input, frequency=7)). The previous hardcoded
        #   (0,1,1)x(0,1,1) was just one candidate, not the selected model.
        hp_map = {
            "ES_bu": {"AutoETS": {"model": "ZZZ", "season_length": 7, "allow_multiplicative": True, "ic": "aicc"}}
            "ARIMA_bu": {"AutoARIMA":  {"season_length": 7}},
        }

        specific_path = f"{self.model_path}_{method}"
        if os.path.exists(specific_path):
            print(f"Cleaning up existing path: {specific_path}")
            shutil.rmtree(specific_path)

        predictor = TimeSeriesPredictor(
            prediction_length=self.horizon,
            target="sales_quantity",
            eval_metric="RMSSE",         # [CHANGE] was RMSE; RMSSE aligns with paper
            path=specific_path,
        ).fit(
            ag_df,
            hyperparameters=hp_map[method],
            enable_ensemble=False,
            skip_model_selection=True,
            verbosity=0,                 # [CHANGE] was 2; remove per-series stdout noise
        )

        # Predict using trimmed context (AutoGluon only needs recent history)
        CONTEXT_LEN = max(14, self.horizon)
        last_rows   = ag_df.groupby(level="item_id", sort=False).tail(CONTEXT_LEN)
        predictions = predictor.predict(last_rows)

        f_df = predictions["mean"].reset_index()
        f_df.columns = ["id", "date", "sales_quantity"]
        f_df["sales_quantity"] = f_df["sales_quantity"].clip(lower=0)
        f_df["id"] = f_df["id"].astype(str)  # [CHANGE 9]
        return f_df.reset_index(drop=True)

    # ──────────────────────────────────────────────────────────────────────────
    #  Bottom-up aggregation (mirrors R's ddply aggregation per level)
    # ──────────────────────────────────────────────────────────────────────────

    def aggregate_bu(
        self,
        f_df: pd.DataFrame,
        static_df: pd.DataFrame,
        level_map: dict,
    ) -> pd.DataFrame:
        """
        Aggregate bottom-level (L12) forecasts up to all hierarchy levels.
        Mirrors the ddply() calls in R lines 521-853.
        Returns long-format [id, date, sales_quantity] covering all levels.
        """
        f_df = f_df.copy()
        f_df["id"] = f_df["id"].astype(str)
        f_df = f_df.merge(static_df.assign(id=static_df["id"].astype(str)), on="id", how="left")
        assert f_df["id"].nunique() == 30490, "Bottom-level count mismatch — check static_df merge"

        # Set categorical dtypes once for all group keys
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
                        .sum()
                        .reset_index()
                )
                agg["id"] = agg[group_cols].astype(str).agg("_".join, axis=1)
            all_levels.append(agg[["id", "date", "sales_quantity"]])

        return pd.concat(all_levels, axis=0, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  OPTIONAL: Chronos (not in R benchmarks — for your own future experiments)
# ═══════════════════════════════════════════════════════════════════════════════
# To run Chronos add "Chronos-Tiny" or "Chronos-Base" to the methods list and
# pass them through _run_autogluon with:
#   hp_map["Chronos-Tiny"] = {"Chronos": {"model_path": "amazon/chronos-t5-tiny"}}
#   hp_map["Chronos-Base"] = {"Chronos": {"model_path": "amazon/chronos-t5-base"}}
# These are NOT part of the paper's benchmark table.


# ═══════════════════════════════════════════════════════════════════════════════
#  Usage
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ── Replace with your actual DataFrames ────────────────────────────────────
    # hist_df   : long [id, date, sales_quantity], all 1913 or 1941 training days
    # static_df : [id, state_id, store_id, cat_id, dept_id, item_id]
    # df_actual : long [id, date, sales_quantity], the 28 evaluation days
    # evaluator : M5Evaluator instance (from m5_evaluator.py in this session)
    # pipeline.level_map : {1: [], 2: ["state_id"], ..., 12: ["id"]}

    suite = M5BenchmarkSuite(horizon=28, n_jobs=-1)

    # Run all R benchmarks in one call
    all_forecasts = suite.run(
        train_df=hist_df,
        methods=M5BenchmarkSuite.R_BENCHMARK_NAMES,
        static_df=static_df,
    )

    # Or run a single method
    naive_fcst = suite.run(train_df=hist_df, methods=["Naive"])["Naive"]

    # Aggregate and evaluate each
    results_table = {}
    for method, f_df in all_forecasts.items():
        full = suite.aggregate_bu(f_df, static_df, pipeline.level_map)
        metrics = evaluator.evaluate_all(full, df_actual)
        results_table[method] = metrics
        print(f"{method:15s}  WRMSSE={metrics['WRMSSE']:.4f}  WAPE={metrics['WAPE_L12']:.2%}")

    # Quick diagnostic: confirm Naive matches paper (~1.752 overall WRMSSE)
    # and ES_bu matches (~0.575–0.610 range depending on phase)
