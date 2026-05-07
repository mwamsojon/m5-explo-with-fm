"""
M5DataPipeline — data preparation for the M5 forecasting competition.

Handles loading, aggregation to any of the 12 hierarchy levels, train/future
splitting, series trimming, and pre-computation of WRMSSE weights & scales.

Typical usage
-------------
    from m5_dataprep import M5DataPipeline

    pipeline = M5DataPipeline(config={"tag": "sales_only"})
    hist_df, hist_df_trimmed, future_df, static_df, weights_scales = (
        pipeline.get_prepared_data(DATA_PATH, CUTOFF_DAY, level=12)
    )

Output files written alongside each other in the cache folder
-------------------------------------------------------------
    hist.parquet          — full history up to cutoff
    hist_trimmed.parquet  — each series trimmed to its first non-zero sale
    future.parquet        — rows after cutoff (no target column)
    static.parquet        — one row per series with hierarchy columns
    weights_scales.parquet — per-level WRMSSE weights & scale denominators
                             columns: level (int), id (str), scale (float), weight (float)
"""

from __future__ import annotations

import gc
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


class M5DataPipeline:
    """
    Prepares M5 data at any of the 12 competition hierarchy levels and caches
    the results so subsequent runs just read from disk.

    Parameters
    ----------
    config : dict
        "tag"       — subfolder name under base_dir, used to separate experiments
        "model_tag" — model identifier, used for forecast output paths
    target_col : str
        Name of the demand column (default "sales_quantity").
    """

    def __init__(self, config, target_col="sales_quantity"):
        self.target = target_col
        self.date   = "date"
        self.id     = "id"
        self.base_dir   = "/mnt/lab/nmwamsojo/prepared_data"
        self.tag        = config.get("tag", "default")
        self.model_tag  = config.get("model_tag", "default_model")

        # Maps each M5 hierarchy level (1-12) to the columns it groups by.
        # Level 1 = total, level 12 = individual item×store series.
        self.level_map  = {
            1: [], 2: ["state_id"], 3: ["store_id"], 4: ["cat_id"], 5: ["dept_id"],
            6: ["state_id", "cat_id"], 7: ["state_id", "dept_id"],
            8: ["store_id", "cat_id"], 9: ["store_id", "dept_id"],
            10: ["item_id"], 11: ["state_id", "item_id"], 12: ["id"]
        }
        self.covariates  = ["wm_yr_wk", "wday", "month", "year",
                            "event_name_1", "event_type_1",
                            "snap_CA", "snap_TX", "snap_WI", "sell_price"]
        self.static_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]

    # ── Path helpers ──────────────────────────────────────────────────────────

    def _get_base_folder(self, cutoff_day, level):
        return os.path.join(self.base_dir, self.tag,
                            f"level_{level}", cutoff_day.replace("-", ""))

    def _get_cache_paths(self, cutoff_day, level):
        folder = self._get_base_folder(cutoff_day, level)
        return {"folder": folder,
                "hist":   os.path.join(folder, "hist.parquet"),
                "future": os.path.join(folder, "future.parquet"),
                "static": os.path.join(folder, "static.parquet")}

    def get_forecast_paths(self, cutoff_day, level, model_tag=None):
        m_tag  = model_tag or self.model_tag
        folder = os.path.join(self._get_base_folder(cutoff_day, level), "models", m_tag)
        return {"folder":   folder,
                "forecast": os.path.join(folder, "forecasts.parquet"),
                "metrics":  os.path.join(folder, "metrics.json")}

    # ── dtype helpers ─────────────────────────────────────────────────────────

    def optimize_dtypes(self, df):
        """
        Downcast numeric columns to the smallest type that fits the data.
        Halves RAM for a typical M5 DataFrame (float64 → float32, int64 → int16/32).
        Also converts the id column to a categorical to avoid storing duplicate strings.
        Called both after prepare() and after loading from the parquet cache.
        """
        if self.target in df.columns:
            df[self.target] = pd.to_numeric(df[self.target], downcast="float")
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="float")
        for col in df.select_dtypes(include=["int64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")
        if "id" in df.columns:
            df["id"] = df["id"].astype("category")
        return df

    # ── Core preparation ──────────────────────────────────────────────────────

    def prepare(self, path, cutoff_day, level=12):
        """
        Load the raw M5 parquet, aggregate to the requested hierarchy level,
        and split into historical / future / static DataFrames.

        Only reads the columns that are actually needed for the target level
        (e.g. "state_id" for level 2 but not "item_id") to keep memory low.
        """
        print(f"--- Loading: {path} ---")
        aggr_cols    = self.level_map[level]
        schema_names = set(pq.read_schema(path).names)

        # Some older M5 parquet files use "sold" instead of "sales_quantity"
        target_on_disk = "sold" if "sold" in schema_names else self.target
        needed = ({self.id, self.date, target_on_disk}
                  | (set(aggr_cols)        & schema_names)
                  | (set(self.covariates)  & schema_names)
                  | (set(self.static_cols) & schema_names))
        df = pd.read_parquet(path, columns=list(needed))
        if target_on_disk != self.target:
            df.rename(columns={target_on_disk: self.target}, inplace=True)

        df[self.date] = pd.to_datetime(df[self.date], utc=False, cache=True)

        # Event columns come as NaN for non-event days; fill with "none" so
        # they can stay as categoricals without triggering groupby warnings.
        for col in [c for c in df.columns if "event" in c]:
            if hasattr(df[col], "cat"):
                if "none" not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories("none")
                df[col] = df[col].fillna("none")
            else:
                df[col] = df[col].fillna("none").astype("category")

        # Cast to the smallest sensible types before the groupby — the aggregation
        # will then operate on smaller arrays, which is noticeably faster at scale.
        type_map = {"snap": np.int8, "wday": np.int8, "month": np.int8,
                    "year": np.int16, "wm_yr_wk": np.int16,
                    "sell_price": np.float32, self.target: np.float32}
        for prefix, dtype in type_map.items():
            for c in [c for c in df.columns if prefix in c]:
                df[c] = df[c].astype(dtype)
        for col in aggr_cols + self.static_cols:
            if col in df.columns and df[col].dtype == object:
                df[col] = df[col].astype("category")

        print(f"--- Aggregating to Level {level} ---")
        agg_map = {self.target: "sum"}
        for col in self.covariates:
            if col in df.columns:
                if "snap" in col:    agg_map[col] = "max"
                elif "event" in col: agg_map[col] = "first"
                else:                agg_map[col] = "mean"
        # Keep static cols (state_id, store_id, …) as "first" — same value per group
        for col in self.static_cols:
            if col in df.columns and col not in aggr_cols:
                agg_map[col] = "first"

        df_aggr = (df.groupby(aggr_cols + [self.date], observed=True, sort=False)
                     .agg(agg_map).reset_index())
        del df
        gc.collect()

        # Build a clean string id for each aggregated series
        if len(aggr_cols) == 0:
            df_aggr[self.id] = "All"
        elif len(aggr_cols) == 1:
            df_aggr[self.id] = df_aggr[aggr_cols[0]].astype(str)
        else:
            def _get_str_vals(s):
                if hasattr(s, "cat"):
                    return s.cat.categories.astype(str).values[s.cat.codes.values]
                return s.astype(str).values
            parts = [_get_str_vals(df_aggr[c]) for c in aggr_cols]
            ids   = parts[0]
            for part in parts[1:]:
                ids = np.char.add(np.char.add(ids, "_"), part)
            df_aggr[self.id] = ids

        df_aggr[self.id] = (df_aggr[self.id]
                             .astype(str)
                             .str.replace("_evaluation", "", regex=False)
                             .astype("category"))
        df_aggr["id"] = df_aggr["id"].astype(str)
        df_aggr = df_aggr.sort_values([self.id, self.date]).reset_index(drop=True)

        cutoff    = pd.Timestamp(cutoff_day)
        mask      = df_aggr[self.date] <= cutoff
        hist_df   = df_aggr[mask].copy()
        future_df = df_aggr[~mask].copy()
        if self.target in future_df.columns:
            future_df.drop(columns=[self.target], inplace=True)

        static_present = [c for c in self.static_cols if c in df_aggr.columns]
        static_df = (df_aggr[[self.id] + static_present]
                       .drop_duplicates().reset_index(drop=True))

        print(f"Done. {df_aggr[self.id].nunique():,} series | "
              f"Hist: {len(hist_df):,} rows | Future: {len(future_df):,} rows")
        del df_aggr
        gc.collect()

        return hist_df, future_df, static_df

    def trim_series_to_active(self, df, id_col="id", target_col="sales_quantity"):
        """
        Slice each series so it starts at its first non-zero sale.
        Leading zeros before a product was even on the shelves add noise to
        scale denominators, so the evaluator (and Chronos) work on trimmed series.
        Series with no sales at all are left untouched.
        """
        first_active = (
            df[df[target_col] > 0]
            .groupby(id_col, observed=True)["date"]
            .min()
            .rename("_first_active")
        )
        df = df.merge(first_active, on=id_col, how="left")
        # Series with no sales get NaN for _first_active — keep all their rows
        df = df[df["date"] >= df["_first_active"].fillna(df["date"].min())]
        return df.drop(columns="_first_active").reset_index(drop=True)

    # ── Weights & scales ──────────────────────────────────────────────────────

    @staticmethod
    def _compute_scales(array: np.ndarray) -> np.ndarray:
        """
        RMSSE scale denominator: mean squared first-difference of each row's
        active window (the slice from first non-zero value to end).

        Two paths, chosen per row:
          Fast — if the first column is already non-zero, there are no leading
                 zeros and we can diff the whole row with a single numpy call.
                 This covers every L1-L11 aggregated series.
          Slow — if there are leading zeros (common at L12 where items enter
                 the market at different times), we scan for the first non-zero
                 index and diff only the active slice. This is a Python loop but
                 only runs for the rows that actually need it.
        """
        n      = array.shape[0]
        scales = np.empty(n, dtype=np.float64)

        # Split rows into "dense" (no leading zeros) and "sparse" (has leading zeros)
        dense_rows  = np.where(array[:, 0] != 0)[0]
        sparse_rows = np.where(array[:, 0] == 0)[0]

        # Fast path — fully vectorised for the dense rows
        if len(dense_rows) > 0:
            diffs = np.diff(array[dense_rows], axis=1)          # shape (k, T-1)
            scales[dense_rows] = np.mean(diffs ** 2, axis=1)

        # Slow path — row-by-row for sparse rows (leading-zero series)
        for i in sparse_rows:
            row = array[i]
            nz  = np.flatnonzero(row)
            if len(nz) < 2:
                scales[i] = 1e-8
                continue
            active = row[nz[0]:]
            if len(active) < 2:
                scales[i] = 1e-8
                continue
            d = np.diff(active)
            scales[i] = float(np.mean(d * d))

        # Guard against zero-scale series (e.g. constant series, all-zero)
        return np.maximum(scales, 1e-8)

    def _build_weights_scales(self, hist_df, hist_df_trimmed, static_df):
        """
        Compute WRMSSE scale denominators and revenue weights for all 12 levels.

        Mirrors the logic in M5Evaluator._build_hierarchy:
          - Scales for L1-L11 use the raw (un-trimmed) wide matrix.
          - Scales for L12 use the trimmed wide matrix.
          - Weights at every level = normalised sum of last-28-day revenue
            (trimmed sales × sell_price).

        Memory strategy
        ---------------
        1. Only 3-column slices of hist_df / hist_df_trimmed are copied for
           pivoting — not the full ~15-column DataFrames.
        2. raw_wide is freed as soon as we've joined the static columns.
        3. For revenue weights we only ever need the last 28 dates, so the
           price pivot is built from a filtered subset of rows (28 dates × N
           series instead of 1913 × N), which is ~68× smaller than a full
           revenue matrix.

        Returns
        -------
        DataFrame with columns: level (int), id (str), scale (float), weight (float)
        """
        # ── Static lookup: id → state_id, store_id, cat_id, dept_id, item_id ──
        static_index = static_df.set_index("id").copy()
        static_index.index = static_index.index.astype(str)
        hier_cols = [c for c in ["state_id", "store_id", "cat_id", "dept_id", "item_id"]
                     if c in static_index.columns]

        # ── Raw wide matrix (all days, for L1-L11 scale denominators) ─────────
        # Copy only the 3 columns we need — not the full DataFrame.
        raw_slice = hist_df[["id", "date", self.target]].copy()
        raw_slice["id"] = raw_slice["id"].astype(str)
        raw_wide = (raw_slice
                    .pivot(index="id", columns="date", values=self.target)
                    .fillna(0).sort_index(axis=1))
        del raw_slice

        # Join hierarchy cols and immediately free the wide matrix to save RAM
        raw_joined    = raw_wide.join(static_index[hier_cols])
        raw_date_cols = list(raw_wide.columns)
        del raw_wide
        gc.collect()

        # ── Trimmed wide matrix (first sale → cutoff, for L12 scale denom) ───
        price_col = "sell_price"
        has_price = price_col in hist_df_trimmed.columns
        trim_cols = ["id", "date", self.target] + ([price_col] if has_price else [])
        trim_slice = hist_df_trimmed[trim_cols].copy()
        trim_slice["id"] = trim_slice["id"].astype(str)

        trimmed_wide = (trim_slice
                        .pivot(index="id", columns="date", values=self.target)
                        .fillna(0).sort_index(axis=1))

        trimmed_joined    = trimmed_wide.join(static_index[hier_cols])
        trimmed_date_cols = list(trimmed_wide.columns)
        last28_cols       = trimmed_date_cols[-28:]
        del trimmed_wide
        gc.collect()

        # ── Revenue last-28 matrix (trimmed × price, for weight computation) ──
        # Weights only use the sum of the last 28 days of revenue, so we build
        # a tiny price pivot over those 28 dates only — not the full 1913 days.
        if has_price:
            price_last28 = (
                trim_slice.loc[trim_slice["date"].isin(last28_cols),
                               ["id", "date", price_col]]
                .pivot(index="id", columns="date", values=price_col)
                .reindex(index=trimmed_joined.index, columns=last28_cols)
                .ffill(axis=1).bfill(axis=1).fillna(1.0)
            )
            # Multiply trimmed sales by price in the last 28-day window
            rev_last28 = trimmed_joined[last28_cols].multiply(price_last28)
            del price_last28
        else:
            # No price available — fall back to unit-volume weights
            rev_last28 = trimmed_joined[last28_cols].copy()

        rev_joined = rev_last28.join(static_index[hier_cols])
        del trim_slice, rev_last28
        gc.collect()

        # ── Per-level scale and weight computation ────────────────────────────
        records = []
        for lvl, cols in self.level_map.items():

            # Pick the right matrix for scale computation
            if lvl == 12:
                # L12: trimmed series (each row starts at first sale)
                agg_scale = (trimmed_joined.groupby(cols, observed=True)[trimmed_date_cols].sum()
                             if cols else trimmed_joined[trimmed_date_cols].sum().to_frame().T)
            else:
                # L1-L11: raw series (full history, needed for the BUG 1 fix in
                # the evaluator — scale denom uses all days, not just active ones)
                agg_scale = (raw_joined.groupby(cols, observed=True)[raw_date_cols].sum()
                             if cols else raw_joined[raw_date_cols].sum().to_frame().T)

            # Revenue aggregation — rev_joined already has only 28 date columns
            rev_agg = (rev_joined.groupby(cols, observed=True)[last28_cols].sum()
                       if cols else rev_joined[last28_cols].sum().to_frame().T)

            scales  = self._compute_scales(agg_scale.values.astype(np.float64))
            w_vals  = rev_agg.values.sum(axis=1)   # sum over the 28 revenue cols
            w_sum   = w_vals.sum()
            weights = w_vals / w_sum if w_sum > 0 else np.ones(len(w_vals)) / len(w_vals)

            # Build the series label for this level (e.g. "CA_FOODS" for level 6)
            if not cols:
                ids = ["All"]
            elif len(cols) == 1:
                ids = agg_scale.index.astype(str).tolist()
            else:
                ids = ["_".join(str(v) for v in idx) for idx in agg_scale.index]

            for i, id_ in enumerate(ids):
                records.append({"level": lvl, "id": id_,
                                "scale": float(scales[i]), "weight": float(weights[i])})
            print(f"    Level {lvl:>2}: {len(ids):,} series", flush=True)

        del raw_joined, trimmed_joined, rev_joined
        gc.collect()
        return pd.DataFrame(records)

    # ── Main entry point ──────────────────────────────────────────────────────

    def get_prepared_data(self, path, cutoff_day, level=12, force_reprepare=False):
        """
        Return (hist_df, hist_df_trimmed, future_df, static_df, weights_scales).

        On first call (or when force_reprepare=True): runs the full pipeline and
        writes all five parquet files to disk.
        On subsequent calls: reads directly from those cached files.

        The cache key is (tag, level, cutoff_day) — change any of them to get a
        fresh preparation without touching the others.
        """
        paths               = self._get_cache_paths(cutoff_day, level)
        hist_trimmed_path   = paths["hist"].replace("hist", "hist_trimmed")
        weights_scales_path = paths["hist"].replace("hist", "weights_scales")

        all_files_exist = (os.path.exists(paths["hist"])
                           and os.path.exists(hist_trimmed_path)
                           and os.path.exists(weights_scales_path))

        if all_files_exist and not force_reprepare:
            print(f"--- Cache Hit: Data found in {paths['folder']} ---")
            hist_df         = pd.read_parquet(paths["hist"])
            hist_df_trimmed = pd.read_parquet(hist_trimmed_path)
            future_df       = pd.read_parquet(paths["future"])
            static_df       = pd.read_parquet(paths["static"])
            weights_scales  = pd.read_parquet(weights_scales_path)
            # Parquet round-trips can promote float32 → float64; downcast back
            for df in (hist_df, hist_df_trimmed, future_df):
                self.optimize_dtypes(df)
            return hist_df, hist_df_trimmed, future_df, static_df, weights_scales

        reason = "Force Reprepare" if force_reprepare else "Cache Miss"
        print(f"--- {reason}: Preparing Level {level} for {cutoff_day} ---")

        hist_df, future_df, static_df = self.prepare(path, cutoff_day, level)
        hist_df_trimmed = self.trim_series_to_active(hist_df)

        print("  Building weights & scales …", flush=True)
        weights_scales = self._build_weights_scales(hist_df, hist_df_trimmed, static_df)

        os.makedirs(paths["folder"], exist_ok=True)
        hist_df.to_parquet(paths["hist"],              index=False)
        hist_df_trimmed.to_parquet(hist_trimmed_path,  index=False)
        future_df.to_parquet(paths["future"],          index=False)
        static_df.to_parquet(paths["static"],          index=False)
        weights_scales.to_parquet(weights_scales_path, index=False)

        return hist_df, hist_df_trimmed, future_df, static_df, weights_scales
