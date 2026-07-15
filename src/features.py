"""
features.py – Canonicalize city pairs, compute leave-one-out city aggregates,
and assemble symmetric pair features for pairs_2025q4.parquet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# LCC / ULCC carrier codes
LCC_CODES: frozenset[str] = frozenset({"WN", "NK", "F9", "G4", "B6", "MX"})

SEED = 42


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------

def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add canonical pair columns so that every (citymarketid_1, citymarketid_2)
    pair is represented with the smaller ID first.

    Adds: cmid_a, cmid_b, city_a, city_b, pair_id
    """
    id1 = df["citymarketid_1"].astype(int)
    id2 = df["citymarketid_2"].astype(int)

    swap = id1 > id2

    df = df.copy()
    df["cmid_a"] = np.where(swap, id2, id1)
    df["cmid_b"] = np.where(swap, id1, id2)
    df["city_a"] = np.where(swap, df["city2"].values, df["city1"].values)
    df["city_b"] = np.where(swap, df["city1"].values, df["city2"].values)
    df["pair_id"] = df["cmid_a"].astype(str) + "__" + df["cmid_b"].astype(str)

    dups = df["pair_id"].duplicated().sum()
    assert dups == 0, f"Found {dups} duplicate canonical pair_ids after canonicalization"

    return df


# ---------------------------------------------------------------------------
# Leave-one-out city aggregate transformer
# ---------------------------------------------------------------------------

class LOOCityAggregator:
    """
    Compute city-level aggregates with leave-one-out exclusion.

    For each pair row, the aggregate for city A excludes that specific row
    so the target fare does not leak through the city-level statistic.

    The transformer can be fit on a subset (e.g. training fold) and applied
    to any rows, producing LOO aggregates from the fit data.

    Parameters
    ----------
    fit_df : DataFrame that has been canonicalized and has columns:
        cmid_a, cmid_b, city_a, city_b, fare, nsmiles, passengers,
        carrier_low, lf_ms

    After calling fit(), call transform(df) where df is the target rows
    (may equal fit_df for the full-data pipeline).
    """

    # Aggregate columns produced per city
    AGG_COLS = [
        "degree",
        "mean_fare",
        "median_fare",
        "fare_residual",   # city fare-level effect: mean(fare - fare_hat_from_log_dist)
        "total_pax",
        "lcc_share",
        "mean_lf_ms",
    ]

    def __init__(self) -> None:
        self._fit_df: pd.DataFrame | None = None
        self._log_dist_reg: LinearRegression | None = None

    def fit(self, df: pd.DataFrame) -> "LOOCityAggregator":
        """
        Fit the log-distance -> fare OLS on df, then store df for LOO lookups.
        df must already be canonicalized.
        """
        # Fit log-distance OLS to compute fare residuals
        mask = df["nsmiles"].notna() & df["fare"].notna() & (df["nsmiles"] > 0)
        X = np.log(df.loc[mask, "nsmiles"].astype(float)).values.reshape(-1, 1)
        y = df.loc[mask, "fare"].values

        reg = LinearRegression(fit_intercept=True)
        reg.fit(X, y)
        self._log_dist_reg = reg

        df = df.copy()
        df["_fare_hat"] = np.nan
        df.loc[mask, "_fare_hat"] = reg.predict(X)
        df["_fare_resid"] = df["fare"] - df["_fare_hat"]
        df["_is_lcc"] = df["carrier_low"].isin(LCC_CODES).astype(float)

        self._fit_df = df
        return self

    def _city_stats_excluding(self, city_cmid: int, exclude_pair_id: str) -> dict:
        """
        Compute city aggregates for city_cmid, excluding the row identified by
        exclude_pair_id (the pair that city_cmid is part of).
        """
        fdf = self._fit_df
        # rows where this city appears (either side)
        mask = (fdf["cmid_a"] == city_cmid) | (fdf["cmid_b"] == city_cmid)
        # leave-one-out exclusion
        mask &= fdf["pair_id"] != exclude_pair_id
        sub = fdf[mask]

        degree = int(mask.sum())  # number of OTHER pairs this city is in

        if degree == 0:
            return {
                "degree": 0,
                "mean_fare": np.nan,
                "median_fare": np.nan,
                "fare_residual": np.nan,
                "total_pax": np.nan,
                "lcc_share": np.nan,
                "mean_lf_ms": np.nan,
            }

        return {
            "degree": degree,
            "mean_fare": float(sub["fare"].mean()),
            "median_fare": float(sub["fare"].median()),
            "fare_residual": float(sub["_fare_resid"].mean()),
            "total_pax": float(sub["passengers"].sum()),
            "lcc_share": float(sub["_is_lcc"].mean()),
            "mean_lf_ms": float(sub["lf_ms"].mean()),
        }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add LOO city-level aggregates for both cities in each pair, then
        compute symmetric (mean, absdiff) pair features.

        Returns df with new columns appended.
        """
        assert self._fit_df is not None, "Call .fit() before .transform()"

        records_a: list[dict] = []
        records_b: list[dict] = []

        for _, row in df.iterrows():
            pid = row["pair_id"]
            stats_a = self._city_stats_excluding(int(row["cmid_a"]), pid)
            stats_b = self._city_stats_excluding(int(row["cmid_b"]), pid)
            records_a.append(stats_a)
            records_b.append(stats_b)

        agg_a = pd.DataFrame(records_a, index=df.index).add_prefix("a_")
        agg_b = pd.DataFrame(records_b, index=df.index).add_prefix("b_")
        df = pd.concat([df, agg_a, agg_b], axis=1)

        # Symmetric pair features
        def _sym(col: str, suffix: str) -> tuple[str, str]:
            return f"{suffix}_mean", f"{suffix}_absdiff"

        feature_map = {
            "degree": "deg",
            "fare_residual": "farelevel",
            "median_fare": "medfare",
            "total_pax": "pax",
            "lcc_share": "lcc_share",
            "mean_lf_ms": "lfms",
        }

        for agg_col, feat_prefix in feature_map.items():
            ca = f"a_{agg_col}"
            cb = f"b_{agg_col}"
            df[f"{feat_prefix}_mean"] = (df[ca] + df[cb]) / 2
            df[f"{feat_prefix}_absdiff"] = (df[ca] - df[cb]).abs()

        return df

    def get_city_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a city-level summary table (LOO aggregates for each city,
        averaged across the city's own pairs).

        Returns a DataFrame with one row per city.
        """
        assert self._fit_df is not None
        fdf = self._fit_df

        city_ids = pd.concat([fdf["cmid_a"], fdf["cmid_b"]]).unique()
        rows = []
        for cid in city_ids:
            # Get city name from fit_df
            name_a = fdf.loc[fdf["cmid_a"] == cid, "city_a"]
            name_b = fdf.loc[fdf["cmid_b"] == cid, "city_b"]
            city_name = pd.concat([name_a, name_b]).iloc[0] if len(name_a) + len(name_b) > 0 else ""

            # For the city table we compute aggregates over ALL pairs the city is in
            # (no exclusion – used for city_coords.parquet summary, not as model input)
            mask = (fdf["cmid_a"] == cid) | (fdf["cmid_b"] == cid)
            sub = fdf[mask]
            rows.append({
                "cmid": cid,
                "city": city_name,
                "degree": int(mask.sum()),
                "mean_fare": float(sub["fare"].mean()),
                "median_fare": float(sub["fare"].median()),
                "fare_residual": float(sub["_fare_resid"].mean()),
                "total_pax": float(sub["passengers"].sum()),
                "lcc_share": float(sub["_is_lcc"].mean()),
                "mean_lf_ms": float(sub["lf_ms"].mean()),
            })

        return pd.DataFrame(rows).sort_values("cmid").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main feature assembly
# ---------------------------------------------------------------------------

def build_features(model_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a cleaned, geocoded 2025-Q4 DataFrame (from data.get_model_data),
    return the full feature table matching the output contract.

    Columns produced:
      pair_id, city_a, city_b, cmid_a, cmid_b,
      fare,
      nsmiles, log_nsmiles, gc_distance, calibrated_distance,
      deg_mean, deg_absdiff, farelevel_mean, farelevel_absdiff,
      medfare_mean, medfare_absdiff, pax_mean, pax_absdiff,
      lcc_share_mean, lcc_share_absdiff, lfms_mean, lfms_absdiff,
      raw_passengers, raw_carrier_lg, raw_large_ms, raw_fare_lg,
      raw_carrier_low, raw_lf_ms, raw_fare_low
    """
    df = canonicalize(model_df)

    # Distance features
    df["log_nsmiles"] = np.log(df["nsmiles"].astype(float).replace(0, np.nan))

    # LOO city aggregates
    agg = LOOCityAggregator()
    agg.fit(df)
    df = agg.transform(df)

    # Rename raw columns (banned features kept for EDA)
    raw_rename = {
        "passengers": "raw_passengers",
        "carrier_lg": "raw_carrier_lg",
        "large_ms": "raw_large_ms",
        "fare_lg": "raw_fare_lg",
        "carrier_low": "raw_carrier_low",
        "lf_ms": "raw_lf_ms",
        "fare_low": "raw_fare_low",
    }
    df = df.rename(columns=raw_rename)

    # Select output columns in contract order
    out_cols = [
        "pair_id", "city_a", "city_b", "cmid_a", "cmid_b",
        "fare",
        "nsmiles", "log_nsmiles", "gc_distance", "calibrated_distance",
        "deg_mean", "deg_absdiff",
        "farelevel_mean", "farelevel_absdiff",
        "medfare_mean", "medfare_absdiff",
        "pax_mean", "pax_absdiff",
        "lcc_share_mean", "lcc_share_absdiff",
        "lfms_mean", "lfms_absdiff",
        "raw_passengers", "raw_carrier_lg", "raw_large_ms", "raw_fare_lg",
        "raw_carrier_low", "raw_lf_ms", "raw_fare_low",
    ]

    missing = [c for c in out_cols if c not in df.columns]
    assert not missing, f"Missing output columns: {missing}"

    return df[out_cols].reset_index(drop=True), agg
