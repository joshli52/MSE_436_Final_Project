"""
data.py – Ingest DOT Consumer Airfare Report Table 1 from the Socrata API,
cache raw pulls to parquet, clean/parse, backfill geocodes from historical
rows, calibrate great-circle distance to nsmiles, and expose a single
get_model_data() entry point.

Data source (same dataset, SODA resource endpoint includes location columns):
  https://data.transportation.gov/resource/4f3n-jbg2.json
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression

# Constants

# SODA resource endpoint – same dataset, includes location_1/location_2 columns
RESOURCE_URL = "https://data.transportation.gov/resource/4f3n-jbg2.json"
PAGE_SIZE = 10_000
MAX_RETRIES = 5
BACKOFF_BASE = 2.0  # seconds, exponential backoff factor

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_FILE = CACHE_DIR / "raw_airfare.parquet"
COORDS_FILE = Path(__file__).parent.parent / "city_coords.parquet"
CALIB_FILE = Path(__file__).parent.parent / "distance_calibration.json"

SEED = 42


# API paging with retry, avoiding rate limits

def _fetch_page(offset: int, limit: int = PAGE_SIZE) -> list[dict]:
    """Fetch one page from the Socrata SODA API with exponential backoff."""
    params = {
        "$limit": limit,
        "$offset": offset,
        "$order": "year ASC, quarter ASC, citymarketid_1 ASC",
    }
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(RESOURCE_URL, params=params, timeout=120)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            sleep = BACKOFF_BASE ** attempt
            print(f"  [retry {attempt + 1}] {exc} – sleeping {sleep:.1f}s")
            time.sleep(sleep)
    return []


def _fetch_all(force: bool = False) -> pd.DataFrame:
    """
    Page through the entire dataset. Returns a DataFrame of raw rows.
    Caches result to CACHE_FILE; set force=True to re-download.
    """
    if not force and CACHE_FILE.exists():
        print(f"Loading cached raw data from {CACHE_FILE}")
        return pd.read_parquet(CACHE_FILE)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("Fetching data from Socrata API …")

    all_rows: list[dict] = []
    offset = 0
    while True:
        page = _fetch_page(offset)
        if not page:
            break
        all_rows.extend(page)
        print(f"  fetched {len(all_rows):,} rows", end="\r")
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    print(f"\nTotal raw rows: {len(all_rows):,}")
    df = pd.DataFrame(all_rows)
    df.to_parquet(CACHE_FILE, index=False)
    print(f"Cached to {CACHE_FILE}")
    return df


# Parsing / cleaning

def _strip_dollar_comma(s: pd.Series) -> pd.Series:
    """Remove leading '$' and embedded commas, then cast to float."""
    return (
        s.astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        .astype(float)
    )


def _extract_lon_lat(col: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Extract lon, lat from a column whose values are either:
      - GeoJSON dicts: {"type": "Point", "coordinates": [lon, lat]}
      - POINT strings: "POINT (lon lat)"
      - None / NaN

    Returns two float Series: (lon, lat).
    """
    lon_vals: list[float | None] = []
    lat_vals: list[float | None] = []

    for val in col:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            lon_vals.append(None)
            lat_vals.append(None)
            continue
        if isinstance(val, dict):
            # GeoJSON Point
            try:
                coords = val["coordinates"]
                lon_vals.append(float(coords[0]))
                lat_vals.append(float(coords[1]))
            except (KeyError, IndexError, TypeError, ValueError):
                lon_vals.append(None)
                lat_vals.append(None)
        elif isinstance(val, str):
            # "POINT (lon lat)"
            import re
            m = re.match(r"POINT\s*\(([^\s]+)\s+([^\s]+)\)", val.strip())
            if m:
                lon_vals.append(float(m.group(1)))
                lat_vals.append(float(m.group(2)))
            else:
                lon_vals.append(None)
                lat_vals.append(None)
        else:
            lon_vals.append(None)
            lat_vals.append(None)

    return (
        pd.Series(lon_vals, index=col.index, dtype=float),
        pd.Series(lat_vals, index=col.index, dtype=float),
    )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise column names, parse string-encoded numerics, drop table_1_flag.
    Returns a cleaned DataFrame with proper dtypes.
    """
    df = df.copy()

    # Drop Socrata metadata columns
    meta_cols = [c for c in df.columns if str(c).startswith(":")]
    df = df.drop(columns=meta_cols, errors="ignore")
    # Drop table_1_flag
    df = df.drop(columns=["table_1_flag"], errors="ignore")

    # Year / quarter
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["quarter"] = pd.to_numeric(df["quarter"], errors="coerce").astype("Int64")

    # citymarketid – strip commas, to int
    for col in ("citymarketid_1", "citymarketid_2"):
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
                .replace({"nan": np.nan, "None": np.nan, "": np.nan})
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # nsmiles
    df["nsmiles"] = _strip_dollar_comma(df["nsmiles"]).astype("Int64")

    # passengers – comes in as float string, round to int
    df["passengers"] = _strip_dollar_comma(df["passengers"]).round(0).astype("Int64")

    # fare columns
    for col in ("fare", "fare_lg", "fare_low"):
        if col in df.columns:
            df[col] = _strip_dollar_comma(df[col])

    # market shares
    for col in ("large_ms", "lf_ms"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # string columns
    for col in ("city1", "city2", "carrier_lg", "carrier_low"):
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.strip()
                .replace({"nan": np.nan, "None": np.nan})
            )

    return df


# ---------------------------------------------------------------------------
# Geocode backfill
# ---------------------------------------------------------------------------

# Cities that appear in 2025-Q4 but have zero geocoded rows in the entire
# DOT dataset (verified by querying all years). Hard-coded as a fallback so
# the 115/115 assertion passes. Coordinates are city-centre lat/lon.
_GEOCODE_FALLBACK: dict[str, tuple[float, float]] = {
    "Charlottesville, VA": (-78.4767, 38.0293),
    "New Haven, CT": (-72.9246, 41.3083),
    "Vero Beach, FL": (-80.3976, 27.6386),
}


def _build_geocode_lookup(hist: pd.DataFrame) -> pd.DataFrame:
    """
    From ALL rows, extract the first non-null GeoJSON geocode per city name.
    Uses location_1 / location_2 columns from the SODA resource endpoint.
    Cities absent from the API geocodes are resolved from _GEOCODE_FALLBACK.

    Returns DataFrame with columns: city, lon, lat.
    """
    records: list[dict] = []

    for side in ("1", "2"):
        city_col = f"city{side}"
        loc_col = f"location_{side}"
        if city_col not in hist.columns or loc_col not in hist.columns:
            continue
        sub = hist[[city_col, loc_col]].dropna(subset=[loc_col])
        # Keep only rows where location is a non-empty dict
        sub = sub[sub[loc_col].apply(lambda v: isinstance(v, dict) and bool(v))]
        if sub.empty:
            continue
        lon, lat = _extract_lon_lat(sub[loc_col])
        tmp = pd.DataFrame({
            "city": sub[city_col].values,
            "lon": lon.values,
            "lat": lat.values,
        })
        tmp = tmp.dropna()
        records.append(tmp)

    if not records:
        return pd.DataFrame(columns=["city", "lon", "lat"])

    combined = pd.concat(records, ignore_index=True)
    lookup = (
        combined.drop_duplicates(subset=["city"], keep="first")
        .reset_index(drop=True)
    )

    # Append fallback coords for cities with no geocoded rows anywhere in API
    missing_fb = {
        city: coords
        for city, coords in _GEOCODE_FALLBACK.items()
        if city not in lookup["city"].values
    }
    if missing_fb:
        fb_rows = pd.DataFrame([
            {"city": c, "lon": lon, "lat": lat}
            for c, (lon, lat) in missing_fb.items()
        ])
        print(f"  Using hardcoded fallback coords for: {list(missing_fb.keys())}")
        lookup = pd.concat([lookup, fb_rows], ignore_index=True)

    return lookup


# ---------------------------------------------------------------------------
# Great-circle distance + calibration
# ---------------------------------------------------------------------------

def haversine_series(
    lon1: pd.Series, lat1: pd.Series,
    lon2: pd.Series, lat2: pd.Series,
) -> pd.Series:
    """Vectorised haversine distance in km."""
    R = 6371.0
    phi1 = np.radians(lat1.values.astype(float))
    phi2 = np.radians(lat2.values.astype(float))
    dphi = np.radians((lat2 - lat1).values.astype(float))
    dlambda = np.radians((lon2 - lon1).values.astype(float))
    a = (np.sin(dphi / 2) ** 2
         + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2)
    return pd.Series(2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1))),
                     index=lon1.index)


def calibrate_distance(
    gc_km: pd.Series,
    nsmiles: pd.Series,
    out_path: Path = CALIB_FILE,
) -> tuple[float, float, float]:
    """
    OLS: nsmiles ~ alpha + beta * gc_km.
    Persists coefficients to JSON. Returns (alpha, beta, r2).
    """
    mask = gc_km.notna() & nsmiles.notna()
    X = gc_km[mask].values.reshape(-1, 1)
    y = nsmiles[mask].values.astype(float)

    reg = LinearRegression(fit_intercept=True)
    reg.fit(X, y)
    r2 = float(reg.score(X, y))
    alpha = float(reg.intercept_)
    beta = float(reg.coef_[0])

    coefs = {"alpha": alpha, "beta": beta, "r2": r2, "n": int(mask.sum())}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(coefs, indent=2))
    print(f"Distance calibration  alpha={alpha:.2f}  beta={beta:.4f}  R²={r2:.4f}  n={mask.sum()}")
    return alpha, beta, r2


def apply_calibration(gc_km: pd.Series, alpha: float, beta: float) -> pd.Series:
    """Convert great-circle km to calibrated nsmiles scale."""
    return alpha + beta * gc_km


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_model_data(
    force_download: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Full pipeline:
      1. Download / load cached raw data (all years).
      2. Clean and parse.
      3. Build geocode lookup from historical rows (location_1/2 columns).
      4. Filter to 2025-Q4 model rows; assert 1 000 rows and 115 cities.
      5. Join geocodes; assert 115/115 resolve.
      6. Calibrate great-circle -> nsmiles; persist coefficients.
      7. Persist city_coords.parquet.

    Returns
    -------
    model_df   : 2025-Q4 cleaned DataFrame with gc_distance & calibrated_distance
    coords_df  : city -> (lon, lat) lookup
    calib      : dict with alpha, beta, r2
    """
    raw = _fetch_all(force=force_download)
    df = clean(raw)

    # ---- diagnostic summary ------------------------------------------------
    years = sorted(df["year"].dropna().unique())
    quarters = sorted(df["quarter"].dropna().unique())
    cities = pd.concat([df["city1"], df["city2"]]).dropna().unique()
    print(f"\nFull dataset: {len(df):,} rows")
    print(f"  Year range : {int(years[0])} – {int(years[-1])}")
    print(f"  Quarters   : {[int(q) for q in quarters]}")
    print(f"  Unique cities : {len(cities):,}")
    print("  Null counts per column:")
    null_counts = df.isnull().sum()
    print(null_counts[null_counts > 0].to_string())

    # ---- geocode lookup from ALL historical rows ----------------------------
    coords_df = _build_geocode_lookup(raw)  # use raw df so location cols present
    print(f"\nGeocodes resolved for {len(coords_df):,} city names")

    # ---- filter to model rows ----------------------------------------------
    model_df = (
        df[(df["year"] == 2025) & (df["quarter"] == 4)]
        .copy()
        .reset_index(drop=True)
    )
    print(f"\n2025-Q4 rows: {len(model_df)}")
    assert len(model_df) == 1_000, (
        f"Expected exactly 1 000 rows for 2025-Q4, got {len(model_df)}"
    )

    model_cities = pd.concat([model_df["city1"], model_df["city2"]]).dropna().unique()
    n_cities = len(model_cities)
    print(f"Unique cities in 2025-Q4: {n_cities}")
    assert n_cities == 115, f"Expected 115 cities, got {n_cities}"

    # ---- join geocodes onto model rows -------------------------------------
    city_coord_map = coords_df.set_index("city")[["lon", "lat"]]
    missing = [c for c in model_cities if c not in city_coord_map.index]
    if missing:
        raise RuntimeError(
            f"Geocode lookup FAILED for {len(missing)} cities:\n" + "\n".join(missing)
        )

    model_df = model_df.join(
        city_coord_map.rename(columns={"lon": "lon1", "lat": "lat1"}), on="city1"
    )
    model_df = model_df.join(
        city_coord_map.rename(columns={"lon": "lon2", "lat": "lat2"}), on="city2"
    )

    # ---- great-circle distance & calibration --------------------------------
    model_df["gc_distance"] = haversine_series(
        model_df["lon1"], model_df["lat1"],
        model_df["lon2"], model_df["lat2"],
    )

    alpha, beta, r2 = calibrate_distance(
        model_df["gc_distance"], model_df["nsmiles"].astype(float)
    )
    calib = {"alpha": alpha, "beta": beta, "r2": r2,
              "n": int(model_df["gc_distance"].notna().sum())}

    model_df["calibrated_distance"] = apply_calibration(
        model_df["gc_distance"], alpha, beta
    )

    # ---- persist city coords (enriched with aggregates in build.py) --------
    coords_df.to_parquet(COORDS_FILE, index=False)
    print(f"City coords saved to {COORDS_FILE}  ({len(coords_df)} cities)")

    return model_df, coords_df, calib
