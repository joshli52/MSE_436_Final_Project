"""
build.py – CLI that runs the full data pipeline and writes output files.

Usage:
    python build.py [--force-download] [--out-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from project root or from src/
sys.path.insert(0, str(Path(__file__).parent))

from data import get_model_data, COORDS_FILE, CALIB_FILE
from features import build_features

DEFAULT_OUT_DIR = Path(__file__).parent.parent


def print_report(pairs: pd.DataFrame, city_coords: pd.DataFrame) -> None:
    """Print a short data quality report to stdout."""
    print("\n" + "=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)

    n_pairs = len(pairs)
    n_cities = len(city_coords)
    all_possible = n_cities * (n_cities - 1) // 2
    coverage = n_pairs / all_possible

    print(f"\nRow / city counts")
    print(f"  Pairs in output      : {n_pairs:,}")
    print(f"  Unique cities        : {n_cities}")
    print(f"  All possible pairs   : {all_possible:,}")
    print(f"  Observed pair coverage: {coverage:.2%}")

    fare = pairs["fare"].dropna()
    print(f"\nFare distribution (one-way average, USD)")
    print(f"  min    : ${fare.min():.2f}")
    print(f"  p10    : ${fare.quantile(0.10):.2f}")
    print(f"  median : ${fare.median():.2f}")
    print(f"  mean   : ${fare.mean():.2f}")
    print(f"  p90    : ${fare.quantile(0.90):.2f}")
    print(f"  max    : ${fare.max():.2f}")
    print(f"  null   : {fare.isna().sum()}")

    deg = city_coords["degree"]
    print(f"\nCity degree distribution (# pairs per city)")
    print(f"  min    : {int(deg.min())}")
    print(f"  median : {deg.median():.1f}")
    print(f"  mean   : {deg.mean():.1f}")
    print(f"  max    : {int(deg.max())}")

    print(f"\nDistance calibration")
    calib_path = CALIB_FILE
    if calib_path.exists():
        coef = json.loads(calib_path.read_text())
        print(f"  alpha={coef['alpha']:.2f}  beta={coef['beta']:.4f}  R²={coef['r2']:.4f}")

    print("\nNull counts in output parquet:")
    null_counts = pairs.isnull().sum()
    nc = null_counts[null_counts > 0]
    if len(nc):
        print(nc.to_string())
    else:
        print("  (none)")

    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pairs_2025q4.parquet flight fare dataset")
    parser.add_argument("--force-download", action="store_true",
                        help="Re-download raw data even if cache exists")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="Directory to write output files (default: project root)")
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: ingest + clean + geocode + distance calibration -----------
    print("\n[1/3] Ingesting and cleaning data …")
    model_df, coords_df, calib = get_model_data(force_download=args.force_download)

    # ---- Step 2: build features --------------------------------------------
    print("\n[2/3] Building features …")
    pairs_df, agg = build_features(model_df)

    # ---- Step 3: enrich city_coords with city aggregates -------------------
    print("\n[3/3] Writing output files …")
    city_table = agg.get_city_table(model_df)

    # Merge lon/lat into city table
    city_name_to_coords = coords_df.set_index("city")[["lon", "lat"]]
    city_table = city_table.join(city_name_to_coords, on="city", how="left")

    city_out = out_dir / "city_coords.parquet"
    city_table.to_parquet(city_out, index=False)
    print(f"  city_coords.parquet  -> {city_out}  ({len(city_table)} rows)")

    pairs_out = out_dir / "pairs_2025q4.parquet"
    pairs_df.to_parquet(pairs_out, index=False)
    print(f"  pairs_2025q4.parquet -> {pairs_out}  ({len(pairs_df)} rows, {len(pairs_df.columns)} cols)")

    calib_out = out_dir / "distance_calibration.json"
    calib_out.write_text(json.dumps(calib, indent=2))
    print(f"  distance_calibration.json -> {calib_out}")

    # ---- Data quality report -----------------------------------------------
    print_report(pairs_df, city_table)


if __name__ == "__main__":
    main()
