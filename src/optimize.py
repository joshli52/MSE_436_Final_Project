"""
optimize.py – Rank candidate conference host cities by total attendee travel cost.

For each candidate host city j and each source city i:
  - If i == j                : cost = 0
  - If pair is observed      : use ACTUAL fare from pairs_2025q4.parquet
  - Else                     : use model prediction 

Round-trip = 2 × fare (one-way average).

Usage (as a module)::

    from optimize import load_artifacts, run_optimizer
    arts = load_artifacts()
    result = run_optimizer(sources, arts)
"""

from __future__ import annotations

import difflib
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from model import (
    MODEL_FEATURES, SEED, TARGET,
    _add_log_cal_dist, _make_city_matrix,
)

ROOT = Path(__file__).parent.parent

# ── Artifact loading ──────────────────────────────────────────────────────────

def load_artifacts(model_dir: Path | None = None, root: Path = ROOT) -> dict:
    """
    Load the full inference bundle: pairs table, city coords, calibration
    coefficients, and the shipped model.

    Returns a dict with keys:
      pairs_df, coords_df, calib, model_bundle, pairs_lookup, all_cities
    """
    mdir = model_dir or root / "models"

    pairs_df  = pd.read_parquet(root / "pairs_2025q4.parquet")
    coords_df = pd.read_parquet(root / "city_coords.parquet")
    calib     = json.loads((root / "distance_calibration.json").read_text())

    shipped = json.loads((mdir / "shipped.json").read_text())["model"]
    with open(mdir / f"{shipped}.pkl", "rb") as f:
        model_bundle = pickle.load(f)

    # Build fast pair lookup: frozenset({cmid_a, cmid_b}) -> (fare, pair_id)
    pairs_lookup: dict[frozenset, tuple[float, str]] = {}
    for _, row in pairs_df.iterrows():
        key = frozenset({int(row["cmid_a"]), int(row["cmid_b"])})
        pairs_lookup[key] = (float(row["fare"]), str(row["pair_id"]))

    all_cities = sorted(set(coords_df["city"]))

    return {
        "pairs_df":     pairs_df,
        "coords_df":    coords_df,
        "calib":        calib,
        "model_bundle": model_bundle,
        "pairs_lookup": pairs_lookup,
        "all_cities":   all_cities,
    }


# ── City name matching ────────────────────────────────────────────────────────

def match_city(name: str, all_cities: list[str]) -> str:
    """
    Fuzzy-match `name` against the canonical city list.
    Returns the best match, or raises ValueError with near misses listed.
    """
    # Exact match first (case-insensitive)
    name_lc = name.strip().lower()
    for city in all_cities:
        if city.lower() == name_lc:
            return city

    # Fuzzy match
    matches = difflib.get_close_matches(name, all_cities, n=5, cutoff=0.4)
    if not matches:
        raise ValueError(
            f"City not found: {name!r}\n"
            f"No close matches in the 115-city dataset."
        )
    # If the top match is clearly dominant, accept it
    ratios = {m: difflib.SequenceMatcher(None, name_lc, m.lower()).ratio() for m in matches}
    best   = max(ratios, key=ratios.get)
    if ratios[best] >= 0.75:
        return best
    raise ValueError(
        f"City not found: {name!r}\n"
        f"Near misses (use one of these exact strings):\n"
        + "\n".join(f"  {m}" for m in matches)
    )


# ── Inference helpers ─────────────────────────────────────────────────────────

def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R    = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp   = math.radians(lat2 - lat1)
    dl   = math.radians(lon2 - lon1)
    a    = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(max(0, min(1, a))))


def _inference_features(
    city_a: str,
    city_b: str,
    coords_df: pd.DataFrame,
    calib: dict,
) -> dict[str, float]:
    """
    Build the MODEL_FEATURES dict for an unseen (city_a, city_b) pair.
    Uses city_coords.parquet for geocodes and city-level aggregates.
    """
    coords = coords_df.set_index("city")
    ca, cb = coords.loc[city_a], coords.loc[city_b]

    gc_km  = _haversine_km(ca["lon"], ca["lat"], cb["lon"], cb["lat"])
    cal_d  = calib["alpha"] + calib["beta"] * gc_km

    def sym(col_a: float, col_b: float) -> tuple[float, float]:
        return (col_a + col_b) / 2, abs(col_a - col_b)

    deg_m,  deg_a   = sym(ca["degree"],       cb["degree"])
    fl_m,   fl_a    = sym(ca["fare_residual"], cb["fare_residual"])
    mf_m,   mf_a    = sym(ca["median_fare"],   cb["median_fare"])
    pax_m,  pax_a   = sym(ca["total_pax"],     cb["total_pax"])
    lcc_m,  lcc_a   = sym(ca["lcc_share"],     cb["lcc_share"])
    lfms_m, lfms_a  = sym(ca["mean_lf_ms"],    cb["mean_lf_ms"])

    return {
        "calibrated_distance": cal_d,
        "log_cal_dist":        math.log(max(cal_d, 1.0)),
        "gc_distance":         gc_km,
        "deg_mean":            deg_m,
        "deg_absdiff":         deg_a,
        "farelevel_mean":      fl_m,
        "farelevel_absdiff":   fl_a,
        "medfare_mean":        mf_m,
        "medfare_absdiff":     mf_a,
        "pax_mean":            pax_m,
        "pax_absdiff":         pax_a,
        "lcc_share_mean":      lcc_m,
        "lcc_share_absdiff":   lcc_a,
        "lfms_mean":           lfms_m,
        "lfms_absdiff":        lfms_a,
    }


def _predict_fare(
    city_a: str,
    city_b: str,
    coords_df: pd.DataFrame,
    calib: dict,
    model_bundle: dict,
    all_cities: list[str],
) -> float:
    """Run inference with the shipped model for an unseen pair."""
    mtype = model_bundle["type"]
    feats = _inference_features(city_a, city_b, coords_df, calib)

    if mtype == "baseline1":
        model = model_bundle["model"]
        return float(model.predict([[feats["calibrated_distance"]]])[0])

    if mtype == "baseline2":
        model  = model_bundle["model"]
        cities = model_bundle["cities"]
        cidx   = {c: i for i, c in enumerate(cities)}
        log_d  = math.log(max(feats["calibrated_distance"], 1.0))
        city_v = np.zeros(len(cities))
        if city_a in cidx:
            city_v[cidx[city_a]] = 1.0
        if city_b in cidx:
            city_v[cidx[city_b]] = 1.0
        X = np.array([[log_d, *city_v]])
        return float(model.predict(X)[0])

    if mtype == "lgbm":
        model = model_bundle["model"]
        X     = pd.DataFrame([[feats[f] for f in MODEL_FEATURES]], columns=MODEL_FEATURES)
        return float(model.predict(X)[0])

    raise ValueError(f"Unknown model type: {mtype}")


# ── Pair cost lookup (observed or predicted) ──────────────────────────────────

def _pair_fare(
    city_i: str,
    cmid_i: int,
    city_j: str,
    cmid_j: int,
    pairs_lookup: dict,
    coords_df: pd.DataFrame,
    calib: dict,
    model_bundle: dict,
    all_cities: list[str],
) -> tuple[float, bool]:
    """
    Return (fare, is_imputed).
    Uses actual fare if the pair is in the top-1,000; otherwise predicts.
    """
    key = frozenset({cmid_i, cmid_j})
    if key in pairs_lookup:
        return pairs_lookup[key][0], False
    fare = _predict_fare(city_i, city_j, coords_df, calib, model_bundle, all_cities)
    return fare, True


# ── Optimizer ─────────────────────────────────────────────────────────────────

def run_optimizer(
    sources: list[dict],
    artifacts: dict,
    candidates: list[str] | None = None,
) -> pd.DataFrame:
    """
    Rank all candidate host cities by total round-trip travel cost.

    Parameters
    ----------
    sources    : list of {"city": str, "attendees": int}
    artifacts  : from load_artifacts()
    candidates : city strings to evaluate as host; defaults to all 115

    Returns
    -------
    DataFrame sorted by total_cost with columns:
      host, total_cost, cost_per_attendee,
      n_imputed, imputed_attendee_share, breakdown (list of per-source dicts)
    """
    pairs_lookup = artifacts["pairs_lookup"]
    coords_df    = artifacts["coords_df"]
    calib        = artifacts["calib"]
    model_bundle = artifacts["model_bundle"]
    all_cities   = artifacts["all_cities"]

    cmid_map = coords_df.set_index("city")["cmid"].to_dict()

    # Resolve and validate source cities
    resolved_sources = []
    for s in sources:
        city_raw  = s["city"]
        attendees = int(s["attendees"])
        city      = match_city(city_raw, all_cities)
        if city != city_raw:
            print(f"  Resolved {city_raw!r} → {city!r}")
        cmid = int(cmid_map[city])
        resolved_sources.append({"city": city, "cmid": cmid, "attendees": attendees})

    total_attendees = sum(s["attendees"] for s in resolved_sources)

    if candidates is None:
        candidates = all_cities

    rows = []
    for host in candidates:
        if host not in cmid_map:
            continue
        host_cmid = int(cmid_map[host])

        total_cost          = 0.0
        imputed_cost        = 0.0
        n_imputed           = 0
        source_breakdowns   = []

        for src in resolved_sources:
            if src["city"] == host:
                fare       = 0.0
                imputed    = False
            else:
                fare, imputed = _pair_fare(
                    src["city"], src["cmid"],
                    host, host_cmid,
                    pairs_lookup, coords_df, calib, model_bundle, all_cities,
                )
            round_trip = 2 * fare
            leg_cost   = src["attendees"] * round_trip
            total_cost += leg_cost
            if imputed:
                imputed_cost += leg_cost
                n_imputed    += 1

            source_breakdowns.append({
                "source":     src["city"],
                "attendees":  src["attendees"],
                "fare_ow":    round(fare, 2),
                "leg_cost":   round(leg_cost, 2),
                "imputed":    imputed,
            })

        imputed_share = imputed_cost / total_cost if total_cost > 0 else 0.0
        rows.append({
            "host":                  host,
            "total_cost":            round(total_cost, 2),
            "cost_per_attendee":     round(total_cost / total_attendees, 2),
            "n_imputed_legs":        n_imputed,
            "imputed_attendee_share": round(imputed_share, 4),
            "breakdown":             source_breakdowns,
        })

    result = (
        pd.DataFrame(rows)
        .sort_values("total_cost")
        .reset_index(drop=True)
    )
    result.index += 1   # 1-based rank
    return result


# Formatted output

def format_ranking(result: pd.DataFrame, top_n_detail: int = 5) -> str:
    """Return a printable ranked table with per-source detail for the top N."""
    lines = []
    lines.append(f"\n{'Rank':<5} {'Host City':<42} {'Total Cost':>12} {'$/Attendee':>12}")
    lines.append("-" * 75)

    for rank, row in result.iterrows():
        lines.append(
            f"{rank:<5} {row['host']:<42} ${row['total_cost']:>11,.0f} "
            f"${row['cost_per_attendee']:>11,.0f}"
        )

    # Detail breakdown for top N
    lines.append(f"\n── Top {top_n_detail} host detail ──")
    for rank, row in result.head(top_n_detail).iterrows():
        lines.append(f"\n  #{rank} {row['host']}  (total ${row['total_cost']:,.0f})")
        lines.append(f"  {'Source':<42} {'Attend':>6} {'Fare OW':>9} {'Leg Cost':>10}")
        lines.append("  " + "-" * 72)
        for bd in sorted(row["breakdown"], key=lambda x: -x["leg_cost"]):
            lines.append(
                f"  {bd['source']:<42} {bd['attendees']:>6} "
                f"${bd['fare_ow']:>8,.2f} ${bd['leg_cost']:>9,.0f}"
            )

    return "\n".join(lines)
