"""
model.py – Train three fare models with dual cross-validation and persist.

Models
------
  baseline1 : OLS  fare ~ calibrated_distance
  baseline2 : Ridge  fare ~ log(calibrated_distance) + city dummies (symmetric)
  lgbm      : LightGBM on allowed features, LOO aggregates refit per CV fold

CV splits
---------
  random    : 5-fold random over pairs – measures interpolation
  cold_city : 5-fold GroupKFold on cities – measures cold-city generalisation

Usage
-----
    python model.py [--pairs PATH] [--out-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).parent))
from features import LOOCityAggregator

ROOT = Path(__file__).parent.parent
PAIRS_FILE  = ROOT / "pairs_2025q4.parquet"
MODEL_DIR   = ROOT / "models"

SEED   = 42
TARGET = "fare"

# Features available at inference for ALL pairs (observed and unseen).
# nsmiles / log_nsmiles are BANNED: unavailable for unobserved pairs.
# calibrated_distance (= alpha + beta*gc_km) is the universal proxy.
DIST_FEATURES = ["calibrated_distance", "log_cal_dist", "gc_distance"]
AGG_FEATURES  = [
    "deg_mean",       "deg_absdiff",
    "farelevel_mean", "farelevel_absdiff",
    "medfare_mean",   "medfare_absdiff",
    "pax_mean",       "pax_absdiff",
    "lcc_share_mean", "lcc_share_absdiff",
    "lfms_mean",      "lfms_absdiff",
]
MODEL_FEATURES = DIST_FEATURES + AGG_FEATURES   # 15 cols


# ── Metric helpers ────────────────────────────────────────────────────────────

def _rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def _mape(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs((y - yhat) / y)) * 100)


def _metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    return {
        "mae":  float(mean_absolute_error(y, yhat)),
        "rmse": _rmse(y, yhat),
        "mape": _mape(y, yhat),
        "r2":   float(r2_score(y, yhat)),
    }


# ── Data helpers ──────────────────────────────────────────────────────────────

def _add_log_cal_dist(df: pd.DataFrame) -> pd.DataFrame:
    """Add log_cal_dist = log(calibrated_distance) column."""
    df = df.copy()
    df["log_cal_dist"] = np.log(df["calibrated_distance"].clip(lower=1.0))
    return df


def _reconstruct_raw(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Restore the original column names needed by LOOCityAggregator
    (passengers, carrier_low, lf_ms) from their raw_* counterparts.
    """
    df = pairs.copy()
    df["passengers"]  = df["raw_passengers"]
    df["carrier_low"] = df["raw_carrier_low"]
    df["lf_ms"]       = df["raw_lf_ms"]
    return df


def _make_city_matrix(df: pd.DataFrame, cities: list[str]) -> np.ndarray:
    """
    Symmetric city dummy matrix of shape (n_pairs, n_cities).
    Entry [i, j] = 1 if cities[j] is either city_a or city_b of pair i.
    """
    cidx = {c: k for k, c in enumerate(cities)}
    X    = np.zeros((len(df), len(cities)))
    for row_i, (_, row) in enumerate(df.iterrows()):
        for col in ("city_a", "city_b"):
            if row[col] in cidx:
                X[row_i, cidx[row[col]]] = 1.0
    return X


def _degree_bucket(deg: float) -> str:
    if deg <= 2:
        return "deg≤2"
    if deg <= 10:
        return "deg3-10"
    return "deg>10"


# ── CV split generators ───────────────────────────────────────────────────────

def random_splits(n: int, n_folds: int = 5, seed: int = SEED):
    """Yield (train_idx, test_idx) for stratified random 5-fold."""
    rng  = np.random.default_rng(seed)
    idx  = rng.permutation(n)
    fold = np.arange(n) % n_folds
    fold_assigned = np.empty(n, dtype=int)
    fold_assigned[idx] = fold
    for k in range(n_folds):
        yield np.where(fold_assigned != k)[0], np.where(fold_assigned == k)[0]


def cold_city_splits(df: pd.DataFrame, n_folds: int = 5, seed: int = SEED):
    """
    Yield (train_idx, test_idx) where test contains ALL pairs touching any
    city in the held-out group.  Pairs are test if EITHER city is held out.
    """
    cities = sorted(set(df["city_a"]) | set(df["city_b"]))
    rng    = np.random.default_rng(seed)
    perm   = rng.permutation(len(cities))
    c2f    = {cities[perm[i]]: i % n_folds for i in range(len(cities))}

    ca, cb = df["city_a"].values, df["city_b"].values
    for k in range(n_folds):
        test_mask = np.array([
            c2f.get(ca[i], -1) == k or c2f.get(cb[i], -1) == k
            for i in range(len(df))
        ])
        train_idx = np.where(~test_mask)[0]
        test_idx  = np.where( test_mask)[0]
        if len(train_idx) > 0 and len(test_idx) > 0:
            yield train_idx, test_idx


# ── Per-model CV routines ─────────────────────────────────────────────────────

def _cv_baseline1(
    raw_df: pd.DataFrame,
    splits: list[tuple],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS: fare ~ calibrated_distance."""
    preds, actuals, idxs = [], [], []
    for tr_idx, te_idx in splits:
        tr, te = raw_df.iloc[tr_idx], raw_df.iloc[te_idx]
        m = LinearRegression().fit(tr[["calibrated_distance"]].values, tr[TARGET].values)
        preds.extend(m.predict(te[["calibrated_distance"]].values))
        actuals.extend(te[TARGET].values)
        idxs.extend(te_idx)
    return np.array(preds), np.array(actuals), np.array(idxs)


def _cv_baseline2(
    raw_df: pd.DataFrame,
    splits: list[tuple],
    all_cities: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ridge: fare ~ log(calibrated_distance) + symmetric city dummies."""
    preds, actuals, idxs = [], [], []
    for tr_idx, te_idx in splits:
        tr, te = raw_df.iloc[tr_idx], raw_df.iloc[te_idx]
        log_tr = np.log(tr["calibrated_distance"].clip(1.0)).values.reshape(-1, 1)
        log_te = np.log(te["calibrated_distance"].clip(1.0)).values.reshape(-1, 1)
        X_tr   = np.hstack([log_tr, _make_city_matrix(tr, all_cities)])
        X_te   = np.hstack([log_te, _make_city_matrix(te, all_cities)])
        m      = Ridge(alpha=10.0, random_state=SEED).fit(X_tr, tr[TARGET].values)
        preds.extend(m.predict(X_te))
        actuals.extend(te[TARGET].values)
        idxs.extend(te_idx)
    return np.array(preds), np.array(actuals), np.array(idxs)


def _cv_lgbm(
    raw_df: pd.DataFrame,
    splits: list[tuple],
    params: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    LightGBM with LOO city aggregates refit on each training fold.
    params must NOT include n_jobs or random_state (added here).
    """
    preds, actuals, idxs = [], [], []
    for tr_idx, te_idx in splits:
        tr_raw = raw_df.iloc[tr_idx].copy()
        te_raw = raw_df.iloc[te_idx].copy()

        # Refit LOO aggregator on training fold only
        agg    = LOOCityAggregator().fit(tr_raw)
        tr_f   = _add_log_cal_dist(agg.transform(tr_raw))
        te_f   = _add_log_cal_dist(agg.transform(te_raw))

        m = lgb.LGBMRegressor(**params, random_state=SEED, n_jobs=1, verbose=-1)
        # Pass DataFrames (not numpy) so feature names stay consistent
        m.fit(tr_f[MODEL_FEATURES], tr_f[TARGET].values)
        preds.extend(m.predict(te_f[MODEL_FEATURES]))
        actuals.extend(te_f[TARGET].values)
        idxs.extend(te_idx)
    return np.array(preds), np.array(actuals), np.array(idxs)


# ── LightGBM hyperparameter random search ────────────────────────────────────

def _tune_lgbm(
    raw_df: pd.DataFrame,
    n_iter: int = 20,
    seed: int = SEED,
) -> dict:
    """
    Random search over LightGBM params using random-5-fold RMSE.
    Returns the best param dict found.
    """
    rng = np.random.default_rng(seed)

    def _sample() -> dict:
        return {
            "n_estimators":      int(rng.choice([150, 200, 250, 300])),
            "num_leaves":        int(rng.choice([15, 31])),
            "max_depth":         4,
            "min_child_samples": int(rng.choice([10, 20, 30])),
            "lambda_l1":         float(rng.choice([0.0, 0.1, 0.5, 1.0])),
            "lambda_l2":         float(rng.choice([0.0, 0.1, 0.5, 1.0])),
            "learning_rate":     float(rng.choice([0.05, 0.08, 0.10])),
            "subsample":         float(rng.choice([0.7, 0.8, 1.0])),
            "colsample_bytree":  float(rng.choice([0.7, 0.8, 1.0])),
        }

    best_rmse, best_params = np.inf, _sample()
    splits = list(random_splits(len(raw_df), seed=seed))

    print(f"  LightGBM random search ({n_iter} trials) …")
    for trial in range(n_iter):
        params = _sample()
        p, a, _ = _cv_lgbm(raw_df, splits, params)
        rmse = _rmse(a, p)
        if rmse < best_rmse:
            best_rmse, best_params = rmse, dict(params)
            print(f"    trial {trial:2d}  RMSE={rmse:.2f}  leaves={params['num_leaves']}  "
                  f"trees={params['n_estimators']}  lr={params['learning_rate']}")

    print(f"  Best RMSE={best_rmse:.2f}  params={best_params}")
    return best_params


# ── Degree-bucket breakdown ───────────────────────────────────────────────────

def _degree_metrics(
    preds: np.ndarray,
    actuals: np.ndarray,
    idxs: np.ndarray,
    raw_df: pd.DataFrame,
    coords: pd.DataFrame,
) -> dict[str, dict]:
    """Break out MAE and R² by city degree bucket (minimum of pair's two cities)."""
    deg_map = coords.set_index("city")["degree"].to_dict()
    bucket_data: dict[str, dict] = {}
    for pred, actual, idx in zip(preds, actuals, idxs):
        row = raw_df.iloc[idx]
        deg = min(deg_map.get(row["city_a"], 999), deg_map.get(row["city_b"], 999))
        bkt = _degree_bucket(deg)
        bucket_data.setdefault(bkt, {"preds": [], "actuals": []})
        bucket_data[bkt]["preds"].append(pred)
        bucket_data[bkt]["actuals"].append(actual)
    return {
        bkt: {**_metrics(np.array(v["actuals"]), np.array(v["preds"])), "n": len(v["preds"])}
        for bkt, v in bucket_data.items()
    }


# ── Final model fitting ───────────────────────────────────────────────────────

def fit_all_models(
    raw_df: pd.DataFrame,
    all_cities: list[str],
    lgbm_params: dict,
) -> tuple:
    """Fit Baseline1, Baseline2, LightGBM on the full dataset."""
    # Baseline 1
    b1 = LinearRegression().fit(
        raw_df[["calibrated_distance"]].values, raw_df[TARGET].values
    )
    # Baseline 2
    log_d = np.log(raw_df["calibrated_distance"].clip(1.0)).values.reshape(-1, 1)
    X_b2  = np.hstack([log_d, _make_city_matrix(raw_df, all_cities)])
    b2    = Ridge(alpha=10.0, random_state=SEED).fit(X_b2, raw_df[TARGET].values)
    # LightGBM with full-data LOO aggregator
    agg    = LOOCityAggregator().fit(raw_df)
    full   = _add_log_cal_dist(agg.transform(raw_df))
    lgbm_m = lgb.LGBMRegressor(**lgbm_params, random_state=SEED, n_jobs=1, verbose=-1)
    lgbm_m.fit(full[MODEL_FEATURES], full[TARGET].values)
    return b1, b2, lgbm_m, agg, all_cities


# ── Main ──────────────────────────────────────────────────────────────────────

def main(pairs_file: Path = PAIRS_FILE, out_dir: Path = ROOT) -> None:
    pairs  = pd.read_parquet(pairs_file)
    coords = pd.read_parquet(out_dir / "city_coords.parquet")

    raw_df     = _add_log_cal_dist(_reconstruct_raw(pairs))
    all_cities = sorted(set(raw_df["city_a"]) | set(raw_df["city_b"]))

    rand_splits = list(random_splits(len(raw_df)))
    cold_splits = list(cold_city_splits(raw_df))

    # ── Tune LightGBM once on random split ──
    lgbm_params = _tune_lgbm(raw_df)

    print("\n=== Cross-validation ===")
    metrics_rows: list[dict] = []

    for split_name, splits in [("random-5fold", rand_splits), ("cold-city", cold_splits)]:
        print(f"\n-- {split_name} --")

        p1, a1, i1 = _cv_baseline1(raw_df, splits)
        m1 = _metrics(a1, p1)
        print(f"  Baseline1  MAE={m1['mae']:.2f}  RMSE={m1['rmse']:.2f}  "
              f"MAPE={m1['mape']:.1f}%  R²={m1['r2']:.3f}")
        metrics_rows.append({"model": "baseline1", "split": split_name, **m1})

        p2, a2, i2 = _cv_baseline2(raw_df, splits, all_cities)
        m2 = _metrics(a2, p2)
        print(f"  Baseline2  MAE={m2['mae']:.2f}  RMSE={m2['rmse']:.2f}  "
              f"MAPE={m2['mape']:.1f}%  R²={m2['r2']:.3f}")
        metrics_rows.append({"model": "baseline2", "split": split_name, **m2})

        p3, a3, i3 = _cv_lgbm(raw_df, splits, lgbm_params)
        m3 = _metrics(a3, p3)
        print(f"  LightGBM   MAE={m3['mae']:.2f}  RMSE={m3['rmse']:.2f}  "
              f"MAPE={m3['mape']:.1f}%  R²={m3['r2']:.3f}")
        metrics_rows.append({"model": "lgbm", "split": split_name, **m3})

    # ── Degree-bucket breakdown ──
    print("\n=== Error by city degree bucket (LightGBM, random-5fold) ===")
    p3r, a3r, i3r = p3, a3, i3  # last ran on random-5fold? No - last was cold.
    # Re-run on random split to get degree breakdown
    p3r, a3r, i3r = _cv_lgbm(raw_df, list(random_splits(len(raw_df))), lgbm_params)
    deg_met = _degree_metrics(p3r, a3r, i3r, raw_df, coords)
    for bkt in sorted(deg_met):
        dm = deg_met[bkt]
        print(f"  {bkt:<10}  n={dm['n']:4d}  MAE={dm['mae']:.2f}  R²={dm['r2']:.3f}")

    # ── Decide shipped model ──
    b2_rmse = next(r["rmse"] for r in metrics_rows
                   if r["model"] == "baseline2" and r["split"] == "random-5fold")
    lg_rmse = next(r["rmse"] for r in metrics_rows
                   if r["model"] == "lgbm"      and r["split"] == "random-5fold")
    lgbm_wins  = lg_rmse < b2_rmse * 0.95   # must be >5% better to ship LGBM
    shipped    = "lgbm" if lgbm_wins else "baseline2"
    print(f"\nShipped model: {shipped}")
    if not lgbm_wins:
        print("  NOTE: LightGBM does not clearly beat Baseline 2; shipping Baseline 2.")

    # ── Fit final models on full data ──
    print("\nFitting final models on full data …")
    b1, b2, lgbm_m, agg, all_cities = fit_all_models(raw_df, all_cities, lgbm_params)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "baseline1.pkl", "wb") as f:
        pickle.dump({"type": "baseline1", "model": b1}, f)
    with open(MODEL_DIR / "baseline2.pkl", "wb") as f:
        pickle.dump({"type": "baseline2", "model": b2, "cities": all_cities}, f)
    with open(MODEL_DIR / "lgbm.pkl", "wb") as f:
        pickle.dump({"type": "lgbm", "model": lgbm_m,
                     "loo_aggregator": agg, "features": MODEL_FEATURES}, f)
    (MODEL_DIR / "shipped.json").write_text(json.dumps({"model": shipped}, indent=2))
    pd.DataFrame(metrics_rows).to_json(MODEL_DIR / "metrics.json",
                                       orient="records", indent=2)
    print(f"Models saved to {MODEL_DIR}")

    # ── Final metrics table ──
    print("\n" + "=" * 70)
    print("METRICS TABLE")
    print("=" * 70)
    fmt = "{:<12} {:<16} {:>8} {:>8} {:>8} {:>7}"
    print(fmt.format("Model", "Split", "MAE", "RMSE", "MAPE%", "R²"))
    print("-" * 70)
    for r in metrics_rows:
        print(fmt.format(
            r["model"], r["split"],
            f"{r['mae']:.2f}", f"{r['rmse']:.2f}",
            f"{r['mape']:.1f}", f"{r['r2']:.3f}",
        ))
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fare models")
    parser.add_argument("--pairs",   type=Path, default=PAIRS_FILE)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    main(args.pairs, args.out_dir)
