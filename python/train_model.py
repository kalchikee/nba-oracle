#!/usr/bin/env python3
"""
NBA Oracle v4.0 — ML Meta-Model Training
Trains a Logistic Regression on historical features built by build_dataset.py.
Exports model artifacts to data/model/ for the TypeScript runtime.

Workflow:
  1. python python/build_dataset.py   → data/training_data.csv
  2. python python/train_model.py     → data/model/*.json

Output files (data/model/):
  coefficients.json   — LR feature weights
  scaler.json         — StandardScaler (mean + scale per feature)
  calibration.json    — Isotonic regression calibration thresholds
  metadata.json       — Training metadata + walk-forward CV metrics
"""

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

# ─── Config ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "data" / "model"
CSV_PATH = PROJECT_ROOT / "data" / "training_data.csv"
DB_PATH = PROJECT_ROOT / "data" / "nba_oracle.db"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = [
    "elo_diff",
    "net_rtg_diff",
    "off_rtg_diff",
    "def_rtg_diff",
    "pace_diff",
    "pythagorean_diff",
    "log5_prob",
    "efg_pct_diff",
    "tov_pct_diff",
    "oreb_pct_diff",
    "ft_rate_diff",
    "three_pt_rate_diff",
    "three_pt_pct_diff",
    "ts_pct_diff",
    "ast_pct_diff",
    "stl_pct_diff",
    "blk_pct_diff",
    "team_10d_net_rtg_diff",
    "team_10d_off_rtg_diff",
    "momentum_diff",
    "rest_days_diff",
    "b2b_home",
    "b2b_away",
    "altitude_factor",
    "lineup_net_rtg_diff",
    "vegas_home_prob",
    "mc_win_pct",
]

# ─── Data loaders ─────────────────────────────────────────────────────────────

def load_from_csv() -> pd.DataFrame:
    if not CSV_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")
    return df


def load_from_db() -> pd.DataFrame:
    """Fall back to predictions already stored in the SQLite DB."""
    if not DB_PATH.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT p.game_date, p.feature_vector, p.mc_win_pct,
               p.correct, p.home_team
        FROM predictions p
        WHERE p.correct IS NOT NULL AND p.feature_vector IS NOT NULL
        ORDER BY p.game_date
        """,
        conn,
    )
    conn.close()

    if df.empty:
        return df

    def parse_fv(fv_str):
        try:
            fv = json.loads(fv_str)
            return {k: float(v) if v is not None else 0.0 for k, v in fv.items()}
        except Exception:
            return {}

    fv_df = pd.DataFrame(df["feature_vector"].apply(parse_fv).tolist())
    fv_df["mc_win_pct"] = df["mc_win_pct"].values
    df["label"] = df["correct"].astype(int)  # 1 = model was right ≠ home win
    # For this purpose label = home win (need actual result)
    # Use mc_win_pct ≥ 0.5 as home prediction and correct = 1 means that was right
    # Actually safer: label = 1 if home team won, reconstruct from correct + mc
    # If mc ≥ 0.5 and correct=1, home won. If mc < 0.5 and correct=1, away won.
    df["label"] = np.where(
        (df["mc_win_pct"] >= 0.5) & (df["correct"] == 1), 1,
        np.where((df["mc_win_pct"] < 0.5) & (df["correct"] == 0), 1, 0)
    )

    df = pd.concat([df.reset_index(drop=True), fv_df.reset_index(drop=True)], axis=1)
    print(f"Loaded {len(df)} rows from SQLite DB")
    return df


def load_data() -> pd.DataFrame:
    df = load_from_csv()
    if not df.empty:
        return df
    df = load_from_db()
    if not df.empty:
        print("Note: using DB predictions as fallback — run build_dataset.py for better training data")
        return df
    return pd.DataFrame()

# ─── Feature prep ─────────────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    for f in FEATURE_NAMES:
        if f not in df.columns:
            df[f] = 0.0
    X = df[FEATURE_NAMES].fillna(0.0).astype(float)
    y = df["label"].astype(int)
    return X, y

# ─── Walk-forward cross-validation ────────────────────────────────────────────

WALK_FORWARD_SPLITS = [
    ("2021-22", "2018-01-01", "2021-10-01", "2022-06-30"),
    ("2022-23", "2018-01-01", "2022-10-01", "2023-06-30"),
    ("2023-24", "2018-01-01", "2023-10-01", "2024-06-30"),
    ("2024-25", "2018-01-01", "2024-10-01", "2025-06-30"),
]

def walk_forward_cv(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> dict:
    print("\nWalk-forward cross-validation:")
    print(f"{'Split':<12} {'Train':<8} {'Test':<8} {'Brier':<8} {'Acc':<8} {'HiConv':<10} {'LogLoss'}")
    print("─" * 72)

    briers, accuracies, log_losses, hc_accs = [], [], [], []

    for test_label, train_start, train_end, test_end in WALK_FORWARD_SPLITS:
        train_mask = (df["game_date"] >= train_start) & (df["game_date"] < train_end)
        test_mask = (df["game_date"] >= train_end) & (df["game_date"] <= test_end)

        if train_mask.sum() < 100 or test_mask.sum() < 50:
            print(f"{test_label:<12} {'—':<8} {'—':<8}  insufficient data")
            continue

        X_tr, y_tr = X[train_mask].values, y[train_mask].values
        X_te, y_te = X[test_mask].values, y[test_mask].values

        # Scale
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Train
        model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
        model.fit(X_tr_s, y_tr)

        # Calibrate on held-out 20% of train
        split = int(0.8 * len(X_tr_s))
        cal = IsotonicRegression(out_of_bounds="clip")
        raw_cal = model.predict_proba(X_tr_s[split:])[:, 1]
        cal.fit(raw_cal, y_tr[split:])

        # Evaluate on test
        raw_te = model.predict_proba(X_te_s)[:, 1]
        preds = np.clip(cal.predict(raw_te), 0.01, 0.99)

        brier = brier_score_loss(y_te, preds)
        acc = accuracy_score(y_te, preds >= 0.5)
        ll = log_loss(y_te, preds)

        # High conviction: ≥67% on either side
        hc_mask = (preds >= 0.67) | (preds <= 0.33)
        hc_acc = accuracy_score(y_te[hc_mask], preds[hc_mask] >= 0.5) if hc_mask.sum() >= 10 else None

        briers.append(brier)
        accuracies.append(acc)
        log_losses.append(ll)
        if hc_acc is not None:
            hc_accs.append(hc_acc)

        hc_str = f"{hc_acc:.3f} ({hc_mask.sum()})" if hc_acc is not None else "—"
        print(
            f"{test_label:<12} {train_mask.sum():<8} {test_mask.sum():<8} "
            f"{brier:<8.4f} {acc:<8.3f} {hc_str:<10} {ll:.4f}"
        )

    if not briers:
        print("  Not enough data for walk-forward CV")
        return {"avg_brier": 0.25, "avg_accuracy": 0.50, "avg_log_loss": 0.70, "avg_hc_accuracy": None}

    avg_brier = float(np.mean(briers))
    avg_acc = float(np.mean(accuracies))
    avg_ll = float(np.mean(log_losses))
    avg_hc = float(np.mean(hc_accs)) if hc_accs else None

    print("─" * 72)
    hc_str = f"{avg_hc:.3f}" if avg_hc else "—"
    print(f"{'AVERAGE':<12} {'':8} {'':8} {avg_brier:<8.4f} {avg_acc:<8.3f} {hc_str:<10} {avg_ll:.4f}")

    return {
        "avg_brier": avg_brier,
        "avg_accuracy": avg_acc,
        "avg_log_loss": avg_ll,
        "avg_hc_accuracy": avg_hc,
    }

# ─── Final model training ─────────────────────────────────────────────────────

def train_final(X: pd.DataFrame, y: pd.Series) -> dict:
    print("\nTraining final model on all data...")

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X.values)

    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
    model.fit(X_s, y.values)

    # Calibrate on last 15% of data (most recent games)
    n = len(X_s)
    holdout = int(0.85 * n)
    raw_cal = model.predict_proba(X_s[holdout:])[:, 1]
    y_cal = y.values[holdout:]

    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(raw_cal, y_cal)

    # Feature importance
    feature_names = X.columns.tolist()
    importance = sorted(
        zip(feature_names, np.abs(model.coef_[0])),
        key=lambda x: x[1], reverse=True
    )
    print("\nTop 12 Feature Importances (|coefficient| after scaling):")
    for name, imp in importance[:12]:
        bar = "█" * int(imp * 30)
        print(f"  {name:<35} {imp:.4f}  {bar}")

    # Build artifacts
    coefficients = {"_intercept": float(model.intercept_[0])}
    for name, coef in zip(feature_names, model.coef_[0]):
        coefficients[name] = float(coef)

    scaler_artifact = {
        "feature_names": feature_names,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }

    calibration_artifact = {
        "method": "isotonic",
        "x_thresholds": cal.X_thresholds_.tolist(),
        "y_thresholds": cal.y_thresholds_.tolist(),
        "n_thresholds": len(cal.X_thresholds_),
    }

    return {
        "coefficients": coefficients,
        "scaler": scaler_artifact,
        "calibration": calibration_artifact,
    }

# ─── Save artifacts ───────────────────────────────────────────────────────────

def save_artifacts(artifacts: dict, cv_metrics: dict, n_samples: int,
                   date_range: tuple[str, str]) -> None:
    coeff_str = json.dumps(artifacts["coefficients"], sort_keys=True)
    weights_hash = hashlib.md5(coeff_str.encode()).hexdigest()[:12]

    metadata = {
        "version": "4.0.0",
        "model_type": "logistic_regression_isotonic_calibration",
        "feature_names": artifacts["scaler"]["feature_names"],
        "train_dates": f"{date_range[0]} to {date_range[1]}",
        "n_samples": n_samples,
        "avg_brier": cv_metrics["avg_brier"],
        "avg_accuracy": cv_metrics["avg_accuracy"],
        "avg_log_loss": cv_metrics["avg_log_loss"],
        "avg_hc_accuracy": cv_metrics.get("avg_hc_accuracy"),
        "weights_hash": weights_hash,
        "trained_at": datetime.now().isoformat(),
    }

    files = {
        "coefficients.json": artifacts["coefficients"],
        "scaler.json": artifacts["scaler"],
        "calibration.json": artifacts["calibration"],
        "metadata.json": metadata,
    }

    for fname, data in files.items():
        path = MODEL_DIR / fname
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved {path.name}")

    print(f"\n{'='*55}")
    print(f"Model trained on {n_samples:,} games ({date_range[0]} → {date_range[1]})")
    print(f"Walk-forward avg Brier:    {cv_metrics['avg_brier']:.4f}  (target: <0.220)")
    print(f"Walk-forward avg Accuracy: {cv_metrics['avg_accuracy']:.3f}  (target: >0.650)")
    if cv_metrics.get("avg_hc_accuracy"):
        print(f"High-conviction accuracy:  {cv_metrics['avg_hc_accuracy']:.3f}  (target: >0.730)")
    print(f"\nModel artifacts saved to {MODEL_DIR}")
    print("Restart the pipeline to activate: npm start")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train NBA Oracle ML model")
    parser.add_argument("--min-games", type=int, default=100,
                        help="Minimum games required to train (default: 100)")
    args = parser.parse_args()

    print("NBA Oracle v4.0 — ML Model Training")
    print("=" * 55)

    df = load_data()

    if df.empty or len(df) < args.min_games:
        n = len(df) if not df.empty else 0
        print(f"\nNot enough training data ({n} games, need {args.min_games}).")
        print("\nRun this first:")
        print("  python python/build_dataset.py")
        sys.exit(0)

    date_range = (str(df["game_date"].min()), str(df["game_date"].max()))
    print(f"Training data: {len(df):,} games ({date_range[0]} → {date_range[1]})")
    print(f"Home win rate: {df['label'].mean():.3f}")

    X, y = prepare_features(df)
    print(f"Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")

    # Walk-forward CV
    cv_metrics = walk_forward_cv(df, X, y)

    # Train final model
    artifacts = train_final(X, y)

    # Save
    print("\nSaving model artifacts...")
    save_artifacts(artifacts, cv_metrics, n_samples=len(df), date_range=date_range)


if __name__ == "__main__":
    main()
