#!/usr/bin/env python3
"""
NBA Playoff Model Trainer
Combines regular season + playoff data (playoff games weighted 3x).
Walk-forward CV by season, then trains final playoff model.
Output: model/playoff_coefficients.json, playoff_scaler.json, playoff_metadata.json

Usage: python python/train_playoff_model.py
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, brier_score_loss
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR  = PROJECT_ROOT / "data"
MODEL_DIR = DATA_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

REG_CSV     = DATA_DIR / "training_data.csv"
PLAYOFF_CSV = DATA_DIR / "playoff_data.csv"

FEATURE_NAMES = [
    "elo_diff", "net_rtg_diff", "win_pct_diff",
    "ppg_diff", "papg_diff", "off_rtg_diff", "def_rtg_diff",
    "rest_days_diff", "is_home",
    "series_game_num", "series_deficit", "is_elimination_game",
]
PLAYOFF_WEIGHT = 3.0


def load_combined(feature_cols):
    reg = pd.read_csv(REG_CSV)
    reg["is_playoff"] = 0
    reg["weight"] = 1.0

    po = pd.read_csv(PLAYOFF_CSV)
    po["is_playoff"] = 1
    po["weight"] = PLAYOFF_WEIGHT

    # Align columns
    for c in feature_cols:
        if c not in reg.columns:
            reg[c] = 0.0
        if c not in po.columns:
            po[c] = 0.0

    combined = pd.concat([reg[feature_cols + ["label", "season", "is_playoff", "weight"]],
                          po[feature_cols + ["label", "season", "is_playoff", "weight"]]], ignore_index=True)
    return combined


def main():
    print("NBA Playoff Model Trainer")
    print("=" * 40)

    if not PLAYOFF_CSV.exists():
        print(f"No playoff data at {PLAYOFF_CSV}. Run fetch_playoff_data.py first.")
        sys.exit(1)

    feature_cols = [c for c in FEATURE_NAMES]
    df = load_combined(feature_cols)
    print(f"Combined: {len(df)} rows ({int((df['is_playoff']==0).sum())} reg + {int((df['is_playoff']==1).sum())} playoff)")

    po_only = df[df["is_playoff"] == 1]
    seasons = sorted(po_only["season"].unique())
    print(f"Playoff seasons: {seasons}")
    print()

    # Walk-forward CV on playoff games only (train on reg + prior playoff, test on this playoff season)
    print("Walk-forward results (test = playoff games only):")
    print(f"  {'Season':>8}  {'N':>4}  {'LR':>6}  {'XGB':>6}  {'Ens':>6}")
    print(f"  {'-'*8}  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*6}")

    lr_accs, xgb_accs, ens_accs = [], [], []

    all_seasons_sorted = sorted(df["season"].unique())

    for i, test_season in enumerate(seasons):
        # Training: all regular season data + all playoff seasons before this one
        train_df = df[
            (df["is_playoff"] == 0) |
            ((df["is_playoff"] == 1) & (df["season"].isin(seasons[:i])))
        ]
        test_df = df[(df["is_playoff"] == 1) & (df["season"] == test_season)]

        if len(train_df) < 50 or len(test_df) < 5:
            continue

        X_tr = train_df[feature_cols].fillna(0).values
        y_tr = train_df["label"].values
        w_tr = train_df["weight"].values
        X_te = test_df[feature_cols].fillna(0).values
        y_te = test_df["label"].values

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(X_tr_s, y_tr, sample_weight=w_tr)
        lr_p = np.clip(lr.predict_proba(X_te_s)[:, 1], 0.01, 0.99)
        lr_acc = accuracy_score(y_te, lr_p >= 0.5)
        lr_accs.append(lr_acc)

        xgb_p, ens_p = None, lr_p
        if HAS_XGB:
            xgb = XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_weight=5, verbosity=0, eval_metric='logloss',
            )
            xgb.fit(X_tr, y_tr, sample_weight=w_tr)
            xgb_p = np.clip(xgb.predict_proba(X_te)[:, 1], 0.01, 0.99)
            ens_p = (lr_p + xgb_p) / 2
            xgb_acc = accuracy_score(y_te, xgb_p >= 0.5)
            ens_acc = accuracy_score(y_te, ens_p >= 0.5)
            xgb_accs.append(xgb_acc)
            ens_accs.append(ens_acc)
            print(f"  {test_season:>8}  {len(test_df):>4}  {lr_acc:.3f}  {xgb_acc:.3f}  {ens_acc:.3f}")
        else:
            print(f"  {test_season:>8}  {len(test_df):>4}  {lr_acc:.3f}")

    if lr_accs:
        print(f"\nSummary:")
        print(f"  LR avg accuracy:      {np.mean(lr_accs):.4f}")
        if xgb_accs:
            print(f"  XGBoost avg accuracy: {np.mean(xgb_accs):.4f}")
        if ens_accs:
            print(f"  Ensemble avg accuracy:{np.mean(ens_accs):.4f}")

    # Train final model on ALL data
    print("\nTraining final model on all data...")
    X_all = df[feature_cols].fillna(0).values
    y_all = df["label"].values
    w_all = df["weight"].values
    sc_final = StandardScaler()
    X_all_s = sc_final.fit_transform(X_all)
    lr_final = LogisticRegression(C=1.0, max_iter=1000)
    lr_final.fit(X_all_s, y_all, sample_weight=w_all)

    # Save artifacts
    coef_path = MODEL_DIR / "playoff_coefficients.json"
    scaler_path = MODEL_DIR / "playoff_scaler.json"
    meta_path = MODEL_DIR / "playoff_metadata.json"

    with open(coef_path, "w") as f:
        json.dump({
            "intercept": float(lr_final.intercept_[0]),
            "coefficients": lr_final.coef_[0].tolist(),
            "feature_names": feature_cols,
        }, f, indent=2)

    with open(scaler_path, "w") as f:
        json.dump({
            "mean": sc_final.mean_.tolist(),
            "scale": sc_final.scale_.tolist(),
            "feature_names": feature_cols,
        }, f, indent=2)

    with open(meta_path, "w") as f:
        json.dump({
            "sport": "NBA",
            "model_type": "playoff_logistic_regression",
            "playoff_weight": PLAYOFF_WEIGHT,
            "cv_accuracy_lr": float(np.mean(lr_accs)) if lr_accs else None,
            "cv_accuracy_xgb": float(np.mean(xgb_accs)) if xgb_accs else None,
            "feature_names": feature_cols,
            "playoff_seasons": seasons,
        }, f, indent=2)

    print(f"Saved playoff model to {MODEL_DIR}/playoff_*.json")


if __name__ == "__main__":
    main()
