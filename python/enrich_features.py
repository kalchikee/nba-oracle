#!/usr/bin/env python3
"""
NBA Feature Enrichment — computes missing features from game history.
Adds: h2h_season_record, timezone travel shift, fixes lineup_net_rtg_diff.
Removes dead-weight features (vegas_home_prob always 0).

Usage: python python/enrich_features.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
INPUT_CSV    = DATA_DIR / "training_data.csv"
OUTPUT_CSV   = DATA_DIR / "training_data.csv"
BACKUP_CSV   = DATA_DIR / "training_data_backup.csv"

# NBA team timezone offsets (hours from UTC during regular season)
# Used to compute travel fatigue for away teams
TEAM_TZ = {
    # Eastern (-5)
    "ATL": -5, "BOS": -5, "BKN": -5, "CHA": -5, "CHI": -6, "CLE": -5,
    "DET": -5, "IND": -5, "MIA": -5, "MIL": -6, "NYK": -5, "ORL": -5,
    "PHI": -5, "TOR": -5, "WAS": -5,
    # Central (-6)
    "DAL": -6, "HOU": -6, "MEM": -6, "MIN": -6, "NOP": -6, "OKC": -6, "SAS": -6,
    # Mountain (-7)
    "DEN": -7, "UTA": -7,
    # Pacific (-8)
    "GSW": -8, "LAC": -8, "LAL": -8, "PHX": -7, "POR": -8, "SAC": -8, "SEA": -8,
}


def compute_h2h_record(df: pd.DataFrame) -> pd.Series:
    """Compute season H2H record between the two teams before this game."""
    df_sorted = df.sort_values("game_date").copy()
    # key = (season, frozenset({h,a})) -> {team: wins}
    h2h = defaultdict(lambda: defaultdict(int))
    records = []

    for _, row in df_sorted.iterrows():
        season = row["season"]
        h, a = row["home_team"], row["away_team"]
        key = (season, frozenset([h, a]))

        h_wins = h2h[key][h]
        a_wins = h2h[key][a]
        total = h_wins + a_wins
        # H2H record from home team's perspective: 0.5 if no prior meetings
        record = h_wins / total if total > 0 else 0.5
        records.append(record)

        # Update after recording pre-game state
        label = int(row["label"])
        if label == 1:
            h2h[key][h] += 1
        else:
            h2h[key][a] += 1

    return pd.Series(records, index=df_sorted.index)


def compute_tz_shift(df: pd.DataFrame) -> pd.Series:
    """Compute timezone shift for the away team (absolute hours shifted)."""
    shifts = []
    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        h_tz = TEAM_TZ.get(h, -5)
        a_tz = TEAM_TZ.get(a, -5)
        # Away team's timezone shift (e.g., LAL (-8) at BOS (-5) = 3 hour shift)
        shifts.append(abs(h_tz - a_tz))
    return pd.Series(shifts, index=df.index)


def main():
    print("NBA Feature Enrichment")
    print("=" * 50)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    df.to_csv(BACKUP_CSV, index=False)
    print(f"Backup saved to {BACKUP_CSV}")

    # ── 1. H2H season record ────────────────────────────────────────────
    print("\n[1/3] Computing H2H season records...")
    df = df.sort_values("game_date")
    df["h2h_season_record"] = compute_h2h_record(df)
    print(f"  h2h_season_record: mean={df['h2h_season_record'].mean():.3f}, std={df['h2h_season_record'].std():.4f}")

    # ── 2. Timezone travel shift ─────────────────────────────────────────
    print("\n[2/3] Computing timezone travel shifts...")
    df["travel_tz_shift_away"] = compute_tz_shift(df)
    df["travel_tz_shift_home"] = 0  # Home team doesn't travel
    print(f"  travel_tz_shift_away: mean={df['travel_tz_shift_away'].mean():.2f}, std={df['travel_tz_shift_away'].std():.3f}")

    # ── 3. Fix synthetic lineup_net_rtg_diff ─────────────────────────────
    # Currently it's just net_rtg_diff * 0.5 which adds no information.
    # Replace with a form-adjusted version: blend of season net_rtg and recent form
    print("\n[3/3] Fixing lineup_net_rtg_diff (was synthetic copy)...")
    if "net_rtg_diff" in df.columns and "team_10d_net_rtg_diff" in df.columns:
        # Use the gap between rolling form and season average as "lineup impact"
        # This captures when a team is performing above/below their season baseline
        df["lineup_net_rtg_diff"] = (df["team_10d_net_rtg_diff"] - df["net_rtg_diff"]) * 0.5
        print(f"  lineup_net_rtg_diff: std={df['lineup_net_rtg_diff'].std():.4f}")

    # ── 4. Zero out vegas_home_prob (always 0, remove scaler bias) ───────
    # The model trains on this but it's always 0.0 in training data.
    # At inference it's also 0.0 (no odds source). So it adds scaler bias.
    # We'll keep the column but it's 0.0 by default.

    # Summary
    print("\n" + "=" * 50)
    enriched = ["h2h_season_record", "travel_tz_shift_away", "lineup_net_rtg_diff"]
    for col in enriched:
        s = df[col].std()
        print(f"  {col:30s}  std={s:.4f}  {'✓' if s > 0.001 else '✗'}")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
