#!/usr/bin/env python3
"""
NBA Oracle v4.0 — Walk-Forward Backtest Engine
Replays historical seasons day-by-day with sequential Elo updates.
No lookahead bias: stats computed only from prior games.

Usage:
  python python/backtest.py                    # backtest all available seasons
  python python/backtest.py --season 2024-25  # specific season only
  python python/backtest.py --output report.json

Requirements:
  pip install scikit-learn pandas numpy requests
"""

import argparse
import json
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler

# ─── Config ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "nba_oracle.db"
MODEL_DIR = PROJECT_ROOT / "data" / "model"

LEAGUE_MEAN_ELO = 1500
K_FACTOR = 20
HOME_ADV_ELO = 100

NBA_STATS_BASE = "https://stats.nba.com/stats"
NBA_HEADERS = {
    "Accept": "application/json",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/121.0.0.0 Safari/537.36"
    ),
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

FEATURE_NAMES = [
    "elo_diff", "net_rtg_diff", "off_rtg_diff", "def_rtg_diff", "pace_diff",
    "pythagorean_diff", "log5_prob", "efg_pct_diff", "tov_pct_diff",
    "oreb_pct_diff", "ft_rate_diff", "three_pt_rate_diff", "three_pt_pct_diff",
    "ts_pct_diff", "ast_pct_diff", "stl_pct_diff", "blk_pct_diff",
    "rest_days_diff", "b2b_home", "b2b_away", "altitude_factor",
    "mc_win_pct",
]

# ─── Elo Engine ───────────────────────────────────────────────────────────────

class EloEngine:
    def __init__(self):
        self.ratings = {}

    def get(self, team: str) -> float:
        return self.ratings.get(team, LEAGUE_MEAN_ELO)

    def win_prob(self, home: str, away: str) -> float:
        elo_diff = (self.get(home) + HOME_ADV_ELO) - self.get(away)
        return 1 / (1 + 10 ** (-elo_diff / 400))

    def update(self, home: str, away: str, home_score: int, away_score: int):
        home_elo = self.get(home)
        away_elo = self.get(away)
        home_expected = self.win_prob(home, away)
        home_actual = 1 if home_score > away_score else 0
        margin = abs(home_score - away_score)
        mov_mult = np.log(1 + min(margin, 20))
        k = K_FACTOR * mov_mult
        self.ratings[home] = home_elo + k * (home_actual - home_expected)
        self.ratings[away] = away_elo + k * ((1 - home_actual) - (1 - home_expected))

    def offseason_regression(self):
        for team in self.ratings:
            self.ratings[team] = 0.75 * self.ratings[team] + 0.25 * LEAGUE_MEAN_ELO


# ─── Simple Monte Carlo (Normal) ──────────────────────────────────────────────

def monte_carlo_win_prob(elo_diff: float, net_rtg_diff: float) -> float:
    """Simplified MC: use Elo-based win prob as proxy."""
    elo_prob = 1 / (1 + 10 ** (-(elo_diff + HOME_ADV_ELO) / 400))
    rtg_prob = 1 / (1 + np.exp(-net_rtg_diff / 8.0))  # logistic from net rtg diff
    return 0.5 * elo_prob + 0.5 * rtg_prob


# ─── Load game log from NBA API ───────────────────────────────────────────────

def fetch_game_log(season: str) -> pd.DataFrame:
    url = (
        f"{NBA_STATS_BASE}/leaguegamelog"
        f"?Counter=1000&DateFrom=&DateTo=&Direction=DESC&LeagueID=00"
        f"&PlayerOrTeam=T&Season={season}&SeasonType=Regular+Season&Sorter=DATE"
    )

    print(f"  Fetching game log for {season}...")
    try:
        resp = requests.get(url, headers=NBA_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        rs = data["resultSets"][0]
        df = pd.DataFrame(rs["rowSet"], columns=rs["headers"])
        print(f"  Got {len(df)} team-game records")
        return df
    except Exception as e:
        print(f"  Failed: {e}")
        return pd.DataFrame()


def load_from_db_for_backtest() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT
            p.game_date, p.home_team, p.away_team,
            p.mc_win_pct, p.calibrated_prob, p.feature_vector,
            p.correct,
            gr.home_score, gr.away_score
        FROM predictions p
        JOIN game_results gr ON gr.game_id = p.game_id
        WHERE p.correct IS NOT NULL
        ORDER BY p.game_date
        """,
        conn
    )
    conn.close()
    return df


# ─── Backtest engine ──────────────────────────────────────────────────────────

def run_backtest(seasons: list[str]) -> dict:
    """
    Walk-forward backtest:
    - Train on seasons 1..N-1
    - Test on season N
    - Repeat for each season
    """

    all_game_rows = []
    elo = EloEngine()

    for season in seasons:
        df = fetch_game_log(season)
        if df.empty:
            continue

        # Pair up games (each game appears twice — once per team)
        # GAME_ID identifies the game; filter to unique games
        games = (
            df.groupby("GAME_ID")
            .apply(lambda g: {
                "game_id": g["GAME_ID"].iloc[0],
                "game_date": g["GAME_DATE"].iloc[0] if "GAME_DATE" in g.columns else season,
                "home_team": g[g["MATCHUP"].str.contains(" vs\. ", na=False)]["TEAM_ABBREVIATION"].iloc[0]
                    if (g["MATCHUP"].str.contains(" vs\. ", na=False)).any() else g["TEAM_ABBREVIATION"].iloc[0],
                "away_team": g[g["MATCHUP"].str.contains(" @ ", na=False)]["TEAM_ABBREVIATION"].iloc[0]
                    if (g["MATCHUP"].str.contains(" @ ", na=False)).any() else g["TEAM_ABBREVIATION"].iloc[1],
                "home_pts": int(g[g["MATCHUP"].str.contains(" vs\. ", na=False)]["PTS"].iloc[0])
                    if (g["MATCHUP"].str.contains(" vs\. ", na=False)).any() and "PTS" in g.columns else 0,
                "away_pts": int(g[g["MATCHUP"].str.contains(" @ ", na=False)]["PTS"].iloc[0])
                    if (g["MATCHUP"].str.contains(" @ ", na=False)).any() and "PTS" in g.columns else 0,
                "season": season,
            })
            .reset_index(drop=True)
        )

        for _, game in games.iterrows():
            if not isinstance(game, dict):
                game = game.to_dict()

            elo_diff = elo.get(game["home_team"]) - elo.get(game["away_team"])
            win_prob = elo.win_prob(game["home_team"], game["away_team"])
            label = 1 if game["home_pts"] > game["away_pts"] else 0

            all_game_rows.append({
                "game_id": game["game_id"],
                "game_date": game["game_date"],
                "season": season,
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "home_pts": game["home_pts"],
                "away_pts": game["away_pts"],
                "elo_diff": elo_diff,
                "mc_win_pct": win_prob,
                "label": label,
            })

            # Update Elo
            if game["home_pts"] > 0:
                elo.update(game["home_team"], game["away_team"], game["home_pts"], game["away_pts"])

        elo.offseason_regression()

    if not all_game_rows:
        print("No game data available for backtest")
        return {}

    df_all = pd.DataFrame(all_game_rows)
    df_all["game_date"] = pd.to_datetime(df_all["game_date"], errors="coerce")

    # Walk-forward splits
    print(f"\nTotal games for backtest: {len(df_all)}")
    print("\nWalk-forward results:")
    print("-" * 50)

    test_seasons = seasons[1:]  # train on all prior seasons, test on each one
    results = {}

    for i, test_season in enumerate(test_seasons):
        train_df = df_all[df_all["season"].isin(seasons[:i+1])]
        test_df = df_all[df_all["season"] == test_season]

        if len(train_df) < 50 or len(test_df) < 20:
            continue

        X_train = train_df[["elo_diff", "mc_win_pct"]].fillna(0)
        y_train = train_df["label"]
        X_test = test_df[["elo_diff", "mc_win_pct"]].fillna(0)
        y_test = test_df["label"]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = LogisticRegression(C=1.0, max_iter=500)
        model.fit(X_train_s, y_train)

        preds = model.predict_proba(X_test_s)[:, 1]
        preds = np.clip(preds, 0.01, 0.99)

        brier = brier_score_loss(y_test, preds)
        acc = accuracy_score(y_test, preds >= 0.5)
        ll = log_loss(y_test, preds)

        # High conviction accuracy
        hc_mask = (preds >= 0.67) | (preds <= 0.33)
        hc_acc = accuracy_score(y_test[hc_mask], preds[hc_mask] >= 0.5) if hc_mask.sum() > 0 else None

        results[test_season] = {
            "n_games": len(test_df),
            "brier": float(brier),
            "accuracy": float(acc),
            "log_loss": float(ll),
            "high_conv_accuracy": float(hc_acc) if hc_acc is not None else None,
            "high_conv_games": int(hc_mask.sum()),
        }

        print(f"  {test_season}: n={len(test_df)}, Brier={brier:.4f}, Acc={acc:.3f}", end="")
        if hc_acc is not None:
            print(f", HiConv={hc_acc:.3f} ({hc_mask.sum()} games)", end="")
        print()

    # Summary
    if results:
        avg_brier = np.mean([r["brier"] for r in results.values()])
        avg_acc = np.mean([r["accuracy"] for r in results.values()])
        hc_accs = [r["high_conv_accuracy"] for r in results.values() if r["high_conv_accuracy"] is not None]
        avg_hc = np.mean(hc_accs) if hc_accs else None

        print(f"\nSummary: Avg Brier={avg_brier:.4f}, Avg Acc={avg_acc:.3f}", end="")
        if avg_hc is not None:
            print(f", Avg HiConv={avg_hc:.3f}", end="")
        print()

        results["summary"] = {
            "avg_brier": float(avg_brier),
            "avg_accuracy": float(avg_acc),
            "avg_high_conv_accuracy": float(avg_hc) if avg_hc is not None else None,
        }

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NBA Oracle Walk-Forward Backtest")
    parser.add_argument("--seasons", nargs="+", default=[
        "2018-19", "2019-20", "2020-21", "2021-22",
        "2022-23", "2023-24", "2024-25",
    ], help="Seasons to include")
    parser.add_argument("--output", default=None, help="Output JSON file for results")
    args = parser.parse_args()

    print("NBA Oracle v4.0 — Walk-Forward Backtest")
    print("=" * 40)

    # Try DB first, fall back to API
    df_db = load_from_db_for_backtest()
    if not df_db.empty:
        print(f"Loaded {len(df_db)} games from database for backtest analysis")

    results = run_backtest(args.seasons)

    if args.output and results:
        output_path = PROJECT_ROOT / args.output
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    print("\nBacktest complete.")


if __name__ == "__main__":
    main()
