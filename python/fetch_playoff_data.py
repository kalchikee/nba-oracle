#!/usr/bin/env python3
"""
NBA Playoff Data Fetcher — last 5 seasons of playoff games.
Uses end-of-regular-season Elo + season stats as features.
Output: data/playoff_data.csv

Usage: python python/fetch_playoff_data.py
"""
import sys, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

try:
    from nba_api.stats.endpoints import LeagueGameLog
except ImportError:
    print("ERROR: pip install nba_api"); sys.exit(1)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache" / "python"
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

REG_CSV   = DATA_DIR / "training_data.csv"
OUT_CSV   = DATA_DIR / "playoff_data.csv"

PLAYOFF_SEASONS = [
    "2018-19",
    # Skip 2019-20: COVID bubble (neutral site — no meaningful home court)
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
]

K_FACTOR    = 20.0
HOME_ADV    = 100.0
LEAGUE_ELO  = 1500.0

FEATURE_NAMES = [
    "elo_diff", "net_rtg_diff", "win_pct_diff",
    "ppg_diff", "papg_diff", "off_rtg_diff", "def_rtg_diff",
    "rest_days_diff", "is_home",
    # Series context — all NBA rounds are best-of-7
    "series_game_num",    # 1–7: which game in the series
    "series_deficit",     # home_wins - away_wins at tip-off (negative = home trails)
    "is_elimination_game", # 1 if either team faces series elimination this game
    "is_playin",          # 1 if this is a Play-In Tournament game (not a full series)
]

def fetch_game_log(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    cache = CACHE_DIR / f"nba_{season}_{season_type.replace(' ','_')}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    headers = {
        "Host": "stats.nba.com",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "x-nba-stats-origin": "stats",
        "x-nba-stats-token": "true",
        "Origin": "https://www.nba.com",
        "Referer": "https://www.nba.com/",
    }
    try:
        log = LeagueGameLog(
            season=season,
            season_type_all_star=season_type,
            timeout=60,
            headers=headers,
        )
        df = log.get_data_frames()[0]
        df.to_csv(cache, index=False)
        time.sleep(2)
        return df
    except Exception as e:
        print(f"  Failed {season} {season_type}: {e}")
        return pd.DataFrame()


def build_elo_from_reg(reg_df: pd.DataFrame, season: str) -> dict:
    """Replay regular season games to get end-of-season Elo per team."""
    elo = defaultdict(lambda: LEAGUE_ELO)
    season_df = reg_df[reg_df["season"] == season].copy()
    if season_df.empty:
        return dict(elo)
    season_df = season_df.sort_values("game_date")
    for _, row in season_df.iterrows():
        h, a = row["home_team"], row["away_team"]
        he, ae = elo[h] + HOME_ADV, elo[a]
        exp = 1 / (1 + 10 ** ((ae - he) / 400))
        act = 1 if row["label"] == 1 else 0
        elo[h] += K_FACTOR * (act - exp)
        elo[a] += K_FACTOR * ((1 - act) - (1 - exp))
    return dict(elo)


def build_season_stats(reg_df: pd.DataFrame, season: str) -> dict:
    """Compute per-team season averages from regular season data."""
    s = reg_df[reg_df["season"] == season]
    stats = {}
    teams = set(s["home_team"].tolist() + s["away_team"].tolist())
    for team in teams:
        home_g = s[s["home_team"] == team]
        away_g = s[s["away_team"] == team]
        pts_for  = list(home_g["home_pts"]) + list(away_g["away_pts"])
        pts_agst = list(home_g["away_pts"]) + list(away_g["home_pts"])
        wins = len(home_g[home_g["label"] == 1]) + len(away_g[away_g["label"] == 0])
        games = len(home_g) + len(away_g)
        net_rtg = home_g["net_rtg_diff"].mean() if not home_g.empty else 0
        off_rtg = home_g["off_rtg_diff"].mean() if not home_g.empty else 0
        def_rtg = home_g["def_rtg_diff"].mean() if not home_g.empty else 0
        stats[team] = {
            "win_pct":  wins / games if games > 0 else 0.5,
            "ppg":      np.mean(pts_for) if pts_for else 110,
            "papg":     np.mean(pts_agst) if pts_agst else 110,
            "net_rtg":  float(net_rtg),
            "off_rtg":  float(off_rtg),
            "def_rtg":  float(def_rtg),
        }
    return stats


def pair_games(log_df: pd.DataFrame) -> list:
    """Pair NBA team-game rows into single game rows."""
    games = []
    for gid, grp in log_df.groupby("GAME_ID"):
        home_row = grp[grp["MATCHUP"].str.contains(r" vs\. ", na=False, regex=True)]
        away_row = grp[grp["MATCHUP"].str.contains(r" @ ", na=False, regex=True)]
        if home_row.empty or away_row.empty:
            continue
        h = home_row.iloc[0]
        a = away_row.iloc[0]
        games.append({
            "game_id":   gid,
            "game_date": h.get("GAME_DATE", ""),
            "home_team": h["TEAM_ABBREVIATION"],
            "away_team": a["TEAM_ABBREVIATION"],
            "home_pts":  float(h.get("PTS", 0) or 0),
            "away_pts":  float(a.get("PTS", 0) or 0),
        })
    return games


def add_series_context(games: list) -> list:
    """
    Add series context to each game.
    NBA format: all rounds are best-of-7. Play-In Tournament games are
    single games (not a series) — flagged with is_playin=1.

    A "series" is identified by the frozenset of the two teams playing.
    Games are processed in date order so wins accumulate correctly.
    """
    # The Play-In typically happens in mid-April before the first round.
    # Play-In game IDs in the NBA API have a different game_id prefix, but
    # the simplest heuristic is: if a pair of teams plays only 1 or 2 games
    # total (across the whole playoff log), those are Play-In games.
    # We'll detect this after building series records.

    # Step 1: sort by date
    sorted_games = sorted(games, key=lambda g: g["game_date"])

    # Step 2: track series records
    # series_key = frozenset({home_team, away_team})
    # We reset when a new instance of the same matchup starts (different round)
    # — but since teams only meet once per season in the NBA playoffs, each
    # frozenset is unique per season.
    series_wins: dict = defaultdict(lambda: defaultdict(int))  # key -> team -> wins

    result = []
    for g in sorted_games:
        h, a = g["home_team"], g["away_team"]
        key = frozenset([h, a])

        h_wins = series_wins[key][h]
        a_wins = series_wins[key][a]
        game_num = h_wins + a_wins + 1  # 1-indexed game number in series

        # Series deficit from the home team's perspective
        series_deficit = h_wins - a_wins

        # Elimination: any team at 3 wins means the other is one loss from out
        # (best-of-7: first to 4 wins). Game is elimination if loser is out.
        is_elimination = int((h_wins == 3) or (a_wins == 3))

        # Update wins after recording pre-game state
        label = 1 if g["home_pts"] > g["away_pts"] else 0
        if label == 1:
            series_wins[key][h] += 1
        else:
            series_wins[key][a] += 1

        result.append({
            **g,
            "series_game_num":    game_num,
            "series_deficit":     series_deficit,
            "is_elimination_game": is_elimination,
            "is_playin":          0,  # marked in second pass below
        })

    # Step 3: detect Play-In games — series that never reached game 3
    # (Play-In games are 1-or-2 game "series" between teams that don't
    # appear again in the proper bracket)
    series_total: dict = defaultdict(int)
    for g in result:
        key = frozenset([g["home_team"], g["away_team"]])
        series_total[key] += 1

    for g in result:
        key = frozenset([g["home_team"], g["away_team"]])
        if series_total[key] <= 2:
            g["is_playin"] = 1
            # Play-In has no meaningful series context — reset to neutral
            g["series_game_num"]     = 1
            g["series_deficit"]      = 0
            g["is_elimination_game"] = 1  # every Play-In game is do-or-die

    return result


def main():
    print("NBA Playoff Data Fetcher")
    print("=" * 40)

    if not REG_CSV.exists():
        print(f"Regular season CSV not found: {REG_CSV}")
        sys.exit(1)

    reg_df = pd.read_csv(REG_CSV)
    reg_df["game_date"] = pd.to_datetime(reg_df["game_date"], errors="coerce")

    all_rows = []

    for season in PLAYOFF_SEASONS:
        print(f"\nSeason {season}")

        # End-of-regular-season context
        elo    = build_elo_from_reg(reg_df, season)
        stats  = build_season_stats(reg_df, season)

        print(f"  Fetching playoff games...")
        po_log = fetch_game_log(season, "Playoffs")
        if po_log.empty:
            print("  No data — skipping")
            continue

        games = pair_games(po_log)
        games = add_series_context(games)
        print(f"  Got {len(games)} playoff games ({sum(1 for g in games if g['is_playin'])} Play-In)")

        prev_dates: dict = {}  # team -> last game date

        for g in sorted(games, key=lambda x: x["game_date"]):
            h, a = g["home_team"], g["away_team"]
            if h not in stats or a not in stats:
                continue

            game_dt = pd.to_datetime(g["game_date"], errors="coerce")
            rest_h = (game_dt - prev_dates.get(h, game_dt)).days if h in prev_dates else 2
            rest_a = (game_dt - prev_dates.get(a, game_dt)).days if a in prev_dates else 2
            prev_dates[h] = game_dt
            prev_dates[a] = game_dt

            h_elo = elo.get(h, LEAGUE_ELO)
            a_elo = elo.get(a, LEAGUE_ELO)
            hs = stats[h]
            as_ = stats[a]

            label = 1 if g["home_pts"] > g["away_pts"] else 0

            row = {
                "season":       season,
                "game_id":      g["game_id"],
                "game_date":    g["game_date"],
                "home_team":    h,
                "away_team":    a,
                "home_pts":     g["home_pts"],
                "away_pts":     g["away_pts"],
                "label":        label,
                "is_playoff":   1,
                "elo_diff":     h_elo - a_elo,
                "net_rtg_diff": hs["net_rtg"] - as_["net_rtg"],
                "off_rtg_diff": hs["off_rtg"] - as_["off_rtg"],
                "def_rtg_diff": hs["def_rtg"] - as_["def_rtg"],
                "win_pct_diff": hs["win_pct"] - as_["win_pct"],
                "ppg_diff":     hs["ppg"]     - as_["ppg"],
                "papg_diff":    hs["papg"]    - as_["papg"],
                "rest_days_diff": rest_h - rest_a,
                "is_home":      1,
                # Series context (all NBA rounds = best-of-7)
                "series_game_num":     g["series_game_num"],
                "series_deficit":      g["series_deficit"],
                "is_elimination_game": g["is_elimination_game"],
                "is_playin":           g["is_playin"],
            }

            # Update Elo after game (higher K in playoffs = faster adjustment)
            h_exp = 1 / (1 + 10 ** ((a_elo - (h_elo + HOME_ADV)) / 400))
            elo[h] = h_elo + K_FACTOR * (label - h_exp)
            elo[a] = a_elo + K_FACTOR * ((1 - label) - (1 - h_exp))

            all_rows.append(row)

    if not all_rows:
        print("\nNo playoff data fetched.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} playoff games to {OUT_CSV}")
    print(f"Seasons: {df['season'].unique().tolist()}")
    print(f"Home win rate: {df['label'].mean():.3f}")


if __name__ == "__main__":
    main()
