#!/usr/bin/env python3
"""
NBA Oracle v4.0 -- Historical Dataset Builder
Uses the nba_api package (handles auth + throttling automatically).
Fetches 5+ seasons, computes features with NO lookahead bias.

Usage:
  python python/build_dataset.py
  python python/build_dataset.py --seasons 2022-23 2023-24 2024-25

Output:
  data/training_data.csv  -- labeled feature vectors for train_model.py
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from nba_api.stats.endpoints import LeagueGameLog
    from nba_api.stats.library.parameters import SeasonTypeAllStar
except ImportError:
    print("ERROR: nba_api not installed.")
    print("Run: pip install nba_api")
    sys.exit(1)

# ---- Config ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache" / "python"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = DATA_DIR / "training_data.csv"

HIGH_ALTITUDE_TEAMS = {"DEN"}

DEFAULT_SEASONS = [
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
]

# ---- Elo engine --------------------------------------------------------------

LEAGUE_ELO = 1500.0
K_FACTOR = 20.0
HOME_ADV_ELO = 100.0


class EloTracker:
    def __init__(self):
        self.ratings: dict = defaultdict(lambda: LEAGUE_ELO)

    def win_prob(self, home: str, away: str) -> float:
        diff = (self.ratings[home] + HOME_ADV_ELO) - self.ratings[away]
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    def get_diff(self, home: str, away: str) -> float:
        return float(self.ratings[home] - self.ratings[away])

    def update(self, home: str, away: str, home_pts: float, away_pts: float):
        expected = self.win_prob(home, away)
        actual = 1.0 if home_pts > away_pts else 0.0
        margin = abs(home_pts - away_pts)
        mov = np.log(1.0 + min(margin, 20.0))
        k = K_FACTOR * mov
        self.ratings[home] += k * (actual - expected)
        self.ratings[away] += k * ((1.0 - actual) - (1.0 - expected))

    def offseason_regression(self):
        for t in list(self.ratings.keys()):
            self.ratings[t] = 0.75 * self.ratings[t] + 0.25 * LEAGUE_ELO


# ---- Stat helpers ------------------------------------------------------------

def safe(val, default=0.0) -> float:
    try:
        v = float(val)
        return v if not np.isnan(v) else default
    except (TypeError, ValueError):
        return default


def est_poss(fga, oreb, tov, fta) -> float:
    return max(1.0, float(fga) - float(oreb) + float(tov) + 0.44 * float(fta))


def pythagorean(ortg: float, drtg: float) -> float:
    exp = 14.0
    return ortg ** exp / max(1e-6, ortg ** exp + drtg ** exp)


def log5(pa: float, pb: float) -> float:
    pa = max(0.01, min(0.99, pa))
    pb = max(0.01, min(0.99, pb))
    return (pa - pa * pb) / max(1e-8, pa + pb - 2.0 * pa * pb)


# ---- Rolling team stats tracker ---------------------------------------------

class TeamStatsTracker:
    def __init__(self):
        self.games: dict = defaultdict(list)

    def record(self, team: str, pts_for: float, pts_against: float,
               fgm: float, fga: float, fg3m: float, fg3a: float,
               fta: float, ftm: float, oreb: float, dreb: float,
               ast: float, stl: float, blk: float, tov: float,
               won: bool, date: str):
        poss = est_poss(fga, oreb, tov, fta)
        opp_poss = est_poss(fga, dreb, tov, fta)
        ortg = (pts_for / poss) * 100.0
        drtg = (pts_against / opp_poss) * 100.0
        efg = (fgm + 0.5 * fg3m) / max(1.0, fga)
        ts = pts_for / max(1.0, 2.0 * (fga + 0.44 * fta))
        tov_pct = tov / max(1.0, poss)
        oreb_pct = oreb / max(1.0, oreb + dreb)
        ft_rate = fta / max(1.0, fga)
        three_rate = fg3a / max(1.0, fga)
        three_pct = fg3m / max(1.0, fg3a) if fg3a > 0 else 0.36
        ast_pct = ast / max(1.0, fgm)
        stl_pct = stl / max(1.0, poss / 100.0)
        blk_pct = blk / max(1.0, poss / 100.0)
        self.games[team].append({
            "won": won, "ortg": ortg, "drtg": drtg,
            "net_rtg": ortg - drtg, "pace": poss,
            "efg": efg, "ts": ts, "tov_pct": tov_pct, "oreb_pct": oreb_pct,
            "ft_rate": ft_rate, "three_rate": three_rate, "three_pct": three_pct,
            "ast_pct": ast_pct, "stl_pct": stl_pct, "blk_pct": blk_pct,
        })

    def stats(self, team: str, window: int = 82) -> dict:
        g = self.games[team][-window:]
        if not g:
            return self._default()
        n = len(g)
        avg = lambda k: sum(x[k] for x in g) / n
        return {
            "win_pct": sum(x["won"] for x in g) / n,
            "ortg": avg("ortg"), "drtg": avg("drtg"), "net_rtg": avg("net_rtg"),
            "pace": avg("pace"), "efg": avg("efg"), "ts": avg("ts"),
            "tov_pct": avg("tov_pct"), "oreb_pct": avg("oreb_pct"),
            "ft_rate": avg("ft_rate"), "three_rate": avg("three_rate"),
            "three_pct": avg("three_pct"), "ast_pct": avg("ast_pct"),
            "stl_pct": avg("stl_pct"), "blk_pct": avg("blk_pct"), "n": n,
        }

    def rolling10(self, team: str) -> dict:
        return self.stats(team, window=10)

    def reset(self):
        self.games.clear()

    @staticmethod
    def _default() -> dict:
        return {
            "win_pct": 0.5, "ortg": 113.0, "drtg": 113.0, "net_rtg": 0.0,
            "pace": 99.5, "efg": 0.530, "ts": 0.570, "tov_pct": 0.130,
            "oreb_pct": 0.230, "ft_rate": 0.200, "three_rate": 0.400,
            "three_pct": 0.360, "ast_pct": 0.600, "stl_pct": 0.090,
            "blk_pct": 0.080, "n": 0,
        }


# ---- Fetch game log via nba_api ----------------------------------------------

def fetch_game_log(season: str, retries: int = 3) -> pd.DataFrame:
    cache_path = CACHE_DIR / f"gamelog_{season.replace('-','')}.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"  Loaded from cache: {len(df)} team-game records")
        return df

    for attempt in range(retries):
        try:
            print(f"  Fetching from nba_api (attempt {attempt+1})...", flush=True)
            gl = LeagueGameLog(
                season=season,
                season_type_all_star="Regular Season",
                player_or_team_abbreviation="T",
                timeout=60,
            )
            df = gl.get_data_frames()[0]
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
            df = df.sort_values("GAME_DATE").reset_index(drop=True)
            df.to_parquet(cache_path, index=False)
            print(f"  Got {len(df)} team-game records ({len(df)//2} games)")
            return df
        except Exception as e:
            wait = 2 ** attempt * 3
            print(f"  Error: {e}")
            if attempt < retries - 1:
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

    return pd.DataFrame()


# ---- Build features for one season ------------------------------------------

def build_season(season: str, gl: pd.DataFrame,
                 elo: EloTracker, tracker: TeamStatsTracker,
                 min_games: int = 15) -> list:
    rows = []
    last_date: dict = {}
    skipped = 0

    by_id = gl.groupby("GAME_ID")
    unique = (gl[["GAME_ID", "GAME_DATE"]]
              .drop_duplicates("GAME_ID")
              .sort_values("GAME_DATE"))

    for _, urow in unique.iterrows():
        gid = urow["GAME_ID"]
        gdate = urow["GAME_DATE"]
        team_rows = by_id.get_group(gid)

        if len(team_rows) != 2:
            skipped += 1
            continue

        # Identify home / away from MATCHUP column
        home_r = away_r = None
        for _, tr in team_rows.iterrows():
            m = str(tr.get("MATCHUP", ""))
            if " vs. " in m:
                home_r = tr
            elif " @ " in m:
                away_r = tr

        if home_r is None or away_r is None:
            skipped += 1
            continue

        home = str(home_r["TEAM_ABBREVIATION"])
        away = str(away_r["TEAM_ABBREVIATION"])
        home_pts = safe(home_r.get("PTS", 0))
        away_pts = safe(away_r.get("PTS", 0))

        if home_pts == 0 and away_pts == 0:
            skipped += 1
            continue

        # Skip early season before teams have enough data
        h_n = tracker.stats(home)["n"]
        a_n = tracker.stats(away)["n"]
        if h_n < min_games or a_n < min_games:
            # Still record the game for future use
            _record(tracker, home_r, away_pts, home_pts > away_pts, gdate)
            _record(tracker, away_r, home_pts, away_pts > home_pts, gdate)
            elo.update(home, away, home_pts, away_pts)
            last_date[home] = gdate
            last_date[away] = gdate
            continue

        # Snapshot BEFORE recording result
        hs = tracker.stats(home)
        as_ = tracker.stats(away)
        hr10 = tracker.rolling10(home)
        ar10 = tracker.rolling10(away)

        elo_diff = elo.get_diff(home, away)
        net_diff = hs["net_rtg"] - as_["net_rtg"]
        mc_prob = 0.5 * elo.win_prob(home, away) + 0.5 * (1.0 / (1.0 + np.exp(-net_diff / 8.0)))

        # Rest days
        h_rest = min(7, (gdate - last_date[home]).days) if home in last_date else 3
        a_rest = min(7, (gdate - last_date[away]).days) if away in last_date else 3

        label = 1 if home_pts > away_pts else 0

        rows.append({
            "season": season,
            "game_id": gid,
            "game_date": gdate.strftime("%Y-%m-%d"),
            "home_team": home,
            "away_team": away,
            "home_pts": home_pts,
            "away_pts": away_pts,
            "label": label,
            # --- features ---
            "elo_diff": round(elo_diff, 2),
            "net_rtg_diff": round(net_diff, 3),
            "off_rtg_diff": round(hs["ortg"] - as_["ortg"], 3),
            "def_rtg_diff": round(hs["drtg"] - as_["drtg"], 3),
            "pace_diff": round(hs["pace"] - as_["pace"], 3),
            "pythagorean_diff": round(pythagorean(hs["ortg"], hs["drtg"]) - pythagorean(as_["ortg"], as_["drtg"]), 4),
            "log5_prob": round(log5(hs["win_pct"], as_["win_pct"]), 4),
            "efg_pct_diff": round(hs["efg"] - as_["efg"], 4),
            "tov_pct_diff": round(hs["tov_pct"] - as_["tov_pct"], 4),
            "oreb_pct_diff": round(hs["oreb_pct"] - as_["oreb_pct"], 4),
            "ft_rate_diff": round(hs["ft_rate"] - as_["ft_rate"], 4),
            "three_pt_rate_diff": round(hs["three_rate"] - as_["three_rate"], 4),
            "three_pt_pct_diff": round(hs["three_pct"] - as_["three_pct"], 4),
            "ts_pct_diff": round(hs["ts"] - as_["ts"], 4),
            "ast_pct_diff": round(hs["ast_pct"] - as_["ast_pct"], 4),
            "stl_pct_diff": round(hs["stl_pct"] - as_["stl_pct"], 4),
            "blk_pct_diff": round(hs["blk_pct"] - as_["blk_pct"], 4),
            "team_10d_net_rtg_diff": round(hr10["net_rtg"] - ar10["net_rtg"], 3),
            "team_10d_off_rtg_diff": round(hr10["ortg"] - ar10["ortg"], 3),
            "momentum_diff": round(
                (hr10["net_rtg"] - hs["net_rtg"]) - (ar10["net_rtg"] - as_["net_rtg"]), 3),
            "rest_days_diff": h_rest - a_rest,
            "b2b_home": 1 if h_rest == 1 else 0,
            "b2b_away": 1 if a_rest == 1 else 0,
            "travel_tz_shift_home": 0,
            "travel_tz_shift_away": 1,
            "is_home": 1,
            "altitude_factor": 1.5 if home in HIGH_ALTITUDE_TEAMS else 0.0,
            "star_player_impact_diff": 0.0,
            "injury_impact_diff": 0.0,
            "lineup_net_rtg_diff": round(net_diff * 0.5, 3),
            "bench_net_rtg_diff": 0.0,
            "clutch_net_rtg_diff": 0.0,
            "h2h_season_record": 0.5,
            "vegas_home_prob": 0.0,
            "mc_win_pct": round(mc_prob, 4),
        })

        # Record result
        _record(tracker, home_r, away_pts, home_pts > away_pts, gdate)
        _record(tracker, away_r, home_pts, away_pts > home_pts, gdate)
        elo.update(home, away, home_pts, away_pts)
        last_date[home] = gdate
        last_date[away] = gdate

    print(f"  Built {len(rows)} feature rows (skipped {skipped})")
    return rows


def _record(tracker: TeamStatsTracker, tr, opp_pts: float, won: bool, gdate):
    tracker.record(
        team=str(tr["TEAM_ABBREVIATION"]),
        pts_for=safe(tr.get("PTS", 0)),
        pts_against=opp_pts,
        fgm=safe(tr.get("FGM", 0)), fga=safe(tr.get("FGA", 1)),
        fg3m=safe(tr.get("FG3M", 0)), fg3a=safe(tr.get("FG3A", 0)),
        fta=safe(tr.get("FTA", 0)), ftm=safe(tr.get("FTM", 0)),
        oreb=safe(tr.get("OREB", 0)), dreb=safe(tr.get("DREB", 0)),
        ast=safe(tr.get("AST", 0)), stl=safe(tr.get("STL", 0)),
        blk=safe(tr.get("BLK", 0)), tov=safe(tr.get("TOV", 0)),
        won=won, date=str(gdate)[:10],
    )


# ---- Main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    parser.add_argument("--min-games", type=int, default=15,
                        help="Games a team must have played before its row is included")
    args = parser.parse_args()

    print("NBA Oracle v4.0 -- Building Historical Training Dataset")
    print("=" * 55)
    print(f"Seasons : {', '.join(args.seasons)}")
    print(f"Output  : {OUTPUT_PATH}")
    print()

    elo = EloTracker()
    tracker = TeamStatsTracker()
    all_rows = []

    for season in args.seasons:
        print(f"\nSeason: {season}")
        try:
            gl = fetch_game_log(season)
        except Exception as e:
            print(f"  FAILED: {e} -- skipping season")
            continue

        if gl.empty:
            print("  Empty game log -- skipping")
            continue

        rows = build_season(season, gl, elo, tracker, min_games=args.min_games)
        all_rows.extend(rows)

        elo.offseason_regression()
        tracker.reset()
        time.sleep(1)   # small pause between seasons

    if not all_rows:
        print("\nERROR: No data collected. Check internet connection and try again.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    total = len(df)
    home_wr = df["label"].mean()

    print(f"\n{'='*55}")
    print(f"Total games   : {total:,}")
    print(f"Date range    : {df['game_date'].min()} to {df['game_date'].max()}")
    print(f"Home win rate : {home_wr:.3f} ({home_wr*100:.1f}%)")

    by_season = df.groupby("season")["label"].agg(["count", "mean"]).rename(
        columns={"count": "games", "mean": "home_win_pct"})
    print("\nPer season:")
    print(by_season.to_string())

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {total:,} rows to {OUTPUT_PATH}")
    print("Next: python python/train_model.py")


if __name__ == "__main__":
    main()
