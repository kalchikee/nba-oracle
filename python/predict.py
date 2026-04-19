#!/usr/bin/env python3
"""
NBA Oracle v4.1 — Live Predictions
Fetches today's NBA games from ESPN, rebuilds Elo from history,
and runs the trained logistic regression model. Auto-switches to the
playoff model from mid-April through June.

Usage:
  python python/predict.py              # today's games
  python python/predict.py --date 20260415
"""
import argparse, json, math, time, requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = DATA_DIR / "model"
HIST_CSV  = DATA_DIR / "training_data.csv"

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
HEADERS   = {"User-Agent": "NBA-Oracle/4.1"}

INITIAL_ELO = 1500.0
K_FACTOR    = 20.0
HOME_ADV    = 100.0


# ── Season detection ───────────────────────────────────────────────────────────

def is_playoff_season(date_str: str) -> bool:
    """NBA playoffs: mid-April through mid-June."""
    try:
        d = datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        return False
    return (d.month == 4 and d.day >= 13) or d.month in (5, 6) or (d.month == 7 and d.day <= 1)


# ── ESPN helpers ───────────────────────────────────────────────────────────────

def fetch_json(url: str) -> dict:
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    return {}


def fetch_games(date_str: str) -> list:
    # Use seasontype=3 during playoffs to get series context
    playoff = is_playoff_season(date_str)
    season_param = "&seasontype=3" if playoff else ""
    data = fetch_json(f"{ESPN_BASE}/scoreboard?dates={date_str}&limit=20{season_param}")
    games = []
    for ev in data.get("events", []):
        status = ev.get("status", {}).get("type", {}).get("name", "")
        comp   = (ev.get("competitions") or [{}])[0]
        cs     = comp.get("competitors", [])
        home   = next((c for c in cs if c.get("homeAway") == "home"), None)
        away   = next((c for c in cs if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue

        # Extract series wins from ESPN playoff competition object
        h_wins, a_wins = 0, 0
        series = comp.get("series", {})
        if series:
            for sc in series.get("competitors", []):
                wins = int(sc.get("wins", 0) or 0)
                if sc.get("homeAway") == "home":
                    h_wins = wins
                elif sc.get("homeAway") == "away":
                    a_wins = wins

        game_num   = h_wins + a_wins + 1
        deficit    = h_wins - a_wins          # positive = home leads series
        elim_game  = int(h_wins == 3 or a_wins == 3)  # NBA best-of-7, 4 wins needed

        games.append({
            "event_name":          ev.get("name", ""),
            "status":              status,
            "home_abbr":           home.get("team", {}).get("abbreviation", "").upper(),
            "home_id":             home.get("team", {}).get("id", ""),
            "home_name":           home.get("team", {}).get("displayName", ""),
            "away_abbr":           away.get("team", {}).get("abbreviation", "").upper(),
            "away_id":             away.get("team", {}).get("id", ""),
            "away_name":           away.get("team", {}).get("displayName", ""),
            "series_game_num":     game_num,
            "series_deficit":      deficit,
            "is_elimination_game": elim_game,
        })
    return games


def fetch_team_record(team_id: str) -> dict:
    """Get season record + home/away win% from ESPN."""
    data = fetch_json(f"{ESPN_BASE}/teams/{team_id}?enable=record,stats")
    items = data.get("team", {}).get("record", {}).get("items", [])
    result = {"win_pct": 0.5, "home_win_pct": 0.5, "away_win_pct": 0.5, "games_played": 0}
    for item in items:
        rtype = item.get("type", "")
        stats = {s["name"]: s["value"] for s in item.get("stats", [])}
        gp    = stats.get("gamesPlayed", 0) or 0
        wins  = stats.get("wins", 0) or 0
        pct   = wins / gp if gp > 0 else 0.5
        if rtype == "total":
            result["win_pct"] = pct
            result["games_played"] = int(gp)
        elif rtype == "home":
            result["home_win_pct"] = pct
        elif rtype == "road":
            result["away_win_pct"] = pct
    return result


# ── Elo reconstruction ─────────────────────────────────────────────────────────

def build_elo_from_history() -> dict:
    if not HIST_CSV.exists():
        return {}
    df = pd.read_csv(HIST_CSV, usecols=["season", "game_date", "home_team",
                                         "away_team", "label"])
    df = df.sort_values("game_date")
    elo = defaultdict(lambda: INITIAL_ELO)
    last_season = None

    for _, row in df.iterrows():
        season = row["season"]
        if last_season and season != last_season:
            for t in list(elo.keys()):
                elo[t] = 0.75 * elo[t] + 0.25 * INITIAL_ELO  # offseason regression
        last_season = season
        h, a  = str(row["home_team"]).upper(), str(row["away_team"]).upper()
        hw    = int(row["label"])
        rh, ra = elo[h], elo[a]
        exp_h  = 1.0 / (1.0 + 10 ** ((ra - (rh + HOME_ADV)) / 400.0))
        delta  = K_FACTOR * (hw - exp_h)
        elo[h] = rh + delta
        elo[a] = ra - delta

    return dict(elo)


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(date_str: str) -> dict | None:
    try:
        # Playoff season: use playoff model if available
        if is_playoff_season(date_str):
            po = MODEL_DIR / "playoff_coefficients.json"
            ps = MODEL_DIR / "playoff_scaler.json"
            pm = MODEL_DIR / "playoff_metadata.json"
            if po.exists() and ps.exists() and pm.exists():
                po_data = json.loads(po.read_text())
                # Convert to dict-key format for predict_proba
                coeff = dict(zip(po_data["feature_names"], po_data["coefficients"]))
                coeff["_intercept"] = po_data["intercept"]
                meta = json.loads(pm.read_text())
                meta["feature_names"] = po_data["feature_names"]
                scaler = json.loads(ps.read_text())
                print("  [PLAYOFFS] Using NBA playoff model")
                return {"coeff": coeff, "scaler": scaler,
                        "calib": {"x_thresholds": [], "y_thresholds": []}, "meta": meta}
        coeff  = json.loads((MODEL_DIR / "coefficients.json").read_text())
        scaler = json.loads((MODEL_DIR / "scaler.json").read_text())
        calib  = json.loads((MODEL_DIR / "calibration.json").read_text())
        meta   = json.loads((MODEL_DIR / "metadata.json").read_text())
        return {"coeff": coeff, "scaler": scaler, "calib": calib, "meta": meta}
    except Exception as e:
        print(f"  Model load failed: {e}")
        return None


def predict_proba(model: dict, fv: dict) -> float:
    features  = model["meta"]["feature_names"]
    coeff_map = model["coeff"]
    intercept = coeff_map.get("_intercept", coeff_map.get("intercept", 0.0))
    mean      = model["scaler"]["mean"]
    scale     = model["scaler"]["scale"]

    x = np.array([(fv.get(f, 0.0) - mean[i]) / (scale[i] if scale[i] != 0 else 1.0)
                  for i, f in enumerate(features)])
    coeff  = np.array([coeff_map.get(f, 0.0) for f in features])
    logit  = float(np.dot(coeff, x)) + intercept
    raw    = 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, logit))))

    calib = model["calib"]
    bins  = calib.get("x_thresholds", calib.get("bins", []))
    cals  = calib.get("y_thresholds", calib.get("calibrated", []))
    if not bins or not cals or raw <= bins[0]:
        p = cals[0] if cals else raw
    elif raw >= bins[-1]:
        p = cals[-1]
    else:
        p = raw
        for i in range(len(bins) - 1):
            if bins[i] <= raw <= bins[i + 1]:
                t = (raw - bins[i]) / (bins[i + 1] - bins[i])
                p = cals[i] + t * (cals[i + 1] - cals[i])
                break

    # Cap at 85%: no single NBA game is more predictable than this
    return max(0.15, min(0.85, p))


# ── Feature builder ────────────────────────────────────────────────────────────

def build_features(elo_ratings: dict, h_abbr: str, a_abbr: str,
                   h_rec: dict, a_rec: dict,
                   series_game_num: int = 1, series_deficit: int = 0,
                   is_elimination_game: int = 0) -> dict:
    rh = elo_ratings.get(h_abbr, INITIAL_ELO)
    ra = elo_ratings.get(a_abbr, INITIAL_ELO)
    h_wp = h_rec["win_pct"]
    a_wp = a_rec["win_pct"]
    pyth_diff = h_wp - a_wp

    return {
        "elo_diff":              rh - ra,
        "net_rtg_diff":          pyth_diff * 20,   # rough proxy
        "off_rtg_diff":          pyth_diff * 10,
        "def_rtg_diff":          pyth_diff * 10,
        "pace_diff":             0.0,
        "pythagorean_diff":      pyth_diff,
        "log5_prob":             h_wp / (h_wp + a_wp) if (h_wp + a_wp) > 0 else 0.5,
        "efg_pct_diff":          0.0,
        "tov_pct_diff":          0.0,
        "oreb_pct_diff":         0.0,
        "ft_rate_diff":          0.0,
        "three_pt_rate_diff":    0.0,
        "three_pt_pct_diff":     0.0,
        "ts_pct_diff":           0.0,
        "ast_pct_diff":          0.0,
        "stl_pct_diff":          0.0,
        "blk_pct_diff":          0.0,
        "team_10d_net_rtg_diff": 0.0,
        "team_10d_off_rtg_diff": 0.0,
        "momentum_diff":         0.0,
        "rest_days_diff":        0.0,
        "b2b_home":              0.0,
        "b2b_away":              0.0,
        "altitude_factor":       1.0,
        "lineup_net_rtg_diff":   0.0,
        "vegas_home_prob":       0.0,
        "mc_win_pct":            h_wp,
        # playoff model features
        "win_pct_diff":          h_wp - a_wp,
        "ppg_diff":              0.0,
        "papg_diff":             0.0,
        "is_neutral":            0.0,
        # Series context (all NBA rounds are best-of-7)
        "series_game_num":       float(series_game_num),
        "series_deficit":        float(series_deficit),
        "is_elimination_game":   float(is_elimination_game),
    }


# ── Printing ───────────────────────────────────────────────────────────────────

def pad(s: str, w: int) -> str:
    return s[:w].ljust(w)


def print_predictions(results: list, date_str: str) -> None:
    width = 85
    print("\n" + "=" * width)
    print(f"  NBA ORACLE v4.1  |  {date_str}  |  {len(results)} games")
    print("=" * width)
    print("  " + pad("MATCHUP", 30) + pad("HOME WIN%", 11) + pad("AWAY WIN%", 11) + "PICK")
    print("-" * width)
    for r in sorted(results, key=lambda x: -max(x["home_prob"], x["away_prob"])):
        matchup  = f"{r['home_abbr']} vs {r['away_abbr']}"
        home_pct = f"{r['home_prob']*100:.1f}%"
        away_pct = f"{r['away_prob']*100:.1f}%"
        pick     = r["home_abbr"] if r["home_prob"] >= r["away_prob"] else r["away_abbr"]
        star     = " *" if max(r["home_prob"], r["away_prob"]) >= 0.65 else ""
        print(f"  {pad(matchup, 30)}{pad(home_pct, 11)}{pad(away_pct, 11)}{pick}{star}")
    print("-" * width)
    print("* = high confidence (>= 65%)\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    args = parser.parse_args()
    date_str = args.date

    print(f"=== NBA Oracle v4.1 — Predictions for {date_str} ===\n")

    model = load_model(date_str)
    if not model:
        print("ERROR: No model found. Run: python python/train_model.py")
        return

    print("Loading Elo ratings from history...")
    elo = build_elo_from_history()
    print(f"  {len(elo)} teams rated")

    print(f"\nFetching games for {date_str}...")
    games = fetch_games(date_str)

    if not games:
        for offset in list(range(1, 8)) + list(range(-1, -8, -1)):
            d = (datetime.strptime(date_str, "%Y%m%d") + timedelta(days=offset)).strftime("%Y%m%d")
            games = fetch_games(d)
            if games:
                label = "next" if offset > 0 else "most recent"
                print(f"  No games today — showing {label} games ({d})")
                date_str = d
                break

    if not games:
        print("No games found.")
        return

    scheduled = [g for g in games if "SCHEDULED" in g["status"]] or games
    print(f"  Found {len(scheduled)} game(s)\n")

    results = []
    for game in scheduled:
        h_rec = fetch_team_record(game["home_id"])
        a_rec = fetch_team_record(game["away_id"])
        time.sleep(0.1)

        fv     = build_features(elo, game["home_abbr"], game["away_abbr"], h_rec, a_rec,
                               series_game_num     = game["series_game_num"],
                               series_deficit      = game["series_deficit"],
                               is_elimination_game = game["is_elimination_game"])
        home_p = predict_proba(model, fv)

        results.append({
            "home_abbr": game["home_abbr"],
            "away_abbr": game["away_abbr"],
            "home_prob": home_p,
            "away_prob": 1.0 - home_p,
        })

    print_predictions(results, date_str)


if __name__ == "__main__":
    main()
