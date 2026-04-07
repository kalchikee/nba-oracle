// NBA Oracle v4.0 — Vegas Odds Client
// Uses The Odds API (free tier: 500 requests/month)
// Falls back to manual vegas_lines.json if API not configured.

import { existsSync, readFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import fetch from 'node-fetch';
import { logger } from '../logger.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const MANUAL_LINES_PATH = resolve(__dirname, '../../data/vegas_lines.json');

// ─── Manual lines JSON format ─────────────────────────────────────────────────
// data/vegas_lines.json (optional, create manually):
// {
//   "2026-04-06": {
//     "BOS_MIA": { "homeML": -165, "awayML": 140 },
//     ...
//   }
// }

interface ManualLines {
  [date: string]: {
    [matchupKey: string]: { homeML: number; awayML: number };
  };
}

interface OddsApiBookmaker {
  key: string;
  markets: Array<{
    key: string;
    outcomes: Array<{ name: string; price: number }>;
  }>;
}

interface OddsApiGame {
  id: string;
  home_team: string;
  away_team: string;
  bookmakers: OddsApiBookmaker[];
}

// ─── Implied probability from moneyline ───────────────────────────────────────

export function mlToImplied(ml: number): number {
  if (ml > 0) return 100 / (ml + 100);
  return Math.abs(ml) / (Math.abs(ml) + 100);
}

export function removeVig(homeML: number, awayML: number): { homeProb: number; awayProb: number; vig: number } {
  const rawHome = mlToImplied(homeML);
  const rawAway = mlToImplied(awayML);
  const total = rawHome + rawAway;
  const vig = total - 1.0;
  return {
    homeProb: rawHome / total,
    awayProb: rawAway / total,
    vig,
  };
}

// ─── Persistent odds storage (loaded once per run) ───────────────────────────

let _gameOddsMap: Map<string, { homeImpliedProb: number; awayImpliedProb: number; homeML: number; awayML: number }> | null = null;

export function getOddsForGame(gameId: string): { homeImpliedProb: number; awayImpliedProb: number; homeML: number; awayML: number } | null {
  return _gameOddsMap?.get(gameId) ?? null;
}

export function hasAnyOdds(): boolean {
  return (_gameOddsMap?.size ?? 0) > 0;
}

// ─── Load manual lines ────────────────────────────────────────────────────────

export function loadManualLines(date: string): Map<string, { homeML: number; awayML: number }> {
  if (!existsSync(MANUAL_LINES_PATH)) return new Map();

  try {
    const raw = readFileSync(MANUAL_LINES_PATH, 'utf-8');
    const lines = JSON.parse(raw) as ManualLines;
    const dayLines = lines[date];
    if (!dayLines) return new Map();

    const map = new Map<string, { homeML: number; awayML: number }>();
    for (const [key, val] of Object.entries(dayLines)) {
      map.set(key, val);
    }
    return map;
  } catch (err) {
    logger.warn({ err }, 'Failed to parse manual vegas_lines.json');
    return new Map();
  }
}

// ─── Fetch from The Odds API ──────────────────────────────────────────────────

export async function loadOddsApiLines(
  date: string
): Promise<Map<string, { homeML: number; awayML: number }>> {
  const apiKey = process.env.ODDS_API_KEY;
  if (!apiKey) {
    logger.debug('ODDS_API_KEY not set — skipping live odds fetch');
    return new Map();
  }

  const url = `https://api.the-odds-api.com/v4/sports/basketball_nba/odds/?apiKey=${apiKey}&regions=us&markets=h2h&dateFormat=iso&oddsFormat=american`;

  try {
    const resp = await fetch(url, { signal: AbortSignal.timeout(10000) });
    if (!resp.ok) {
      logger.warn({ status: resp.status }, 'Odds API returned error');
      return new Map();
    }

    const games = (await resp.json()) as OddsApiGame[];
    const map = new Map<string, { homeML: number; awayML: number }>();

    for (const game of games) {
      // Use consensus across bookmakers
      const allHomeOdds: number[] = [];
      const allAwayOdds: number[] = [];

      for (const book of game.bookmakers) {
        const h2h = book.markets.find(m => m.key === 'h2h');
        if (!h2h) continue;
        const homeOdds = h2h.outcomes.find(o => o.name === game.home_team)?.price;
        const awayOdds = h2h.outcomes.find(o => o.name === game.away_team)?.price;
        if (homeOdds !== undefined) allHomeOdds.push(homeOdds);
        if (awayOdds !== undefined) allAwayOdds.push(awayOdds);
      }

      if (allHomeOdds.length === 0) continue;

      const avgHomeML = allHomeOdds.reduce((a, b) => a + b, 0) / allHomeOdds.length;
      const avgAwayML = allAwayOdds.reduce((a, b) => a + b, 0) / allAwayOdds.length;

      // Build key from team names — try to match NBA abbreviations
      const homeAbbr = resolveTeamAbbr(game.home_team);
      const awayAbbr = resolveTeamAbbr(game.away_team);
      if (homeAbbr && awayAbbr) {
        map.set(`${awayAbbr}@${homeAbbr}`, { homeML: Math.round(avgHomeML), awayML: Math.round(avgAwayML) });
      }
    }

    logger.info({ games: map.size }, 'Odds API lines loaded');
    return map;
  } catch (err) {
    logger.warn({ err }, 'Failed to fetch Odds API lines');
    return new Map();
  }
}

// ─── Initialize odds for today ────────────────────────────────────────────────

export async function initializeOdds(date: string): Promise<void> {
  _gameOddsMap = new Map();

  // Manual lines take priority
  const manualLines = loadManualLines(date);
  if (manualLines.size > 0) {
    logger.info({ lines: manualLines.size }, 'Using manual Vegas lines');
    for (const [key, line] of manualLines.entries()) {
      const { homeProb, awayProb } = removeVig(line.homeML, line.awayML);
      _gameOddsMap.set(key, {
        homeImpliedProb: homeProb,
        awayImpliedProb: awayProb,
        homeML: line.homeML,
        awayML: line.awayML,
      });
    }
    return;
  }

  // Live lines from Odds API
  const apiLines = await loadOddsApiLines(date);
  for (const [key, line] of apiLines.entries()) {
    const { homeProb, awayProb } = removeVig(line.homeML, line.awayML);
    _gameOddsMap.set(key, {
      homeImpliedProb: homeProb,
      awayImpliedProb: awayProb,
      homeML: line.homeML,
      awayML: line.awayML,
    });
  }
}

// ─── Team name → abbreviation resolver ───────────────────────────────────────

const TEAM_NAME_TO_ABBR: Record<string, string> = {
  'Atlanta Hawks': 'ATL',
  'Boston Celtics': 'BOS',
  'Brooklyn Nets': 'BKN',
  'Charlotte Hornets': 'CHA',
  'Chicago Bulls': 'CHI',
  'Cleveland Cavaliers': 'CLE',
  'Dallas Mavericks': 'DAL',
  'Denver Nuggets': 'DEN',
  'Detroit Pistons': 'DET',
  'Golden State Warriors': 'GSW',
  'Houston Rockets': 'HOU',
  'Indiana Pacers': 'IND',
  'Los Angeles Clippers': 'LAC',
  'Los Angeles Lakers': 'LAL',
  'Memphis Grizzlies': 'MEM',
  'Miami Heat': 'MIA',
  'Milwaukee Bucks': 'MIL',
  'Minnesota Timberwolves': 'MIN',
  'New Orleans Pelicans': 'NOP',
  'New York Knicks': 'NYK',
  'Oklahoma City Thunder': 'OKC',
  'Orlando Magic': 'ORL',
  'Philadelphia 76ers': 'PHI',
  'Phoenix Suns': 'PHX',
  'Portland Trail Blazers': 'POR',
  'Sacramento Kings': 'SAC',
  'San Antonio Spurs': 'SAS',
  'Toronto Raptors': 'TOR',
  'Utah Jazz': 'UTA',
  'Washington Wizards': 'WAS',
};

function resolveTeamAbbr(fullName: string): string | null {
  return TEAM_NAME_TO_ABBR[fullName] ?? null;
}
