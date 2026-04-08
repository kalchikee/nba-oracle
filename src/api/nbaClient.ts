// NBA Oracle v4.0 — ESPN-based API client
// Uses ESPN's public APIs (no key required, no IP restrictions, works from GitHub Actions).
// stats.nba.com is blocked by Cloudflare on cloud IPs — ESPN is the reliable alternative.

import { mkdirSync, readFileSync, writeFileSync, existsSync, statSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import fetch from 'node-fetch';
import { logger } from '../logger.js';
import type { NBAGame, NBATeam, NBAPlayer, InjuryReport, InjuredPlayer, GameResult } from '../types.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const CACHE_DIR = process.env.CACHE_DIR ?? resolve(__dirname, '../../cache');
const CACHE_TTL_MS = (Number(process.env.CACHE_TTL_HOURS ?? 6)) * 60 * 60 * 1000;

mkdirSync(CACHE_DIR, { recursive: true });

const ESPN_BASE     = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba';
const ESPN_WEB_BASE = 'https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba';

// ─── Team abbreviation ↔ NBA team ID mapping ──────────────────────────────────
export const TEAM_ID_TO_ABBR: Record<number, string> = {
  1610612737: 'ATL', 1610612738: 'BOS', 1610612751: 'BKN', 1610612766: 'CHA',
  1610612741: 'CHI', 1610612739: 'CLE', 1610612742: 'DAL', 1610612743: 'DEN',
  1610612765: 'DET', 1610612744: 'GSW', 1610612745: 'HOU', 1610612754: 'IND',
  1610612746: 'LAC', 1610612747: 'LAL', 1610612763: 'MEM', 1610612748: 'MIA',
  1610612749: 'MIL', 1610612750: 'MIN', 1610612740: 'NOP', 1610612752: 'NYK',
  1610612760: 'OKC', 1610612753: 'ORL', 1610612755: 'PHI', 1610612756: 'PHX',
  1610612757: 'POR', 1610612758: 'SAC', 1610612759: 'SAS', 1610612761: 'TOR',
  1610612762: 'UTA', 1610612764: 'WAS',
};

export const ABBR_TO_TEAM_ID: Record<string, number> = Object.fromEntries(
  Object.entries(TEAM_ID_TO_ABBR).map(([id, abbr]) => [abbr, Number(id)])
);

// ESPN team ID → NBA abbreviation (ESPN uses different numeric IDs)
const ESPN_ID_TO_ABBR: Record<number, string> = {
  1: 'ATL', 2: 'BOS', 17: 'BKN', 30: 'CHA', 4: 'CHI', 5: 'CLE',
  6: 'DAL', 7: 'DEN', 8: 'DET', 9: 'GSW', 10: 'HOU', 11: 'IND',
  12: 'LAC', 13: 'LAL', 29: 'MEM', 14: 'MIA', 15: 'MIL', 16: 'MIN',
  3: 'NOP', 18: 'NYK', 25: 'OKC', 19: 'ORL', 20: 'PHI', 21: 'PHX',
  22: 'POR', 23: 'SAC', 24: 'SAS', 28: 'TOR', 26: 'UTA', 27: 'WAS',
};

const ABBR_TO_ESPN_ID: Record<string, number> = Object.fromEntries(
  Object.entries(ESPN_ID_TO_ABBR).map(([id, abbr]) => [abbr, Number(id)])
);

const HIGH_ALTITUDE_TEAMS = new Set(['DEN']);

// ─── Cache helpers ─────────────────────────────────────────────────────────────

function cacheKey(url: string): string {
  return url.replace(/[^a-zA-Z0-9]/g, '_').slice(0, 200) + '.json';
}

function readCache<T>(key: string): T | null {
  const path = resolve(CACHE_DIR, key);
  if (!existsSync(path)) return null;
  const stat = statSync(path);
  if (Date.now() - stat.mtimeMs > CACHE_TTL_MS) return null;
  try { return JSON.parse(readFileSync(path, 'utf-8')) as T; }
  catch { return null; }
}

function writeCache(key: string, data: unknown): void {
  try { writeFileSync(resolve(CACHE_DIR, key), JSON.stringify(data), 'utf-8'); }
  catch (err) { logger.warn({ err }, 'Failed to write cache'); }
}

// ─── Fetch with retry ──────────────────────────────────────────────────────────

async function fetchWithRetry<T>(url: string, attempts = 3): Promise<T> {
  const key = cacheKey(url);
  const cached = readCache<T>(key);
  if (cached !== null) { logger.debug({ url }, 'Cache HIT'); return cached; }

  let lastError: Error | null = null;
  for (let attempt = 0; attempt < attempts; attempt++) {
    try {
      const resp = await fetch(url, {
        headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36' },
        signal: AbortSignal.timeout(15000),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = (await resp.json()) as T;
      writeCache(key, data);
      return data;
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));
      if (attempt < attempts - 1) await new Promise(r => setTimeout(r, (attempt + 1) * 2000));
    }
  }
  throw lastError ?? new Error(`Failed: ${url}`);
}

// ─── Current season ────────────────────────────────────────────────────────────

export function getCurrentSeason(): string {
  const now = new Date();
  const year = now.getFullYear();
  return (now.getMonth() + 1) >= 10
    ? `${year}-${String(year + 1).slice(2)}`
    : `${year - 1}-${String(year).slice(2)}`;
}

function getCurrentSeasonYear(): number {
  const now = new Date();
  return (now.getMonth() + 1) >= 10 ? now.getFullYear() + 1 : now.getFullYear();
}

// ─── ESPN scoreboard response types ───────────────────────────────────────────

interface ESPNCompetitor {
  homeAway: 'home' | 'away';
  score?: string;
  records?: Array<{ type: string; summary: string }>;
  team: { id: string; abbreviation: string; displayName: string };
}

interface ESPNEvent {
  id: string;
  date: string;
  status: { type: { name: string; description: string } };
  competitions: Array<{
    competitors: ESPNCompetitor[];
    venue?: { fullName?: string; address?: { city?: string } };
  }>;
}

interface ESPNScoreboardResp { events?: ESPNEvent[] }

// ─── Schedule ─────────────────────────────────────────────────────────────────

export async function fetchSchedule(date: string): Promise<NBAGame[]> {
  const dateStr = date.replace(/-/g, ''); // YYYYMMDD
  const url = `${ESPN_BASE}/scoreboard?dates=${dateStr}&limit=20`;

  let data: ESPNScoreboardResp;
  try {
    data = await fetchWithRetry<ESPNScoreboardResp>(url);
  } catch (err) {
    logger.warn({ err, date }, 'Failed to fetch ESPN schedule — returning empty');
    return [];
  }

  const events = data.events ?? [];
  if (events.length === 0) {
    logger.info({ date }, 'No games on ESPN scoreboard');
    return [];
  }

  const games: NBAGame[] = [];

  for (const event of events) {
    const comp = event.competitions[0];
    if (!comp) continue;

    const home = comp.competitors.find(c => c.homeAway === 'home');
    const away = comp.competitors.find(c => c.homeAway === 'away');
    if (!home || !away) continue;

    const homeAbbr = home.team.abbreviation;
    const awayAbbr = away.team.abbreviation;
    const homeId = ABBR_TO_TEAM_ID[homeAbbr] ?? 0;
    const awayId = ABBR_TO_TEAM_ID[awayAbbr] ?? 0;
    const statusDesc = event.status.type.description;

    games.push({
      gameId:   event.id,
      gameDate: date,
      gameTime: event.date,
      status:   statusDesc,
      homeTeam: {
        teamId:   homeId,
        teamAbbr: homeAbbr,
        teamName: home.team.displayName,
        score:    home.score ? Number(home.score) : undefined,
      },
      awayTeam: {
        teamId:   awayId,
        teamAbbr: awayAbbr,
        teamName: away.team.displayName,
        score:    away.score ? Number(away.score) : undefined,
      },
      arena:     comp.venue?.fullName ?? '',
      arenaCity: comp.venue?.address?.city ?? '',
    });
  }

  logger.info({ date, games: games.length }, 'Schedule fetched (ESPN)');
  return games;
}

// ─── Team stats (ESPN) ────────────────────────────────────────────────────────

interface ESPNStatCategory {
  name: string;
  stats: Array<{ name: string; value: number }>;
}

interface ESPNTeamStatsResp {
  team?: { id?: string; abbreviation?: string; record?: { items?: Array<{ description?: string; stats?: Array<{ name: string; value: number }> }> } };
  results?: { stats?: { categories?: ESPNStatCategory[] } };
}

function getStat(categories: ESPNStatCategory[], statName: string): number | undefined {
  for (const cat of categories) {
    const s = cat.stats.find(s => s.name === statName);
    if (s !== undefined) return s.value;
  }
  return undefined;
}

let _teamStatsCache: Map<number, NBATeam> | null = null;
let _teamStatsCacheTime = 0;

export async function fetchAllTeamStats(_season?: string): Promise<Map<number, NBATeam>> {
  const now = Date.now();
  if (_teamStatsCache && now - _teamStatsCacheTime < CACHE_TTL_MS) return _teamStatsCache;

  const teamMap = new Map<number, NBATeam>();

  // Fetch all 30 teams' statistics in parallel
  const entries = Object.entries(ABBR_TO_ESPN_ID);
  const results = await Promise.allSettled(
    entries.map(([abbr, espnId]) =>
      fetchWithRetry<ESPNTeamStatsResp>(`${ESPN_WEB_BASE}/teams/${espnId}/statistics`)
        .then(data => ({ abbr, espnId, data }))
    )
  );

  for (const result of results) {
    if (result.status !== 'fulfilled') continue;
    const { abbr, data } = result.value;

    const categories = data.results?.stats?.categories ?? [];

    // Points scored / allowed
    const pts    = getStat(categories, 'avgPoints')         ?? 110;
    const oppPts = getStat(categories, 'avgPointsAllowed')  ?? 110;

    // Shooting
    const fgm  = getStat(categories, 'avgFieldGoalsMade')               ?? 40;
    const fga  = getStat(categories, 'avgFieldGoalsAttempted')           ?? 88;
    const fg3m = getStat(categories, 'avgThreePointFieldGoalsMade')      ?? 12;
    const fg3a = getStat(categories, 'avgThreePointFieldGoalsAttempted') ?? 32;
    const ftm  = getStat(categories, 'avgFreeThrowsMade')                ?? 14;
    const fta  = getStat(categories, 'avgFreeThrowsAttempted')           ?? 18;
    const oreb = getStat(categories, 'avgOffensiveRebounds')             ?? 9;
    const tov  = getStat(categories, 'avgTurnovers')                     ?? 13;

    // Derived advanced stats (approximations without real possession data)
    const possEst   = fga - oreb + tov + 0.44 * fta;        // possession estimate
    const offRtg    = possEst > 0 ? (pts / possEst) * 100 : pts;
    const defRtg    = possEst > 0 ? (oppPts / possEst) * 100 : oppPts;
    const netRtg    = offRtg - defRtg;
    const efgPct    = fga > 0 ? (fgm + 0.5 * fg3m) / fga : 0.52;
    const tovPct    = possEst > 0 ? (tov / possEst) * 100 : 13;
    const orbPct    = 0.23; // no opponent data available — use league avg
    const ftRate    = fga > 0 ? fta / fga : 0.20;
    const threePtPct = fg3a > 0 ? fg3m / fg3a : 0.36;
    const threePtRate = fga > 0 ? fg3a / fga : 0.40;
    const tsPct     = (2 * (fga + 0.44 * fta)) > 0 ? pts / (2 * (fga + 0.44 * fta)) : 0.56;

    // Pythagorean win probability (Hollinger exponent = 14)
    const pythagoreanWinPct =
      (Math.pow(offRtg, 14)) / (Math.pow(offRtg, 14) + Math.pow(defRtg, 14));

    // W/L from team record (if available)
    const recordItems = data.team?.record?.items ?? [];
    const overallRecord = recordItems.find(r => r.description === 'Overall');
    const winsLossesStats = overallRecord?.stats ?? [];
    const w = winsLossesStats.find(s => s.name === 'wins')?.value   ?? 41;
    const l = winsLossesStats.find(s => s.name === 'losses')?.value ?? 41;
    const gp = w + l;

    const nbaId = ABBR_TO_TEAM_ID[abbr] ?? 0;

    teamMap.set(nbaId, {
      teamId:   nbaId,
      teamAbbr: abbr,
      teamName: abbr,
      w, l,
      winPct:   gp > 0 ? w / gp : 0.5,
      offRtg,
      defRtg,
      netRtg,
      pace:     100,  // league average — not available via ESPN without play-by-play
      efgPct,
      tovPct,
      orbPct,
      ftRate,
      threePtPct,
      threePtRate,
      tsPct,
      astPct:             getStat(categories, 'avgAssists')     ? (getStat(categories, 'avgAssists')! / pts) : 0.60,
      stlPct:             0.09,
      blkPct:             0.08,
      pythagoreanWinPct,
      clutchNetRtg:       0,
    });
  }

  if (teamMap.size > 0) {
    _teamStatsCache = teamMap;
    _teamStatsCacheTime = now;
    logger.info({ teams: teamMap.size }, 'Team stats loaded (ESPN)');
  } else {
    logger.warn('ESPN team stats returned 0 teams — using defaults');
  }

  return teamMap;
}

// ─── Rolling stats ─────────────────────────────────────────────────────────────
// ESPN doesn't support last-N-games queries — returns season averages as proxy.

export async function fetchTeamRollingStats(
  teamAbbr: string,
  _lastNGames = 10,
): Promise<{ netRtg: number; offRtg: number }> {
  const allStats = await fetchAllTeamStats();
  const nbaId = ABBR_TO_TEAM_ID[teamAbbr];
  if (!nbaId) return { netRtg: 0, offRtg: 110 };
  const team = allStats.get(nbaId);
  return team ? { netRtg: team.netRtg, offRtg: team.offRtg } : { netRtg: 0, offRtg: 110 };
}

// ─── Player stats ──────────────────────────────────────────────────────────────
// ESPN player stats require per-player API calls (not practical at scale).
// Return empty — player impact features will default to 0.

let _playerStatsCache: Map<number, NBAPlayer> | null = null;
let _playerStatsCacheTime = 0;

export async function fetchAllPlayerStats(_season?: string): Promise<Map<number, NBAPlayer>> {
  const now = Date.now();
  if (_playerStatsCache && now - _playerStatsCacheTime < CACHE_TTL_MS) return _playerStatsCache;

  const playerMap = new Map<number, NBAPlayer>();

  // Attempt to load from ESPN roster for each team — best effort
  const entries = Object.entries(ABBR_TO_ESPN_ID);
  const rosterResults = await Promise.allSettled(
    entries.map(([abbr, espnId]) =>
      fetchWithRetry<{ athletes?: Array<{ id?: string; displayName?: string; statistics?: { splits?: { categories?: ESPNStatCategory[] } } }> }>(
        `${ESPN_WEB_BASE}/teams/${espnId}/roster`
      ).then(data => ({ abbr, espnId, data }))
    )
  );

  for (const result of rosterResults) {
    if (result.status !== 'fulfilled') continue;
    const { abbr, data } = result.value;
    const nbaTeamId = ABBR_TO_TEAM_ID[abbr] ?? 0;

    for (const athlete of data.athletes ?? []) {
      const playerId = Number(athlete.id ?? 0);
      if (!playerId) continue;
      playerMap.set(playerId, {
        playerId,
        playerName:    athlete.displayName ?? '',
        teamId:        nbaTeamId,
        teamAbbr:      abbr,
        position:      '',
        minutesPerGame: 20,
        usageRate:     0.20,
        bpm:           0,
        offBpm:        0,
        defBpm:        0,
        vorp:          0,
        onNetRtg:      0,
      });
    }
  }

  _playerStatsCache = playerMap;
  _playerStatsCacheTime = now;
  logger.info({ players: playerMap.size }, 'Player roster loaded (ESPN)');
  return playerMap;
}

export async function getPlayersByTeam(teamAbbr: string, _season?: string): Promise<NBAPlayer[]> {
  const allPlayers = await fetchAllPlayerStats();
  const teamId = ABBR_TO_TEAM_ID[teamAbbr];
  if (!teamId) return [];
  return Array.from(allPlayers.values()).filter(p => p.teamId === teamId);
}

// ─── Injuries (ESPN — already working) ────────────────────────────────────────

interface ESPNTeamInjury {
  team?: { abbreviation?: string };
  injuries?: Array<{
    athlete?: { id?: string; displayName?: string };
    status?: string;
    shortComment?: string;
  }>;
}

interface ESPNInjuryRoot {
  items?: ESPNTeamInjury[];
  injuries?: Array<{
    athlete?: { id?: string; displayName?: string; team?: { abbreviation?: string } };
    status?: string;
    shortComment?: string;
  }>;
}

export async function fetchInjuries(): Promise<InjuryReport[]> {
  const url = `${ESPN_BASE}/injuries`;
  try {
    const data = await fetchWithRetry<ESPNInjuryRoot>(url);
    const teamMap = new Map<string, InjuredPlayer[]>();

    if (data.items && Array.isArray(data.items)) {
      for (const item of data.items) {
        const teamAbbr = item.team?.abbreviation ?? 'UNK';
        for (const injury of item.injuries ?? []) {
          if (!injury.athlete) continue;
          const status = injury.status ?? 'Questionable';
          if (!['Out', 'Doubtful', 'Questionable', 'Day-To-Day', 'GTD'].includes(status)) continue;
          if (!teamMap.has(teamAbbr)) teamMap.set(teamAbbr, []);
          teamMap.get(teamAbbr)!.push({
            playerId:   Number(injury.athlete.id ?? 0),
            playerName: injury.athlete.displayName ?? 'Unknown',
            status,
            description: injury.shortComment ?? '',
          });
        }
      }
    }

    if (data.injuries && Array.isArray(data.injuries)) {
      for (const injury of data.injuries) {
        if (!injury.athlete) continue;
        const teamAbbr = injury.athlete.team?.abbreviation ?? 'UNK';
        const status = injury.status ?? 'Questionable';
        if (!['Out', 'Doubtful', 'Questionable', 'Day-To-Day', 'GTD'].includes(status)) continue;
        if (!teamMap.has(teamAbbr)) teamMap.set(teamAbbr, []);
        teamMap.get(teamAbbr)!.push({
          playerId:   Number(injury.athlete.id ?? 0),
          playerName: injury.athlete.displayName ?? 'Unknown',
          status,
          description: injury.shortComment ?? '',
        });
      }
    }

    const reports: InjuryReport[] = [];
    for (const [teamAbbr, players] of teamMap.entries()) {
      reports.push({ teamId: ABBR_TO_TEAM_ID[teamAbbr] ?? 0, teamAbbr, players });
    }
    logger.info({ teams: reports.length }, 'Injury reports fetched (ESPN)');
    return reports;
  } catch (err) {
    logger.warn({ err }, 'Failed to fetch injuries — continuing without');
    return [];
  }
}

// ─── Head-to-head (simplified — ESPN doesn't surface this easily) ─────────────

export async function fetchHeadToHeadRecord(
  _homeAbbr: string,
  _awayAbbr: string,
): Promise<{ homeWins: number; awayWins: number }> {
  return { homeWins: 0, awayWins: 0 }; // feature will default to neutral
}

// ─── Last game date (for rest days) ───────────────────────────────────────────

export async function fetchTeamLastGameDate(
  teamAbbr: string,
  beforeDate: string,
): Promise<string | null> {
  const espnId = ABBR_TO_ESPN_ID[teamAbbr];
  if (!espnId) return null;

  const seasonYear = getCurrentSeasonYear();
  const url = `${ESPN_BASE}/teams/${espnId}/schedule?season=${seasonYear}`;

  try {
    const data = await fetchWithRetry<{ events?: Array<{ date: string; competitions?: Array<{ status?: { type?: { completed?: boolean } } }> }> }>(url);
    const events = (data.events ?? []).filter(e => {
      const comp = e.competitions?.[0];
      return comp?.status?.type?.completed === true;
    });

    // Find the latest completed game before beforeDate
    let latestDate: string | null = null;
    for (const event of events) {
      const d = event.date.split('T')[0];
      if (d < beforeDate && (!latestDate || d > latestDate)) {
        latestDate = d;
      }
    }
    return latestDate;
  } catch {
    return null;
  }
}

// ─── Completed game results (for recap and Elo updates) ───────────────────────

export async function fetchCompletedResults(date: string): Promise<GameResult[]> {
  const dateStr = date.replace(/-/g, '');
  const url = `${ESPN_BASE}/scoreboard?dates=${dateStr}&limit=20`;

  try {
    const data = await fetchWithRetry<ESPNScoreboardResp>(url);
    const events = data.events ?? [];
    const results: GameResult[] = [];

    for (const event of events) {
      const statusName = event.status.type.name;
      if (statusName !== 'STATUS_FINAL') continue;

      const comp = event.competitions[0];
      if (!comp) continue;

      const home = comp.competitors.find(c => c.homeAway === 'home');
      const away = comp.competitors.find(c => c.homeAway === 'away');
      if (!home || !away) continue;

      results.push({
        game_id:    event.id,
        date,
        home_team:  home.team.abbreviation,
        away_team:  away.team.abbreviation,
        home_score: Number(home.score ?? 0),
        away_score: Number(away.score ?? 0),
        arena:      comp.venue?.fullName ?? '',
        lineups:    '{}',
      });
    }

    logger.info({ date, results: results.length }, 'Completed results fetched (ESPN)');
    return results;
  } catch (err) {
    logger.warn({ err, date }, 'Failed to fetch completed results');
    return [];
  }
}

// ─── High altitude check ───────────────────────────────────────────────────────

export function isHighAltitude(teamAbbr: string): boolean {
  return HIGH_ALTITUDE_TEAMS.has(teamAbbr);
}
