// NBA Oracle v4.0 — NBA Stats API + ESPN Injuries Client
// Free APIs — no key required.
// NBA Stats API requires specific headers to avoid 403 errors.
// Includes JSON caching (1hr TTL) + exponential backoff retry.

import { mkdirSync, readFileSync, writeFileSync, existsSync, statSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import fetch from 'node-fetch';
import { logger } from '../logger.js';
import type { NBAGame, NBATeam, NBAPlayer, InjuryReport, InjuredPlayer, GameResult } from '../types.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const CACHE_DIR = process.env.CACHE_DIR ?? resolve(__dirname, '../../cache');
const CACHE_TTL_MS = (Number(process.env.CACHE_TTL_HOURS ?? 1)) * 60 * 60 * 1000;

mkdirSync(CACHE_DIR, { recursive: true });

const NBA_STATS_BASE = 'https://stats.nba.com/stats';
const ESPN_BASE = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba';

// ─── Team abbreviation mapping (NBA team ID → abbr) ───────────────────────────
export const TEAM_ID_TO_ABBR: Record<number, string> = {
  1610612737: 'ATL',
  1610612738: 'BOS',
  1610612751: 'BKN',
  1610612766: 'CHA',
  1610612741: 'CHI',
  1610612739: 'CLE',
  1610612742: 'DAL',
  1610612743: 'DEN',
  1610612765: 'DET',
  1610612744: 'GSW',
  1610612745: 'HOU',
  1610612754: 'IND',
  1610612746: 'LAC',
  1610612747: 'LAL',
  1610612763: 'MEM',
  1610612748: 'MIA',
  1610612749: 'MIL',
  1610612750: 'MIN',
  1610612740: 'NOP',
  1610612752: 'NYK',
  1610612760: 'OKC',
  1610612753: 'ORL',
  1610612755: 'PHI',
  1610612756: 'PHX',
  1610612757: 'POR',
  1610612758: 'SAC',
  1610612759: 'SAS',
  1610612761: 'TOR',
  1610612762: 'UTA',
  1610612764: 'WAS',
};

export const ABBR_TO_TEAM_ID: Record<string, number> = Object.fromEntries(
  Object.entries(TEAM_ID_TO_ABBR).map(([id, abbr]) => [abbr, Number(id)])
);

// ─── High-altitude arenas ──────────────────────────────────────────────────────
const HIGH_ALTITUDE_TEAMS = new Set(['DEN']); // Denver ~5280ft

// ─── Cache helpers ────────────────────────────────────────────────────────────

function cacheKey(url: string): string {
  return url.replace(/[^a-zA-Z0-9]/g, '_').slice(0, 200) + '.json';
}

function readCache<T>(key: string): T | null {
  const path = resolve(CACHE_DIR, key);
  if (!existsSync(path)) return null;
  const stat = statSync(path);
  if (Date.now() - stat.mtimeMs > CACHE_TTL_MS) return null;
  try {
    return JSON.parse(readFileSync(path, 'utf-8')) as T;
  } catch {
    return null;
  }
}

function writeCache(key: string, data: unknown): void {
  const path = resolve(CACHE_DIR, key);
  try {
    writeFileSync(path, JSON.stringify(data), 'utf-8');
  } catch (err) {
    logger.warn({ err }, 'Failed to write cache');
  }
}

// ─── NBA Stats API headers (required to avoid 403) ───────────────────────────

const NBA_HEADERS = {
  'Accept': 'application/json, text/plain, */*',
  'Accept-Language': 'en-US,en;q=0.9',
  'Connection': 'keep-alive',
  'Host': 'stats.nba.com',
  'Origin': 'https://www.nba.com',
  'Referer': 'https://www.nba.com/',
  'Sec-Fetch-Dest': 'empty',
  'Sec-Fetch-Mode': 'cors',
  'Sec-Fetch-Site': 'same-site',
  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
  'x-nba-stats-origin': 'stats',
  'x-nba-stats-token': 'true',
};

// ─── Exponential backoff fetch ────────────────────────────────────────────────

async function fetchWithRetry<T>(url: string, useNbaHeaders = false, attempts = 3): Promise<T> {
  const key = cacheKey(url);
  const cached = readCache<T>(key);
  if (cached !== null) {
    logger.debug({ url }, 'Cache HIT');
    return cached;
  }

  let lastError: Error | null = null;
  for (let attempt = 0; attempt < attempts; attempt++) {
    try {
      logger.debug({ url, attempt }, 'Fetching');
      const headers = useNbaHeaders
        ? NBA_HEADERS
        : { 'User-Agent': 'NBAOracle/4.0 (educational)' };

      const resp = await fetch(url, {
        headers,
        signal: AbortSignal.timeout(20000),
      });

      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status} for ${url}`);
      }

      const data = (await resp.json()) as T;
      writeCache(key, data);
      return data;
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));
      if (attempt < attempts - 1) {
        const delay = Math.pow(2, attempt) * 1000;
        logger.warn({ url, attempt, delay, err: lastError.message }, 'Retrying after delay');
        await new Promise(r => setTimeout(r, delay));
      }
    }
  }

  throw lastError ?? new Error(`Failed to fetch ${url}`);
}

// ─── NBA Stats API result row helpers ────────────────────────────────────────

interface NBAStatsResponse {
  resultSets: Array<{
    name: string;
    headers: string[];
    rowSet: Array<Array<string | number | null>>;
  }>;
}

function rowsToObjects(headers: string[], rows: Array<Array<string | number | null>>): Record<string, string | number | null>[] {
  return rows.map(row =>
    Object.fromEntries(headers.map((h, i) => [h, row[i]]))
  );
}

// ─── Current season helper ────────────────────────────────────────────────────

export function getCurrentSeason(): string {
  const now = new Date();
  const year = now.getFullYear();
  const month = now.getMonth() + 1; // 1-indexed
  // NBA season starts in October; if before October, we're in the prior season
  if (month >= 10) {
    return `${year}-${String(year + 1).slice(2)}`;
  }
  return `${year - 1}-${String(year).slice(2)}`;
}

// ─── Schedule fetcher ─────────────────────────────────────────────────────────

export async function fetchSchedule(date: string): Promise<NBAGame[]> {
  // Format: MM/DD/YYYY for NBA API
  const [year, month, day] = date.split('-');
  const apiDate = `${month}%2F${day}%2F${year}`;

  const url = `${NBA_STATS_BASE}/scoreboardv2?GameDate=${apiDate}&LeagueID=00&DayOffset=0`;

  let data: NBAStatsResponse;
  try {
    data = await fetchWithRetry<NBAStatsResponse>(url, true);
  } catch (err) {
    logger.warn({ err, date }, 'Failed to fetch NBA schedule — returning empty');
    return [];
  }

  const gamesSet = data.resultSets.find(rs => rs.name === 'GameHeader');
  if (!gamesSet || gamesSet.rowSet.length === 0) {
    logger.info({ date }, 'No games found in scoreboardv2');
    return [];
  }

  const lineScoreSet = data.resultSets.find(rs => rs.name === 'LineScore');
  const lineScoreRows = lineScoreSet ? rowsToObjects(lineScoreSet.headers, lineScoreSet.rowSet) : [];

  const games: NBAGame[] = [];

  for (const rawRow of gamesSet.rowSet) {
    const row = Object.fromEntries(gamesSet.headers.map((h, i) => [h, rawRow[i]]));

    const gameId = String(row['GAME_ID']);
    const statusText = String(row['GAME_STATUS_TEXT'] ?? 'Scheduled');
    const arena = String(row['ARENA_NAME'] ?? '');
    const arenaCity = String(row['ARENA_CITY'] ?? '');
    const gameTimeUtc = String(row['GAME_DATE_EST'] ?? date) + 'T' + String(row['GAME_STATUS_TEXT'] ?? '00:00:00').replace(' ET', '');

    // Line scores for home and away teams
    const homeLines = lineScoreRows.filter(r => r['GAME_ID'] === gameId && r['TEAM_ABBREVIATION'] !== null);
    const awayLine = homeLines[0];
    const homeLine = homeLines[1];

    if (!awayLine || !homeLine) continue;

    const homeTeamId = Number(homeLine['TEAM_ID']);
    const awayTeamId = Number(awayLine['TEAM_ID']);

    games.push({
      gameId,
      gameDate: date,
      gameTime: gameTimeUtc,
      status: statusText,
      homeTeam: {
        teamId: homeTeamId,
        teamAbbr: TEAM_ID_TO_ABBR[homeTeamId] ?? String(homeLine['TEAM_ABBREVIATION']),
        teamName: String(homeLine['TEAM_CITY_NAME'] ?? '') + ' ' + String(homeLine['TEAM_NICKNAME'] ?? ''),
        score: homeLine['PTS'] !== null ? Number(homeLine['PTS']) : undefined,
      },
      awayTeam: {
        teamId: awayTeamId,
        teamAbbr: TEAM_ID_TO_ABBR[awayTeamId] ?? String(awayLine['TEAM_ABBREVIATION']),
        teamName: String(awayLine['TEAM_CITY_NAME'] ?? '') + ' ' + String(awayLine['TEAM_NICKNAME'] ?? ''),
        score: awayLine['PTS'] !== null ? Number(awayLine['PTS']) : undefined,
      },
      arena,
      arenaCity,
    });
  }

  logger.info({ date, games: games.length }, 'Schedule fetched');
  return games;
}

// ─── Team stats fetcher ───────────────────────────────────────────────────────

let _teamStatsCache: Map<number, NBATeam> | null = null;
let _teamStatsCacheTime = 0;

export async function fetchAllTeamStats(season?: string): Promise<Map<number, NBATeam>> {
  const now = Date.now();
  if (_teamStatsCache && now - _teamStatsCacheTime < CACHE_TTL_MS) {
    return _teamStatsCache;
  }

  const s = season ?? getCurrentSeason();

  // Fetch base stats and advanced stats in parallel
  const baseUrl = `${NBA_STATS_BASE}/leaguedashteamstats?Conference=&DateFrom=&DateTo=&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=${s}&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=`;
  const advUrl = `${NBA_STATS_BASE}/leaguedashteamstats?Conference=&DateFrom=&DateTo=&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Advanced&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=${s}&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=`;

  let baseData: NBAStatsResponse;
  let advData: NBAStatsResponse;

  try {
    [baseData, advData] = await Promise.all([
      fetchWithRetry<NBAStatsResponse>(baseUrl, true),
      fetchWithRetry<NBAStatsResponse>(advUrl, true),
    ]);
  } catch (err) {
    logger.error({ err }, 'Failed to fetch team stats');
    return new Map();
  }

  const baseSet = baseData.resultSets[0];
  const advSet = advData.resultSets[0];

  if (!baseSet || !advSet) return new Map();

  const baseRows = rowsToObjects(baseSet.headers, baseSet.rowSet);
  const advRows = rowsToObjects(advSet.headers, advSet.rowSet);

  // Build advanced stats lookup by team ID
  const advLookup = new Map<number, Record<string, string | number | null>>();
  for (const r of advRows) {
    advLookup.set(Number(r['TEAM_ID']), r);
  }

  const teamMap = new Map<number, NBATeam>();

  for (const base of baseRows) {
    const teamId = Number(base['TEAM_ID']);
    const adv = advLookup.get(teamId) ?? {};

    const w = Number(base['W'] ?? 0);
    const l = Number(base['L'] ?? 0);
    const gp = w + l;

    const offRtg = Number(adv['OFF_RATING'] ?? base['PTS'] ?? 110);
    const defRtg = Number(adv['DEF_RATING'] ?? 110);
    const netRtg = offRtg - defRtg;
    const pace = Number(adv['PACE'] ?? 100);

    const pythagoreanWinPct = gp > 0
      ? Math.pow(offRtg, 14) / (Math.pow(offRtg, 14) + Math.pow(defRtg, 14))
      : 0.5;

    teamMap.set(teamId, {
      teamId,
      teamAbbr: TEAM_ID_TO_ABBR[teamId] ?? String(base['TEAM_ABBREVIATION'] ?? ''),
      teamName: String(base['TEAM_NAME'] ?? ''),
      w,
      l,
      winPct: gp > 0 ? w / gp : 0.5,
      offRtg,
      defRtg,
      netRtg,
      pace,
      efgPct: Number(base['EFG_PCT'] ?? adv['EFG_PCT'] ?? 0.52),
      tovPct: Number(base['TM_TOV_PCT'] ?? adv['TM_TOV_PCT'] ?? 13),
      orbPct: Number(adv['OREB_PCT'] ?? 0.23),
      ftRate: Number(base['FTA_RATE'] ?? 0.20),
      threePtPct: Number(base['FG3_PCT'] ?? 0.36),
      threePtRate: Number(base['FG3A_RATE'] ?? 0.40),
      tsPct: Number(adv['TS_PCT'] ?? 0.56),
      astPct: Number(adv['AST_PCT'] ?? 0.60),
      stlPct: Number(adv['STL_PCT'] ?? 0.09),
      blkPct: Number(adv['BLK_PCT'] ?? 0.08),
      pythagoreanWinPct,
      clutchNetRtg: 0, // will be filled separately if available
    });
  }

  _teamStatsCache = teamMap;
  _teamStatsCacheTime = now;

  logger.info({ teams: teamMap.size, season: s }, 'Team stats loaded');
  return teamMap;
}

// ─── Rolling team stats (last N games) ───────────────────────────────────────

export async function fetchTeamRollingStats(
  teamAbbr: string,
  lastNGames = 10,
  season?: string
): Promise<{ netRtg: number; offRtg: number }> {
  const s = season ?? getCurrentSeason();
  const teamId = ABBR_TO_TEAM_ID[teamAbbr];
  if (!teamId) return { netRtg: 0, offRtg: 0 };

  const url = `${NBA_STATS_BASE}/leaguedashteamstats?Conference=&DateFrom=&DateTo=&Division=&GameScope=&GameSegment=&LastNGames=${lastNGames}&LeagueID=00&Location=&MeasureType=Advanced&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=${s}&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=${teamId}&TwoWay=0&VsConference=&VsDivision=`;

  try {
    const data = await fetchWithRetry<NBAStatsResponse>(url, true);
    const rs = data.resultSets[0];
    if (!rs || rs.rowSet.length === 0) return { netRtg: 0, offRtg: 0 };
    const rows = rowsToObjects(rs.headers, rs.rowSet);
    const row = rows[0];
    return {
      netRtg: Number(row['NET_RATING'] ?? 0),
      offRtg: Number(row['OFF_RATING'] ?? 110),
    };
  } catch (err) {
    logger.debug({ err, teamAbbr }, 'Rolling stats fetch failed');
    return { netRtg: 0, offRtg: 0 };
  }
}

// ─── Player stats (BPM from advanced stats) ──────────────────────────────────

let _playerStatsCache: Map<number, NBAPlayer> | null = null;
let _playerStatsCacheTime = 0;

export async function fetchAllPlayerStats(season?: string): Promise<Map<number, NBAPlayer>> {
  const now = Date.now();
  if (_playerStatsCache && now - _playerStatsCacheTime < CACHE_TTL_MS) {
    return _playerStatsCache;
  }

  const s = season ?? getCurrentSeason();
  const url = `${NBA_STATS_BASE}/leaguedashplayerstats?College=&Conference=&Country=&DateFrom=&DateTo=&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Advanced&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=${s}&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight=`;

  try {
    const data = await fetchWithRetry<NBAStatsResponse>(url, true);
    const rs = data.resultSets[0];
    if (!rs) return new Map();

    const rows = rowsToObjects(rs.headers, rs.rowSet);
    const playerMap = new Map<number, NBAPlayer>();

    for (const row of rows) {
      const playerId = Number(row['PLAYER_ID']);
      const teamId = Number(row['TEAM_ID']);
      const minPerGame = Number(row['MIN'] ?? 0);

      // Only include players with meaningful playing time
      if (minPerGame < 10) continue;

      playerMap.set(playerId, {
        playerId,
        playerName: String(row['PLAYER_NAME'] ?? ''),
        teamId,
        teamAbbr: TEAM_ID_TO_ABBR[teamId] ?? '',
        position: String(row['PLAYER_POSITION'] ?? ''),
        minutesPerGame: minPerGame,
        usageRate: Number(row['USG_PCT'] ?? 0.20),
        bpm: Number(row['NET_RATING'] ?? 0), // NBA API "NET_RATING" is on/off, closest to BPM
        offBpm: Number(row['OFF_RATING'] ?? 0) - 110,
        defBpm: 110 - Number(row['DEF_RATING'] ?? 110),
        vorp: Number(row['PIE'] ?? 0) * 10, // PIE × 10 as VORP proxy
        onNetRtg: Number(row['NET_RATING'] ?? 0),
      });
    }

    _playerStatsCache = playerMap;
    _playerStatsCacheTime = now;

    logger.info({ players: playerMap.size, season: s }, 'Player stats loaded');
    return playerMap;
  } catch (err) {
    logger.warn({ err }, 'Failed to fetch player stats — returning empty map');
    return new Map();
  }
}

// ─── Players by team ──────────────────────────────────────────────────────────

export async function getPlayersByTeam(teamAbbr: string, season?: string): Promise<NBAPlayer[]> {
  const allPlayers = await fetchAllPlayerStats(season);
  const teamId = ABBR_TO_TEAM_ID[teamAbbr];
  if (!teamId) return [];
  return Array.from(allPlayers.values()).filter(p => p.teamId === teamId);
}

// ─── Injury reports (ESPN) ────────────────────────────────────────────────────
// ESPN NBA injuries endpoint returns a grouped structure by team.

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
  // Some responses are flat arrays
  injuries?: Array<{
    athlete?: { id?: string; displayName?: string; team?: { abbreviation?: string } };
    status?: string;
    shortComment?: string;
  }>;
}

export async function fetchInjuries(): Promise<InjuryReport[]> {
  const url = `${ESPN_BASE}/injuries`;

  try {
    const data = await fetchWithRetry<ESPNInjuryRoot>(url, false);
    const teamMap = new Map<string, InjuredPlayer[]>();

    // Handle grouped format: { items: [{ team: {...}, injuries: [...] }] }
    if (data.items && Array.isArray(data.items)) {
      for (const item of data.items) {
        const teamAbbr = item.team?.abbreviation ?? 'UNK';
        if (!item.injuries) continue;

        for (const injury of item.injuries) {
          if (!injury.athlete) continue;
          const status = injury.status ?? 'Questionable';
          if (!['Out', 'Doubtful', 'Questionable', 'Day-To-Day', 'GTD'].includes(status)) continue;

          const player: InjuredPlayer = {
            playerId: Number(injury.athlete.id ?? 0),
            playerName: injury.athlete.displayName ?? 'Unknown',
            status,
            description: injury.shortComment ?? '',
          };

          if (!teamMap.has(teamAbbr)) teamMap.set(teamAbbr, []);
          teamMap.get(teamAbbr)!.push(player);
        }
      }
    }

    // Handle flat format: { injuries: [{ athlete: { team: {...} }, ... }] }
    if (data.injuries && Array.isArray(data.injuries)) {
      for (const injury of data.injuries) {
        if (!injury.athlete) continue;
        const teamAbbr = injury.athlete.team?.abbreviation ?? 'UNK';
        const status = injury.status ?? 'Questionable';
        if (!['Out', 'Doubtful', 'Questionable', 'Day-To-Day', 'GTD'].includes(status)) continue;

        const player: InjuredPlayer = {
          playerId: Number(injury.athlete.id ?? 0),
          playerName: injury.athlete.displayName ?? 'Unknown',
          status,
          description: injury.shortComment ?? '',
        };

        if (!teamMap.has(teamAbbr)) teamMap.set(teamAbbr, []);
        teamMap.get(teamAbbr)!.push(player);
      }
    }

    const reports: InjuryReport[] = [];
    for (const [teamAbbr, players] of teamMap.entries()) {
      const teamId = ABBR_TO_TEAM_ID[teamAbbr] ?? 0;
      reports.push({ teamId, teamAbbr, players });
    }

    logger.info({ teams: reports.length }, 'Injury reports fetched');
    return reports;
  } catch (err) {
    logger.warn({ err }, 'Failed to fetch injury reports — continuing without');
    return [];
  }
}

// ─── Head-to-head season record ───────────────────────────────────────────────

export async function fetchHeadToHeadRecord(
  homeAbbr: string,
  awayAbbr: string,
  season?: string
): Promise<{ homeWins: number; awayWins: number }> {
  const s = season ?? getCurrentSeason();
  const homeId = ABBR_TO_TEAM_ID[homeAbbr];
  const awayId = ABBR_TO_TEAM_ID[awayAbbr];
  if (!homeId || !awayId) return { homeWins: 0, awayWins: 0 };

  const url = `${NBA_STATS_BASE}/teamgamelog?TeamID=${homeId}&Season=${s}&SeasonType=Regular+Season`;

  try {
    const data = await fetchWithRetry<NBAStatsResponse>(url, true);
    const rs = data.resultSets[0];
    if (!rs) return { homeWins: 0, awayWins: 0 };

    const rows = rowsToObjects(rs.headers, rs.rowSet);
    let homeWins = 0;
    let awayWins = 0;

    for (const row of rows) {
      const matchup = String(row['MATCHUP'] ?? '');
      const wl = String(row['WL'] ?? '');

      const isVsAway = matchup.includes(awayAbbr);
      if (!isVsAway) continue;

      if (wl === 'W') homeWins++;
      else awayWins++;
    }

    return { homeWins, awayWins };
  } catch {
    return { homeWins: 0, awayWins: 0 };
  }
}

// ─── Recent game schedule (for rest days calculation) ─────────────────────────

export async function fetchTeamLastGameDate(teamAbbr: string, beforeDate: string, season?: string): Promise<string | null> {
  const s = season ?? getCurrentSeason();
  const teamId = ABBR_TO_TEAM_ID[teamAbbr];
  if (!teamId) return null;

  const url = `${NBA_STATS_BASE}/teamgamelog?TeamID=${teamId}&Season=${s}&SeasonType=Regular+Season`;

  try {
    const data = await fetchWithRetry<NBAStatsResponse>(url, true);
    const rs = data.resultSets[0];
    if (!rs) return null;

    const rows = rowsToObjects(rs.headers, rs.rowSet);
    // Game log is newest first; find the last game before our target date
    for (const row of rows) {
      const gameDate = String(row['GAME_DATE'] ?? '');
      // NBA API format: "APR 05, 2026" → convert to YYYY-MM-DD
      const parsed = parseDateStr(gameDate);
      if (parsed && parsed < beforeDate) {
        return parsed;
      }
    }
    return null;
  } catch {
    return null;
  }
}

function parseDateStr(nbaDateStr: string): string | null {
  try {
    const d = new Date(nbaDateStr);
    if (isNaN(d.getTime())) return null;
    return d.toISOString().split('T')[0];
  } catch {
    return null;
  }
}

// ─── Completed game results (for Elo updates + accuracy tracking) ─────────────

export async function fetchCompletedResults(date: string): Promise<GameResult[]> {
  const [year, month, day] = date.split('-');
  const apiDate = `${month}%2F${day}%2F${year}`;
  const url = `${NBA_STATS_BASE}/scoreboardv2?GameDate=${apiDate}&LeagueID=00&DayOffset=0`;

  try {
    const data = await fetchWithRetry<NBAStatsResponse>(url, true);

    const gamesSet = data.resultSets.find(rs => rs.name === 'GameHeader');
    const lineScoreSet = data.resultSets.find(rs => rs.name === 'LineScore');

    if (!gamesSet || !lineScoreSet) return [];

    const lineScoreRows = rowsToObjects(lineScoreSet.headers, lineScoreSet.rowSet);
    const gameHeaders = rowsToObjects(gamesSet.headers, gamesSet.rowSet);

    const results: GameResult[] = [];

    for (const game of gameHeaders) {
      const gameId = String(game['GAME_ID']);
      const statusText = String(game['GAME_STATUS_TEXT'] ?? '');

      // Only include final games
      if (!statusText.toLowerCase().includes('final')) continue;

      const gameLines = lineScoreRows.filter(r => r['GAME_ID'] === gameId);
      if (gameLines.length < 2) continue;

      const awayLine = gameLines[0];
      const homeLine = gameLines[1];

      results.push({
        game_id: gameId,
        date,
        home_team: TEAM_ID_TO_ABBR[Number(homeLine['TEAM_ID'])] ?? String(homeLine['TEAM_ABBREVIATION']),
        away_team: TEAM_ID_TO_ABBR[Number(awayLine['TEAM_ID'])] ?? String(awayLine['TEAM_ABBREVIATION']),
        home_score: Number(homeLine['PTS'] ?? 0),
        away_score: Number(awayLine['PTS'] ?? 0),
        arena: String(game['ARENA_NAME'] ?? ''),
        lineups: '{}',
      });
    }

    return results;
  } catch (err) {
    logger.warn({ err, date }, 'Failed to fetch completed results');
    return [];
  }
}

// ─── High altitude check ──────────────────────────────────────────────────────

export function isHighAltitude(teamAbbr: string): boolean {
  return HIGH_ALTITUDE_TEAMS.has(teamAbbr);
}
