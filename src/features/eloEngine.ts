// NBA Oracle v4.0 — Elo Rating Engine
// K-factor: 20 (higher than MLB due to fewer games)
// Margin of victory: log(1 + margin) capped at 20 pts
// Home court adjustment: +100 Elo points in expected score
// Offseason regression: 75% carry + 25% league mean (1500)

import { getElo, upsertElo, getAllElos } from '../db/database.js';
import { logger } from '../logger.js';

const LEAGUE_MEAN_ELO = 1500;
const K_FACTOR = 20;
const HOME_ADVANTAGE_ELO = 100; // Elo points added to home team expected score
const MOV_CAP = 20;             // Cap margin of victory at 20 points

export const ALL_NBA_ABBRS = [
  'ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN',
  'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA',
  'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHX',
  'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS',
];

// ─── Seed all teams with default Elo if not present ───────────────────────────

export function seedElos(): void {
  for (const abbr of ALL_NBA_ABBRS) {
    const existing = getElo(abbr);
    if (existing === LEAGUE_MEAN_ELO) {
      // already at default, ensure it's persisted
      upsertElo({ teamAbbr: abbr, rating: LEAGUE_MEAN_ELO, updatedAt: new Date().toISOString() });
    }
  }
}

// ─── Expected win probability from Elo ───────────────────────────────────────

export function eloWinProb(homeElo: number, awayElo: number): number {
  // Home advantage baked into expected score calculation
  const eloDiff = (homeElo + HOME_ADVANTAGE_ELO) - awayElo;
  return 1 / (1 + Math.pow(10, -eloDiff / 400));
}

export function getEloDiff(homeAbbr: string, awayAbbr: string): number {
  const homeElo = getElo(homeAbbr);
  const awayElo = getElo(awayAbbr);
  return homeElo - awayElo;
}

// ─── Update Elo after a game ──────────────────────────────────────────────────

export function updateEloAfterGame(
  homeAbbr: string,
  awayAbbr: string,
  homeScore: number,
  awayScore: number
): void {
  const homeElo = getElo(homeAbbr);
  const awayElo = getElo(awayAbbr);

  const homeExpected = eloWinProb(homeElo, awayElo);
  const homeActual = homeScore > awayScore ? 1 : 0;

  // Margin of victory multiplier: log(1 + |margin|) capped at log(1 + 20)
  const margin = Math.abs(homeScore - awayScore);
  const movMultiplier = Math.log(1 + Math.min(margin, MOV_CAP));

  // Adjust K by MOV
  const adjustedK = K_FACTOR * movMultiplier;

  const homeNewElo = homeElo + adjustedK * (homeActual - homeExpected);
  const awayNewElo = awayElo + adjustedK * ((1 - homeActual) - (1 - homeExpected));

  const now = new Date().toISOString();
  upsertElo({ teamAbbr: homeAbbr, rating: Math.round(homeNewElo), updatedAt: now });
  upsertElo({ teamAbbr: awayAbbr, rating: Math.round(awayNewElo), updatedAt: now });

  logger.debug(
    {
      home: homeAbbr,
      away: awayAbbr,
      homeElo: homeNewElo.toFixed(0),
      awayElo: awayNewElo.toFixed(0),
      margin,
    },
    'Elo updated'
  );
}

// ─── Offseason regression ─────────────────────────────────────────────────────
// Call this at the start of each new season.

export function applyOffseasonRegression(): void {
  const ratings = getAllElos();
  const now = new Date().toISOString();

  for (const r of ratings) {
    const newRating = 0.75 * r.rating + 0.25 * LEAGUE_MEAN_ELO;
    upsertElo({ teamAbbr: r.teamAbbr, rating: Math.round(newRating), updatedAt: now });
  }

  logger.info({ teams: ratings.length }, 'Offseason Elo regression applied (75% carry + 25% mean)');
}

// ─── Log5 formula ─────────────────────────────────────────────────────────────
// Probability that team A beats team B given their win rates.

export function log5Prob(homeWinPct: number, awayWinPct: number): number {
  // Prevent division by zero
  const pA = Math.max(0.01, Math.min(0.99, homeWinPct));
  const pB = Math.max(0.01, Math.min(0.99, awayWinPct));
  const pAvg = 0.5; // league average

  return (pA - pA * pB) / (pA + pB - 2 * pA * pB);
}
