// NBA Oracle v4.0 â€” Feature Engineering
// Computes all 30+ features as home-vs-away differences
// Features: Elo, net rating, efficiency, form, fatigue, player impact, clutch, venue

import { logger } from '../logger.js';
import type { NBAGame, NBATeam, NBAPlayer, FeatureVector } from '../types.js';
import {
  fetchAllTeamStats, fetchTeamRollingStats, fetchInjuries,
  fetchHeadToHeadRecord, fetchTeamLastGameDate, isHighAltitude,
  getPlayersByTeam, getCurrentSeason, TEAM_ID_TO_ABBR,
} from '../api/nbaClient.js';
import { getEloDiff, log5Prob } from './eloEngine.js';

const LEAGUE_AVG_NET_RTG = 0.0;
const LEAGUE_AVG_OFF_RTG = 113.0;

// â”€â”€â”€ Main feature computation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

export async function computeFeatures(
  game: NBAGame,
  gameDate: string
): Promise<FeatureVector> {
  const homeAbbr = game.homeTeam.teamAbbr;
  const awayAbbr = game.awayTeam.teamAbbr;

  logger.debug({ home: homeAbbr, away: awayAbbr }, 'Computing features');

  const season = getCurrentSeason();

  // Fetch all data in parallel for speed
  const [
    teamStats,
    homeRolling,
    awayRolling,
    injuries,
    h2h,
    homeLastGame,
    awayLastGame,
    homePlayers,
    awayPlayers,
  ] = await Promise.all([
    fetchAllTeamStats(season),
    fetchTeamRollingStats(homeAbbr, 10),
    fetchTeamRollingStats(awayAbbr, 10),
    fetchInjuries(),
    fetchHeadToHeadRecord(homeAbbr, awayAbbr),
    fetchTeamLastGameDate(homeAbbr, gameDate),
    fetchTeamLastGameDate(awayAbbr, gameDate),
    getPlayersByTeam(homeAbbr, season),
    getPlayersByTeam(awayAbbr, season),
  ]);

  const homeTeam = teamStats.get(game.homeTeam.teamId);
  const awayTeam = teamStats.get(game.awayTeam.teamId);

  if (!homeTeam || !awayTeam) {
    logger.warn({ home: homeAbbr, away: awayAbbr }, 'Missing team stats â€” using defaults');
  }

  const home = homeTeam ?? defaultTeam(homeAbbr);
  const away = awayTeam ?? defaultTeam(awayAbbr);

  // â”€â”€ Elo diff â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const eloDiff = getEloDiff(homeAbbr, awayAbbr);

  // â”€â”€ Team strength diffs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const netRtgDiff = home.netRtg - away.netRtg;
  const offRtgDiff = home.offRtg - away.offRtg;
  const defRtgDiff = home.defRtg - away.defRtg; // lower is better for defense
  const paceDiff = home.pace - away.pace;

  // â”€â”€ Pythagorean â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const pythagoreanDiff = (home.pythagoreanWinPct ?? 0.5) - (away.pythagoreanWinPct ?? 0.5);

  // â”€â”€ Log5 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const log5 = log5Prob(home.winPct, away.winPct);

  // â”€â”€ Shooting efficiency diffs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const efgDiff = home.efgPct - away.efgPct;
  const tovDiff = home.tovPct - away.tovPct;          // lower is better
  const orebDiff = home.orbPct - away.orbPct;
  const ftRateDiff = home.ftRate - away.ftRate;
  const threePtRateDiff = home.threePtRate - away.threePtRate;
  const threePtPctDiff = home.threePtPct - away.threePtPct;
  const tsDiff = home.tsPct - away.tsPct;
  const astDiff = home.astPct - away.astPct;
  const stlDiff = home.stlPct - away.stlPct;
  const blkDiff = home.blkPct - away.blkPct;

  // â”€â”€ Rolling form (anti-recency bias) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  // Raw recent performance is mostly noise; season baseline is more predictive.
  // Blend: 55% season baseline + 45% recent 10-game rolling average.
  // This prevents hot/cold streaks from overly swinging win probabilities.
  const BASELINE_W = 0.55;
  const RECENT_W   = 0.45;

  const homeBlendedNetRtg = BASELINE_W * home.netRtg + RECENT_W * homeRolling.netRtg;
  const awayBlendedNetRtg = BASELINE_W * away.netRtg + RECENT_W * awayRolling.netRtg;
  const team10dNetRtgDiff = homeBlendedNetRtg - awayBlendedNetRtg;

  const homeBlendedOffRtg = BASELINE_W * home.offRtg + RECENT_W * homeRolling.offRtg;
  const awayBlendedOffRtg = BASELINE_W * away.offRtg + RECENT_W * awayRolling.offRtg;
  const team10dOffRtgDiff = homeBlendedOffRtg - awayBlendedOffRtg;

  // Momentum: cap at Â±5 net rating points to prevent outlier streaks dominating
  const homeMomentum = Math.max(-5, Math.min(5, homeRolling.netRtg - home.netRtg));
  const awayMomentum = Math.max(-5, Math.min(5, awayRolling.netRtg - away.netRtg));
  const momentumDiff = homeMomentum - awayMomentum;

  // â”€â”€ Rest & fatigue â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const today = new Date(gameDate);
  const homeRestDays = homeLastGame
    ? Math.round((today.getTime() - new Date(homeLastGame).getTime()) / (1000 * 60 * 60 * 24))
    : 3; // assume 3 days rest if unknown
  const awayRestDays = awayLastGame
    ? Math.round((today.getTime() - new Date(awayLastGame).getTime()) / (1000 * 60 * 60 * 24))
    : 3;

  const restDaysDiff = homeRestDays - awayRestDays;
  const b2bHome = homeRestDays === 1 ? 1 : 0;
  const b2bAway = awayRestDays === 1 ? 1 : 0;

  // Travel timezone shift â€” simplified: road team always "traveled" more
  // In a full implementation, this would use actual game city coordinates
  const travelTzShiftHome = 0;  // home team doesn't travel
  const travelTzShiftAway = 1;  // away team travels (default 1 tz)

  // â”€â”€ Altitude factor â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  // Denver Nuggets home games: visiting team takes altitude penalty
  const altitudeFactor = isHighAltitude(homeAbbr) ? 1.5 : 0;

  // â”€â”€ Player impact â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const homeInjuries = injuries.find(ir => ir.teamAbbr === homeAbbr);
  const awayInjuries = injuries.find(ir => ir.teamAbbr === awayAbbr);

  const { starImpact: homeStarImpact, injuryImpact: homeInjuryImpact, lineupNetRtg: homeLineupNetRtg, benchNetRtg: homeBenchNetRtg }
    = computePlayerImpact(homePlayers, homeInjuries?.players ?? []);
  const { starImpact: awayStarImpact, injuryImpact: awayInjuryImpact, lineupNetRtg: awayLineupNetRtg, benchNetRtg: awayBenchNetRtg }
    = computePlayerImpact(awayPlayers, awayInjuries?.players ?? []);

  const starPlayerImpactDiff = homeStarImpact - awayStarImpact;
  const injuryImpactDiff = homeInjuryImpact - awayInjuryImpact;
  const lineupNetRtgDiff = homeLineupNetRtg - awayLineupNetRtg;
  const benchNetRtgDiff = homeBenchNetRtg - awayBenchNetRtg;

  // â”€â”€ Clutch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const clutchNetRtgDiff = (home.clutchNetRtg ?? 0) - (away.clutchNetRtg ?? 0);

  // â”€â”€ H2H â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const h2hTotal = h2h.homeWins + h2h.awayWins;
  const h2hRecord = h2hTotal > 0 ? h2h.homeWins / h2hTotal : 0.5;

  const features: FeatureVector = {
    elo_diff: eloDiff,
    net_rtg_diff: netRtgDiff,
    off_rtg_diff: offRtgDiff,
    def_rtg_diff: defRtgDiff,
    pace_diff: paceDiff,
    pythagorean_diff: pythagoreanDiff,
    log5_prob: log5,

    efg_pct_diff: efgDiff,
    tov_pct_diff: tovDiff,
    oreb_pct_diff: orebDiff,
    ft_rate_diff: ftRateDiff,
    three_pt_rate_diff: threePtRateDiff,
    three_pt_pct_diff: threePtPctDiff,
    ts_pct_diff: tsDiff,
    ast_pct_diff: astDiff,
    stl_pct_diff: stlDiff,
    blk_pct_diff: blkDiff,

    team_10d_net_rtg_diff: team10dNetRtgDiff,
    team_10d_off_rtg_diff: team10dOffRtgDiff,
    momentum_diff: momentumDiff,

    rest_days_diff: restDaysDiff,
    b2b_home: b2bHome,
    b2b_away: b2bAway,
    travel_tz_shift_home: travelTzShiftHome,
    travel_tz_shift_away: travelTzShiftAway,

    is_home: 1,
    altitude_factor: altitudeFactor,

    star_player_impact_diff: starPlayerImpactDiff,
    injury_impact_diff: injuryImpactDiff,
    lineup_net_rtg_diff: lineupNetRtgDiff,
    bench_net_rtg_diff: benchNetRtgDiff,

    clutch_net_rtg_diff: clutchNetRtgDiff,
    h2h_season_record: h2hRecord,

    vegas_home_prob: 0, // filled at pipeline level
  };

  logger.debug(
    {
      home: homeAbbr,
      away: awayAbbr,
      eloDiff: eloDiff.toFixed(0),
      netRtgDiff: netRtgDiff.toFixed(2),
      restDaysDiff,
      b2bHome,
      b2bAway,
    },
    'Features computed'
  );

  return features;
}

// â”€â”€â”€ Player impact computation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

interface PlayerImpactResult {
  starImpact: number;       // sum of top-3 BPM
  injuryImpact: number;     // net WAR lost (negative = team hurt)
  lineupNetRtg: number;     // estimated starting 5 net rating
  benchNetRtg: number;      // estimated bench net rating
}

function computePlayerImpact(players: NBAPlayer[], injured: { playerName: string; status: string }[]): PlayerImpactResult {
  const injuredNames = new Set(injured.filter(p => p.status === 'Out' || p.status === 'Doubtful').map(p => p.playerName.toLowerCase()));

  // Active players
  const activePlayers = players.filter(p => !injuredNames.has(p.playerName.toLowerCase()));

  // Sort by BPM for star impact
  const sorted = [...activePlayers].sort((a, b) => b.bpm - a.bpm);
  const top3Bpm = sorted.slice(0, 3).reduce((sum, p) => sum + p.bpm, 0);

  // Starting lineup: top 5 by minutes
  const byMinutes = [...activePlayers].sort((a, b) => b.minutesPerGame - a.minutesPerGame);
  const starters = byMinutes.slice(0, 5);
  const bench = byMinutes.slice(5, 10);

  const lineupNetRtg = starters.length > 0
    ? starters.reduce((sum, p) => sum + (p.onNetRtg ?? p.bpm), 0) / starters.length
    : 0;

  const benchNetRtg = bench.length > 0
    ? bench.reduce((sum, p) => sum + (p.onNetRtg ?? p.bpm), 0) / bench.length
    : 0;

  // Injury impact: sum WAR (VORP proxy) of Out/Doubtful players
  const injuredPlayers = players.filter(p => injuredNames.has(p.playerName.toLowerCase()));
  const injuryImpact = -injuredPlayers.reduce((sum, p) => sum + Math.max(0, p.vorp), 0);

  return {
    starImpact: top3Bpm,
    injuryImpact,
    lineupNetRtg,
    benchNetRtg,
  };
}

// â”€â”€â”€ Default team stats fallback â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function defaultTeam(abbr: string): NBATeam {
  return {
    teamId: 0,
    teamAbbr: abbr,
    teamName: abbr,
    w: 0, l: 0, winPct: 0.5,
    offRtg: LEAGUE_AVG_OFF_RTG,
    defRtg: LEAGUE_AVG_OFF_RTG,
    netRtg: LEAGUE_AVG_NET_RTG,
    pace: 100,
    efgPct: 0.530,
    tovPct: 13.0,
    orbPct: 0.230,
    ftRate: 0.200,
    threePtPct: 0.360,
    threePtRate: 0.400,
    tsPct: 0.570,
    astPct: 0.600,
    stlPct: 0.090,
    blkPct: 0.080,
    pythagoreanWinPct: 0.5,
    clutchNetRtg: 0,
  };
}
