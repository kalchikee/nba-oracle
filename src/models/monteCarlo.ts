// NBA Oracle v4.0 — Monte Carlo Simulation Engine
// 10,000 Normal distribution simulations (NBA scores ≈ Normal, unlike MLB's Poisson)
// Expected points estimation with efficiency, pace, rest, home, and lineup adjustments

import type { FeatureVector, MonteCarloResult, ExpectedScoreEstimate } from '../types.js';

const N_SIMULATIONS = 10_000;
const LEAGUE_AVG_OFF_RTG = 113.0;
const LEAGUE_AVG_DEF_RTG = 113.0;
const LEAGUE_AVG_PACE = 99.5;
const LEAGUE_AVG_SCORE_STD = 12.5;  // NBA game score standard deviation

// ─── Normal random (Box-Muller transform) ────────────────────────────────────

function normalRandom(mean: number, std: number): number {
  const u1 = Math.random();
  const u2 = Math.random();
  const z = Math.sqrt(-2 * Math.log(Math.max(1e-10, u1))) * Math.cos(2 * Math.PI * u2);
  return mean + z * std;
}

// ─── Expected points estimation ───────────────────────────────────────────────
//
// exp_pts = team_off_rtg × (opp_def_rtg / league_avg_def_rtg) × expected_pace / 100
//           × rest_adj × home_adj × lineup_adj × b2b_adj × altitude_adj

export function estimateExpectedPoints(
  features: FeatureVector,
  isHome: boolean
): ExpectedScoreEstimate {
  // Decode team-level ORtg and DRtg from diff features
  // feature.off_rtg_diff = home_off - away_off, so:
  // home_off = league_avg + off_rtg_diff/2
  // away_off = league_avg - off_rtg_diff/2
  const homeOffRtg = LEAGUE_AVG_OFF_RTG + features.off_rtg_diff / 2;
  const awayOffRtg = LEAGUE_AVG_OFF_RTG - features.off_rtg_diff / 2;

  // def_rtg_diff = home_def - away_def; lower DRtg is better
  const homeDefRtg = LEAGUE_AVG_DEF_RTG + features.def_rtg_diff / 2;
  const awayDefRtg = LEAGUE_AVG_DEF_RTG - features.def_rtg_diff / 2;

  // Pace: expected possessions ≈ average of both team paces
  const homePace = LEAGUE_AVG_PACE + features.pace_diff / 2;
  const awayPace = LEAGUE_AVG_PACE - features.pace_diff / 2;
  const expectedPace = (homePace + awayPace) / 2;

  // Expected raw scores (off_rtg × opp_def/league_avg × pace/100)
  const homeRawPts = homeOffRtg * (awayDefRtg / LEAGUE_AVG_DEF_RTG) * (expectedPace / 100);
  const awayRawPts = awayOffRtg * (homeDefRtg / LEAGUE_AVG_DEF_RTG) * (expectedPace / 100);

  // Rest adjustment: B2B penalty
  const homeB2bPenalty = features.b2b_home === 1 ? -3.0 : 0;
  const awayB2bPenalty = features.b2b_away === 1 ? -3.0 : 0;

  // Rest days diff: each day of extra rest ≈ +0.3 pts
  const restBonus = Math.max(-3, Math.min(3, features.rest_days_diff * 0.3));

  // Home court advantage: ~3.0 points in NBA
  const homeAdv = 3.0;

  // Altitude: Denver away team penalty
  const altitudePenalty = features.altitude_factor > 0 ? features.altitude_factor : 0;

  // Lineup adjustment: star player impact
  // Each BPM point ≈ 0.5 pts per game influence
  const lineupAdj = features.lineup_net_rtg_diff * 0.15;
  const injuryAdj = features.injury_impact_diff * 0.20;

  // Final expected points
  const homeExpPts = Math.max(80, homeRawPts + homeAdv + restBonus + homeB2bPenalty + lineupAdj / 2 + injuryAdj / 2 - altitudePenalty / 2);
  const awayExpPts = Math.max(80, awayRawPts - awayB2bPenalty - altitudePenalty + lineupAdj / 2 - injuryAdj / 2);

  // Score standard deviation — teams with higher pace have slightly higher variance
  const paceScaleFactor = expectedPace / LEAGUE_AVG_PACE;
  const homeStd = LEAGUE_AVG_SCORE_STD * paceScaleFactor;
  const awayStd = LEAGUE_AVG_SCORE_STD * paceScaleFactor;

  return { homeExpPts, awayExpPts, homeStd, awayStd, expectedPace };
}

// ─── Monte Carlo simulation ───────────────────────────────────────────────────

export function runMonteCarlo(features: FeatureVector): MonteCarloResult {
  const { homeExpPts, awayExpPts, homeStd, awayStd } = estimateExpectedPoints(features, true);

  let homeWins = 0;
  let totalHomeScore = 0;
  let totalAwayScore = 0;
  let blowouts = 0;
  const scores: Array<[number, number]> = [];
  const margins: number[] = [];

  for (let i = 0; i < N_SIMULATIONS; i++) {
    let homeScore = Math.round(normalRandom(homeExpPts, homeStd));
    let awayScore = Math.round(normalRandom(awayExpPts, awayStd));

    // Prevent negative scores
    homeScore = Math.max(60, homeScore);
    awayScore = Math.max(60, awayScore);

    // Handle ties with overtime: add ~10 pts per team (extra 5 min)
    let finalHome = homeScore;
    let finalAway = awayScore;
    if (homeScore === awayScore) {
      const otHome = Math.round(normalRandom(10, 4));
      const otAway = Math.round(normalRandom(10, 4));
      finalHome += otHome;
      finalAway += otAway;
      // If still tied, add another OT
      if (finalHome === finalAway) {
        finalHome += Math.round(Math.random() * 4 + 2);
      }
    }

    if (finalHome > finalAway) homeWins++;

    totalHomeScore += finalHome;
    totalAwayScore += finalAway;

    const margin = Math.abs(finalHome - finalAway);
    if (margin >= 15) blowouts++;
    margins.push(finalHome - finalAway);
    scores.push([finalHome, finalAway]);
  }

  const winProbability = homeWins / N_SIMULATIONS;
  const avgHomeScore = totalHomeScore / N_SIMULATIONS;
  const avgAwayScore = totalAwayScore / N_SIMULATIONS;
  const expectedSpread = avgHomeScore - avgAwayScore;

  // Most likely score: round expected values
  const mostLikelyScore: [number, number] = [Math.round(homeExpPts), Math.round(awayExpPts)];

  // Upset probability: team with lower Elo wins
  // If home has positive elo_diff, home is favored, so upset = away wins
  const isHomeFavored = features.elo_diff > 0;
  const upsetProb = isHomeFavored ? 1 - winProbability : winProbability;

  return {
    win_probability: winProbability,
    away_win_probability: 1 - winProbability,
    spread: expectedSpread,
    total_points: avgHomeScore + avgAwayScore,
    most_likely_score: mostLikelyScore,
    upset_probability: upsetProb,
    blowout_probability: blowouts / N_SIMULATIONS,
    home_exp_pts: homeExpPts,
    away_exp_pts: awayExpPts,
    simulations: N_SIMULATIONS,
  };
}
