// NBA Oracle v4.0 — Core Type Definitions

// ─── NBA API types ────────────────────────────────────────────────────────────

export interface NBATeam {
  teamId: number;
  teamAbbr: string;
  teamName: string;
  // Base stats (per game)
  w: number;
  l: number;
  winPct: number;
  offRtg: number;      // offensive rating (points per 100 possessions)
  defRtg: number;      // defensive rating (points allowed per 100 possessions)
  netRtg: number;      // offRtg - defRtg
  pace: number;        // possessions per 48 min
  efgPct: number;      // effective FG%
  tovPct: number;      // turnover percentage
  orbPct: number;      // offensive rebound rate
  ftRate: number;      // free throw rate (FTA/FGA)
  threePtPct: number;  // 3-point %
  threePtRate: number; // 3PA/FGA
  tsPct: number;       // true shooting %
  astPct: number;      // assist percentage
  stlPct: number;      // steal percentage
  blkPct: number;      // block percentage
  // Rolling form (last 10 days)
  rolling10dNetRtg?: number;
  rolling10dOffRtg?: number;
  // Home/Away splits
  homeNetRtg?: number;
  awayNetRtg?: number;
  // Clutch
  clutchNetRtg?: number;
  // Pythagorean
  pythagoreanWinPct?: number;
}

export interface NBAPlayer {
  playerId: number;
  playerName: string;
  teamId: number;
  teamAbbr: string;
  position: string;
  minutesPerGame: number;
  usageRate: number;
  bpm: number;        // Box Plus/Minus
  offBpm: number;     // Offensive BPM
  defBpm: number;     // Defensive BPM
  vorp: number;       // Value Over Replacement Player
  // On/Off net rating
  onNetRtg?: number;
  offNetRtg?: number;
  // Injury status
  injured?: boolean;
  injuryStatus?: string; // 'Out' | 'Doubtful' | 'Questionable' | 'Probable'
}

export interface NBAGame {
  gameId: string;
  gameDate: string;    // YYYY-MM-DD
  gameTime: string;    // ISO datetime
  status: string;      // 'Scheduled' | 'Live' | 'Final'
  homeTeam: NBAGameTeam;
  awayTeam: NBAGameTeam;
  arena: string;
  arenaCity: string;
}

export interface NBAGameTeam {
  teamId: number;
  teamAbbr: string;
  teamName: string;
  score?: number;
  lineup?: NBAPlayer[];
}

export interface InjuryReport {
  teamId: number;
  teamAbbr: string;
  players: InjuredPlayer[];
}

export interface InjuredPlayer {
  playerId: number;
  playerName: string;
  status: string;    // 'Out' | 'Doubtful' | 'Questionable' | 'GTD'
  description: string;
}

// ─── Feature vector ───────────────────────────────────────────────────────────

export interface FeatureVector {
  // Team strength
  elo_diff: number;              // home Elo - away Elo
  net_rtg_diff: number;          // home net rating - away net rating
  off_rtg_diff: number;          // home ORtg - away ORtg
  def_rtg_diff: number;          // home DRtg - away DRtg (negative = home better defense)
  pace_diff: number;             // home pace - away pace
  pythagorean_diff: number;      // home pythagorean win% - away
  log5_prob: number;             // Log5 head-to-head win probability (home)

  // Shooting efficiency differentials (home - away)
  efg_pct_diff: number;
  tov_pct_diff: number;
  oreb_pct_diff: number;
  ft_rate_diff: number;
  three_pt_rate_diff: number;
  three_pt_pct_diff: number;
  ts_pct_diff: number;
  ast_pct_diff: number;
  stl_pct_diff: number;
  blk_pct_diff: number;

  // Form (trailing)
  team_10d_net_rtg_diff: number;
  team_10d_off_rtg_diff: number;
  momentum_diff: number;          // (last-10 win% - season win%) differential

  // Fatigue & travel
  rest_days_diff: number;         // home rest days - away rest days
  b2b_home: number;               // 1 if home is on B2B
  b2b_away: number;               // 1 if away is on B2B
  travel_tz_shift_home: number;   // timezone shifts for home team
  travel_tz_shift_away: number;   // timezone shifts for away team

  // Venue
  is_home: number;                // always 1 (feature vector always from home perspective)
  altitude_factor: number;        // extra penalty for high altitude (Denver)

  // Player impact
  star_player_impact_diff: number; // sum of top-3 BPM (home - away)
  injury_impact_diff: number;      // WAR lost to injuries (home - away, negative = home hurt more)
  lineup_net_rtg_diff: number;     // projected starting 5 net rating diff
  bench_net_rtg_diff: number;      // bench unit net rating diff

  // Clutch
  clutch_net_rtg_diff: number;    // clutch net rating diff

  // Matchup
  h2h_season_record: number;      // home wins in season series (normalized: 0.5 = even)

  // Vegas (filled at prediction time if available)
  vegas_home_prob: number;        // vig-removed implied probability (0 if unavailable)
}

// ─── Model outputs ────────────────────────────────────────────────────────────

export interface ExpectedScoreEstimate {
  homeExpPts: number;
  awayExpPts: number;
  homeStd: number;
  awayStd: number;
  expectedPace: number;
}

export interface MonteCarloResult {
  win_probability: number;       // home win probability
  away_win_probability: number;
  spread: number;                // home team expected spread (positive = home favored)
  total_points: number;          // expected total points
  most_likely_score: [number, number]; // [home, away]
  upset_probability: number;    // lower-elo team wins
  blowout_probability: number;  // margin >= 15
  home_exp_pts: number;
  away_exp_pts: number;
  simulations: number;
}

export interface Prediction {
  game_date: string;
  game_id: string;
  home_team: string;
  away_team: string;
  arena: string;
  feature_vector: FeatureVector;
  mc_win_pct: number;            // raw Monte Carlo
  calibrated_prob: number;       // after ML calibration (falls back to MC)
  vegas_prob?: number;           // implied from odds
  edge?: number;                 // calibrated_prob - vegas_prob
  model_version: string;
  home_exp_pts: number;
  away_exp_pts: number;
  total_points: number;
  spread: number;
  most_likely_score: string;
  upset_probability: number;
  blowout_probability: number;
  actual_winner?: string;
  correct?: boolean;
  created_at: string;
}

export interface EloRating {
  teamAbbr: string;
  rating: number;
  updatedAt: string;
}

export interface AccuracyLog {
  date: string;
  brier_score: number;
  log_loss: number;
  accuracy: number;
  high_conv_accuracy: number;
  vs_vegas_brier: number;
}

export interface CalibrationLog {
  date: string;
  game_id: string;
  model_prob: number;
  vegas_prob: number;
  edge: number;
  outcome: number; // 1 = home win, 0 = away win
}

export interface GameResult {
  game_id: string;
  date: string;
  home_team: string;
  away_team: string;
  home_score: number;
  away_score: number;
  arena: string;
  lineups: string; // JSON
}

export interface PipelineOptions {
  date?: string;
  forceRefresh?: boolean;
  verbose?: boolean;
}

// ─── Edge detection ───────────────────────────────────────────────────────────

export type EdgeCategory =
  | 'none'        // |edge| < 3%
  | 'small'       // 3–6%
  | 'meaningful'  // 6–10%
  | 'large'       // 10–15%
  | 'extreme';    // ≥15%

export interface EdgeResult {
  modelProb: number;
  vegasProb: number;
  rawHomeImplied: number;
  rawAwayImplied: number;
  vigPct: number;
  edge: number;
  edgeCategory: EdgeCategory;
  homeFavorite: boolean;
}
