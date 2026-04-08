// NBA Oracle v4.0 — Daily Pipeline
// Orchestrates: Fetch → Features → Monte Carlo → ML Model → Edge → Store → Print

import { logger } from './logger.js';
import { fetchSchedule } from './api/nbaClient.js';
import { computeFeatures } from './features/featureEngine.js';
import { runMonteCarlo } from './models/monteCarlo.js';
import { upsertPrediction, initDb } from './db/database.js';
import { loadModel, predict as mlPredict, isModelLoaded, getModelInfo } from './models/metaModel.js';
import { computeEdge, formatEdge } from './features/marketEdge.js';
import { initializeOdds, hasAnyOdds, getOddsForGame, loadOddsApiLines } from './api/oddsClient.js';
import { seedElos } from './features/eloEngine.js';
import type { NBAGame, Prediction, PipelineOptions } from './types.js';

const MODEL_VERSION = '4.0.0';

// ─── Main pipeline ────────────────────────────────────────────────────────────

export async function runPipeline(options: PipelineOptions = {}): Promise<Prediction[]> {
  const today = new Date().toISOString().split('T')[0];
  const gameDate = options.date ?? today;

  logger.info({ gameDate, version: MODEL_VERSION }, '=== NBA Oracle v4.0 Pipeline Start ===');

  // 1. Initialize database
  await initDb();

  // 2. Seed Elo ratings (idempotent — only seeds teams that don't exist)
  seedElos();

  // 3. Attempt to load ML meta-model
  const modelLoaded = loadModel();
  if (modelLoaded) {
    const info = getModelInfo();
    logger.info(
      { version: info?.version, avgBrier: info?.avgBrier, trainDates: info?.trainDates },
      'Using ML meta-model for calibrated predictions'
    );
  } else {
    logger.info('ML model not found — using Monte Carlo win probability as calibrated_prob');
    logger.info('Run: npm run train  (python python/train_model.py) to train the model');
  }

  // 4. Initialize Vegas odds (manual lines or Odds API)
  await initializeOdds(gameDate);
  if (hasAnyOdds()) {
    logger.info('Vegas lines loaded — will compute market edge for each game');
  }

  // 5. Fetch today's schedule
  const games = await fetchSchedule(gameDate);
  if (games.length === 0) {
    logger.warn({ gameDate }, 'No games found for date');
    return [];
  }

  logger.info({ gameDate, games: games.length }, 'Schedule fetched');

  // 6. Process each game
  const predictions: Prediction[] = [];
  let processed = 0;
  let failed = 0;

  for (const game of games) {
    try {
      const pred = await processGame(game, gameDate, modelLoaded);
      if (pred) {
        predictions.push(pred);
        processed++;
      }
    } catch (err) {
      failed++;
      logger.error(
        { err, gameId: game.gameId, home: game.homeTeam.teamAbbr, away: game.awayTeam.teamAbbr },
        'Failed to process game'
      );
    }
  }

  logger.info({ processed, failed, total: games.length }, 'Pipeline complete');

  // 7. Print formatted predictions table
  if (options.verbose !== false) {
    printPredictions(predictions, gameDate, modelLoaded);
  }

  return predictions;
}

// ─── Single game processing ───────────────────────────────────────────────────

async function processGame(
  game: NBAGame,
  gameDate: string,
  modelLoaded: boolean
): Promise<Prediction | null> {
  const homeAbbr = game.homeTeam.teamAbbr;
  const awayAbbr = game.awayTeam.teamAbbr;

  logger.info({ gameId: game.gameId, matchup: `${awayAbbr} @ ${homeAbbr}` }, 'Processing game');

  // Skip non-scheduled games
  if (game.status.toLowerCase().includes('final') || game.status.toLowerCase().includes('in progress')) {
    logger.info({ status: game.status }, 'Skipping non-upcoming game');
    return null;
  }

  // ── Step A: Compute feature vector ─────────────────────────────────────────
  const features = await computeFeatures(game, gameDate);

  // ── Step B: Monte Carlo simulation ─────────────────────────────────────────
  const mc = runMonteCarlo(features);

  // ── Step C: ML calibration ──────────────────────────────────────────────────
  // The ML model's mc_win_pct feature must match the training computation exactly:
  //   mc_prob = 0.5 * elo_win_prob + 0.5 * sigmoid(net_rtg_diff / 8)
  // The full MC simulation is used for expected scores/spread display, NOT as the ML input.
  const eloWinProb = 1 / (1 + Math.pow(10, -features.elo_diff / 400));
  const logisticNetRtg = 1 / (1 + Math.exp(-features.net_rtg_diff / 8.0));
  const trainingMcProb = 0.5 * eloWinProb + 0.5 * logisticNetRtg;

  let calibrated_prob: number;

  if (modelLoaded && isModelLoaded()) {
    calibrated_prob = mlPredict(features, trainingMcProb);
    logger.debug(
      {
        gameId: game.gameId,
        mc_prob: mc.win_probability.toFixed(3),
        ml_prob: calibrated_prob.toFixed(3),
        delta: (calibrated_prob - mc.win_probability).toFixed(3),
      },
      'ML model applied'
    );
  } else {
    calibrated_prob = mc.win_probability;
  }

  // ── Step D: Market edge computation ────────────────────────────────────────
  let vegas_prob: number | undefined;
  let edge: number | undefined;

  const matchupKey = `${awayAbbr}@${homeAbbr}`;
  const gameOdds = getOddsForGame(matchupKey);

  if (gameOdds) {
    vegas_prob = gameOdds.homeImpliedProb;
    edge = calibrated_prob - vegas_prob;

    const edgeResult = computeEdge(calibrated_prob, gameOdds.homeML, gameOdds.awayML);
    logger.info(
      { gameId: game.gameId, matchup: matchupKey },
      formatEdge(edgeResult)
    );
  }

  // Inject Vegas prob into features for ML use
  if (vegas_prob !== undefined) {
    features.vegas_home_prob = vegas_prob;
  }

  // ── Step E: Build prediction record ────────────────────────────────────────
  const prediction: Prediction = {
    game_date: gameDate,
    game_id: game.gameId,
    home_team: homeAbbr,
    away_team: awayAbbr,
    arena: game.arena,
    feature_vector: features,
    mc_win_pct: mc.win_probability,
    calibrated_prob,
    vegas_prob,
    edge,
    model_version: MODEL_VERSION,
    home_exp_pts: mc.home_exp_pts,
    away_exp_pts: mc.away_exp_pts,
    total_points: mc.total_points,
    spread: mc.spread,
    most_likely_score: `${mc.most_likely_score[0]}-${mc.most_likely_score[1]}`,
    upset_probability: mc.upset_probability,
    blowout_probability: mc.blowout_probability,
    created_at: new Date().toISOString(),
  };

  // ── Step F: Store in DB ────────────────────────────────────────────────────
  upsertPrediction(prediction);

  return prediction;
}

// ─── Console output ───────────────────────────────────────────────────────────

function printPredictions(
  predictions: Prediction[],
  gameDate: string,
  mlModelActive = false,
): void {
  if (predictions.length === 0) {
    console.log(`\nNo predictions for ${gameDate}\n`);
    return;
  }

  const modelLabel = mlModelActive ? 'ML+Isotonic' : 'Monte Carlo';
  const hasEdge = predictions.some(p => p.edge !== undefined);
  const totalWidth = hasEdge ? 115 : 100;

  console.log('\n' + '═'.repeat(totalWidth));
  console.log(
    `  NBA ORACLE v4.0  ·  Predictions for ${gameDate}  ·  ${predictions.length} games  ·  [${modelLabel}]`
  );
  console.log('═'.repeat(totalWidth));

  const headerCols = [
    pad('MATCHUP', 22),
    pad('ARENA', 22),
    pad('CAL WIN%', 10),
    pad('MC WIN%', 9),
    pad('HOME EXP', 10),
    pad('AWAY EXP', 10),
    pad('TOTAL', 7),
    pad('PROJ SCORE', 11),
  ];
  if (hasEdge) headerCols.push(pad('EDGE', 9));

  console.log('\n' + headerCols.join('  '));
  console.log('─'.repeat(totalWidth));

  // Sort by confidence (most confident first)
  const sorted = [...predictions].sort((a, b) =>
    Math.abs(b.calibrated_prob - 0.5) - Math.abs(a.calibrated_prob - 0.5)
  );

  for (const p of sorted) {
    const calPct = (p.calibrated_prob * 100).toFixed(1) + '%';
    const mcPct = (p.mc_win_pct * 100).toFixed(1) + '%';
    const matchup = `${p.away_team} @ ${p.home_team}`;

    const confidence = Math.abs(p.calibrated_prob - 0.5);
    const marker = confidence >= 0.17 ? ' ★' : confidence >= 0.10 ? ' ·' : '  ';

    const rowCols = [
      pad(matchup, 22),
      pad(p.arena.slice(0, 21), 22),
      pad(`${calPct} (H)`, 10),
      pad(mcPct, 9),
      pad(p.home_exp_pts.toFixed(1), 10),
      pad(p.away_exp_pts.toFixed(1), 10),
      pad(p.total_points.toFixed(1), 7),
      pad(p.most_likely_score, 11),
    ];

    if (hasEdge) {
      if (p.edge !== undefined) {
        const sign = p.edge >= 0 ? '+' : '';
        rowCols.push(pad(`${sign}${(p.edge * 100).toFixed(1)}%`, 9));
      } else {
        rowCols.push(pad('—', 9));
      }
    }

    console.log(rowCols.join('  ') + marker);
  }

  console.log('─'.repeat(totalWidth));
  console.log('\nLegend: ★ = high conviction (≥67% cal prob)  · = strong (≥60%)');
  if (hasEdge) console.log('EDGE: model vs vig-removed Vegas probability (+ = model favors home)');

  // Summary
  const avgCalWin = predictions.reduce((s, p) => s + p.calibrated_prob, 0) / predictions.length;
  const avgTotal = predictions.reduce((s, p) => s + p.total_points, 0) / predictions.length;
  const highConv = predictions.filter(p => Math.abs(p.calibrated_prob - 0.5) >= 0.17).length;
  const edgePicks = predictions.filter(p => p.edge !== undefined && Math.abs(p.edge) >= 0.06);

  let summary = (
    `\nSummary: avg cal win% = ${(avgCalWin * 100).toFixed(1)}%` +
    `  |  avg total pts = ${avgTotal.toFixed(1)}` +
    `  |  ${highConv} high-conviction picks`
  );
  if (edgePicks.length > 0) summary += `  |  ${edgePicks.length} edge picks (≥6%)`;
  console.log(summary);
  console.log('═'.repeat(totalWidth) + '\n');

  // Edge picks table
  if (edgePicks.length > 0) {
    console.log('─'.repeat(72));
    console.log('  EDGE PICKS (model vs Vegas disagreement ≥ 6%)');
    console.log('─'.repeat(72));
    const edgeSorted = [...edgePicks].sort((a, b) => Math.abs(b.edge!) - Math.abs(a.edge!));
    for (const p of edgeSorted) {
      if (p.edge === undefined || p.vegas_prob === undefined) continue;
      const matchup = `${p.away_team} @ ${p.home_team}`;
      const side = p.edge >= 0 ? p.home_team : p.away_team;
      const sign = p.edge >= 0 ? '+' : '';
      const absEdge = Math.abs(p.edge);
      const tier = absEdge >= 0.15 ? 'EXTREME' : absEdge >= 0.10 ? 'LARGE' : 'MEANINGFUL';
      console.log(
        `  ${pad(matchup, 22)}  Model: ${(p.calibrated_prob * 100).toFixed(1)}%` +
        `  Vegas: ${(p.vegas_prob * 100).toFixed(1)}%` +
        `  Edge: ${sign}${(p.edge * 100).toFixed(1)}%  Lean: ${side}  [${tier}]`
      );
    }
    console.log('─'.repeat(72) + '\n');
  }
}

function pad(str: string, width: number): string {
  if (str.length >= width) return str.slice(0, width);
  return str + ' '.repeat(width - str.length);
}

export { processGame };
