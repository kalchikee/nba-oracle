// NBA Oracle v4.0 — Results Processor
// Fetches completed game results, updates Elo ratings, and computes accuracy metrics.

import { logger } from '../logger.js';
import {
  getPredictionsByDate, updatePredictionResult,
  upsertGameResult, upsertAccuracyLog, upsertCalibration,
} from '../db/database.js';
import { updateEloAfterGame } from '../features/eloEngine.js';
import { fetchCompletedResults } from '../api/nbaClient.js';
import type { Prediction, AccuracyLog } from '../types.js';

interface GameWithResult {
  prediction: Prediction;
  homeScore: number;
  awayScore: number;
}

interface DayMetrics {
  accuracy: number;
  brier: number;
  logLoss: number;
  highConvAccuracy: number | null;
  vsVegasBrier: number;
}

// ─── Process results for a date ───────────────────────────────────────────────

export async function processResults(date: string): Promise<{
  games: GameWithResult[];
  metrics: DayMetrics;
}> {
  logger.info({ date }, 'Processing results');

  // 1. Fetch completed game results from NBA API
  const completedGames = await fetchCompletedResults(date);

  // 2. Load predictions from DB for this date
  const predictions = getPredictionsByDate(date);

  if (predictions.length === 0) {
    logger.warn({ date }, 'No predictions found for this date');
    return { games: [], metrics: emptyMetrics() };
  }

  // 3. Match results to predictions
  const games: GameWithResult[] = [];

  for (const result of completedGames) {
    // Find matching prediction
    const pred = predictions.find(
      p => p.home_team === result.home_team && p.away_team === result.away_team
    );

    if (!pred) {
      logger.debug({ home: result.home_team, away: result.away_team }, 'No prediction found for result');
      continue;
    }

    const winner = result.home_score > result.away_score ? result.home_team : result.away_team;
    const correct = (pred.calibrated_prob >= 0.5 && winner === pred.home_team) ||
                    (pred.calibrated_prob < 0.5 && winner === pred.away_team);

    // Update prediction in DB
    updatePredictionResult(pred.game_id, winner, correct);
    pred.actual_winner = winner;
    pred.correct = correct;

    // Store game result
    upsertGameResult({
      game_id: result.game_id,
      date: result.date,
      home_team: result.home_team,
      away_team: result.away_team,
      home_score: result.home_score,
      away_score: result.away_score,
      arena: result.arena,
      lineups: '{}',
    });

    // Update Elo ratings
    updateEloAfterGame(result.home_team, result.away_team, result.home_score, result.away_score);

    // Store calibration log entry
    if (pred.vegas_prob !== undefined) {
      upsertCalibration({
        date,
        game_id: pred.game_id,
        model_prob: pred.calibrated_prob,
        vegas_prob: pred.vegas_prob,
        edge: pred.edge ?? 0,
        outcome: result.home_score > result.away_score ? 1 : 0,
      });
    }

    games.push({ prediction: pred, homeScore: result.home_score, awayScore: result.away_score });
  }

  // 4. Compute accuracy metrics
  const metrics = computeMetrics(games);

  // 5. Store accuracy log
  if (games.length > 0) {
    const accuracyLog: AccuracyLog = {
      date,
      brier_score: metrics.brier,
      log_loss: metrics.logLoss,
      accuracy: metrics.accuracy,
      high_conv_accuracy: metrics.highConvAccuracy ?? 0,
      vs_vegas_brier: metrics.vsVegasBrier,
    };
    upsertAccuracyLog(accuracyLog);
  }

  logger.info(
    { date, games: games.length, accuracy: metrics.accuracy.toFixed(3), brier: metrics.brier.toFixed(4) },
    'Results processed'
  );

  return { games, metrics };
}

// ─── Metric computation ───────────────────────────────────────────────────────

function computeMetrics(games: GameWithResult[]): DayMetrics {
  if (games.length === 0) return emptyMetrics();

  let correct = 0;
  let brierSum = 0;
  let logLossSum = 0;
  let highConvCorrect = 0;
  let highConvTotal = 0;
  let vegasBrierSum = 0;
  let vegasCount = 0;

  for (const { prediction: pred, homeScore, awayScore } of games) {
    const outcome = homeScore > awayScore ? 1 : 0;
    const p = pred.calibrated_prob;

    if (pred.correct) correct++;

    brierSum += Math.pow(p - outcome, 2);
    logLossSum -= outcome * Math.log(Math.max(1e-10, p)) + (1 - outcome) * Math.log(Math.max(1e-10, 1 - p));

    // High-conviction: ≥67% confidence on either side
    const maxProb = Math.max(p, 1 - p);
    if (maxProb >= 0.67) {
      highConvTotal++;
      if (pred.correct) highConvCorrect++;
    }

    // Vegas comparison
    if (pred.vegas_prob !== undefined) {
      vegasBrierSum += Math.pow(pred.vegas_prob - outcome, 2);
      vegasCount++;
    }
  }

  const n = games.length;
  const modelBrier = brierSum / n;
  const vegasBrier = vegasCount > 0 ? vegasBrierSum / vegasCount : 0;

  return {
    accuracy: correct / n,
    brier: modelBrier,
    logLoss: logLossSum / n,
    highConvAccuracy: highConvTotal > 0 ? highConvCorrect / highConvTotal : null,
    vsVegasBrier: vegasCount > 0 ? modelBrier - vegasBrier : 0,
  };
}

function emptyMetrics(): DayMetrics {
  return { accuracy: 0, brier: 0, logLoss: 0, highConvAccuracy: null, vsVegasBrier: 0 };
}
