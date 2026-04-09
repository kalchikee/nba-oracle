// NBA Oracle v4.0 — Discord Webhook Alert Module
// Message 1 (morning): all picks + which games to bet on
// Message 2 (1 AM):    results recap — how many correct

import fetch from 'node-fetch';
import { logger } from '../logger.js';
import { getPredictionsByDate, initDb, getRecentAccuracy, getSeasonRecord } from '../db/database.js';
import { getConfidenceTier } from '../features/marketEdge.js';
import type { Prediction } from '../types.js';

// ─── Colors ───────────────────────────────────────────────────────────────────

const COLORS = {
  morning: 0x1a6ef5,         // bright blue
  bet: 0x27ae60,             // green — bet recommendations section
  recap_good: 0x2ecc71,      // green — good night
  recap_bad: 0xe74c3c,       // red — bad night
  recap_neutral: 0x95a5a6,   // gray — neutral
} as const;

// ─── Discord types ────────────────────────────────────────────────────────────

interface DiscordField {
  name: string;
  value: string;
  inline?: boolean;
}

interface DiscordEmbed {
  title?: string;
  description?: string;
  color?: number;
  fields?: DiscordField[];
  footer?: { text: string };
  timestamp?: string;
}

interface DiscordPayload {
  content?: string;
  embeds: DiscordEmbed[];
}

// ─── Webhook sender ───────────────────────────────────────────────────────────

async function sendWebhook(payload: DiscordPayload): Promise<boolean> {
  const webhookUrl = process.env.DISCORD_WEBHOOK_URL;
  if (!webhookUrl) {
    logger.warn('DISCORD_WEBHOOK_URL not set — skipping Discord alert');
    return false;
  }

  try {
    const resp = await fetch(webhookUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(10000),
    });

    if (!resp.ok) {
      const text = await resp.text();
      logger.error({ status: resp.status, body: text }, 'Discord webhook error');
      return false;
    }

    logger.info('Discord alert sent');
    return true;
  } catch (err) {
    logger.error({ err }, 'Failed to send Discord webhook');
    return false;
  }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function pct(prob: number): string {
  return (prob * 100).toFixed(1) + '%';
}

function getWinner(pred: Prediction): { team: string; winPct: number } {
  if (pred.calibrated_prob >= 0.5) return { team: pred.home_team, winPct: pred.calibrated_prob };
  return { team: pred.away_team, winPct: 1 - pred.calibrated_prob };
}

function confidenceBar(prob: number): string {
  // prob is the higher of the two sides
  const p = Math.max(prob, 1 - prob);
  if (p >= 0.72) return '🔥🔥🔥';
  if (p >= 0.67) return '🔥🔥';
  if (p >= 0.60) return '🔥';
  if (p >= 0.55) return '✅';
  return '🪙';
}

function shouldBet(pred: Prediction): boolean {
  const p = Math.max(pred.calibrated_prob, 1 - pred.calibrated_prob);
  return p >= 0.67;
}

function betReason(pred: Prediction): string {
  const p = Math.max(pred.calibrated_prob, 1 - pred.calibrated_prob);
  const tier = getConfidenceTier(pred.calibrated_prob);

  const reasons: string[] = [];
  if (tier === 'extreme') reasons.push('model has extreme conviction (72%+)');
  else if (tier === 'high_conviction') reasons.push('model has high conviction (67–72%)');

  if (pred.edge !== undefined && Math.abs(pred.edge) >= 0.06) {
    const sign = pred.edge >= 0 ? '+' : '';
    reasons.push(`${sign}${pct(pred.edge)} edge vs Vegas`);
  }

  const fv = pred.feature_vector as unknown as Record<string, number>;
  if ((fv['b2b_away'] ?? 0) === 1 && pred.calibrated_prob >= 0.5) {
    reasons.push('away team on back-to-back');
  }
  if ((fv['rest_days_diff'] ?? 0) >= 2 && pred.calibrated_prob >= 0.5) {
    reasons.push('home team has rest advantage');
  }
  if ((fv['injury_impact_diff'] ?? 0) < -1.0 && pred.calibrated_prob < 0.5) {
    reasons.push('home team missing key players');
  }

  return reasons.length > 0 ? reasons.join(' · ') : `${pct(p)} win probability`;
}

// ─── MESSAGE 1: Morning Picks + Bet Recommendations ──────────────────────────

export async function sendMorningBriefing(date: string): Promise<boolean> {
  await initDb();
  const predictions = getPredictionsByDate(date);

  if (predictions.length === 0) {
    logger.warn({ date }, 'No predictions for morning briefing');
    return false;
  }

  // Sort by confidence descending
  const sorted = [...predictions].sort((a, b) =>
    Math.abs(b.calibrated_prob - 0.5) - Math.abs(a.calibrated_prob - 0.5)
  );

  const betPicks = sorted.filter(p => shouldBet(p));
  const recentAccuracy = getRecentAccuracy(7);
  const avgAcc = recentAccuracy.length > 0
    ? recentAccuracy.reduce((s, a) => s + a.accuracy, 0) / recentAccuracy.length
    : null;
  const season = getSeasonRecord();

  // ── Embed 1: All Picks ────────────────────────────────────────────────────
  const picksFields: DiscordField[] = [];

  for (const pred of sorted) {
    const { team, winPct } = getWinner(pred);
    const matchup = `${pred.away_team} @ ${pred.home_team}`;
    const conf = confidenceBar(pred.calibrated_prob);
    const spread = pred.spread >= 0
      ? `${pred.home_team} -${pred.spread.toFixed(1)}`
      : `${pred.away_team} -${Math.abs(pred.spread).toFixed(1)}`;

    picksFields.push({
      name: `${conf} ${matchup}`,
      value: [
        `**Pick:** ${team}  |  **Win%:** ${pct(winPct)}`,
        `**Spread:** ${spread}  |  **Total:** ${pred.total_points.toFixed(0)} pts`,
        `**Proj:** ${pred.most_likely_score}`,
      ].join('\n'),
      inline: false,
    });
  }

  const seasonLine = season.total > 0
    ? `📈 Season: **${season.correct}-${season.total - season.correct}** (${((season.correct / season.total) * 100).toFixed(1)}%)` +
      (season.betTotal > 0 ? `  ·  💰 Bets: **${season.betCorrect}-${season.betTotal - season.betCorrect}** (${((season.betCorrect / season.betTotal) * 100).toFixed(1)}%)` : '')
    : '📈 Season: **0-0** (tracking starts tonight)';

  const picksEmbed: DiscordEmbed = {
    title: `🏀 NBA Oracle — Picks for ${date}`,
    description: [
      `${predictions.length} games today`,
      avgAcc !== null ? `7-day accuracy: **${(avgAcc * 100).toFixed(1)}%**` : '',
      seasonLine,
    ].filter(Boolean).join('  ·  '),
    color: COLORS.morning,
    fields: picksFields.slice(0, 20),
    footer: { text: '🔥🔥🔥 = Extreme  🔥🔥 = High Conviction  🔥 = Strong  ✅ = Lean  🪙 = Coin Flip' },
    timestamp: new Date().toISOString(),
  };

  // ── Embed 2: Bet Recommendations ─────────────────────────────────────────
  let betEmbed: DiscordEmbed;

  if (betPicks.length === 0) {
    betEmbed = {
      title: '💰 Bet Recommendations',
      description: '**No bets today.** No games clear the 67% confidence threshold. Skip today.',
      color: COLORS.recap_neutral,
    };
  } else {
    const betFields: DiscordField[] = betPicks.map(pred => {
      const { team, winPct } = getWinner(pred);
      const matchup = `${pred.away_team} @ ${pred.home_team}`;
      const tier = getConfidenceTier(pred.calibrated_prob);
      const tierLabel = tier === 'extreme' ? '🔥 EXTREME' : '⭐ HIGH CONVICTION';

      return {
        name: `${tierLabel}: ${matchup}`,
        value: [
          `**BET:** ${team} moneyline`,
          `**Confidence:** ${pct(winPct)}`,
          `**Why:** ${betReason(pred)}`,
        ].join('\n'),
        inline: false,
      };
    });

    betEmbed = {
      title: `💰 Bet Recommendations — ${betPicks.length} bet${betPicks.length !== 1 ? 's' : ''} today`,
      description: 'Only games where the model has ≥67% confidence. Historical accuracy on these: **73–76%**.',
      color: COLORS.bet,
      fields: betFields,
      footer: { text: 'Bet responsibly. Past performance does not guarantee future results.' },
    };
  }

  return sendWebhook({ embeds: [picksEmbed, betEmbed] });
}

// ─── MESSAGE 2: 1 AM Results Recap ───────────────────────────────────────────

export async function sendEveningRecap(
  date: string,
  games: Array<{ prediction: Prediction; homeScore: number; awayScore: number }>,
  metrics: { accuracy: number; brier: number; highConvAccuracy: number | null }
): Promise<boolean> {
  const season = getSeasonRecord();
  const gradedGames = games.filter(g => g.prediction.correct !== undefined);
  const correct = gradedGames.filter(g => g.prediction.correct).length;
  const total = gradedGames.length;

  if (games.length === 0) {
    return sendWebhook({
      embeds: [{
        title: `🌙 NBA Oracle — Recap for ${date}`,
        description: 'No completed games found yet. Results may still be in progress.',
        color: COLORS.recap_neutral,
        timestamp: new Date().toISOString(),
      }],
    });
  }

  const accPct = total > 0 ? (correct / total) * 100 : 0;
  const recapColor = total === 0 ? COLORS.recap_neutral : accPct >= 65 ? COLORS.recap_good : accPct >= 50 ? COLORS.recap_neutral : COLORS.recap_bad;
  const accEmoji = total === 0 ? '⚪' : accPct >= 65 ? '🟢' : accPct >= 50 ? '🟡' : '🔴';

  // Bet picks performance
  const betGames = gradedGames.filter(g => shouldBet(g.prediction));
  const betCorrect = betGames.filter(g => g.prediction.correct).length;

  // Individual game lines
  const gameLines = games.map(({ prediction: pred, homeScore, awayScore }) => {
    const noPrediction = pred.correct === undefined && pred.calibrated_prob === 0.5 && pred.mc_win_pct === 0.5;
    if (noPrediction) {
      // Results-only — system wasn't running that day
      const winner = homeScore > awayScore ? pred.home_team : pred.away_team;
      return `⚪ **${pred.away_team} @ ${pred.home_team}**: ${pred.away_team} ${awayScore}–${homeScore} ${pred.home_team} *(no pick — ${winner} won)*`;
    }
    const { team: pickedTeam } = getWinner(pred);
    const isCorrect = pred.correct ? '✅' : '❌';
    const wasBet = shouldBet(pred) ? ' 💰' : '';
    return `${isCorrect}${wasBet} **${pred.away_team} @ ${pred.home_team}**: ${pred.away_team} ${awayScore}–${homeScore} ${pred.home_team} *(picked ${pickedTeam})*`;
  }).join('\n');

  const seasonSummary = season.total > 0
    ? `📈 Season record: **${season.correct}-${season.total - season.correct}** (${((season.correct / season.total) * 100).toFixed(1)}%)` +
      (season.betTotal > 0 ? `\n💰 Season bets: **${season.betCorrect}-${season.betTotal - season.betCorrect}** (${((season.betCorrect / season.betTotal) * 100).toFixed(1)}%)` : '')
    : '📈 Season record: **0-0** (tracking starts tonight)';

  const summaryLines = [
    total > 0
      ? `**${accEmoji} Tonight: ${correct}/${total} correct (${accPct.toFixed(0)}%)**`
      : `**⚪ ${games.length} games played — no picks were made this day**`,
    betGames.length > 0
      ? `**💰 Tonight bets: ${betCorrect}/${betGames.length} correct (${((betCorrect / betGames.length) * 100).toFixed(0)}%)**`
      : '**💰 No bets placed today**',
    metrics.highConvAccuracy !== null
      ? `**High-conviction (67%+): ${(metrics.highConvAccuracy * 100).toFixed(0)}%**`
      : '',
    `Brier score: ${metrics.brier.toFixed(4)}`,
    seasonSummary,
  ].filter(Boolean).join('\n');

  const embed: DiscordEmbed = {
    title: `🌙 NBA Oracle — Results for ${date}`,
    color: recapColor,
    fields: [
      {
        name: '📊 Summary',
        value: summaryLines,
        inline: false,
      },
      {
        name: '🎯 Game-by-Game',
        value: gameLines || 'No results available.',
        inline: false,
      },
    ],
    footer: { text: 'NBA Oracle v4.0 · Brier score tracks calibration over time' },
    timestamp: new Date().toISOString(),
  };

  return sendWebhook({ embeds: [embed] });
}
