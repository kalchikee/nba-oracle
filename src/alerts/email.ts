// NBA Oracle v4.0 — Email Alerts (Resend)
// Optional. Set RESEND_API_KEY, RESEND_FROM, RESEND_TO in .env

import fetch from 'node-fetch';
import { logger } from '../logger.js';
import { getConfidenceTier } from '../features/marketEdge.js';
import type { Prediction } from '../types.js';

const RESEND_API = 'https://api.resend.com/emails';

async function sendEmail(subject: string, html: string): Promise<boolean> {
  const apiKey = process.env.RESEND_API_KEY;
  if (!apiKey) return false;

  const from = process.env.RESEND_FROM ?? 'nba-oracle@yourdomain.com';
  const to = process.env.RESEND_TO;
  if (!to) return false;

  try {
    const resp = await fetch(RESEND_API, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ from, to, subject, html }),
      signal: AbortSignal.timeout(10000),
    });

    if (!resp.ok) {
      const text = await resp.text();
      logger.warn({ status: resp.status, body: text }, 'Resend email error');
      return false;
    }

    logger.info({ subject }, 'Email sent via Resend');
    return true;
  } catch (err) {
    logger.warn({ err }, 'Failed to send email');
    return false;
  }
}

function pct(p: number): string {
  return (p * 100).toFixed(1) + '%';
}

function getWinner(pred: Prediction): { team: string; winPct: number } {
  if (pred.calibrated_prob >= 0.5) return { team: pred.home_team, winPct: pred.calibrated_prob };
  return { team: pred.away_team, winPct: 1 - pred.calibrated_prob };
}

// ─── Morning briefing email ───────────────────────────────────────────────────

export async function sendMorningBriefingEmail(date: string, predictions: Prediction[]): Promise<boolean> {
  if (predictions.length === 0) return false;

  const sorted = [...predictions].sort((a, b) =>
    Math.abs(b.calibrated_prob - 0.5) - Math.abs(a.calibrated_prob - 0.5)
  );

  const rows = sorted.map(pred => {
    const { team, winPct } = getWinner(pred);
    const tier = getConfidenceTier(pred.calibrated_prob);
    const tierBadge = tier === 'extreme' ? '🔥' : tier === 'high_conviction' ? '⭐' : tier === 'strong' ? '✅' : '';
    const edgeStr = pred.edge !== undefined
      ? `<td>${pred.edge >= 0 ? '+' : ''}${pct(pred.edge)}</td>`
      : '<td>—</td>';

    return `
      <tr>
        <td>${pred.away_team} @ ${pred.home_team}</td>
        <td>${tierBadge} <strong>${team}</strong> ${pct(winPct)}</td>
        <td>${pct(pred.mc_win_pct)}</td>
        <td>${pred.spread >= 0 ? `Home -${pred.spread.toFixed(1)}` : `Away -${Math.abs(pred.spread).toFixed(1)}`}</td>
        <td>${pred.total_points.toFixed(1)}</td>
        ${edgeStr}
      </tr>`;
  }).join('');

  const html = `
    <h2>🏀 NBA Oracle v4.0 — Morning Briefing (${date})</h2>
    <p><strong>${predictions.length}</strong> games today</p>
    <table border="1" cellpadding="6" style="border-collapse:collapse;font-family:monospace">
      <thead>
        <tr>
          <th>Matchup</th>
          <th>Pick</th>
          <th>MC Win%</th>
          <th>Spread</th>
          <th>Total</th>
          <th>Edge</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
    <p style="color:#999;font-size:12px">NBA Oracle v4.0 · Monte Carlo + ML Meta-model</p>
  `;

  return sendEmail(`NBA Oracle — Predictions for ${date}`, html);
}

// ─── Evening recap email ──────────────────────────────────────────────────────

export async function sendEveningRecapEmail(
  date: string,
  games: Array<{ prediction: Prediction; homeScore: number; awayScore: number }>,
  metrics: { accuracy: number; brier: number; highConvAccuracy: number | null }
): Promise<boolean> {
  const correct = games.filter(g => g.prediction.correct).length;
  const total = games.length;

  const rows = games.map(({ prediction: pred, homeScore, awayScore }) => {
    const isCorrect = pred.correct ? '✅' : '❌';
    const { team: pickedTeam, winPct } = getWinner(pred);
    const actualWinner = homeScore > awayScore ? pred.home_team : pred.away_team;
    return `
      <tr>
        <td>${pred.away_team} @ ${pred.home_team}</td>
        <td>${pred.home_team} ${homeScore} - ${awayScore} ${pred.away_team}</td>
        <td>${pickedTeam} ${pct(winPct)}</td>
        <td>${isCorrect}</td>
      </tr>`;
  }).join('');

  const html = `
    <h2>🌙 NBA Oracle v4.0 — Evening Recap (${date})</h2>
    <p>
      <strong>Record:</strong> ${correct}/${total} (${total > 0 ? ((correct / total) * 100).toFixed(1) : '—'}%) &nbsp;|&nbsp;
      <strong>Brier:</strong> ${metrics.brier.toFixed(4)} &nbsp;|&nbsp;
      ${metrics.highConvAccuracy !== null ? `<strong>High-Conv:</strong> ${(metrics.highConvAccuracy * 100).toFixed(1)}%` : ''}
    </p>
    <table border="1" cellpadding="6" style="border-collapse:collapse;font-family:monospace">
      <thead>
        <tr><th>Matchup</th><th>Final</th><th>Pick</th><th>Result</th></tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
    <p style="color:#999;font-size:12px">NBA Oracle v4.0</p>
  `;

  return sendEmail(`NBA Oracle — Recap for ${date}`, html);
}
