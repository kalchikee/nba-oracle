// NBA Oracle v4.0 — Market Edge Detection
// Compares model probability to Vegas implied probability.
// Edge tiers: <3% = none, 3-6% = small, 6-10% = meaningful, 10-15% = large, ≥15% = extreme

import { mlToImplied, removeVig } from '../api/oddsClient.js';
import type { EdgeResult, EdgeCategory } from '../types.js';

// ─── Compute edge ─────────────────────────────────────────────────────────────

export function computeEdge(
  modelProb: number,
  homeML: number,
  awayML: number
): EdgeResult {
  const { homeProb, awayProb, vig } = removeVig(homeML, awayML);

  const rawHomeImplied = mlToImplied(homeML);
  const rawAwayImplied = mlToImplied(awayML);

  const edge = modelProb - homeProb;

  let edgeCategory: EdgeCategory;
  const absEdge = Math.abs(edge);

  if (absEdge < 0.03) edgeCategory = 'none';
  else if (absEdge < 0.06) edgeCategory = 'small';
  else if (absEdge < 0.10) edgeCategory = 'meaningful';
  else if (absEdge < 0.15) edgeCategory = 'large';
  else edgeCategory = 'extreme';

  return {
    modelProb,
    vegasProb: homeProb,
    rawHomeImplied,
    rawAwayImplied,
    vigPct: vig,
    edge,
    edgeCategory,
    homeFavorite: homeML < 0,
  };
}

// ─── Format edge for logging ──────────────────────────────────────────────────

export function formatEdge(result: EdgeResult): string {
  const sign = result.edge >= 0 ? '+' : '';
  const pct = (result.edge * 100).toFixed(1);
  const tier = result.edgeCategory.toUpperCase();
  return (
    `Edge: ${sign}${pct}% [${tier}] | ` +
    `Model: ${(result.modelProb * 100).toFixed(1)}% | ` +
    `Vegas: ${(result.vegasProb * 100).toFixed(1)}% | ` +
    `Vig: ${(result.vigPct * 100).toFixed(1)}%`
  );
}

// ─── Killer combination check ─────────────────────────────────────────────────
// Model ≥67% AND Vegas ≤61% = high-value spot historically

export function isKillerCombo(modelProb: number, vegasProb: number): boolean {
  return modelProb >= 0.67 && vegasProb <= 0.61;
}

// ─── Confidence tier ──────────────────────────────────────────────────────────

export type ConfidenceTier =
  | 'coin_flip'       // 50-55%
  | 'lean'            // 55-60%
  | 'strong'          // 60-67%
  | 'high_conviction' // 67-72%
  | 'extreme';        // 72%+

export function getConfidenceTier(calibratedProb: number): ConfidenceTier {
  const p = Math.max(calibratedProb, 1 - calibratedProb); // always use the higher prob
  if (p >= 0.72) return 'extreme';
  if (p >= 0.67) return 'high_conviction';
  if (p >= 0.60) return 'strong';
  if (p >= 0.55) return 'lean';
  return 'coin_flip';
}

export function shouldAlert(calibratedProb: number): boolean {
  const tier = getConfidenceTier(calibratedProb);
  return tier === 'high_conviction' || tier === 'extreme';
}

// ─── Signal agreement ─────────────────────────────────────────────────────────
// Counts how many model signals agree with the pick direction.
// NBA signals: Elo, season net rating, rolling net rating, rest advantage, momentum.
// More agreeing signals = pick is backed by multiple independent factors.

export type SignalAgreement = {
  agreeing: number;
  total: number;
  label: 'CONTRARIAN' | 'SPLIT' | 'MAJORITY' | 'CONSENSUS' | 'LOCK';
};

export function getSignalAgreement(
  features: Record<string, number>,
  pickIsHome: boolean,
): SignalAgreement {
  const dir = pickIsHome ? 1 : -1;
  const candidates: Array<number | undefined> = [
    features['elo_diff'],
    features['net_rtg_diff'],
    features['team_10d_net_rtg_diff'],
    features['rest_days_diff'],
    features['momentum_diff'],
  ];
  const valid = candidates.filter((v): v is number => v != null);
  const agreeing = valid.filter(v => v * dir > 0).length;
  const total = valid.length;

  let label: SignalAgreement['label'];
  if (agreeing === total)  label = 'LOCK';
  else if (agreeing >= 4)  label = 'CONSENSUS';
  else if (agreeing >= 3)  label = 'MAJORITY';
  else if (agreeing >= 2)  label = 'SPLIT';
  else                     label = 'CONTRARIAN';

  return { agreeing, total, label };
}
