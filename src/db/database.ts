// NBA Oracle v4.0 — SQLite Database Layer (sql.js — pure JS, no native build)

import initSqlJs, { type Database as SqlJsDatabase } from 'sql.js';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import type { Prediction, GameResult, AccuracyLog, CalibrationLog, EloRating } from '../types.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const DB_PATH = resolve(
  process.env.DB_PATH
    ? process.env.DB_PATH.startsWith('.')
      ? resolve(__dirname, '../../', process.env.DB_PATH)
      : process.env.DB_PATH
    : resolve(__dirname, '../../data/nba_oracle.db')
);

mkdirSync(dirname(DB_PATH), { recursive: true });

let _db: SqlJsDatabase | null = null;
let _SQL: Awaited<ReturnType<typeof initSqlJs>> | null = null;

// ─── Initialization ───────────────────────────────────────────────────────────

export async function initDb(): Promise<SqlJsDatabase> {
  if (_db) return _db;

  _SQL = await initSqlJs();

  if (existsSync(DB_PATH)) {
    const fileBuffer = readFileSync(DB_PATH);
    _db = new _SQL.Database(fileBuffer);
  } else {
    _db = new _SQL.Database();
  }

  initializeSchema(_db);
  persistDb();
  return _db;
}

export function getDb(): SqlJsDatabase {
  if (!_db) throw new Error('Database not initialized. Call initDb() first.');
  return _db;
}

export function persistDb(): void {
  if (!_db) return;
  const data = _db.export();
  writeFileSync(DB_PATH, Buffer.from(data));
}

function run(sql: string, params: (string | number | null | undefined)[] = []): void {
  const db = getDb();
  const stmt = db.prepare(sql);
  stmt.run(params.map(p => p === undefined ? null : p));
  stmt.free();
  persistDb();
}

function queryAll<T = Record<string, unknown>>(sql: string, params: (string | number | null)[] = []): T[] {
  const db = getDb();
  const stmt = db.prepare(sql);
  stmt.bind(params);
  const results: T[] = [];
  while (stmt.step()) {
    results.push(stmt.getAsObject() as T);
  }
  stmt.free();
  return results;
}

function queryOne<T = Record<string, unknown>>(sql: string, params: (string | number | null)[] = []): T | undefined {
  const results = queryAll<T>(sql, params);
  return results[0];
}

// ─── Schema ───────────────────────────────────────────────────────────────────

function initializeSchema(db: SqlJsDatabase): void {
  db.run(`
    CREATE TABLE IF NOT EXISTS elo_ratings (
      team_abbr TEXT PRIMARY KEY,
      rating REAL NOT NULL DEFAULT 1500,
      games_played INTEGER NOT NULL DEFAULT 0,
      season TEXT NOT NULL DEFAULT '2024-25',
      updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS predictions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      game_id TEXT NOT NULL,
      game_date TEXT NOT NULL,
      home_team TEXT NOT NULL,
      away_team TEXT NOT NULL,
      arena TEXT NOT NULL DEFAULT '',
      feature_vector TEXT NOT NULL,
      home_exp_pts REAL NOT NULL DEFAULT 0,
      away_exp_pts REAL NOT NULL DEFAULT 0,
      mc_win_pct REAL NOT NULL,
      calibrated_prob REAL NOT NULL,
      vegas_prob REAL,
      edge REAL,
      total_points REAL NOT NULL DEFAULT 0,
      spread REAL NOT NULL DEFAULT 0,
      most_likely_score TEXT NOT NULL DEFAULT '',
      upset_probability REAL NOT NULL DEFAULT 0,
      blowout_probability REAL NOT NULL DEFAULT 0,
      model_version TEXT NOT NULL DEFAULT '4.0.0',
      actual_winner TEXT,
      correct INTEGER,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS accuracy_log (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      date TEXT NOT NULL UNIQUE,
      brier_score REAL NOT NULL DEFAULT 0,
      log_loss REAL NOT NULL DEFAULT 0,
      accuracy REAL NOT NULL DEFAULT 0,
      high_conv_accuracy REAL NOT NULL DEFAULT 0,
      vs_vegas_brier REAL NOT NULL DEFAULT 0,
      games_evaluated INTEGER NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS game_results (
      game_id TEXT PRIMARY KEY,
      date TEXT NOT NULL,
      home_team TEXT NOT NULL,
      away_team TEXT NOT NULL,
      home_score INTEGER NOT NULL,
      away_score INTEGER NOT NULL,
      arena TEXT NOT NULL DEFAULT '',
      lineups TEXT NOT NULL DEFAULT '{}',
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS calibration_log (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      date TEXT NOT NULL,
      game_id TEXT NOT NULL UNIQUE,
      model_prob REAL NOT NULL,
      vegas_prob REAL NOT NULL DEFAULT 0,
      edge REAL NOT NULL DEFAULT 0,
      outcome INTEGER,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS model_registry (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      version TEXT NOT NULL UNIQUE,
      weights_hash TEXT NOT NULL DEFAULT '',
      train_dates TEXT NOT NULL DEFAULT '',
      test_brier REAL NOT NULL DEFAULT 0,
      test_accuracy REAL NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
  `);

  // Seed model registry if empty
  const cnt = queryOne<{ cnt: number }>('SELECT COUNT(*) as cnt FROM model_registry');
  if (!cnt || cnt.cnt === 0) {
    db.run(
      `INSERT OR IGNORE INTO model_registry (version, weights_hash, train_dates, test_brier, test_accuracy)
       VALUES (?, ?, ?, ?, ?)`,
      ['4.0.0', 'monte-carlo-only', '2018-01-01/2025-06-30', 0, 0]
    );
  }
}

// ─── Elo helpers ──────────────────────────────────────────────────────────────

export function upsertElo(rating: EloRating): void {
  run(
    `INSERT INTO elo_ratings (team_abbr, rating, updated_at)
     VALUES (?, ?, ?)
     ON CONFLICT(team_abbr) DO UPDATE SET
       rating = excluded.rating,
       updated_at = excluded.updated_at`,
    [rating.teamAbbr, rating.rating, rating.updatedAt]
  );
}

export function getElo(teamAbbr: string): number {
  const row = queryOne<{ rating: number }>(
    'SELECT rating FROM elo_ratings WHERE team_abbr = ?',
    [teamAbbr]
  );
  return row?.rating ?? 1500;
}

export function getAllElos(): EloRating[] {
  return queryAll<{ team_abbr: string; rating: number; updated_at: string }>(
    'SELECT team_abbr, rating, updated_at FROM elo_ratings ORDER BY rating DESC'
  ).map(r => ({ teamAbbr: r.team_abbr, rating: r.rating, updatedAt: r.updated_at }));
}

// ─── Prediction helpers ───────────────────────────────────────────────────────

export function upsertPrediction(pred: Prediction): void {
  run(
    `DELETE FROM predictions WHERE game_id = ? AND model_version = ?`,
    [pred.game_id, pred.model_version]
  );
  run(
    `INSERT INTO predictions (
       game_id, game_date, home_team, away_team, arena,
       feature_vector, home_exp_pts, away_exp_pts,
       mc_win_pct, calibrated_prob, vegas_prob, edge,
       total_points, spread, most_likely_score,
       upset_probability, blowout_probability,
       model_version, created_at
     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      pred.game_id, pred.game_date, pred.home_team, pred.away_team, pred.arena,
      JSON.stringify(pred.feature_vector),
      pred.home_exp_pts, pred.away_exp_pts,
      pred.mc_win_pct, pred.calibrated_prob, pred.vegas_prob ?? null, pred.edge ?? null,
      pred.total_points, pred.spread, pred.most_likely_score,
      pred.upset_probability, pred.blowout_probability,
      pred.model_version, pred.created_at,
    ]
  );
}

export function getPredictionsByDate(date: string): Prediction[] {
  const rows = queryAll<Record<string, unknown>>(
    'SELECT * FROM predictions WHERE game_date = ? ORDER BY calibrated_prob DESC',
    [date]
  );
  return rows.map(row => ({
    ...row,
    feature_vector: JSON.parse(row.feature_vector as string),
  })) as Prediction[];
}

export function updatePredictionResult(gameId: string, winner: string, correct: boolean): void {
  run(
    `UPDATE predictions SET actual_winner = ?, correct = ? WHERE game_id = ?`,
    [winner, correct ? 1 : 0, gameId]
  );
}

// ─── Game result helpers ───────────────────────────────────────────────────────

export function upsertGameResult(result: GameResult): void {
  run(
    `INSERT INTO game_results (game_id, date, home_team, away_team, home_score, away_score, arena, lineups)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(game_id) DO UPDATE SET
       home_score = excluded.home_score,
       away_score = excluded.away_score`,
    [
      result.game_id, result.date, result.home_team, result.away_team,
      result.home_score, result.away_score, result.arena, result.lineups,
    ]
  );
}

// ─── Calibration helpers ───────────────────────────────────────────────────────

export function upsertCalibration(entry: CalibrationLog): void {
  run(
    `INSERT INTO calibration_log (date, game_id, model_prob, vegas_prob, edge, outcome)
     VALUES (?, ?, ?, ?, ?, ?)
     ON CONFLICT(game_id) DO UPDATE SET outcome = excluded.outcome`,
    [entry.date, entry.game_id, entry.model_prob, entry.vegas_prob, entry.edge, entry.outcome ?? null]
  );
}

// ─── Accuracy helpers ─────────────────────────────────────────────────────────

export function upsertAccuracyLog(log: AccuracyLog): void {
  run(
    `INSERT INTO accuracy_log (date, brier_score, log_loss, accuracy, high_conv_accuracy, vs_vegas_brier)
     VALUES (?, ?, ?, ?, ?, ?)
     ON CONFLICT(date) DO UPDATE SET
       brier_score = excluded.brier_score,
       log_loss = excluded.log_loss,
       accuracy = excluded.accuracy,
       high_conv_accuracy = excluded.high_conv_accuracy,
       vs_vegas_brier = excluded.vs_vegas_brier`,
    [log.date, log.brier_score, log.log_loss, log.accuracy, log.high_conv_accuracy, log.vs_vegas_brier]
  );
}

export function getRecentAccuracy(days = 30): AccuracyLog[] {
  const cutoff = new Date();
  cutoff.setDate(cutoff.getDate() - days);
  const dateStr = cutoff.toISOString().split('T')[0];
  return queryAll<AccuracyLog>(
    'SELECT * FROM accuracy_log WHERE date >= ? ORDER BY date DESC',
    [dateStr]
  );
}

export interface SeasonRecord {
  correct: number;
  total: number;
  betCorrect: number;
  betTotal: number;
}

export function getSeasonRecord(): SeasonRecord {
  const all = queryOne<{ correct: number; total: number }>(
    `SELECT
       SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct,
       COUNT(*) as total
     FROM predictions WHERE correct IS NOT NULL`
  );
  const bets = queryOne<{ correct: number; total: number }>(
    `SELECT
       SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct,
       COUNT(*) as total
     FROM predictions
     WHERE correct IS NOT NULL
       AND (calibrated_prob >= 0.67 OR calibrated_prob <= 0.33)`
  );
  return {
    correct:    all?.correct  ?? 0,
    total:      all?.total    ?? 0,
    betCorrect: bets?.correct ?? 0,
    betTotal:   bets?.total   ?? 0,
  };
}

export function closeDb(): void {
  if (_db) {
    persistDb();
    _db.close();
    _db = null;
  }
}
