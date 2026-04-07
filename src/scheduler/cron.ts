// NBA Oracle v4.0 — Cron Scheduler
// 6 AM CST: morning picks + bet recommendations
// 1 AM CST: nightly results recap
//
// Automatically goes silent during the NBA off-season (mid-June → mid-October).
// Sends a one-time "season over" message when the Finals end and a
// "season starting" message the week before opening night.
//
// Usage: npm run scheduler  (long-running process)

import 'dotenv/config';
import cron from 'node-cron';
import { logger } from '../logger.js';
import { runPipeline } from '../pipeline.js';
import { initDb, getPredictionsByDate, closeDb } from '../db/database.js';
import { sendMorningBriefing } from '../alerts/discord.js';
import fetch from 'node-fetch';

// ─── NBA season window ────────────────────────────────────────────────────────
// Regular season:  ~Oct 22 → ~Apr 13
// Playoffs:        ~Apr 14 → ~Jun 22  (Finals end by late June)
// Off-season:      ~Jun 23 → ~Oct 21
//
// We define the ACTIVE window as Oct 1 – Jun 25 (inclusive).
// Outside that window the scheduler skips all Discord messages.

const SEASON_START_MONTH = 10; // October (1-indexed)
const SEASON_START_DAY   = 1;
const SEASON_END_MONTH   = 6;  // June
const SEASON_END_DAY     = 25;

function isNBASeason(date: Date = new Date()): boolean {
  const month = date.getMonth() + 1; // 1-indexed
  const day   = date.getDate();

  // Active: Oct 1 → Jun 25
  if (month > SEASON_START_MONTH) return true;          // Nov, Dec
  if (month === SEASON_START_MONTH && day >= SEASON_START_DAY) return true;  // Oct ≥ 1
  if (month < SEASON_END_MONTH) return true;            // Jan – May
  if (month === SEASON_END_MONTH && day <= SEASON_END_DAY) return true;      // Jun ≤ 25
  return false;
}

// ─── Transition state (persisted in memory) ───────────────────────────────────
// We send a one-time "season over" / "season starting" message on each transition.

let lastSeasonState: boolean | null = null;

async function sendSeasonTransitionAlert(nowActive: boolean): Promise<void> {
  const webhookUrl = process.env.DISCORD_WEBHOOK_URL;
  if (!webhookUrl) return;

  const embed = nowActive
    ? {
        title: '🏀 NBA Season Is Starting!',
        description: 'Opening Night is approaching. NBA Oracle is back online — daily picks and bet recommendations resume at **6 AM CST** every game day.',
        color: 0x1a6ef5,
        timestamp: new Date().toISOString(),
      }
    : {
        title: '🏆 NBA Season Is Over',
        description: 'The Championship has been decided. NBA Oracle is going into **off-season mode** — no more daily messages until the next regular season starts in October.',
        color: 0x95a5a6,
        timestamp: new Date().toISOString(),
      };

  try {
    await fetch(webhookUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ embeds: [embed] }),
      signal: AbortSignal.timeout(10000),
    });
    logger.info({ active: nowActive }, '[Scheduler] Season transition alert sent');
  } catch (err) {
    logger.warn({ err }, '[Scheduler] Failed to send season transition alert');
  }
}

// ─── Utility ──────────────────────────────────────────────────────────────────

function todayStr(): string {
  return new Date().toISOString().split('T')[0];
}

// ─── Morning routine (6 AM CST) ──────────────────────────────────────────────

async function runMorningRoutine(): Promise<void> {
  const date = todayStr();
  const now = new Date();

  // Check season state and fire transition alert if needed
  const active = isNBASeason(now);
  if (lastSeasonState !== null && lastSeasonState !== active) {
    await sendSeasonTransitionAlert(active);
  }
  lastSeasonState = active;

  if (!active) {
    logger.info({ date }, '[Scheduler] Off-season — skipping morning routine');
    return;
  }

  logger.info({ date }, '[Scheduler] Morning routine starting');

  try {
    // Run pipeline — returns [] if no games today (All-Star break, travel days, etc.)
    const predictions = await runPipeline({ date, verbose: false });

    if (predictions.length === 0) {
      logger.info({ date }, '[Scheduler] No games today — skipping Discord message');
      return;
    }

    await sendMorningBriefing(date);
    logger.info({ date, games: predictions.length }, '[Scheduler] Morning routine complete');
  } catch (err) {
    logger.error({ err, date }, '[Scheduler] Morning routine failed');
  }
}

// ─── Evening recap (1 AM CST) ─────────────────────────────────────────────────

async function runEveningRoutine(): Promise<void> {
  const date = todayStr();

  if (!isNBASeason()) {
    logger.info({ date }, '[Scheduler] Off-season — skipping recap');
    return;
  }

  logger.info({ date }, '[Scheduler] Evening routine starting');

  try {
    const { sendEveningRecap }      = await import('../alerts/discord.js');
    const { sendEveningRecapEmail } = await import('../alerts/email.js');
    const { processResults }        = await import('../alerts/results.js');

    const { games, metrics } = await processResults(date);

    if (games.length === 0) {
      logger.info({ date }, '[Scheduler] No completed games — skipping recap');
      return;
    }

    await sendEveningRecap(date, games, metrics);
    await sendEveningRecapEmail(date, games, metrics);
    logger.info({ date, games: games.length }, '[Scheduler] Evening routine complete');
  } catch (err) {
    logger.error({ err, date }, '[Scheduler] Evening routine failed');
  }
}

// ─── Start scheduler ──────────────────────────────────────────────────────────

async function startScheduler(): Promise<void> {
  logger.info('[Scheduler] NBA Oracle v4.0 Scheduler starting...');

  await initDb();

  // Set initial season state (no transition alert on startup)
  lastSeasonState = isNBASeason();
  logger.info(
    { active: lastSeasonState },
    lastSeasonState ? '[Scheduler] NBA season is ACTIVE' : '[Scheduler] NBA OFF-SEASON — messages suppressed until Oct 1'
  );

  // 6 AM CST daily — pipeline + morning picks + bet recommendations
  cron.schedule('0 6 * * *', () => {
    logger.info('[Scheduler] 6 AM CST fired');
    void runMorningRoutine();
  }, { timezone: 'America/Chicago' });

  // 1 AM CST daily — results recap
  cron.schedule('0 1 * * *', () => {
    logger.info('[Scheduler] 1 AM CST fired');
    void runEveningRoutine();
  }, { timezone: 'America/Chicago' });

  logger.info('[Scheduler] Cron running: 6 AM morning + 1 AM recap (CST). Active Oct 1 – Jun 25.');

  // If already past 6 AM with no predictions and it's game season, run immediately
  const now  = new Date();
  const hour = now.getHours();
  const date = todayStr();

  if (isNBASeason(now) && hour >= 6 && getPredictionsByDate(date).length === 0) {
    logger.info('[Scheduler] Running morning routine now (missed scheduled window)');
    void runMorningRoutine();
  }
}

// ─── Graceful shutdown ────────────────────────────────────────────────────────

process.on('SIGINT', () => {
  logger.info('[Scheduler] SIGINT — shutting down');
  closeDb();
  process.exit(0);
});

process.on('SIGTERM', () => {
  logger.info('[Scheduler] SIGTERM — shutting down');
  closeDb();
  process.exit(0);
});

process.on('unhandledRejection', (reason) => {
  logger.error({ reason }, '[Scheduler] Unhandled rejection');
});

process.on('uncaughtException', (err) => {
  logger.error({ err }, '[Scheduler] Uncaught exception');
  closeDb();
  process.exit(1);
});

startScheduler();
