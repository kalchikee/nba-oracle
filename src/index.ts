// NBA Oracle v4.0 — CLI Entry Point
// Usage:
//   npm start                              → predictions for today
//   npm start -- --date 2026-04-06        → predictions for specific date
//   npm start -- --alert morning          → send morning briefing
//   npm start -- --alert recap            → send evening recap
//   npm start -- --help                   → show help

import 'dotenv/config';
import { logger } from './logger.js';
import { runPipeline } from './pipeline.js';
import { closeDb, initDb, getPredictionsByDate } from './db/database.js';
import type { PipelineOptions } from './types.js';

// ─── CLI argument parsing ─────────────────────────────────────────────────────

type AlertMode = 'morning' | 'recap' | null;

function parseArgs(): PipelineOptions & { help: boolean; alertMode: AlertMode } {
  const args = process.argv.slice(2);
  const opts: PipelineOptions & { help: boolean; alertMode: AlertMode } = {
    help: false,
    verbose: true,
    forceRefresh: false,
    alertMode: null,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    switch (arg) {
      case '--help':
      case '-h':
        opts.help = true;
        break;
      case '--date':
      case '-d':
        opts.date = args[++i];
        break;
      case '--force-refresh':
      case '-f':
        opts.forceRefresh = true;
        break;
      case '--quiet':
      case '-q':
        opts.verbose = false;
        break;
      case '--alert':
      case '-a': {
        const mode = args[++i];
        if (mode === 'morning' || mode === 'recap') {
          opts.alertMode = mode as AlertMode;
        } else {
          console.error(`Unknown alert mode: "${mode}". Use "morning" or "recap".`);
          process.exit(1);
        }
        break;
      }
      default:
        if (/^\d{4}-\d{2}-\d{2}$/.test(arg)) {
          opts.date = arg;
        }
    }
  }

  return opts;
}

function printHelp(): void {
  console.log(`
NBA Oracle v4.0 — ML Prediction Engine
=======================================

USAGE:
  npm start [options]
  node --loader ts-node/esm src/index.ts [options]

OPTIONS:
  --date, -d YYYY-MM-DD        Run predictions for a specific date (default: today)
  --force-refresh, -f          Bypass cache and re-fetch all data
  --quiet, -q                  Suppress prediction table output
  --alert, -a <morning|recap>  Send a Discord alert for today (or --date)
  --help, -h                   Show this help message

EXAMPLES:
  npm start                              # Today's predictions
  npm start -- --date 2026-04-06        # Specific date
  npm start -- -d 2026-04-06 -f         # Specific date, force fresh data
  npm run alerts:morning                 # Send morning briefing to Discord
  npm run alerts:recap                   # Send evening recap to Discord
  npm run scheduler                      # Start the long-running cron scheduler
  npm run train                          # Train the ML meta-model (Python)
  npm run backtest                       # Run walk-forward backtest (Python)

OUTPUT:
  Predictions stored in ./data/nba_oracle.db (SQLite)
  Cache files in ./cache/
  Logs in ./logs/

ENVIRONMENT (.env):
  DISCORD_WEBHOOK_URL    Discord webhook (optional — alerts skipped if unset)
  RESEND_API_KEY         Resend email API key (optional)
  ODDS_API_KEY           The Odds API key (optional — live Vegas lines)
  LOG_LEVEL              Logging level (default: info)

ARCHITECTURE:
  NBA API → Feature Engineering (30+ features) → Monte Carlo (10k Normal sims)
  → ML Meta-model (Logistic Regression) → Platt Scaling → Edge Detection → SQLite
`);
}

// ─── Alert handlers ───────────────────────────────────────────────────────────

async function runMorningAlert(date: string): Promise<void> {
  const { sendMorningBriefing } = await import('./alerts/discord.js');
  const { sendMorningBriefingEmail } = await import('./alerts/email.js');

  await initDb();

  // Always re-run the pipeline — never serve stale DB predictions
  const predictions = await runPipeline({ date, verbose: false });

  // Discord: picks + bet recommendations (two embeds in one message)
  await sendMorningBriefing(date);
  // Email: optional if RESEND_API_KEY is set
  await sendMorningBriefingEmail(date, predictions);
}

async function runRecapAlert(date: string): Promise<void> {
  const { sendEveningRecap } = await import('./alerts/discord.js');
  const { sendEveningRecapEmail } = await import('./alerts/email.js');
  const { processResults } = await import('./alerts/results.js');

  const { games, metrics } = await processResults(date);
  await sendEveningRecap(date, games, metrics);
  await sendEveningRecapEmail(date, games, metrics);
}

// ─── Entry point ──────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const opts = parseArgs();

  if (opts.help) {
    printHelp();
    process.exit(0);
  }

  if (opts.date && !/^\d{4}-\d{2}-\d{2}$/.test(opts.date)) {
    logger.error({ date: opts.date }, 'Invalid date format. Use YYYY-MM-DD');
    process.exit(1);
  }

  const date = opts.date ?? new Date().toISOString().split('T')[0];

  logger.info({ date, version: '4.0.0', pid: process.pid, alertMode: opts.alertMode ?? 'pipeline' }, 'NBA Oracle starting');

  try {
    if (opts.alertMode === 'morning') {
      await runMorningAlert(date);
      return;
    }

    if (opts.alertMode === 'recap') {
      await runRecapAlert(date);
      return;
    }

    // Force refresh: clear cache
    if (opts.forceRefresh) {
      logger.info('Force refresh: clearing cache');
      const { readdirSync, unlinkSync } = await import('fs');
      const cacheDir = process.env.CACHE_DIR ?? './cache';
      try {
        const files = readdirSync(cacheDir);
        for (const file of files) {
          if (file.endsWith('.json')) unlinkSync(`${cacheDir}/${file}`);
        }
        logger.info({ cleared: files.length }, 'Cache cleared');
      } catch {
        // Cache dir may not exist yet
      }
    }

    const predictions = await runPipeline(opts);

    if (predictions.length === 0) {
      console.log(`\nNo games scheduled for ${date}.\n`);
      process.exit(0);
    }

    logger.info({ predictions: predictions.length }, 'Pipeline completed successfully');

  } catch (err) {
    logger.error({ err }, 'Fatal error');
    process.exit(1);
  } finally {
    closeDb();
  }
}

process.on('unhandledRejection', (reason) => {
  logger.error({ reason }, 'Unhandled promise rejection');
  process.exit(1);
});

process.on('uncaughtException', (err) => {
  logger.error({ err }, 'Uncaught exception');
  closeDb();
  process.exit(1);
});

main();
