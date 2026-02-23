#!/usr/bin/env tsx
import { join } from 'node:path';
import { mkdir, writeFile } from 'node:fs/promises';
import { runBenchmark } from '../lib/benchmark-runner.js';
import { formatReport } from '../lib/report.js';

const DATA_DIR = join(import.meta.dirname, '..', 'data');
const DATASET_DIR = join(DATA_DIR, 'nfcorpus');
const VAULT_DIR = join(DATA_DIR, 'vault');
const RESULTS_DIR = join(DATA_DIR, 'results');

const args = process.argv.slice(2);
const quick = args.includes('--quick');
const maxQueries = quick ? 50 : parseArg('--max-queries');
const algorithm = parseStringArg('--algorithm') ?? 'substring';
const force = args.includes('--force');

async function main() {
  const report = await runBenchmark({
    datasetDir: DATASET_DIR,
    vaultDir: VAULT_DIR,
    maxQueries,
    algorithm,
    force,
  });

  console.log('\n' + formatReport(report));

  // Save JSON result
  await mkdir(RESULTS_DIR, { recursive: true });
  const filename = `${algorithm}-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
  await writeFile(join(RESULTS_DIR, filename), JSON.stringify(report, null, 2));
  console.log(`\nResult saved: ${join(RESULTS_DIR, filename)}`);
}

function parseArg(flag: string): number | undefined {
  const idx = args.indexOf(flag);
  if (idx === -1 || idx + 1 >= args.length) return undefined;
  return parseInt(args[idx + 1]!, 10);
}

function parseStringArg(flag: string): string | undefined {
  const idx = args.indexOf(flag);
  if (idx === -1 || idx + 1 >= args.length) return undefined;
  return args[idx + 1];
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
