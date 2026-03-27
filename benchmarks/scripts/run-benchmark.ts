#!/usr/bin/env tsx
import { fork } from 'node:child_process';
import { join } from 'node:path';
import { mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import pidusage from 'pidusage';
import { formatReport } from '../lib/report.js';
import type { BenchmarkReport } from '../lib/types.js';

const DATA_DIR = join(import.meta.dirname, '..', 'data');
const DATASET_DIR = join(DATA_DIR, 'nfcorpus');
const VAULT_DIR = join(DATA_DIR, 'vault');
const RESULTS_DIR = join(DATA_DIR, 'results');
const WORKER_PATH = fileURLToPath(new URL('./benchmark-worker.ts', import.meta.url));

const args = process.argv.slice(2);
const quick = args.includes('--quick');
const maxQueries = quick ? 50 : parseArg('--max-queries');
const algorithm = parseStringArg('--algorithm') ?? 'substring';
const force = args.includes('--force');

const runOptions = { datasetDir: DATASET_DIR, vaultDir: VAULT_DIR, maxQueries, algorithm, force };

type EmbedderConfig =
  | { adapterType: 'onnx'; model: string }
  | { adapterType: 'gguf'; model: string; quant: string; contextSize?: number | { min?: number; max?: number } };

const GGUF_MODEL = 'jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF';

const embedderConfigs: [EmbedderConfig, string][] = [
  [{ adapterType: 'onnx', model: 'Xenova/all-MiniLM-L6-v2' }, 'Xenova_all-MiniLM-L6-v2'],
  [{ adapterType: 'onnx', model: 'Xenova/multilingual-e5-small' }, 'Xenova_multilingual-e5-small'],
  [{ adapterType: 'gguf', model: GGUF_MODEL, quant: 'IQ1_S' }, `${GGUF_MODEL}:IQ1_S`],
  [{ adapterType: 'gguf', model: GGUF_MODEL, quant: 'IQ1_M' }, `${GGUF_MODEL}:IQ1_M`],
  [{ adapterType: 'gguf', model: GGUF_MODEL, quant: 'IQ2_XXS' }, `${GGUF_MODEL}:IQ2_XXS`],
  [{ adapterType: 'gguf', model: GGUF_MODEL, quant: 'Q2_K' }, `${GGUF_MODEL}:Q2_K`],
  [{ adapterType: 'gguf', model: GGUF_MODEL, quant: 'IQ2_M' }, `${GGUF_MODEL}:IQ2_M`],
  [{ adapterType: 'gguf', model: GGUF_MODEL, quant: 'Q3_K_M' }, `${GGUF_MODEL}:Q3_K_M`],
  [{ adapterType: 'gguf', model: GGUF_MODEL, quant: 'IQ4_XS' }, `${GGUF_MODEL}:IQ4_XS`],
  [{ adapterType: 'gguf', model: GGUF_MODEL, quant: 'IQ4_NL' }, `${GGUF_MODEL}:IQ4_NL`],
  [{ adapterType: 'gguf', model: GGUF_MODEL, quant: 'Q4_K_M' }, `${GGUF_MODEL}:Q4_K_M`],
  [{ adapterType: 'gguf', model: GGUF_MODEL, quant: 'Q5_K_S' }, `${GGUF_MODEL}:Q5_K_S`],
  [{ adapterType: 'gguf', model: GGUF_MODEL, quant: 'Q5_K_M' }, `${GGUF_MODEL}:Q5_K_M`],
  [{ adapterType: 'gguf', model: GGUF_MODEL, quant: 'Q6_K' }, `${GGUF_MODEL}:Q6_K`],
  [{ adapterType: 'gguf', model: GGUF_MODEL, quant: 'Q8_0' }, `${GGUF_MODEL}:Q8_0`],
  [{ adapterType: 'gguf', model: GGUF_MODEL, quant: 'F16' }, `${GGUF_MODEL}:F16`],
];

async function runWithMemory(config: EmbedderConfig): Promise<BenchmarkReport> {
  const phasePeaks: Record<string, number> = {};
  let activePhase: string | null = null;

  const child = fork(WORKER_PATH, [JSON.stringify({ ...config, runOptions })], {
    execArgv: ['--import', 'tsx/esm'],
  });

  const pollMemoryUsage = setInterval(async () => {
    if (!activePhase) return;
    try {
      const stats = await pidusage(child.pid!);
      const mb = stats.memory / 1024 / 1024;
      phasePeaks[activePhase] = Math.max(phasePeaks[activePhase] ?? 0, mb);
    } catch { /* process already exited */ }
  }, 200);

  const report = await new Promise<BenchmarkReport>((resolve, reject) => {
    child.on('message', (msg: any) => {
      switch (msg.type) {
        case 'phase:start': phasePeaks[msg.name] = 0; activePhase = msg.name; break;
        case 'phase:end':   activePhase = null; break;
        case 'result':      resolve(msg.report as BenchmarkReport); break;
      }
    });
    child.on('error', reject);
    child.on('exit', (code) => {
      if (code !== 0) reject(new Error(`Worker exited with code ${code}`));
    });
  });

  clearInterval(pollMemoryUsage);

  report.peakInitMemoryMb = phasePeaks['init'];
  report.peakSearchMemoryMb = phasePeaks['search'];
  return report;
}

async function main() {
  for (const [config, name] of embedderConfigs) {
    const report = await runWithMemory(config);

    console.log('\n' + formatReport(report));

    await mkdir(RESULTS_DIR, { recursive: true });
    const filename = `${name.replace(/\//g, '_')}-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
    await writeFile(join(RESULTS_DIR, filename), JSON.stringify(report, null, 2));
    console.log(`\nResult saved: ${join(RESULTS_DIR, filename)}`);
  }
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
