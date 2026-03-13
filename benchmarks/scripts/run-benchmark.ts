#!/usr/bin/env tsx
import { join } from 'node:path';
import { mkdir, writeFile } from 'node:fs/promises';
import { runBenchmark } from '../lib/benchmark-runner.js';
import { formatReport } from '../lib/report.js';
import type { EmbeddingAdapter } from '../../src/embedding/types.js';
import { OnnxAdapter } from '../../src/embedding/onnx-adapter.js';
import { GgufAdapter } from '../../src/embedding/gguf-adapter.js';

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
  const embedders: [EmbeddingAdapter, string][] = [
    [new OnnxAdapter('Xenova/all-MiniLM-L6-v2'), 'Xenova_all-MiniLM-L6-v2'],
    [new OnnxAdapter('Xenova/multilingual-e5-small'), 'Xenova_multilingual-e5-small'],
    [new GgufAdapter('jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF', 'IQ1_S'), 'jinaai_jina-embeddings-v5-text-nano-retrieval-GGUF:IQ1_S'],
    [new GgufAdapter('jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF', 'IQ1_M'), 'jinaai_jina-embeddings-v5-text-nano-retrieval-GGUF:IQ1_M'],
    [new GgufAdapter('jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF', 'IQ2_XXS'), 'jinaai_jina-embeddings-v5-text-nano-retrieval-GGUF:IQ2_XXS'],
    [new GgufAdapter('jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF', 'Q2_K'), 'jinaai_jina-embeddings-v5-text-nano-retrieval-GGUF:Q2_K'],
    [new GgufAdapter('jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF', 'IQ2_M'), 'jinaai_jina-embeddings-v5-text-nano-retrieval-GGUF:IQ2_M'],
    [new GgufAdapter('jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF', 'Q3_K_M'), 'jinaai_jina-embeddings-v5-text-nano-retrieval-GGUF:Q3_K_M'],
    [new GgufAdapter('jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF', 'IQ4_XS'), 'jinaai_jina-embeddings-v5-text-nano-retrieval-GGUF:IQ4_XS'],
    [new GgufAdapter('jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF', 'IQ4_NL'), 'jinaai_jina-embeddings-v5-text-nano-retrieval-GGUF:IQ4_NL'],
    [new GgufAdapter('jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF', 'Q4_K_M'), 'jinaai_jina-embeddings-v5-text-nano-retrieval-GGUF:Q4_K_M'],
    [new GgufAdapter('jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF', 'Q5_K_S'), 'jinaai_jina-embeddings-v5-text-nano-retrieval-GGUF:Q5_K_S'],
    [new GgufAdapter('jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF', 'Q5_K_M'), 'jinaai_jina-embeddings-v5-text-nano-retrieval-GGUF:Q5_K_M'],
    [new GgufAdapter('jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF', 'Q6_K'), 'jinaai_jina-embeddings-v5-text-nano-retrieval-GGUF:Q6_K'],
    [new GgufAdapter('jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF', 'Q8_0'), 'jinaai_jina-embeddings-v5-text-nano-retrieval-GGUF:Q8_0'],
    [new GgufAdapter('jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF', 'F16'), 'jinaai_jina-embeddings-v5-text-nano-retrieval-GGUF:F16']
  ];

  for (const [embedder, name] of embedders) {
    const report = await runBenchmark({
      embedder,
      datasetDir: DATASET_DIR,
      vaultDir: VAULT_DIR,
      maxQueries,
      algorithm,
      force,
    });

    console.log('\n' + formatReport(report));

    // Save JSON result
    await mkdir(RESULTS_DIR, { recursive: true });
    const filename = `${name}-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
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
