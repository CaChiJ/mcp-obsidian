#!/usr/bin/env tsx
import v8 from 'node:v8';
import { runBenchmark } from '../lib/benchmark-runner.js';
import { OnnxAdapter } from '../../src/embedding/onnx-adapter.js';
import { GgufAdapter } from '../../src/embedding/gguf-adapter.js';
import { jinaRetrievalPreprocessor } from '../../src/embedding/preprocessing.js';
import type { EmbeddingAdapterOptions } from '../../src/embedding/types.js';

// Take a heap snapshot once when V8 heap exceeds this threshold
const HEAP_SNAPSHOT_THRESHOLD_MB = 3000;
let heapSnapshotTaken = false;
setInterval(() => {
  if (heapSnapshotTaken) return;
  const heapMb = process.memoryUsage().heapUsed / 1024 / 1024;
  if (heapMb > HEAP_SNAPSHOT_THRESHOLD_MB) {
    heapSnapshotTaken = true;
    const path = v8.writeHeapSnapshot();
    process.stderr.write(`[heap-snapshot] ${heapMb.toFixed(0)} MB → ${path}\n`);
  }
}, 500);

interface WorkerConfig {
  adapterType: 'onnx' | 'gguf';
  model: string;
  quant?: string;
  contextSize?: number | { min?: number; max?: number };
  runOptions: {
    datasetDir: string;
    vaultDir: string;
    maxQueries?: number;
    algorithm?: string;
    force?: boolean;
  };
}

const config: WorkerConfig = JSON.parse(process.argv[2]!);

const options: EmbeddingAdapterOptions =
  config.adapterType === 'gguf'
    ? { preprocessor: jinaRetrievalPreprocessor, contextSize: config.contextSize }
    : {};

const embedder =
  config.adapterType === 'onnx'
    ? new OnnxAdapter(config.model)
    : new GgufAdapter(config.model, config.quant!, options);

const report = await runBenchmark({
  ...config.runOptions,
  embedder,
  onPhase: (event, name) => process.send!({ type: `phase:${event}`, name }),
});

process.send!({ type: 'result', report });
