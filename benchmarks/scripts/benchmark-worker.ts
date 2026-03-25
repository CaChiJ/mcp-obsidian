#!/usr/bin/env tsx
import { runBenchmark } from '../lib/benchmark-runner.js';
import { OnnxAdapter } from '../../src/embedding/onnx-adapter.js';
import { GgufAdapter } from '../../src/embedding/gguf-adapter.js';
import { jinaRetrievalPreprocessor } from '../../src/embedding/preprocessing.js';
import type { EmbeddingAdapterOptions } from '../../src/embedding/types.js';

interface WorkerConfig {
  adapterType: 'onnx' | 'gguf';
  model: string;
  quant?: string;
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
  config.adapterType === 'gguf' ? { preprocessor: jinaRetrievalPreprocessor } : {};

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
