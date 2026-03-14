import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { runBenchmark } from '../lib/benchmark-runner.js';
import type { EmbeddingAdapter, EmbeddingRequest } from '../../src/embedding/types.js';

class FakeEmbedder implements EmbeddingAdapter {
  readonly dimensions = 2;
  readonly modelId = 'fake-test-model';
  readonly options = {};

  async embed(text: string, _request: EmbeddingRequest): Promise<Float32Array> {
    const normalized = text.toLowerCase();
    return new Float32Array([
      normalized.includes('alpha') ? 1 : 0,
      normalized.includes('beta') ? 1 : 0,
    ]);
  }
}

let tempRoot: string;
let datasetDir: string;
let vaultDir: string;

beforeEach(async () => {
  tempRoot = await mkdtemp(join(tmpdir(), 'bench-runner-'));
  datasetDir = join(tempRoot, 'dataset');
  vaultDir = join(tempRoot, 'vault');

  await mkdir(join(datasetDir, 'qrels'), { recursive: true });
  await mkdir(vaultDir, { recursive: true });

  await writeFile(
    join(datasetDir, 'corpus.jsonl'),
    [
      JSON.stringify({ _id: 'd1', title: 'Alpha Doc', text: 'alpha treatment guidance' }),
      JSON.stringify({ _id: 'd2', title: 'Beta Doc', text: 'beta prevention notes' }),
    ].join('\n'),
  );

  await writeFile(
    join(datasetDir, 'queries.jsonl'),
    JSON.stringify({ _id: 'q1', text: 'alpha' }),
  );

  await writeFile(join(datasetDir, 'qrels', 'test.tsv'), 'q1\td1\t2\n');
});

afterEach(async () => {
  await rm(tempRoot, { recursive: true, force: true });
});

describe('runBenchmark', () => {
  it('initializes the search service before running queries and reports the timing', async () => {
    const report = await runBenchmark({
      embedder: new FakeEmbedder(),
      datasetDir,
      vaultDir,
      maxQueries: 1,
    });

    expect(report.queryCount).toBe(1);
    expect(report.initializationMs).toBeGreaterThanOrEqual(0);
    expect(report.durationMs).toBeGreaterThanOrEqual(report.initializationMs);
    expect(report.latency.mean).toBeGreaterThanOrEqual(0);
  });
});
