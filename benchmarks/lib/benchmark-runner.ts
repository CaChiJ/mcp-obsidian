import { join } from 'node:path';
import { performance } from 'node:perf_hooks';
import { PathFilter } from '../../src/pathfilter.js';
import { loadCorpus, loadQueries, loadQrels } from './dataset-loader.js';
import { buildVault } from './vault-builder.js';
import { ndcg, recall, mrr, latencyStats } from './metrics.js';
import type { BenchmarkReport } from './types.js';
import { SemanticSearchService } from '../../src/search/semantic-search.js';
import { VectorStore } from '../../src/search/vector-store.js';
import { EmbeddingAdapter } from '../../src/embedding/types.js';

export interface RunOptions {
  embedder: EmbeddingAdapter;
  datasetDir: string;
  vaultDir: string;
  maxQueries?: number;
  algorithm?: string;
  force?: boolean;
  onPhase?: (event: 'start' | 'end', name: string) => void;
}

export async function runBenchmark(options: RunOptions): Promise<BenchmarkReport> {
  const { embedder, datasetDir, vaultDir, maxQueries, algorithm = 'substring' } = options;
  const totalStart = performance.now();

  // Load dataset
  const corpus = await loadCorpus(join(datasetDir, 'corpus.jsonl'));
  const queries = await loadQueries(join(datasetDir, 'queries.jsonl'));
  const qrels = await loadQrels(join(datasetDir, 'qrels', 'test.tsv'), new Set(queries.keys()));

  // Build vault (title as filename), get reverse mapping
  const filenameToDocId = await buildVault(corpus, vaultDir, { force: options.force });
  const docIdToFilename = new Map<string, string>();
  for (const [filename, docId] of filenameToDocId) {
    docIdToFilename.set(docId, filename);
  }

  // Prepare search service
  const vectorStore = new VectorStore(vaultDir, embedder);
  const search = new SemanticSearchService(vaultDir, new PathFilter(), vectorStore);
  const initializationStart = performance.now();
  options.onPhase?.('start', 'init');
  await search.initialize();
  options.onPhase?.('end', 'init');
  const initializationMs = performance.now() - initializationStart;

  // Run queries
  const kValues = [5, 10];
  const ndcgSums: Record<number, number> = { 5: 0, 10: 0 };
  const recallSums: Record<number, number> = { 5: 0, 10: 0 };
  const mrrSums: Record<number, number> = { 5: 0, 10: 0 };
  const latencies: number[] = [];

  options.onPhase?.('start', 'search');
  let queryCount = 0;
  for (const [queryId, query] of queries) {
    if (maxQueries && queryCount >= maxQueries) break;

    const relevant = qrels.get(queryId);
    if (!relevant || relevant.size === 0) continue;

    const start = performance.now();
    const results = await search.search({ query: query.text, limit: 20 });
    latencies.push(performance.now() - start);

    // Map filenames back to docIds
    const retrievedDocIds = results.map(r => {
      const filename = r.p.replace(/\.md$/, '');
      return filenameToDocId.get(filename) ?? filename;
    });

    for (const k of kValues) {
      ndcgSums[k]! += ndcg(retrievedDocIds, relevant, k);
      recallSums[k]! += recall(retrievedDocIds, relevant, k);
      mrrSums[k]! += mrr(retrievedDocIds, relevant, k);
    }
    queryCount++;
  }
  options.onPhase?.('end', 'search');

  // Average metrics
  const avg = (sums: Record<number, number>) =>
    Object.fromEntries(kValues.map(k => [k, queryCount > 0 ? sums[k]! / queryCount : 0]));

  return {
    algorithm,
    corpusSize: corpus.size,
    queryCount,
    ndcg: avg(ndcgSums),
    recall: avg(recallSums),
    mrr: avg(mrrSums),
    latency: latencyStats(latencies),
    initializationMs,
    durationMs: performance.now() - totalStart,
  };
}
