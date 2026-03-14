import type { BenchmarkReport } from './types.js';

export function formatReport(report: BenchmarkReport): string {
  const lines = [
    '=== Search Benchmark Report ===',
    `Algorithm:  ${report.algorithm}`,
    `Corpus:     ${report.corpusSize} documents | Queries: ${report.queryCount}`,
    '',
    pad('Metric', 12) + pad('@5', 10) + '@10',
    '─'.repeat(32),
    metricRow('NDCG', report.ndcg),
    metricRow('Recall', report.recall),
    metricRow('MRR', report.mrr),
    '',
    'Latency (ms/query)',
    '─'.repeat(32),
    `Mean: ${fmt(report.latency.mean)}  P50: ${fmt(report.latency.p50)}  P95: ${fmt(report.latency.p95)}  P99: ${fmt(report.latency.p99)}`,
    '',
    `Reference: BEIR BM25 NDCG@10 ≈ 0.325`,
    `Initialization: ${fmt(report.initializationMs)}ms`,
    `Duration: ${(report.durationMs / 1000).toFixed(1)}s`,
  ];
  return lines.join('\n');
}

function metricRow(name: string, values: Record<number, number>): string {
  return pad(name, 12) + pad(fmt4(values[5]!), 10) + fmt4(values[10]!);
}

function pad(s: string, n: number): string {
  return s.padEnd(n);
}

function fmt(n: number): string {
  return n.toFixed(1);
}

function fmt4(n: number): string {
  return n.toFixed(4);
}
