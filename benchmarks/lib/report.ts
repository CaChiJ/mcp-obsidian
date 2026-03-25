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
    `Reference: BEIR BM25 NDCG@10 ≈ 0.325`,
    `Initialization: ${fmt(report.initializationMs)}ms${report.peakInitMemoryMb !== undefined ? ` | Peak Memory: ${report.peakInitMemoryMb.toFixed(1)} MB` : ''}`,
    `Search Latency: Mean ${fmt(report.latency.mean)}ms  P50 ${fmt(report.latency.p50)}ms  P95 ${fmt(report.latency.p95)}ms  P99 ${fmt(report.latency.p99)}ms${report.peakSearchMemoryMb !== undefined ? ` | Peak Memory: ${report.peakSearchMemoryMb.toFixed(1)} MB` : ''}`,
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
