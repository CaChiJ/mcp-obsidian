import type { LatencyStats } from './types.js';

/**
 * NDCG@K — primary metric.
 * relevant: docId → graded relevance score (1, 2, 3)
 */
export function ndcg(retrieved: string[], relevant: Map<string, number>, k: number): number {
  const dcg = computeDCG(retrieved, relevant, k);
  const idcg = computeDCG(
    [...relevant.entries()].sort((a, b) => b[1] - a[1]).map(entry => entry[0]),
    relevant,
    k);

  return idcg === 0 ? 0 : dcg / idcg;
}

export function recall(retrieved: string[], relevant: Map<string, number>, k: number): number {
  const totalRelevant = relevant.size;
  if (totalRelevant === 0) { return 0; }

  let found = 0;

  for (let i = 0; i < Math.min(k, retrieved.length); i++) {
    if (relevant.has(retrieved[i]!)) { found++; }
  }

  return found / totalRelevant;
}

export function mrr(retrieved: string[], relevant: Map<string, number>, k: number): number {
  for (let i = 0; i < Math.min(k, retrieved.length); i++) {
    if (relevant.has(retrieved[i]!)) return 1 / (i + 1);
  }
  return 0;
}

export function latencyStats(times: number[]): LatencyStats {
  const sorted = [...times].sort((a, b) => a - b);
  const n = sorted.length;
  return {
    mean: times.reduce((a, b) => a + b, 0) / n,
    p50: sorted[Math.floor(n * 0.5)]!,
    p95: sorted[Math.floor(n * 0.95)]!,
    p99: sorted[Math.floor(n * 0.99)]!,
  };
}

function computeDCG(retrieved: string[], relevant: Map<string, number>, k: number): number {
  let dcg = 0;
  for (let i = 0; i < Math.min(k, retrieved.length); i++) {
    const rel = relevant.get(retrieved[i]!) ?? 0;
    dcg += (Math.pow(2, rel) - 1) / Math.log2(i + 2);
  }
  return dcg;
}