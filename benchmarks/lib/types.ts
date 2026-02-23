export interface CorpusEntry {
  _id: string;
  title: string;
  text: string;
}

export interface Query {
  _id: string;
  text: string;
}

export interface LatencyStats {
  mean: number;
  p50: number;
  p95: number;
  p99: number;
}

export interface BenchmarkReport {
  algorithm: string;
  corpusSize: number;
  queryCount: number;
  ndcg: Record<number, number>;
  recall: Record<number, number>;
  mrr: Record<number, number>;
  latency: LatencyStats;
  durationMs: number;
}
