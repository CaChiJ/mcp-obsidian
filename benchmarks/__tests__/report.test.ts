import { describe, expect, it } from 'vitest';
import { formatReport } from '../lib/report.js';

describe('formatReport', () => {
  it('includes initialization timing in the benchmark output', () => {
    const output = formatReport({
      algorithm: 'semantic',
      corpusSize: 2,
      queryCount: 1,
      ndcg: { 5: 1, 10: 1 },
      recall: { 5: 1, 10: 1 },
      mrr: { 5: 1, 10: 1 },
      latency: {
        mean: 4.2,
        p50: 4.2,
        p95: 4.2,
        p99: 4.2,
      },
      initializationMs: 12.34,
      durationMs: 250,
    });

    expect(output).toContain('Initialization: 12.3ms');
    expect(output).toContain('Duration: 0.3s');
  });
});
