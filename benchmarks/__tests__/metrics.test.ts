import { describe, it, expect } from 'vitest';
import { ndcg, recall, mrr, latencyStats } from '../lib/metrics.js';

describe('ndcg', () => {
  describe('when ranking is perfect', () => {
    it('should return 1.0', () => {
      const relevant = new Map([['a', 3], ['b', 2], ['c', 1]]);
      expect(ndcg(['a', 'b', 'c'], relevant, 3)).toBeCloseTo(1.0);
    });
  });

  describe('when ranking is reversed', () => {
    it('should return less than perfect ranking', () => {
      const relevant = new Map([['a', 3], ['b', 2], ['c', 1]]);
      const reversed = ndcg(['c', 'b', 'a'], relevant, 3);
      expect(reversed).toBeLessThan(1.0);
      expect(reversed).toBeGreaterThan(0);
    });
  });

  describe('when no relevant docs are retrieved', () => {
    it('should return 0', () => {
      const relevant = new Map([['a', 1]]);
      expect(ndcg(['x', 'y', 'z'], relevant, 3)).toBe(0);
    });
  });

  describe('when relevant set is empty', () => {
    it('should return 0', () => {
      expect(ndcg(['a', 'b'], new Map(), 3)).toBe(0);
    });
  });

  describe('when relevant doc is beyond k', () => {
    it('should not count it', () => {
      const relevant = new Map([['d', 2]]);
      expect(ndcg(['x', 'y', 'z', 'd'], relevant, 3)).toBe(0);
    });
  });

  describe('when retrieved has a mix of relevant and irrelevant', () => {
    it('should compute correct value', () => {
      // retrieved: [a(rel=3), x(0), b(rel=1)] at k=3
      // DCG  = (2^3-1)/log2(2) + 0 + (2^1-1)/log2(4) = 7 + 0 + 0.5 = 7.5
      // IDCG = (2^3-1)/log2(2) + (2^1-1)/log2(3) = 7 + 0.6309 = 7.6309
      // NDCG = 7.5 / 7.6309 ≈ 0.9828
      const relevant = new Map([['a', 3], ['b', 1]]);
      expect(ndcg(['a', 'x', 'b'], relevant, 3)).toBeCloseTo(0.9828, 3);
    });
  });
});

describe('recall', () => {
  describe('when all relevant docs are in top-k', () => {
    it('should return 1.0', () => {
      const relevant = new Map([['a', 1], ['b', 1]]);
      expect(recall(['a', 'b', 'c'], relevant, 3)).toBe(1.0);
    });
  });

  describe('when only some relevant docs are in top-k', () => {
    it('should return the fraction found', () => {
      const relevant = new Map([['a', 1], ['b', 1], ['c', 1], ['d', 1]]);
      expect(recall(['a', 'b', 'x'], relevant, 3)).toBe(0.5);
    });
  });

  describe('when no relevant docs are retrieved', () => {
    it('should return 0', () => {
      const relevant = new Map([['a', 1]]);
      expect(recall(['x', 'y'], relevant, 2)).toBe(0);
    });
  });

  describe('when relevant set is empty', () => {
    it('should return 0', () => {
      expect(recall(['a'], new Map(), 5)).toBe(0);
    });
  });

  describe('when relevant doc exists but beyond k', () => {
    it('should not count it', () => {
      const relevant = new Map([['b', 1]]);
      expect(recall(['x', 'b'], relevant, 1)).toBe(0);
    });
  });
});

describe('mrr', () => {
  describe('when first result is relevant', () => {
    it('should return 1.0', () => {
      const relevant = new Map([['a', 1]]);
      expect(mrr(['a', 'b'], relevant, 5)).toBe(1.0);
    });
  });

  describe('when second result is the first relevant', () => {
    it('should return 0.5', () => {
      const relevant = new Map([['b', 1]]);
      expect(mrr(['a', 'b'], relevant, 5)).toBe(0.5);
    });
  });

  describe('when no relevant doc is within k', () => {
    it('should return 0', () => {
      const relevant = new Map([['c', 1]]);
      expect(mrr(['a', 'b'], relevant, 2)).toBe(0);
    });
  });
});

describe('latencyStats', () => {
  describe('when given a uniform sequence', () => {
    it('should compute correct mean and percentiles', () => {
      const times = Array.from({ length: 100 }, (_, i) => i + 1);
      const stats = latencyStats(times);
      expect(stats.mean).toBe(50.5);
      expect(stats.p50).toBe(51);
      expect(stats.p95).toBe(96);
      expect(stats.p99).toBe(100);
    });
  });

  describe('when given a single value', () => {
    it('should return that value for all stats', () => {
      const stats = latencyStats([42]);
      expect(stats.mean).toBe(42);
      expect(stats.p50).toBe(42);
      expect(stats.p95).toBe(42);
      expect(stats.p99).toBe(42);
    });
  });
});
