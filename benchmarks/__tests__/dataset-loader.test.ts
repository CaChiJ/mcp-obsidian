import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, rm, writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { loadCorpus, loadQueries, loadQrels } from '../lib/dataset-loader.js';

let tmpDir: string;

beforeEach(async () => {
  tmpDir = await mkdtemp(join(tmpdir(), 'bench-loader-'));
});

afterEach(async () => {
  await rm(tmpDir, { recursive: true });
});

describe('loadCorpus', () => {
  describe('when file has valid JSONL entries', () => {
    it('should parse all entries into a map', async () => {
      await writeFile(join(tmpDir, 'corpus.jsonl'), [
        '{"_id":"d1","title":"Alpha","text":"content a"}',
        '{"_id":"d2","title":"Beta","text":"content b"}',
      ].join('\n'));

      const corpus = await loadCorpus(join(tmpDir, 'corpus.jsonl'));
      expect(corpus.size).toBe(2);
      expect(corpus.get('d1')?.title).toBe('Alpha');
      expect(corpus.get('d2')?.text).toBe('content b');
    });
  });
});

describe('loadQueries', () => {
  describe('when queries have mixed word counts', () => {
    it('should only keep queries with ≤3 words', async () => {
      await writeFile(join(tmpDir, 'queries.jsonl'), [
        '{"_id":"q1","text":"acne"}',
        '{"_id":"q2","text":"blood pressure"}',
        '{"_id":"q3","text":"vitamin D benefits"}',
        '{"_id":"q4","text":"how to treat diabetes"}',
      ].join('\n'));

      const queries = await loadQueries(join(tmpDir, 'queries.jsonl'));
      expect(queries.size).toBe(3);
      expect(queries.has('q1')).toBe(true);
      expect(queries.has('q2')).toBe(true);
      expect(queries.has('q3')).toBe(true);
      expect(queries.has('q4')).toBe(false);
    });
  });
});

describe('loadQrels', () => {
  describe('when filtering by query IDs', () => {
    it('should only keep rows matching the given query set', async () => {
      await writeFile(join(tmpDir, 'test.tsv'), [
        'q1\td1\t2',
        'q1\td2\t1',
        'q2\td3\t3',
        'q3\td4\t1',
      ].join('\n'));

      const qrels = await loadQrels(join(tmpDir, 'test.tsv'), new Set(['q1', 'q2']));
      expect(qrels.size).toBe(2);
      expect(qrels.get('q1')?.size).toBe(2);
      expect(qrels.get('q1')?.get('d1')).toBe(2);
      expect(qrels.get('q2')?.get('d3')).toBe(3);
      expect(qrels.has('q3')).toBe(false);
    });
  });

  describe('when relevance score is 0', () => {
    it('should skip that entry', async () => {
      await writeFile(join(tmpDir, 'test.tsv'), [
        'q1\td1\t0',
        'q1\td2\t1',
      ].join('\n'));

      const qrels = await loadQrels(join(tmpDir, 'test.tsv'), new Set(['q1']));
      expect(qrels.get('q1')?.size).toBe(1);
      expect(qrels.get('q1')?.has('d1')).toBe(false);
    });
  });
});
