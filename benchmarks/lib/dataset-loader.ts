import { createReadStream } from 'node:fs';
import { createInterface } from 'node:readline';
import type { CorpusEntry, Query } from './types.js';

const MAX_QUERY_WORDS = 3;

export async function loadCorpus(path: string): Promise<Map<string, CorpusEntry>> {
  const corpus = new Map<string, CorpusEntry>();
  for await (const line of lines(path)) {
    if (!line.trim()) { continue; }

    const entry = JSON.parse(line) as CorpusEntry;
    corpus.set(entry._id, entry);
  }
  return corpus;
}

export async function loadQueries(path: string): Promise<Map<string, Query>> {
  const queries = new Map<string, Query>();
  for await (const line of lines(path)) {
    if (!line.trim()) { continue; }

    const q = JSON.parse(line) as Query;
    if (q.text.trim().split(/\s+/).length <= MAX_QUERY_WORDS) {
      queries.set(q._id, q);
    }
  }
  return queries;
}

/** Loads qrels TSV, keeping only rows whose queryId is in the given set. */
export async function loadQrels(
  path: string,
  queryIds: Set<string>,
): Promise<Map<string, Map<string, number>>> {
  const qrels = new Map<string, Map<string, number>>();
  for await (const line of lines(path)) {
    if (!line.trim()) { continue; }

    const parts = line.split('\t');
    const hasAllColumn = parts.length < 3;
    if (hasAllColumn) { continue; }

    const [queryId, docId, scoreStr] = parts as [string, string, string];
    if (!queryIds.has(queryId)) { continue; }

    const score = parseInt(scoreStr, 10);
    if (score <= 0) { continue; }

    let docs = qrels.get(queryId);
    if (!docs) {
      docs = new Map();
      qrels.set(queryId, docs);
    }
    docs.set(docId, score);
  }
  return qrels;
}

function lines(filePath: string): AsyncIterable<string> {
  return createInterface({
    input: createReadStream(filePath, { encoding: 'utf8' }),
    crlfDelay: Infinity,
  });
}