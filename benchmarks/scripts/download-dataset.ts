#!/usr/bin/env tsx
import { existsSync } from 'node:fs';
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { execSync } from 'node:child_process';
import { Writable } from 'node:stream';
import { createWriteStream } from 'node:fs';

const DATASET_URL = 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip';
const DATA_DIR = join(import.meta.dirname, '..', 'data');
const DATASET_DIR = join(DATA_DIR, 'nfcorpus');
const ZIP_PATH = join(DATA_DIR, 'nfcorpus.zip');

async function main() {
  if (existsSync(join(DATASET_DIR, 'corpus.jsonl'))) {
    console.log('NFCorpus already downloaded. Skipping.');
    return;
  }

  await mkdir(DATA_DIR, { recursive: true });

  console.log(`Downloading NFCorpus from ${DATASET_URL}...`);
  const response = await fetch(DATASET_URL);
  if (!response.ok || !response.body) {
    throw new Error(`Download failed: ${response.status} ${response.statusText}`);
  }

  const fileStream = createWriteStream(ZIP_PATH);
  // @ts-expect-error ReadableStream → Node writable pipe
  await response.body.pipeTo(Writable.toWeb(fileStream));

  console.log('Extracting...');
  execSync(`unzip -o -q "${ZIP_PATH}" -d "${DATA_DIR}"`);

  // Verify expected files
  for (const file of ['corpus.jsonl', 'queries.jsonl', 'qrels/test.tsv']) {
    if (!existsSync(join(DATASET_DIR, file))) {
      throw new Error(`Expected file missing: ${file}`);
    }
  }

  console.log(`Done. Dataset at ${DATASET_DIR}`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
