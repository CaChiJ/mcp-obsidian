import { mkdir, writeFile, readdir } from 'node:fs/promises';
import { join } from 'node:path';
import type { CorpusEntry } from './types.js';

// Characters not allowed in Obsidian note names:
// OS: * " \ / < > : | ?
// Obsidian link syntax: # ^ [ ]
const INVALID_CHARS = /[*"\\/<>:|?#^[\]]/g;

export interface BuildVaultOptions {
  force?: boolean;
}

/** Returns a map of filename (without .md) → docId for result mapping. */
export async function buildVault(
  corpus: Map<string, CorpusEntry>,
  outDir: string,
  options: BuildVaultOptions = {},
): Promise<Map<string, string>> {
  await mkdir(outDir, { recursive: true });

  const existing = await readdir(outDir);
  if (existing.length > 0 && !options.force) {
    console.log(`Vault directory not empty (${existing.length} files). Use force to overwrite.`);
    // Rebuild mapping from existing files — caller still needs it
    return rebuildMapping(corpus);
  }

  console.log(`Building vault: ${corpus.size} documents → ${outDir}`);

  const filenameToDocId = new Map<string, string>();
  const usedNames = new Map<string, number>();
  const writes: Promise<void>[] = [];

  for (const [id, entry] of corpus) {
    let name = sanitize(entry.title || id);

    // Handle duplicate titles
    const count = usedNames.get(name) ?? 0;
    usedNames.set(name, count + 1);
    if (count > 0) { name = `${name} (${count})`; }

    filenameToDocId.set(name, id);
    writes.push(writeFile(join(outDir, `${name}.md`), entry.text + '\n', 'utf-8'));
  }

  await Promise.all(writes);
  console.log('Vault built.');
  return filenameToDocId;
}

function sanitize(title: string): string {
  return title.replace(INVALID_CHARS, '-').trim() || 'untitled';
}

/** Rebuilds the filename→docId mapping without writing files. */
function rebuildMapping(corpus: Map<string, CorpusEntry>): Map<string, string> {
  const mapping = new Map<string, string>();
  const usedNames = new Map<string, number>();

  for (const [id, entry] of corpus) {
    let name = sanitize(entry.title || id);
    const count = usedNames.get(name) ?? 0;
    usedNames.set(name, count + 1);
    if (count > 0) { name = `${name} (${count})`; }
    mapping.set(name, id);
  }

  return mapping;
}