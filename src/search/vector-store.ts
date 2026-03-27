import { readFile, writeFile, mkdir, stat } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import type { EmbeddingAdapter } from '../embedding/types.js';

export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i]! * b[i]!;
        normA += a[i]! * a[i]!;
        normB += b[i]! * b[i]!;
    }
    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom === 0 ? 0 : dot / denom;
}

function splitIntoChunks(text: string, maxChars: number, overlapChars: number): string[] {
    if (text.length <= maxChars) return [text];
    const chunks: string[] = [];
    let start = 0;
    while (start < text.length) {
        chunks.push(text.slice(start, start + maxChars));
        start += maxChars - overlapChars;
    }
    return chunks;
}

interface VectorEntry {
    path: string;
    mtime: number;
    vector: number[];
}

interface CacheFile {
    modelId: string;
    entries: VectorEntry[];
}

export interface VectorSearchResult {
    path: string;
    score: number;
}

export class VectorStore {
    private vectors = new Map<string, { mtime: number; vector: Float32Array }>();
    private readonly cachePath: string;
    private dirty = false;

    constructor(
        private readonly vaultPath: string,
        private readonly embedder: EmbeddingAdapter,
        private readonly chunkSizeChars?: number,
    ) {
        this.cachePath = join(vaultPath, '.mcpvault', 'embeddings.json');
    }

    /** Ensure a document is indexed. Reads file, strips frontmatter, embeds. Skips if up-to-date. */
    async index(relativePath: string): Promise<void> {
        const modifiedTime = await this.getFileModifiedTime(relativePath);

        const canonicalEntry =
            this.vectors.get(relativePath) ?? this.vectors.get(`${relativePath}#0`);
        if (canonicalEntry && canonicalEntry.mtime >= modifiedTime) return;

        // Remove stale chunks before re-indexing (handles chunk count changes)
        for (const key of [...this.vectors.keys()]) {
            if (key === relativePath || key.startsWith(`${relativePath}#`)) {
                this.vectors.delete(key);
            }
        }

        const fullPath = join(this.vaultPath, relativePath);
        const raw = await readFile(fullPath, 'utf-8');
        const frontmatterMatch = raw.match(/^---\n[\s\S]*?\n---\n/);
        const body = frontmatterMatch ? raw.slice(frontmatterMatch[0].length) : raw;

        const chunks = this.chunkSizeChars
            ? splitIntoChunks(body, this.chunkSizeChars, Math.floor(this.chunkSizeChars * 0.1))
            : [body];

        for (let i = 0; i < chunks.length; i++) {
            const key = chunks.length === 1 ? relativePath : `${relativePath}#${i}`;
            const vector = await this.embedder.embed(chunks[i]!, { kind: 'document' });
            this.vectors.set(key, { mtime: modifiedTime, vector });
        }
        this.dirty = true;
    }

    /** Search for top-K most similar documents. */
    async search(query: string, limit: number): Promise<VectorSearchResult[]> {
        const queryVec = await this.embedder.embed(query, { kind: 'query' });

        const bestByPath = new Map<string, number>();
        for (const [key, entry] of this.vectors) {
            const path = key.includes('#') ? key.slice(0, key.lastIndexOf('#')) : key;
            const score = cosineSimilarity(queryVec, entry.vector);
            if (!bestByPath.has(path) || score > bestByPath.get(path)!) {
                bestByPath.set(path, score);
            }
        }

        return [...bestByPath.entries()]
            .map(([path, score]) => ({ path, score }))
            .sort((a, b) => b.score - a.score)
            .slice(0, limit);
    }

    /** Remove a path from the index. */
    remove(relativePath: string): void {
        let removed = false;
        for (const key of [...this.vectors.keys()]) {
            if (key === relativePath || key.startsWith(`${relativePath}#`)) {
                this.vectors.delete(key);
                removed = true;
            }
        }
        if (removed) this.dirty = true;
    }

    get size(): number {
        const paths = new Set<string>();
        for (const key of this.vectors.keys()) {
            paths.add(key.includes('#') ? key.slice(0, key.lastIndexOf('#')) : key);
        }
        return paths.size;
    }

    // ---- Persistence ----

    async loadCache(): Promise<void> {
        try {
            const raw = await readFile(this.cachePath, 'utf-8');
            const cache: CacheFile = JSON.parse(raw);

            if (cache.modelId !== this.embedder.modelId) return; // model changed

            for (const entry of cache.entries) {
                this.vectors.set(entry.path, {
                    mtime: entry.mtime,
                    vector: new Float32Array(entry.vector),
                });
            }
        } catch {
            // no cache or parse error
        }
    }

    async saveCache(): Promise<void> {
        if (!this.dirty) { return; }

        const entries: VectorEntry[] = [];
        for (const [path, entry] of this.vectors) {
            entries.push({
                path,
                mtime: entry.mtime,
                vector: Array.from(entry.vector),
            });
        }

        const cache: CacheFile = { modelId: this.embedder.modelId, entries };
        await mkdir(dirname(this.cachePath), { recursive: true });
        await writeFile(this.cachePath, JSON.stringify(cache));
        this.dirty = false;
    }

    private async getFileModifiedTime(relativePath: string): Promise<number> {
        try {
            const s = await stat(join(this.vaultPath, relativePath));
            return s.mtimeMs;
        } catch {
            return 0;
        }
    }
}
