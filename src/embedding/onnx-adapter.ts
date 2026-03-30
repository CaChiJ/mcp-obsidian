import { preprocessEmbeddingText } from './preprocessing.js';
import type {
    EmbeddingAdapter,
    EmbeddingAdapterOptions,
    EmbeddingRequest,
} from './types.js';

export class OnnxAdapter implements EmbeddingAdapter {
    private pipeline: any; // lazy-loaded
    private _dimensions = 0;
    readonly modelId: string;
    readonly options: EmbeddingAdapterOptions;

    constructor(modelId: string, options: EmbeddingAdapterOptions = {}) {
        this.modelId = modelId;
        this.options = options;
    }

    get dimensions(): number {
        return this._dimensions;
    }

    async embed(text: string, request: EmbeddingRequest): Promise<Float32Array[]> {
        if (!this.pipeline) {
            const { pipeline } = await import('@huggingface/transformers');
            this.pipeline = await pipeline('feature-extraction', this.modelId, {
                dtype: 'q4',
            });
        }

        const formattedText = preprocessEmbeddingText(
            text,
            request.kind,
            this.modelId,
            this.options.preprocessor,
        );
        const tokenIds: number[] = this.pipeline.tokenizer.encode(formattedText);
        const maxTokens: number = this.pipeline.tokenizer.model_max_length;

        if (tokenIds.length <= maxTokens) {
            const output = await this.pipeline(formattedText, { pooling: 'mean', normalize: true });
            const vec = output.data as Float32Array;
            this._dimensions = vec.length;
            return [vec];
        }

        const overlap = Math.floor(maxTokens * 0.1);
        const chunks: Float32Array[] = [];

        for (let start = 0; start < tokenIds.length; start += maxTokens - overlap) {
            const chunkIds = tokenIds.slice(start, start + maxTokens);
            const chunkText: string = this.pipeline.tokenizer.decode(chunkIds);
            const output = await this.pipeline(chunkText, { pooling: 'mean', normalize: true });
            const vec = output.data as Float32Array;
            this._dimensions = vec.length;
            chunks.push(vec);
        }
        return chunks;
    }
}
