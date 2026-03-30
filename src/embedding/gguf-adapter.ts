import { preprocessEmbeddingText } from './preprocessing.js';
import type {
    EmbeddingAdapter,
    EmbeddingAdapterOptions,
    EmbeddingRequest,
} from './types.js';

export class GgufAdapter implements EmbeddingAdapter {
    private context: any; // lazy-loaded
    private _dimensions = 0;
    readonly modelId: string;
    private readonly ggufFile: string;
    readonly options: EmbeddingAdapterOptions;

    /**
     * @param modelId  HuggingFace repo, e.g. "jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF"
     * @param ggufFile Quantization variant, e.g. "Q4_K_M"
     */
    constructor(modelId: string, ggufFile: string, options: EmbeddingAdapterOptions = {}) {
        this.modelId = `${modelId}:${ggufFile}`;
        this.ggufFile = ggufFile;
        this.options = options;
    }

    get dimensions(): number {
        return this._dimensions;
    }

    async embed(text: string, request: EmbeddingRequest): Promise<Float32Array[]> {
        if (!this.context) {
            await this.init();
        }

        const formattedText = preprocessEmbeddingText(
            text,
            request.kind,
            this.modelId,
            this.options.preprocessor,
        );
        const tokens = this.context.model.tokenize(formattedText);
        const maxTokens = this.context._llamaContext.contextSize;

        if (tokens.length <= maxTokens) {
            const embedding = await this.context.getEmbeddingFor(tokens);
            const vec = new Float32Array(embedding.vector);
            this._dimensions = vec.length;
            return [vec];
        }

        const overlap = Math.floor(maxTokens * 0.1);
        const chunks: Float32Array[] = [];

        for (let start = 0; start < tokens.length; start += maxTokens - overlap) {
            const chunkTokens = tokens.slice(start, start + maxTokens);
            const embedding = await this.context.getEmbeddingFor(chunkTokens);
            const vec = new Float32Array(embedding.vector);
            this._dimensions = vec.length;
            chunks.push(vec);
        }
        return chunks;
    }

    private async init(): Promise<void> {
        // Dynamic import — optional dependency.
        const { getLlama, resolveModelFile } = await import('node-llama-cpp');
        const llama = await getLlama();

        const modelPath = await resolveModelFile(
            `hf:${this.modelId.split(':')[0]}:${this.ggufFile}`,
        );

        const model = await llama.loadModel({ modelPath });
        this.context = await model.createEmbeddingContext({
            contextSize: this.options.contextSize ?? "auto",
        });
    }
}
