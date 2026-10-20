import { BaseEmbeddingEngine } from "./base.js";

export class LocalXenovaEmbedder implements BaseEmbeddingEngine {
  public dimension: number;
  private modelName: string;
  private pipelinePromise: Promise<any> | null = null;

  constructor(modelName: string = "Xenova/bge-small-en-v1.5", dimension: number = 384) {
    this.modelName = modelName;
    this.dimension = dimension;
  }

  private async getPipeline(): Promise<any> {
    if (!this.pipelinePromise) {
      this.pipelinePromise = (async () => {
        // Dynamically import @xenova/transformers so it loads on-demand
        const { pipeline } = await import("@xenova/transformers");
        return await pipeline("feature-extraction", this.modelName, {
          quantized: true
        });
      })();
    }
    return this.pipelinePromise;
  }

  public async embedText(text: string): Promise<number[]> {
    const pipe = await this.getPipeline();
    const output = await pipe(text, { pooling: "mean", normalize: true });
    return Array.from(output.data);
  }

  public async embedBatch(texts: string[], batchSize: number = 64): Promise<number[][]> {
    if (texts.length === 0) return [];
    const pipe = await this.getPipeline();
    const results: number[][] = [];

    // Process in chunks to prevent memory spikes while leveraging native ONNX SIMD vectorization
    for (let i = 0; i < texts.length; i += batchSize) {
      const chunk = texts.slice(i, i + batchSize);
      const output = await pipe(chunk, { pooling: "mean", normalize: true });
      if (typeof output.tolist === "function") {
        results.push(...(output.tolist() as number[][]));
      } else {
        results.push(Array.from(output.data));
      }
    }
    return results;
  }
}
