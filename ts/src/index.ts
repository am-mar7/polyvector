import { IndexConfig, IndexOptions, SearchResponse, SearchResultItem } from "./core/models.js";
import { SQLCompiler, ParamStyle, CompiledQuery } from "./core/compiler.js";
import { BaseEmbeddingEngine } from "./embeddings/base.js";
import { LocalXenovaEmbedder } from "./embeddings/xenova.js";

export * from "./core/models.js";
export * from "./core/compiler.js";
export * from "./embeddings/base.js";
export * from "./embeddings/xenova.js";
export * from "./adapters/drizzle.js";
export * from "./adapters/prisma.js";

export interface SearchOptions {
  filters?: Record<string, any>;
  limit?: number;
}

export interface CompileOptions extends SearchOptions {
  paramStyle?: ParamStyle;
}

export class PolyVectorIndex {
  public config: IndexConfig;
  public embeddingEngine: BaseEmbeddingEngine;

  constructor(options: IndexOptions, embeddingEngine?: BaseEmbeddingEngine) {
    this.embeddingEngine = embeddingEngine || new LocalXenovaEmbedder();
    this.config = new IndexConfig(options, this.embeddingEngine.dimension);
  }

  /**
   * Executes DDL statements in a single multi-statement script.
   * Maintains strict dependency order (Extension -> Table -> Indexes)
   * while eliminating multiple sequential roundtrips.
   */
  public async setup(client: any): Promise<string[]> {
    const ddl = SQLCompiler.compileSetupDDL(this.config);
    const script = ddl.join("\n");
    await client.query(script);
    return ddl;
  }

  /**
   * Embeds records locally on CPU and writes them in multi-row parameterized batches.
   */
  public async index(
    records: Record<string, any>[],
    client: any,
    batchSize: number = 100
  ): Promise<{ count: number; vectorsCreated: number; elapsedMs: number }> {
    if (!records || records.length === 0) {
      return { count: 0, vectorsCreated: 0, elapsedMs: 0 };
    }

    const tStart = performance.now();
    let totalVectors = 0;

    for (const vf of this.config.vectorFields) {
      const texts: string[] = [];
      const validRecords: Record<string, any>[] = [];

      for (const r of records) {
        let val = "";
        if (vf.textExtractor) {
          val = vf.textExtractor(r);
        } else if (vf.name in r && r[vf.name] !== null && r[vf.name] !== undefined) {
          val = String(r[vf.name]);
        }
        if (val) {
          texts.push(val);
          validRecords.push(r);
        }
      }

      if (texts.length === 0) continue;

      // Local ONNX vectorized tensor batch inference
      const vectors = await this.embeddingEngine.embedBatch(texts);

      // Multi-row parameterized batch insert
      for (let i = 0; i < validRecords.length; i += batchSize) {
        const chunkRecords = validRecords.slice(i, i + batchSize);
        const chunkVectors = vectors.slice(i, i + batchSize);

        const valueClauses: string[] = [];
        const params: any[] = [];
        let paramIdx = 1;

        for (let j = 0; j < chunkRecords.length; j++) {
          const r = chunkRecords[j];
          const vec = chunkVectors[j];
          const pid = String(r[this.config.idColumn]);
          const vecStr = `[${vec.join(",")}]`;
          const meta: Record<string, any> = {};
          for (const ff of this.config.filterFields) {
            if (ff in r) meta[ff] = r[ff];
          }

          valueClauses.push(
            `($${paramIdx}, $${paramIdx + 1}, CAST($${paramIdx + 2} AS vector), CAST($${paramIdx + 3} AS jsonb))`
          );
          params.push(pid, vf.name, vecStr, JSON.stringify(meta));
          paramIdx += 4;
          totalVectors++;
        }

        const insertSql = `
          INSERT INTO ${this.config.embeddingsTable} (${this.config.parentIdColumn}, field_name, embedding, metadata)
          VALUES ${valueClauses.join(",\n")}
        `;
        await client.query(insertSql, params);
      }
    }

    const elapsed = performance.now() - tStart;
    return {
      count: records.length,
      vectorsCreated: totalVectors,
      elapsedMs: Math.round(elapsed * 100) / 100
    };
  }

  /**
   * Compiles the search query into parameterized SQL without executing it.
   * Ideal for Drizzle, Prisma, TypeORM, or custom database connections.
   */
  public async compileQuery(query: string, options: CompileOptions = {}): Promise<CompiledQuery> {
    const queryVec = await this.embeddingEngine.embedText(query);
    return SQLCompiler.compileSearchSQL(this.config, {
      query,
      queryEmbedding: queryVec,
      filters: options.filters,
      limit: options.limit,
      paramStyle: options.paramStyle || "named"
    });
  }

  /**
   * Executes hybrid search and transparently returns decoupled latency metrics.
   */
  public async search(query: string, options: SearchOptions = {}, client: any): Promise<SearchResponse> {
    const tEmbedStart = performance.now();
    const queryVec = await this.embeddingEngine.embedText(query);
    const embedMs = performance.now() - tEmbedStart;

    const compiled = SQLCompiler.compileSearchSQL(this.config, {
      query,
      queryEmbedding: queryVec,
      filters: options.filters,
      limit: options.limit,
      paramStyle: "positional" // standard for node-postgres ($1, $2)
    });

    const tDbStart = performance.now();
    const res = await client.query(compiled.sql, compiled.params);
    const dbMs = performance.now() - tDbStart;

    const items: SearchResultItem[] = res.rows.map((row: any) => ({
      id: String(row.id),
      score: parseFloat(row.score)
    }));

    return {
      items,
      count: items.length,
      embedLatencyMs: Math.round(embedMs * 100) / 100,
      dbLatencyMs: Math.round(dbMs * 100) / 100,
      totalLatencyMs: Math.round((embedMs + dbMs) * 100) / 100
    };
  }
}
