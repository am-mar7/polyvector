import { IndexConfig } from "./models.js";

export type ParamStyle = "named" | "positional" | "pyformat";

export interface CompileSearchOptions {
  query: string;
  queryEmbedding: number[];
  filters?: Record<string, any>;
  limit?: number;
  candLimit?: number;
  paramStyle?: ParamStyle;
}

export interface CompiledQuery {
  sql: string;
  params: Record<string, any> | any[];
}

export class SQLCompiler {
  /**
   * Compiles DDL statements for pgvector extension, vertical table, and partial HNSW indexes.
   */
  public static compileSetupDDL(config: IndexConfig): string[] {
    const statements: string[] = [
      "CREATE EXTENSION IF NOT EXISTS vector;"
    ];

    // 1. Vertical Embeddings Table
    statements.push(`
      CREATE TABLE IF NOT EXISTS ${config.embeddingsTable} (
        id BIGSERIAL PRIMARY KEY,
        ${config.parentIdColumn} VARCHAR(128) NOT NULL,
        field_name VARCHAR(64) NOT NULL,
        embedding vector(384) NOT NULL,
        metadata JSONB DEFAULT '{}'::jsonb
      );
    `);

    // 2. Partial HNSW Indexes by field (Key Optimization)
    for (const vf of config.vectorFields) {
      const idxName = `idx_${config.table}_hnsw_${vf.name}`;
      statements.push(`
        CREATE INDEX IF NOT EXISTS ${idxName}
        ON ${config.embeddingsTable} USING hnsw (embedding vector_cosine_ops)
        WHERE field_name = '${vf.name}';
      `);
    }

    // 3. GIN index on metadata
    statements.push(`
      CREATE INDEX IF NOT EXISTS idx_${config.table}_meta
      ON ${config.embeddingsTable} USING GIN (metadata);
    `);

    // 4. Composite B-tree index
    statements.push(`
      CREATE INDEX IF NOT EXISTS idx_${config.table}_fld_parent
      ON ${config.embeddingsTable} (field_name, ${config.parentIdColumn});
    `);

    return statements;
  }

  /**
   * Compiles the single-query hybrid search SQL and binds parameters.
   */
  public static compileSearchSQL(
    config: IndexConfig,
    options: CompileSearchOptions
  ): CompiledQuery {
    const limit = options.limit || 10;
    const candLimit = options.candLimit || Math.max(config.defaultCandLimit, limit * 2);
    const paramStyle = options.paramStyle || "named";
    const vecStr = `[${options.queryEmbedding.join(",")}]`;

    const namedParams: Record<string, any> = {
      vec: vecStr,
      q: options.query,
      limit: limit,
      cand_limit: candLimit
    };

    const positionalValues: any[] = [];
    const positionalParamMap: Map<string, number> = new Map();

    const getParamRef = (name: string, value?: any): string => {
      if (paramStyle === "named") {
        return `:${name}`;
      } else if (paramStyle === "pyformat") {
        return `%(${name})s`;
      } else {
        // Positional ($1, $2, ...)
        if (!positionalParamMap.has(name)) {
          positionalValues.push(value !== undefined ? value : namedParams[name]);
          positionalParamMap.set(name, positionalValues.length);
        }
        return `$${positionalParamMap.get(name)}`;
      }
    };

    // Separate Tier-1 (metadata) and Tier-2 (parent table) filters
    const tier1Meta: Record<string, any> = {};
    const tier2Filters: Record<string, any> = {};

    if (options.filters) {
      for (const [k, v] of Object.entries(options.filters)) {
        if (config.filterFields.includes(k)) {
          tier2Filters[k] = v;
        } else {
          tier1Meta[k] = v;
        }
      }
    }

    let metaFilterSql = "";
    if (Object.keys(tier1Meta).length > 0) {
      const metaJson = JSON.stringify(tier1Meta);
      namedParams["meta"] = metaJson;
      metaFilterSql = `AND (metadata @> CAST(${getParamRef("meta", metaJson)} AS jsonb))`;
    }

    const tier2Clauses: string[] = [];
    for (const [k, v] of Object.entries(tier2Filters)) {
      const paramKey = `filter_${k}`;
      if (typeof v === "object" && v !== null && !Array.isArray(v)) {
        for (const [op, val] of Object.entries(v)) {
          const opKey = `${paramKey}_${op}`;
          namedParams[opKey] = val;
          const ref = getParamRef(opKey, val);
          if (op === "eq" || op === "=") tier2Clauses.push(`p.${k} = ${ref}`);
          else if (op === "lte" || op === "<=") tier2Clauses.push(`p.${k} <= ${ref}`);
          else if (op === "gte" || op === ">=") tier2Clauses.push(`p.${k} >= ${ref}`);
          else if (op === "lt" || op === "<") tier2Clauses.push(`p.${k} < ${ref}`);
          else if (op === "gt" || op === ">") tier2Clauses.push(`p.${k} > ${ref}`);
        }
      } else {
        namedParams[paramKey] = v;
        tier2Clauses.push(`p.${k} = ${getParamRef(paramKey, v)}`);
      }
    }

    const tier2Sql = tier2Clauses.length > 0 ? ` AND ${tier2Clauses.join(" AND ")}` : "";

    // Candidate branches
    const branchSqls: string[] = [];

    // Vector Branches
    for (const vf of config.vectorFields) {
      const vecRef = getParamRef("vec", vecStr);
      const candLimitRef = getParamRef("cand_limit", candLimit);
      branchSqls.push(`
        (
          SELECT ${config.parentIdColumn} AS entity_id, (1 - (embedding <=> CAST(${vecRef} AS vector))) * ${vf.weight.toFixed(4)} AS partial_score
          FROM ${config.embeddingsTable}
          WHERE field_name = '${vf.name}'
          ${metaFilterSql}
          ORDER BY embedding <=> CAST(${vecRef} AS vector) ASC
          LIMIT ${candLimitRef}
        )
      `);
    }

    // BM25 Branches (FTS with ts_rank_cd flag 32 length normalization)
    for (const bf of config.bm25Fields) {
      const col = config.ftsColumn;
      const qRef = getParamRef("q", options.query);
      const candLimitRef = getParamRef("cand_limit", candLimit);
      branchSqls.push(`
        (
          SELECT ${config.idColumn} AS entity_id, ts_rank_cd(${col}, plainto_tsquery('english', ${qRef}), 32) * ${bf.weight.toFixed(4)} AS partial_score
          FROM ${config.table}
          WHERE ${col} @@ plainto_tsquery('english', ${qRef})
          ORDER BY partial_score DESC
          LIMIT ${candLimitRef}
        )
      `);
    }

    const unionCandidates = branchSqls.join("\nUNION ALL\n");

    const popSelect = config.popularityColumn ? `, p.${config.popularityColumn}` : "";
    const popOrder = config.popularityColumn ? `, p.${config.popularityColumn} DESC` : "";
    const limitRef = getParamRef("limit", limit);
    const scoredCandLimitRef = getParamRef("cand_limit", candLimit);

    let selectionSql = "";
    if (config.distinctOn) {
      selectionSql = `
        SELECT id, score FROM (
          SELECT DISTINCT ON (p.${config.distinctOn}) p.${config.idColumn} AS id, s.score${popSelect}
          FROM scored s
          JOIN ${config.table} p ON p.${config.idColumn} = s.entity_id
          WHERE 1=1 ${tier2Sql}
          ORDER BY p.${config.distinctOn}, s.score DESC${popOrder}
        ) sub
        ORDER BY score DESC${popOrder}
        LIMIT ${limitRef};
      `;
    } else {
      selectionSql = `
        SELECT p.${config.idColumn} AS id, s.score
        FROM scored s
        JOIN ${config.table} p ON p.${config.idColumn} = s.entity_id
        WHERE 1=1 ${tier2Sql}
        ORDER BY s.score DESC${popOrder}
        LIMIT ${limitRef};
      `;
    }

    const sql = `
      WITH candidates AS (
        ${unionCandidates}
      ),
      scored AS (
        SELECT entity_id, SUM(partial_score) AS score
        FROM candidates
        GROUP BY entity_id
        ORDER BY score DESC
        LIMIT ${scoredCandLimitRef}
      )
      ${selectionSql}
    `.trim();

    return {
      sql,
      params: paramStyle === "positional" ? positionalValues : namedParams
    };
  }
}
