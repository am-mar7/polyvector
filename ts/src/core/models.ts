export type FieldType = "vector" | "bm25" | "filter";

export interface FieldConfig {
  name: string;
  fieldType: FieldType;
  weight: number;
  dimension?: number;
  textExtractor?: (record: Record<string, any>) => string;
}

export interface IndexOptions {
  table: string;
  idColumn?: string;
  embeddingsTable?: string;
  fields?: Record<string, number | { weight?: number; type?: FieldType; extractor?: (record: Record<string, any>) => string }>;
  distinctOn?: string;
  filterFields?: string[];
  ftsColumn?: string;
  parentIdColumn?: string;
  popularityColumn?: string;
  defaultCandLimit?: number;
}

export class IndexConfig {
  public table: string;
  public idColumn: string;
  public embeddingsTable: string;
  public fields: FieldConfig[];
  public distinctOn?: string;
  public filterFields: string[];
  public ftsColumn: string;
  public parentIdColumn: string;
  public popularityColumn?: string;
  public defaultCandLimit: number;

  constructor(options: IndexOptions, defaultDimension: number = 384) {
    this.table = options.table;
    this.idColumn = options.idColumn || "id";
    this.embeddingsTable = options.embeddingsTable || `${this.table}_embeddings`;
    this.distinctOn = options.distinctOn;
    this.filterFields = options.filterFields || [];
    this.ftsColumn = options.ftsColumn || "fts_doc";
    this.popularityColumn = options.popularityColumn;
    this.defaultCandLimit = options.defaultCandLimit || 100;

    if (options.parentIdColumn) {
      this.parentIdColumn = options.parentIdColumn;
    } else {
      const singular = this.table.endsWith("s") ? this.table.slice(0, -1) : this.table;
      this.parentIdColumn = `${singular}_id`;
    }

    this.fields = [];
    if (options.fields) {
      for (const [fName, fCfg] of Object.entries(options.fields)) {
        if (typeof fCfg === "number") {
          this.fields.push({
            name: fName,
            fieldType: "vector",
            weight: fCfg,
            dimension: defaultDimension
          });
        } else if (typeof fCfg === "object" && fCfg !== null) {
          this.fields.push({
            name: fName,
            fieldType: fCfg.type || "vector",
            weight: fCfg.weight !== undefined ? fCfg.weight : 1.0,
            dimension: defaultDimension,
            textExtractor: fCfg.extractor
          });
        }
      }
    }
  }

  public get vectorFields(): FieldConfig[] {
    return this.fields.filter(f => f.fieldType === "vector");
  }

  public get bm25Fields(): FieldConfig[] {
    return this.fields.filter(f => f.fieldType === "bm25");
  }
}

export interface SearchResultItem {
  id: string;
  score: number;
  metadata?: Record<string, any>;
}

export interface SearchResponse {
  items: SearchResultItem[];
  dbLatencyMs: number;
  embedLatencyMs: number;
  totalLatencyMs: number;
  count: number;
}
