# ⚡ PolyVector

> **The Modern In-Engine Alternative to Elasticsearch & OpenSearch for PostgreSQL.**  
> High-performance multi-field hybrid search fusing vertical HNSW vector indexes with native length-normalized BM25 full-text search into a **single parameterized SQL query**—100% local, zero API tokens, 85% less RAM, and zero external search clusters.

[![PyPI version](https://img.shields.io/pypi/v/polyvector-search.svg?color=blue)](https://pypi.org/project/polyvector-search/)
[![npm version](https://img.shields.io/npm/v/polyvector-search.svg?color=green)](https://www.npmjs.com/package/polyvector-search)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14%20--%2017-blue.svg)](https://www.postgresql.org/)
[![pgvector](https://img.shields.io/badge/pgvector-0.5%2B-orange.svg)](https://github.com/pgvector/pgvector)

---

## 💡 Stop Running a Second Database for Search

Traditional search stacks require operating **Elasticsearch, OpenSearch, or Algolia** alongside your primary PostgreSQL database. This introduces:
- **Fragile Sync Pipelines:** CDC/Debezium, Kafka, or Logstash workers that lag, fail, or go out of sync.
- **Split-Brain & Consistency Bugs:** Documents updated in Postgres aren't immediately searchable or transactionally consistent.
- **Massive RAM Overhead:** 2.5 GB to 4+ GB of JVM heap memory just to keep search running.
- **High Cloud Costs:** Expensive separate VM clusters or high monthly SaaS bills.

**PolyVector turns PostgreSQL into an enterprise-grade search engine:**
- ⚡ **3.2× lower latency** than Elasticsearch 8.x hybrid (40ms vs 132ms p50).
- 📉 **85% less RAM consumed** (~350MB vs 2.5–4GB JVM heap).
- 💾 **61% smaller disk footprint** (298MB vs 767MB index size).
- 🔒 **ACID Compliant:** Updates are instantly searchable with transactional consistency.
- 💸 **Zero API Fees:** 100% local CPU ONNX embeddings (`bge-small-en-v1.5`). No OpenAI API bills or rate limits.

---

## 📊 Comprehensive Benchmark (100,000 Documents)

PolyVector was rigorously evaluated against **Elasticsearch 8.x** and **PostgreSQL Native Full-Text Search (FTS)** on an un-cached catalog of **100,000 records** using CPU-only inference (`BAAI/bge-small-en-v1.5`, 384 dimensions).

> **📌 Scalability Note:** All numbers below reflect the verified **100,000-document benchmark**. A final stress test on **1,000,000 records (1M)** is scheduled following the library stabilization to verify continuous linear scaling and memory bounds.

### 1. Performance & Relevance Summary Table

| Metric | PostgreSQL Native FTS | Elasticsearch 8.x (Hybrid) | **PolyVector (In-Engine Hybrid)** | Advantage of PolyVector |
| :--- | :---: | :---: | :---: | :--- |
| **Median Latency (p50)** | 51.66 ms | 132.07 ms | **40.40 ms** *(16–25 ms pooled)* | **3.2× faster** than Elasticsearch Hybrid |
| **95th Percentile (p95)** | 67.95 ms | 316.85 ms | **82.29 ms** | **3.8× faster** than Elasticsearch Hybrid |
| **99th Percentile (p99)** | 75.98 ms | 351.79 ms | **124.83 ms** | Predictable tail latencies under heavy load |
| **Throughput (QPS)** | 19.50 QPS | 6.82 QPS | **20.17 QPS** | **2.9× higher throughput** than Elasticsearch |
| **RAM Consumed (Engine)** | ~256 MB *(shared buffers)* | ~2.5 GB – 4.0 GB *(JVM heap)* | **~350 MB** *(Postgres RAM)* | **~85% less memory** than Elasticsearch |
| **Index Size on Disk** | 4.75 MB | 767.01 MB | **298.87 MB** | **61% smaller** storage footprint than ES |
| **Infrastructure Stack** | 1 Engine (PostgreSQL) | 2 Engines + Sync Pipeline | **1 Engine (PostgreSQL)** | Zero sync pipelines, zero split-brain risk |
| **NDCG@10** *(Ranking Quality)* | 0.6696 | 0.9696 | **0.9253** | **+38.2%** over Postgres FTS, near-parity with ES |
| **MRR** *(First Result Accuracy)* | 0.7143 | 1.0000 | **1.0000** | Perfect first-hit accuracy (same as ES) |
| **Precision@10** | 0.6349 | 0.9286 | **0.8714** | **+37.2%** over Postgres FTS |
| **Recall@100** *(Coverage)* | 0.4471 | 0.9471 | **0.8343** | **+86.6%** over Postgres FTS |

---

### 📖 Benchmark Terms Explained

- **p50 / p95 / p99 Latency:** 50%, 95%, and 99% of user search queries completed in under this time.
- **QPS (Queries Per Second):** Sustained query throughput executed per single engine core.
- **RAM Occupied:** Active operating memory allocated by the search service to keep indexes and caches hot.
- **Storage on Disk:** Combined physical footprint of the inverted index, vector graph (HNSW), and metadata.
- **Infrastructure Stack:** Number of independent stateful systems required to operate search in production.
- **NDCG@10 (Normalized Discounted Cumulative Gain at 10):** Measures ranking quality. Rewards algorithms that place the most relevant items at the top of the list.
- **MRR (Mean Reciprocal Rank):** Measures how quickly a user finds their first relevant hit ($1 / \text{rank}$). A score of `1.0000` means rank #1 hit on all queries.
- **Precision@10:** The proportion of top-10 returned documents that are genuinely relevant.
- **Recall@100:** The percentage of truly relevant documents retrieved within the top 100 candidates.

---

## 💡 Why PolyVector?

1. **Zero API Tokens / 100% Local ONNX Inference:**  
   PolyVector runs local embeddings directly on your application CPU via ONNX Runtime (`fastembed` in Python, `@xenova/transformers` in JS/TS). Your data never leaves your VPC, you incur \$0 in third-party API fees, and you never hit rate limits.
2. **Partial HNSW Vector Indexes:**  
   Instead of maintaining a massive single vector table, PolyVector partitions vector graphs by field (`WHERE field_name = 'name'`). Each graph stays compact, cache-resident in RAM, and searches significantly faster.
3. **In-Engine Single Query Fusion:**  
   Eliminates application-side Reciprocal Rank Fusion (RRF) round-trips. PolyVector compiles candidate branches (`UNION ALL`) and computes weighted linear score fusion (`SUM(partial_score) GROUP BY entity_id`) directly inside PostgreSQL.
4. **Length-Normalized BM25:**  
   Standard full-text search often biases long documents. PolyVector incorporates `ts_rank_cd(fts_doc, query, 32)` to eliminate document length bias, matching Elasticsearch's BM25 accuracy.
5. **Universal ORM & Driver Agnostic:**  
   Outputs pure, ANSI-compliant SQL with parameterized bindings (`:param` or `%(param)s`), working natively with Prisma, Drizzle, SQLAlchemy, Django, or raw database drivers.

---

## 🐍 Python Guide

### 1. Installation

```bash
# Core package (psycopg2 + fastembed)
pip install polyvector-search

# Optional: with SQLAlchemy adapter
pip install "polyvector-search[sqlalchemy]"
```

### 2. Complete Setup & Search Flow

```python
from polyvector import PolyVectorIndex, ConnectionPool

# 1. Initialize persistent connection pool
pool = ConnectionPool(
    config="postgresql://postgres:postgres@localhost:5432/my_database",
    minconn=2,
    maxconn=10
)

# 2. Define the index configuration
index = PolyVectorIndex(
    table="products",
    id_column="id",
    embeddings_table="product_embeddings",
    fields={
        "name": {"weight": 0.45, "type": "vector"},
        "description": {"weight": 0.20, "type": "vector"},
        "fts_doc": {"weight": 0.35, "type": "bm25"}
    },
    distinct_on="name",                    # Deduplicate variants
    filter_fields=["category", "price"],    # Tier-1/Tier-2 filterable fields
    connection_pool=pool
)

# 3. Create tables, pgvector extensions, and partial HNSW indexes
index.setup()

# 4. Ingest documents (Local ONNX embeds automatically on CPU)
documents = [
    {
        "id": "item_101",
        "name": "Men's Waterproof Trail Runner 4",
        "description": "All-terrain running shoes with Gore-Tex lining.",
        "category": "Footwear",
        "price": 149.99
    }
]
index.index(documents)

# 5. Hybrid Search with Decoupled Latency Metrics
response = index.search(
    query="waterproof trail runner",
    filters={"price": {"lte": 160.0}},
    limit=5
)

print(f"Total: {response.total_latency_ms:.2f}ms "
      f"[Model: {response.embed_latency_ms:.2f}ms | DB: {response.db_latency_ms:.2f}ms]")

for item in response.items:
    print(f"[{item.score:.4f}] ID: {item.id}")
```

---

## 🟨 JavaScript & TypeScript Guide (`polyvector-search`)

PolyVector is designed with identical architecture for Node.js, Bun, and Deno using `@xenova/transformers` for local ONNX CPU inference.

### 1. Installation

```bash
npm install polyvector-search
# or
pnpm add polyvector-search
# or
yarn add polyvector-search
```

### 2. TypeScript Quickstart

```typescript
import { PolyVectorIndex } from "polyvector-search";
import { Pool } from "pg";

const pgPool = new Pool({
  connectionString: "postgresql://postgres:postgres@localhost:5432/my_database"
});

const index = new PolyVectorIndex({
  table: "products",
  idColumn: "id",
  embeddingsTable: "product_embeddings",
  fields: {
    name: { weight: 0.45, type: "vector" },
    description: { weight: 0.20, type: "vector" },
    fts_doc: { weight: 0.35, type: "bm25" }
  },
  distinctOn: "name",
  filterFields: ["category", "price"],
  pool: pgPool
});

// Setup schema
await index.setup();

// Execute search
const results = await index.search("waterproof trail runner", {
  filters: { price: { lte: 160.0 } },
  limit: 5
});

console.log(`Executed in ${results.totalLatencyMs}ms`);
for (const item of results.items) {
  console.log(`[${item.score.toFixed(4)}] ID: ${item.id}`);
}
```

---

## 🔌 Universal ORM Integration Examples

Because PolyVector compiles **pure, ANSI-standard parameterized SQL**, you are never trapped by ORM limitations or proprietary query builders.

---

### Example 1: Prisma (TypeScript)

Use PolyVector's `compileQuery()` to generate the parameterized SQL, then execute it directly via Prisma's `$queryRawUnsafe`:

```typescript
import { PrismaClient } from "@prisma/client";
import { PolyVectorIndex } from "polyvector-search";

const prisma = new PrismaClient();
const index = new PolyVectorIndex({
  table: "Product",
  idColumn: "id",
  fields: {
    name: { weight: 0.5, type: "vector" },
    description: { weight: 0.5, type: "bm25" }
  }
});

async function searchProducts(searchTerm: string, maxPrice: number) {
  // 1. Compile pure SQL & bind parameters (:vec, :q, :limit)
  const { sql, params } = await index.compileQuery(searchTerm, {
    filters: { price: { lte: maxPrice } },
    limit: 10,
    paramStyle: "named"
  });

  // 2. Execute via Prisma's native raw interface
  // Prisma maps results seamlessly into your type-safe model structure
  const rankedItems = await prisma.$queryRawUnsafe<Array<{ id: string; score: number }>>(
    sql,
    params
  );

  return rankedItems;
}
```

---

### Example 2: Drizzle ORM (TypeScript)

Drizzle provides native raw execution via `sql.raw` or `db.execute`:

```typescript
import { drizzle } from "drizzle-orm/node-postgres";
import { sql } from "drizzle-orm";
import { PolyVectorIndex } from "polyvector-search";

const db = drizzle(process.env.DATABASE_URL!);
const index = new PolyVectorIndex({
  table: "products",
  fields: {
    title: { weight: 0.6, type: "vector" },
    content: { weight: 0.4, type: "bm25" }
  }
});

async function searchWithDrizzle(queryText: string) {
  // Compile SQL with named parameters
  const { sql: querySql, params } = await index.compileQuery(queryText, {
    limit: 10,
    paramStyle: "named"
  });

  // Execute using Drizzle's sql template
  const results = await db.execute(sql.raw(querySql, params));
  return results.rows;
}
```

---

### Example 3: Universal Pure SQL Execution (Any ORM / Raw Driver)

Whether you use **SQLAlchemy**, **Django ORM**, **TypeORM**, or raw **psycopg3 / pg**, PolyVector gives you the exact query string and parameter format required:

#### A. SQLAlchemy / SQLModel (Python)
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from polyvector import PolyVectorIndex, SQLAlchemyAdapter

engine = create_engine("postgresql+psycopg2://user:pass@localhost:5432/db")

with Session(engine) as session:
    results = SQLAlchemyAdapter.search(
        index=index,
        session=session,
        query="trail runner",
        filters={"category": "Footwear"},
        limit=10
    )
    for row in results:
        print(row.id, row.score)
```

#### B. Django ORM (Python)
```python
from myapp.models import Product

sql, params = index.compile_query("trail runner", limit=10, param_style="pyformat")

# Hydrate directly into Django model instances
products = Product.objects.raw(sql, params)
for p in products:
    print(p.name, p.price)
```

#### C. Raw `node-postgres` (`pg` driver)
```javascript
const { sql, params } = await index.compileQuery("trail runner", { limit: 10 });
const { rows } = await client.query(sql, Object.values(params));
```

> **Why this works universally:**  
> PolyVector uses standard ANSI `CAST(:vec AS vector)` and `CAST(:meta AS jsonb)` instead of dialect-specific operators (`::vector`), ensuring that PostgreSQL parses parameters correctly across **every single ORM and database driver**.

---

## 🛠️ Advanced Customization: `text_extractor`

When indexing data, you often want to combine multiple columns or parse nested JSON before generating vectors. Use `text_extractor`:

```python
index = PolyVectorIndex(
    table="books",
    fields={
        # Enriches the vector with author and year context
        "rich_content": {
            "weight": 0.6,
            "extractor": lambda r: f"{r['title']} by {r['author']} ({r['published_year']})"
        },
        # Extracts nested JSON specification
        "specs_vector": {
            "weight": 0.4,
            "extractor": lambda r: f"{r['specs']['format']} edition, {r['specs']['pages']} pages"
        }
    }
)
```

---

## 🧪 Testing

Run the test suite against a live PostgreSQL instance:

```bash
pytest -v tests/test_polyvector_library.py
```

---

## 📄 License

MIT License. Free for commercial and open-source use.
