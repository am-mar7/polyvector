# ⚡ polyvector/core (JavaScript & TypeScript)

> **High-Performance In-Engine Hybrid Search for PostgreSQL.**  
> Fuses vertical HNSW vectors and length-normalized BM25 full-text search into a **single, unified SQL query**—100% local CPU embeddings with `@xenova/transformers`, **zero API tokens**, and universal ORM compatibility.

---

## 📦 Installation

```bash
npm install polyvector/core
# or
pnpm add polyvector/core
```

---

## 🚀 Quickstart

```typescript
import { PolyVectorIndex } from "polyvector/core";
import pg from "pg";

const client = new pg.Client({
  connectionString: "postgresql://postgres:postgres@localhost:5432/my_database"
});
await client.connect();

// 1. Configure the Index
const index = new PolyVectorIndex({
  table: "products",
  idColumn: "id",
  embeddingsTable: "product_embeddings",
  fields: {
    name: { weight: 0.45, type: "vector" },
    description: { weight: 0.20, type: "vector" },
    fts_doc: { weight: 0.35, type: "bm25" }
  },
  distinctOn: "name",                    // Deduplicate identical items
  filterFields: ["category", "price"]    // Filterable columns
});

// 2. Setup PostgreSQL Schema (Extensions, Vertical Tables, Partial HNSW Indexes)
await index.setup(client);

// 3. Search with Decoupled Latency Profiling
const response = await index.search("waterproof trail runner", {
  filters: { price: { lte: 160 } },
  limit: 5
}, client);

console.log(`Executed in ${response.totalLatencyMs}ms:`);
console.log(`  ├─ CPU Model Inference: ${response.embedLatencyMs} ms`);
console.log(`  └─ PostgreSQL Engine:    ${response.dbLatencyMs} ms`);

for (const item of response.items) {
  console.log(`[${item.score.toFixed(4)}] ID: ${item.id}`);
}
```

---

## 🔌 ORM Adapters

### 1. Drizzle ORM

```typescript
import { drizzle } from "drizzle-orm/node-postgres";
import { PolyVectorIndex, DrizzleAdapter } from "polyvector/core";

const db = drizzle(process.env.DATABASE_URL!);
const index = new PolyVectorIndex({ table: "products" });

const results = await DrizzleAdapter.search(index, db, "running shoes", {
  filters: { category: "Footwear" },
  limit: 10
});
```

### 2. Prisma Client

```typescript
import { PrismaClient } from "@prisma/client";
import { PolyVectorIndex, PrismaAdapter } from "polyvector/core";

const prisma = new PrismaClient();
const index = new PolyVectorIndex({ table: "Product" });

const rankedItems = await PrismaAdapter.search(index, prisma, "running shoes", {
  filters: { price: { lte: 100 } },
  limit: 10
});
```

### 3. Raw Query Compilation (Any SQL Driver)

```typescript
// Compiles pure parameterized SQL (:vec, :q, :limit or $1, $2, ...)
const { sql, params } = await index.compileQuery("running shoes", {
  paramStyle: "named", // or "positional"
  limit: 10
});
```

---

## 📄 License

MIT License. Free for commercial and open-source use.
