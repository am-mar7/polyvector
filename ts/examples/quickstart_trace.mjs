import { PolyVectorIndex, SQLCompiler, LocalXenovaEmbedder } from "../dist/index.js";
import pg from "pg";

const { Client } = pg;

async function runTests() {
  console.log("=================================================");
  console.log("🧪 Testing @polyvector/core (JavaScript / TypeScript)");
  console.log("=================================================");

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
    distinctOn: "name",
    filterFields: ["category", "price"]
  });

  // 2. Test DDL Compilation
  console.log("\n[Test 1] Testing DDL Compilation...");
  const ddl = SQLCompiler.compileSetupDDL(index.config);
  if (ddl.length >= 4 && ddl[1].includes("CREATE TABLE IF NOT EXISTS product_embeddings")) {
    console.log("✅ DDL compilation passed:", ddl.length, "statements generated.");
  } else {
    throw new Error("DDL compilation failed");
  }

  // 3. Test Local CPU Inference via Xenova
  console.log("\n[Test 2] Testing Local ONNX CPU Embedding (Zero API Tokens)...");
  const t0 = performance.now();
  const vector = await index.embeddingEngine.embedText("trail running shoes");
  const embedMs = performance.now() - t0;
  console.log(`✅ Embedding generated in ${embedMs.toFixed(2)}ms!`);
  console.log(`   - Vector dimension: ${vector.length} (expected 384)`);
  if (vector.length !== 384) throw new Error("Incorrect vector dimension");

  // 4. Test Query Compilation (Named for Prisma/Drizzle & Positional for pg)
  console.log("\n[Test 3] Testing Query Compilation...");
  const compiledNamed = await index.compileQuery("trail running shoes", {
    filters: { price: { lte: 150 } },
    limit: 5,
    paramStyle: "named"
  });
  if (compiledNamed.sql.includes("CAST(:vec AS vector)") && compiledNamed.sql.includes("DISTINCT ON (p.name)")) {
    console.log("✅ Named query compilation passed (:vec, :q).");
  } else {
    throw new Error("Named query compilation failed");
  }

  const compiledPos = await index.compileQuery("trail running shoes", {
    filters: { price: { lte: 150 } },
    limit: 5,
    paramStyle: "positional"
  });
  if (compiledPos.sql.includes("CAST($") && Array.isArray(compiledPos.params)) {
    console.log(`✅ Positional query compilation passed ($1..$${compiledPos.params.length}).`);
  } else {
    throw new Error("Positional query compilation failed");
  }

  // 5. Test Live Search Against PostgreSQL 100K Catalog
  console.log("\n[Test 4] Testing Live Search via node-postgres against 100k catalog...");
  const client = new Client({
    host: "localhost",
    port: 5433,
    database: "search_benchmark",
    user: "postgres",
    password: "postgres"
  });

  try {
    await client.connect();
    const searchRes = await index.search("waterproof trail runner", {
      filters: { price: { lte: 160 } },
      limit: 4
    }, client);

    console.log("✅ Live Search Execution Succeeded!");
    console.log(`   ├─ CPU Model Inference: ${searchRes.embedLatencyMs} ms`);
    console.log(`   ├─ PostgreSQL Engine:    ${searchRes.dbLatencyMs} ms`);
    console.log(`   └─ Total Search Latency: ${searchRes.totalLatencyMs} ms`);
    console.log(`\n   Top ${searchRes.count} Results:`);
    for (const item of searchRes.items) {
      console.log(`     - [${item.score.toFixed(4)}] ID: ${item.id}`);
    }
  } catch (err) {
    console.error("Database search error:", err);
    throw err;
  } finally {
    await client.end();
  }

  console.log("\n=================================================");
  console.log("🎉 All TypeScript / JavaScript tests passed successfully!");
  console.log("=================================================");
}

runTests().catch(err => {
  console.error("Test execution failed:", err);
  process.exit(1);
});
