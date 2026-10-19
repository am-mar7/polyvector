#!/usr/bin/env python3
"""
🚀 PolyVector Quickstart & Execution Trace Example
--------------------------------------------------
This script walks through every phase of the PolyVector lifecycle:
  [Step 1] Database Connection Pooling
  [Step 2] Index Definition & DDL Schema Compilation
  [Step 3] Local ONNX CPU Vectorization & Record Ingestion
  [Step 4] In-Engine Hybrid Search (Vectors + BM25 Fusion)
  [Step 5] Relational & Metadata Filtering
  [Step 6] ORM Interoperability (SQLAlchemy & Compiled Raw SQL)

Run this directly:
  python examples/quickstart_trace.py
"""

import sys
import time
from polyvector import PolyVectorIndex, ConnectionPool, SQLAlchemyAdapter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "dbname": "search_benchmark",
    "user": "postgres",
    "password": "postgres"
}

def print_banner(step_num: int, title: str):
    print("\n" + "=" * 70)
    print(f"📍 STEP {step_num}: {title}")
    print("=" * 70)

def main():
    print("""
    ================================================================
    ⚡ PolyVector: In-Engine Hybrid Search (PostgreSQL + Local ONNX)
    ================================================================
    """)

    # -----------------------------------------------------------------------
    # Step 1: Initialize Connection Pool
    # -----------------------------------------------------------------------
    print_banner(1, "Initialize Connection Pool")
    print("Creating thread-safe persistent connection pool...")
    pool = ConnectionPool(config=DB_CONFIG, minconn=2, maxconn=10)
    print("✅ Connection pool created (minconn=2, maxconn=10).")

    # -----------------------------------------------------------------------
    # Step 2: Define Index Configuration
    # -----------------------------------------------------------------------
    print_banner(2, "Define PolyVector Index & Compile DDL")
    index = PolyVectorIndex(
        table="products",
        id_column="id",
        embeddings_table="product_embeddings",
        fields={
            "name": {"weight": 0.45, "type": "vector"},
            "description": {"weight": 0.20, "type": "vector"},
            "fts_doc": {"weight": 0.35, "type": "bm25"}
        },
        distinct_on="name",
        filter_fields=["category", "brand", "availability", "price"],
        connection_pool=pool
    )
    print("Index configured with:")
    print(f"  - Primary Table:       {index.config.table}")
    print(f"  - Embeddings Table:    {index.config.embeddings_table}")
    print(f"  - Vector Fields:       {[f.name for f in index.config.vector_fields]}")
    print(f"  - BM25 Fields:         {[f.name for f in index.config.bm25_fields]}")
    print(f"  - Deduplication:       DISTINCT ON ({index.config.distinct_on})")
    print(f"  - Filter Fields:       {index.config.filter_fields}")

    # Inspect the DDL that setup() creates under the hood
    from polyvector.core.compiler import SQLCompiler
    ddl_statements = SQLCompiler.compile_setup_ddl(index.config)
    print("\nUnder the hood, index.setup() generates these optimized statements:")
    for stmt in ddl_statements:
        clean = " ".join(stmt.strip().split())
        print(f"  SQL> {clean[:95]}...")

    # -----------------------------------------------------------------------
    # Step 3: Record Ingestion (Local CPU ONNX Inference + Batch Insert)
    # -----------------------------------------------------------------------
    print_banner(3, "Ingest Records (Local CPU Embeddings)")
    demo_products = [
        {
            "id": "demo_shoe_001",
            "name": "Men's Waterproof Trail Runner 4",
            "description": "All-weather trail running shoes with Gore-Tex lining and rugged lug traction.",
            "category": "Footwear",
            "brand": "TrailPro",
            "price": 149.99,
            "availability": True,
            "popularity": 85
        },
        {
            "id": "demo_shoe_002",
            "name": "Women's Road Marathon Speed Pro",
            "description": "Ultra-lightweight carbon plate racing shoe engineered for marathon distances.",
            "category": "Footwear",
            "brand": "SpeedLab",
            "price": 199.99,
            "availability": True,
            "popularity": 92
        }
    ]

    print(f"Embedding {len(demo_products)} products locally on CPU (FastEmbed ONNX)...")
    with pool.connection() as conn:
        with conn.cursor() as cur:
            # Insert parent products in transaction
            for p in demo_products:
                cur.execute("""
                    INSERT INTO products (id, name, description, category, brand, price, availability, popularity)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING;
                """, (p["id"], p["name"], p["description"], p["category"], p["brand"], p["price"], p["availability"], p["popularity"]))

        # Call index.index()
        t0 = time.perf_counter()
        ingest_res = index.index(demo_products, conn=conn)
        conn.commit()
        t_ingest = (time.perf_counter() - t0) * 1000.0

    print(f"✅ Ingestion complete in {t_ingest:.2f}ms:")
    print(f"   - Products indexed: {ingest_res['count']}")
    print(f"   - Vertical vectors generated: {ingest_res['vectors_created']}")

    # -----------------------------------------------------------------------
    # Step 4: Hybrid Search (In-Engine Fusion)
    # -----------------------------------------------------------------------
    print_banner(4, "Execute In-Engine Hybrid Search")
    query = "trail runner waterproof"
    print(f"Searching for: '{query}' (limit=4)...")

    response = index.search(query=query, limit=4)

    print("\n📊 Decoupled Latency Profile:")
    print(f"  ├─ CPU Embed Time:      {response.embed_latency_ms:.2f} ms (local ONNX model)")
    print(f"  ├─ PostgreSQL Exec Time: {response.db_latency_ms:.2f} ms (partial HNSW + BM25 + fusion)")
    print(f"  └─ Total Latency:        {response.total_latency_ms:.2f} ms")

    print(f"\nTop {response.count} Results:")
    with pool.connection() as conn:
        with conn.cursor() as cur:
            for i, item in enumerate(response.items, 1):
                cur.execute("SELECT name, category, price FROM products WHERE id = %s;", (item.id,))
                row = cur.fetchone()
                name = row[0] if row else "Unknown"
                cat = row[1] if row else "N/A"
                price = row[2] if row else 0.0
                print(f"  {i}. [{item.score:.4f}] {name} | {cat} | ${price:.2f} (ID: {item.id})")

    # -----------------------------------------------------------------------
    # Step 5: Relational & Metadata Filtering
    # -----------------------------------------------------------------------
    print_banner(5, "Search with Filters (e.g. price <= $160)")
    filtered_resp = index.search(
        query=query,
        filters={"price": {"lte": 160.0}},
        limit=3
    )
    print(f"Results with price <= $160 (Total: {filtered_resp.total_latency_ms:.2f}ms):")
    with pool.connection() as conn:
        with conn.cursor() as cur:
            for item in filtered_resp.items:
                cur.execute("SELECT name, price FROM products WHERE id = %s;", (item.id,))
                row = cur.fetchone()
                if row:
                    print(f"  - [{item.score:.4f}] {row[0]} (${row[1]:.2f})")

    # -----------------------------------------------------------------------
    # Step 6: ORM Interoperability
    # -----------------------------------------------------------------------
    print_banner(6, "ORM Interoperability (SQLAlchemy & Raw Query Compilers)")
    
    # 6A. SQLAlchemy Session Adapter
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session
        engine = create_engine("postgresql+psycopg2://postgres:postgres@localhost:5433/search_benchmark")
        with Session(engine) as session:
            sql_rows = SQLAlchemyAdapter.search(index, session, query="gym training", limit=2)
            print("6A. Native SQLAlchemy Session Search:")
            for r in sql_rows:
                print(f"    SQLAlchemy Row: Product ID = {r[0]}, Score = {r[1]:.4f}")
    except ImportError:
        print("SQLAlchemy not installed, skipping direct session execution.")

    # 6B. Raw Query Compilation for Drizzle / Prisma / Django / asyncpg
    print("\n6B. Raw Parameterized SQL Compilation:")
    sql_named, params_named = index.compile_query("gym training", limit=2, param_style="named")
    print(f"  - Named parameters compiled (:vec, :q, :limit)")
    print(f"  - Vector dimension: {len(eval(params_named['vec']))}")
    print(f"  - Sample SQL snippet:")
    lines = sql_named.splitlines()
    for line in lines[:8]:
        print(f"      {line}")
    print(f"      ...")

    # Cleanup demo products
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM product_embeddings WHERE product_id LIKE 'demo_%';")
            cur.execute("DELETE FROM products WHERE id LIKE 'demo_%';")
        conn.commit()

    pool.close()
    print("\n" + "=" * 70)
    print("🎉 All steps executed and traced successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
