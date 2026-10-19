import pytest
from polyvector import PolyVectorIndex, ConnectionPool, SQLAlchemyAdapter

DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "dbname": "search_benchmark",
    "user": "postgres",
    "password": "postgres"
}

@pytest.fixture(scope="module")
def shared_pool():
    pool = ConnectionPool(config=DB_CONFIG, minconn=2, maxconn=10)
    yield pool
    pool.close()

@pytest.fixture(scope="module")
def product_index(shared_pool):
    return PolyVectorIndex(
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
        connection_pool=shared_pool
    )

def test_polyvector_index_search(product_index):
    # Test Search for 'gym'
    resp = product_index.search(query="gym", limit=4)

    assert resp.count == 4
    assert resp.db_latency_ms > 0.0
    assert resp.embed_latency_ms > 0.0
    assert resp.total_latency_ms == pytest.approx(resp.db_latency_ms + resp.embed_latency_ms, rel=1e-2)

    for item in resp.items:
        assert item.id.startswith("prod_")
        assert item.score > 0.0

def test_polyvector_with_filters(product_index):
    # Search with relational & metadata filters
    resp = product_index.search(
        query="wireless headphones",
        filters={"price": {"lte": 150}},
        limit=5
    )
    assert resp.count > 0
    for item in resp.items:
        assert item.score > 0.0

def test_polyvector_compilation_styles(product_index):
    # Test pyformat (psycopg)
    sql_py, params_py = product_index.compile_query("running shoes", limit=5, param_style="pyformat")
    assert "%(vec)s" in sql_py
    assert "%(q)s" in sql_py
    assert "CAST(%(vec)s AS vector)" in sql_py

    # Test named (:param for SQLAlchemy/Prisma/Drizzle)
    sql_named, params_named = product_index.compile_query("running shoes", limit=5, param_style="named")
    assert ":vec" in sql_named
    assert ":q" in sql_named
    assert "CAST(:vec AS vector)" in sql_named
    assert "%(vec)s" not in sql_named

def test_polyvector_sqlalchemy_adapter(product_index):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    engine = create_engine("postgresql+psycopg2://postgres:postgres@localhost:5433/search_benchmark")
    with Session(engine) as session:
        results = SQLAlchemyAdapter.search(product_index, session, query="leather jacket", limit=3)
        assert len(results) == 3
        for r in results:
            assert str(r[0]).startswith("prod_")
            assert float(r[1]) > 0.0

def test_polyvector_batch_ingest(shared_pool):
    # Ingest test records inside a transaction that rolls back
    index = PolyVectorIndex(
        table="products",
        id_column="id",
        embeddings_table="product_embeddings",
        fields={
            "name": {"weight": 0.5, "type": "vector"},
            "description": {"weight": 0.5, "type": "vector"}
        },
        connection_pool=shared_pool
    )

    sample_records = [
        {
            "id": "test_prod_99901",
            "name": "Custom Titanium Road Bike",
            "description": "Ultra-lightweight aerodynamic racing bicycle",
            "category": "Cycling",
            "price": 1299.99
        },
        {
            "id": "test_prod_99902",
            "name": "Carbon Fiber Water Bottle Cage",
            "description": "Durable cage for road bikes",
            "category": "Cycling",
            "price": 29.99
        }
    ]

    with shared_pool.connection() as conn:
        with conn.cursor() as cur:
            # Insert parent products within transaction
            cur.execute("""
                INSERT INTO products (id, name, description, category, price, popularity)
                VALUES 
                ('test_prod_99901', 'Custom Titanium Road Bike', 'Ultra-lightweight aerodynamic racing bicycle', 'Cycling', 1299.99, 100),
                ('test_prod_99902', 'Carbon Fiber Water Bottle Cage', 'Durable cage for road bikes', 'Cycling', 29.99, 50);
            """)

        # Run index on transactional connection
        res = index.index(sample_records, conn=conn)
        assert res["count"] == 2
        assert res["vectors_created"] == 4  # 2 records * 2 vector fields

        # Verify rows were written
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM product_embeddings WHERE product_id LIKE 'test_prod_%';")
            count = cur.fetchone()[0]
            assert count == 4

        # Roll back transaction to leave database pristine
        conn.rollback()

    # Verify rollback cleaned it up
    with shared_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM product_embeddings WHERE product_id LIKE 'test_prod_%';")
            count = cur.fetchone()[0]
            assert count == 0
            cur.execute("SELECT COUNT(*) FROM products WHERE id LIKE 'test_prod_%';")
            assert cur.fetchone()[0] == 0
