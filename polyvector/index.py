import time
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from psycopg2.extras import execute_values

from polyvector.core.models import (
    IndexConfig, FieldConfig, FieldType, 
    SearchResponse, SearchResultItem
)
from polyvector.core.compiler import SQLCompiler
from polyvector.embeddings.base import BaseEmbeddingEngine
from polyvector.embeddings.local_onnx import LocalONNXEmbedder
from polyvector.pool import ConnectionPool

class PolyVectorIndex:
    """
    PolyVector Index: In-engine multi-field hybrid search for PostgreSQL.
    Combines vertical HNSW vectors with native length-normalized BM25 in a single query.
    """
    def __init__(
        self,
        table: str,
        id_column: str = "id",
        fields: Optional[Dict[str, Union[float, Dict[str, Any]]]] = None,
        distinct_on: Optional[str] = None,
        filter_fields: Optional[List[str]] = None,
        fts_column: str = "fts_doc",
        parent_id_column: Optional[str] = None,
        popularity_column: Optional[str] = None,
        embeddings_table: Optional[str] = None,
        embedding_engine: Optional[BaseEmbeddingEngine] = None,
        connection_pool: Optional[ConnectionPool] = None,
        db_config: Optional[Union[Dict[str, Any], str]] = None
    ):
        self.embedding_engine = embedding_engine or LocalONNXEmbedder()
        self.pool = connection_pool or (ConnectionPool(db_config) if db_config else None)

        # Parse field configurations
        parsed_fields: List[FieldConfig] = []
        if fields:
            for f_name, f_cfg in fields.items():
                if isinstance(f_cfg, (int, float)):
                    # Simple weight shorthand e.g. {"name": 0.45} defaults to vector
                    parsed_fields.append(FieldConfig(
                        name=f_name,
                        field_type=FieldType.VECTOR,
                        weight=float(f_cfg),
                        dimension=self.embedding_engine.dimension
                    ))
                elif isinstance(f_cfg, dict):
                    f_type = FieldType(f_cfg.get("type", "vector"))
                    f_weight = float(f_cfg.get("weight", 1.0))
                    f_extractor = f_cfg.get("extractor")
                    parsed_fields.append(FieldConfig(
                        name=f_name,
                        field_type=f_type,
                        weight=f_weight,
                        dimension=self.embedding_engine.dimension,
                        text_extractor=f_extractor
                    ))

        self.config = IndexConfig(
            table=table,
            id_column=id_column,
            embeddings_table=embeddings_table or f"{table}_embeddings",
            fields=parsed_fields,
            distinct_on=distinct_on,
            filter_fields=filter_fields or [],
            fts_column=fts_column,
            parent_id_column=parent_id_column,
            popularity_column=popularity_column
        )

    def setup(self, conn=None) -> List[str]:
        """
        Executes DDL to set up pgvector extension, vertical embeddings table,
        partial HNSW indexes by field, and metadata indexes.
        """
        statements = SQLCompiler.compile_setup_ddl(self.config)

        def _run_ddl(c):
            c.autocommit = True
            with c.cursor() as cur:
                for stmt in statements:
                    cur.execute(stmt)

        if conn is not None:
            _run_ddl(conn)
        elif self.pool:
            with self.pool.connection() as c:
                _run_ddl(c)
        else:
            raise ValueError("No database connection or connection_pool provided to setup().")

        return statements

    def index(
        self,
        records: List[Dict[str, Any]],
        batch_size: int = 500,
        conn=None
    ) -> Dict[str, Any]:
        """
        Ingests records, generates vertical embeddings for each vector field,
        and writes them into the vertical embeddings table.
        """
        if not records:
            return {"count": 0, "elapsed_ms": 0.0}

        t_start = time.perf_counter()

        # Prepare vertical embedding rows: (parent_id, field_name, embedding_vector_str, metadata_json)
        emb_values = []
        for vf in self.config.vector_fields:
            texts = []
            for r in records:
                if vf.text_extractor:
                    val = vf.text_extractor(r)
                else:
                    val = str(r.get(vf.name, ""))
                texts.append(val)

            # Batch embed on local ONNX
            vectors = self.embedding_engine.embed_batch(texts)

            for r, vec in zip(records, vectors):
                pid = str(r[self.config.id_column])
                vec_str = f"[{','.join(str(x) for x in vec)}]"
                meta = {k: r[k] for k in self.config.filter_fields if k in r}
                emb_values.append((pid, vf.name, vec_str, json.dumps(meta)))

        insert_sql = f"""
            INSERT INTO {self.config.embeddings_table} ({self.config.parent_id_column}, field_name, embedding, metadata)
            VALUES %s
        """

        def _run_insert(c):
            with c.cursor() as cur:
                execute_values(cur, insert_sql, emb_values, page_size=1000)
            if conn is None:
                c.commit()

        if conn is not None:
            _run_insert(conn)
        elif self.pool:
            with self.pool.connection() as c:
                _run_insert(c)
        else:
            raise ValueError("No database connection or connection_pool provided to index().")

        elapsed = (time.perf_counter() - t_start) * 1000.0
        return {"count": len(records), "vectors_created": len(emb_values), "elapsed_ms": round(elapsed, 2)}

    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        conn=None
    ) -> SearchResponse:
        """
        Executes hybrid search:
        1. Embeds query text locally on CPU (~15ms).
        2. Executes unified SQL CTE query in PostgreSQL.
        3. Returns SearchResponse with decoupled timings.
        """
        # 1. Embed Query Text
        t_embed_start = time.perf_counter()
        query_vec = self.embedding_engine.embed_text(query)
        embed_ms = (time.perf_counter() - t_embed_start) * 1000.0

        # 2. Compile SQL
        sql, params = SQLCompiler.compile_search_sql(
            config=self.config,
            query=query,
            query_embedding=query_vec,
            filters=filters,
            limit=limit
        )

        # 3. Execute in PostgreSQL
        def _run_search(c):
            with c.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall()

        t_db_start = time.perf_counter()
        if conn is not None:
            rows = _run_search(conn)
        elif self.pool:
            with self.pool.connection() as c:
                rows = _run_search(c)
        else:
            raise ValueError("No database connection or connection_pool provided to search().")

        db_ms = (time.perf_counter() - t_db_start) * 1000.0

        items = [
            SearchResultItem(
                id=str(r[0]),
                score=float(r[1])
            )
            for r in rows
        ]

        return SearchResponse(
            items=items,
            db_latency_ms=db_ms,
            embed_latency_ms=embed_ms,
            total_latency_ms=embed_ms + db_ms
        )

    def compile_query(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        param_style: str = "pyformat"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Compiles the query into raw SQL and parameters without executing it.
        Supports 'pyformat' (%(param)s) or 'named' (:param) for SQLAlchemy / Prisma / Drizzle.
        """
        query_vec = self.embedding_engine.embed_text(query)
        return SQLCompiler.compile_search_sql(
            config=self.config,
            query=query,
            query_embedding=query_vec,
            filters=filters,
            limit=limit,
            param_style=param_style
        )
