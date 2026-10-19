import json
from typing import Dict, Any, List, Tuple, Optional
from polyvector.core.models import IndexConfig, FieldType, FieldConfig

class SQLCompiler:
    """
    Framework-agnostic and ORM-agnostic SQL Compiler for PolyVector.
    Compiles DDL schema setup and search queries into parameterized SQL.
    """

    @staticmethod
    def compile_setup_ddl(config: IndexConfig) -> List[str]:
        statements = [
            "CREATE EXTENSION IF NOT EXISTS vector;"
        ]

        # 1. Vertical Embeddings Table
        statements.append(f"""
            CREATE TABLE IF NOT EXISTS {config.embeddings_table} (
                id BIGSERIAL PRIMARY KEY,
                {config.parent_id_column} VARCHAR(128) NOT NULL,
                field_name VARCHAR(64) NOT NULL,
                embedding vector(384) NOT NULL,
                metadata JSONB DEFAULT '{{}}'::jsonb
            );
        """)

        # 2. Partial HNSW Indexes by field (Key Optimization)
        for vf in config.vector_fields:
            idx_name = f"idx_{config.table}_hnsw_{vf.name}"
            statements.append(f"""
                CREATE INDEX IF NOT EXISTS {idx_name} 
                ON {config.embeddings_table} USING hnsw (embedding vector_cosine_ops) 
                WHERE field_name = '{vf.name}';
            """)

        # 3. GIN index on metadata
        statements.append(f"""
            CREATE INDEX IF NOT EXISTS idx_{config.table}_meta 
            ON {config.embeddings_table} USING GIN (metadata);
        """)

        # 4. Composite B-tree index on (field_name, parent_id_column)
        statements.append(f"""
            CREATE INDEX IF NOT EXISTS idx_{config.table}_fld_parent 
            ON {config.embeddings_table} (field_name, {config.parent_id_column});
        """)

        return statements

    @staticmethod
    def compile_search_sql(
        config: IndexConfig,
        query: str,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        cand_limit: Optional[int] = None,
        param_style: str = "pyformat"
    ) -> Tuple[str, Dict[str, Any]]:
        cand_limit = cand_limit or max(config.default_cand_limit, limit * 2)
        vec_str = f"[{','.join(str(x) for x in query_embedding)}]"

        def _p(name: str) -> str:
            if param_style == "named":
                return f":{name}"
            return f"%({name})s"

        params: Dict[str, Any] = {
            "vec": vec_str,
            "q": query,
            "limit": limit,
            "cand_limit": cand_limit
        }

        # Separate metadata filters (tier 1) vs parent table filters (tier 2)
        tier1_meta = {}
        tier2_filters = {}
        if filters:
            for k, v in filters.items():
                if k in config.filter_fields:
                    tier2_filters[k] = v
                else:
                    tier1_meta[k] = v

        meta_filter_sql = ""
        if tier1_meta:
            params["meta"] = json.dumps(tier1_meta)
            meta_filter_sql = f"AND (metadata @> CAST({_p('meta')} AS jsonb))"

        tier2_filter_clauses = []
        for k, v in tier2_filters.items():
            param_key = f"filter_{k}"
            if isinstance(v, (int, float, str, bool)):
                tier2_filter_clauses.append(f"p.{k} = {_p(param_key)}")
                params[param_key] = v
            elif isinstance(v, dict):
                for op, val in v.items():
                    op_key = f"{param_key}_{op}"
                    if op in ("eq", "="):
                        tier2_filter_clauses.append(f"p.{k} = {_p(op_key)}")
                    elif op in ("lte", "<="):
                        tier2_filter_clauses.append(f"p.{k} <= {_p(op_key)}")
                    elif op in ("gte", ">="):
                        tier2_filter_clauses.append(f"p.{k} >= {_p(op_key)}")
                    elif op in ("lt", "<"):
                        tier2_filter_clauses.append(f"p.{k} < {_p(op_key)}")
                    elif op in ("gt", ">"):
                        tier2_filter_clauses.append(f"p.{k} > {_p(op_key)}")
                    params[op_key] = val

        tier2_filter_sql = (" AND " + " AND ".join(tier2_filter_clauses)) if tier2_filter_clauses else ""

        # Build Candidate Branches
        branch_sqls = []

        # Vector Branches
        for vf in config.vector_fields:
            branch_sqls.append(f"""
                (
                    SELECT {config.parent_id_column} AS entity_id, (1 - (embedding <=> CAST({_p('vec')} AS vector))) * {vf.weight:.4f} AS partial_score
                    FROM {config.embeddings_table}
                    WHERE field_name = '{vf.name}'
                    {meta_filter_sql}
                    ORDER BY embedding <=> CAST({_p('vec')} AS vector) ASC
                    LIMIT {_p('cand_limit')}
                )
            """)

        # BM25 Branches (FTS with ts_rank_cd flag 32 length normalization)
        for bf in config.bm25_fields:
            col = config.fts_column
            branch_sqls.append(f"""
                (
                    SELECT {config.id_column} AS entity_id, ts_rank_cd({col}, plainto_tsquery('english', {_p('q')}), 32) * {bf.weight:.4f} AS partial_score
                    FROM {config.table}
                    WHERE {col} @@ plainto_tsquery('english', {_p('q')})
                    ORDER BY partial_score DESC
                    LIMIT {_p('cand_limit')}
                )
            """)

        union_candidates = "\nUNION ALL\n".join(branch_sqls)

        pop_select = f", p.{config.popularity_column}" if config.popularity_column else ""
        pop_order = f", p.{config.popularity_column} DESC" if config.popularity_column else ""

        # Handle Distinct Deduplication (if specified)
        if config.distinct_on:
            selection_sql = f"""
                SELECT id, score FROM (
                    SELECT DISTINCT ON (p.{config.distinct_on}) p.{config.id_column} AS id, s.score{pop_select}
                    FROM scored s
                    JOIN {config.table} p ON p.{config.id_column} = s.entity_id
                    WHERE 1=1 {tier2_filter_sql}
                    ORDER BY p.{config.distinct_on}, s.score DESC{pop_order}
                ) sub
                ORDER BY score DESC{pop_order}
                LIMIT {_p('limit')};
            """
        else:
            selection_sql = f"""
                SELECT p.{config.id_column} AS id, s.score
                FROM scored s
                JOIN {config.table} p ON p.{config.id_column} = s.entity_id
                WHERE 1=1 {tier2_filter_sql}
                ORDER BY s.score DESC{pop_order}
                LIMIT {_p('limit')};
            """

        sql = f"""
            WITH candidates AS (
                {union_candidates}
            ),
            scored AS (
                SELECT entity_id, SUM(partial_score) AS score
                FROM candidates
                GROUP BY entity_id
                ORDER BY score DESC
                LIMIT {_p('cand_limit')}
            )
            {selection_sql}
        """

        return sql.strip(), params
