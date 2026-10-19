from typing import Any, Dict, List, Optional
from polyvector.index import PolyVectorIndex

class SQLAlchemyAdapter:
    """
    SQLAlchemy & SQLModel Adapter for PolyVector.
    Allows running PolyVector searches seamlessly within SQLAlchemy sessions.
    """
    @staticmethod
    def search(
        index: PolyVectorIndex,
        session: Any,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ):
        from sqlalchemy import text
        sql, params = index.compile_query(query=query, filters=filters, limit=limit, param_style="named")
        result = session.execute(text(sql), params)
        return result.fetchall()
