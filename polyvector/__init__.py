"""
PolyVector: In-Engine Multi-Field Hybrid Search for PostgreSQL.
Combines vertical HNSW vectors and native length-normalized BM25 in a single query.
"""

from polyvector.index import PolyVectorIndex
from polyvector.core.models import (
    IndexConfig, FieldConfig, FieldType,
    SearchResponse, SearchResultItem
)
from polyvector.embeddings.base import BaseEmbeddingEngine
from polyvector.embeddings.local_onnx import LocalONNXEmbedder
from polyvector.pool import ConnectionPool
from polyvector.adapters.sqlalchemy import SQLAlchemyAdapter

__all__ = [
    "PolyVectorIndex",
    "IndexConfig",
    "FieldConfig",
    "FieldType",
    "SearchResponse",
    "SearchResultItem",
    "BaseEmbeddingEngine",
    "LocalONNXEmbedder",
    "ConnectionPool",
    "SQLAlchemyAdapter"
]

__version__ = "0.1.0"
