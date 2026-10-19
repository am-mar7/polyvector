from enum import Enum
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field

class FieldType(str, Enum):
    VECTOR = "vector"
    BM25 = "bm25"
    FILTER = "filter"

@dataclass
class FieldConfig:
    name: str
    field_type: FieldType = FieldType.VECTOR
    weight: float = 1.0
    dimension: int = 384
    text_extractor: Optional[Callable[[Dict[str, Any]], str]] = None

@dataclass
class IndexConfig:
    table: str
    id_column: str = "id"
    embeddings_table: Optional[str] = None
    fields: List[FieldConfig] = field(default_factory=list)
    distinct_on: Optional[str] = None
    filter_fields: List[str] = field(default_factory=list)
    fts_column: str = "fts_doc"
    parent_id_column: Optional[str] = None
    popularity_column: Optional[str] = None
    default_cand_limit: int = 100

    def __post_init__(self):
        if self.embeddings_table is None:
            self.embeddings_table = f"{self.table}_embeddings"
        if self.parent_id_column is None:
            singular = self.table[:-1] if self.table.endswith("s") else self.table
            self.parent_id_column = f"{singular}_id"

    @property
    def vector_fields(self) -> List[FieldConfig]:
        return [f for f in self.fields if f.field_type == FieldType.VECTOR]

    @property
    def bm25_fields(self) -> List[FieldConfig]:
        return [f for f in self.fields if f.field_type == FieldType.BM25]

@dataclass
class SearchResultItem:
    id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SearchResponse:
    items: List[SearchResultItem]
    db_latency_ms: float
    embed_latency_ms: float
    total_latency_ms: float

    @property
    def count(self) -> int:
        return len(self.items)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "db_latency_ms": round(self.db_latency_ms, 2),
            "embed_latency_ms": round(self.embed_latency_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "items": [
                {
                    "id": it.id,
                    "score": round(it.score, 4),
                    "metadata": it.metadata
                }
                for it in self.items
            ]
        }
