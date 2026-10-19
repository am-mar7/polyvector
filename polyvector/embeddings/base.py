from abc import ABC, abstractmethod
from typing import List

class BaseEmbeddingEngine(ABC):
    def __init__(self, dimension: int, model_name: str):
        self.dimension = dimension
        self.model_name = model_name

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Convert a single query string into a vector of floats."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Convert a batch of text strings into a list of vectors."""
        pass
