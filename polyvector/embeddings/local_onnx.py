from typing import List
from polyvector.embeddings.base import BaseEmbeddingEngine

class LocalONNXEmbedder(BaseEmbeddingEngine):
    """
    High-performance local embedding engine running BAAI/bge-small-en-v1.5 (384 dimensions)
    via ONNX Runtime on CPU.
    
    - 100% offline (Zero API token / network calls)
    - Lightweight (~180 MB memory footprint)
    - Multi-threaded execution
    """
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", dimension: int = 384, threads: int = 8):
        super().__init__(dimension=dimension, model_name=model_name)
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name, threads=threads)

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * self.dimension
        # Pull single vector from generator stream
        emb = next(self.model.embed([text]))
        return emb.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embs = self.model.embed(texts)
        return [e.tolist() for e in embs]
