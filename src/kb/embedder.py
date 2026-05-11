from sentence_transformers import SentenceTransformer
import numpy as np
from loguru import logger


class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception:
            logger.warning("Fallback to dummy embedder")
            self.model = None

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.model is None:
            return np.zeros((len(texts), 384))  # dummy 384-dim

        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return embeddings
        except Exception as e:
            logger.error(f"Embedding lỗi: {e}")
            return np.zeros((len(texts), 384))

    def cosine_sim(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.dot(a, b.T) / (np.linalg.norm(a, axis=1)[:, np.newaxis] * np.linalg.norm(b, axis=1))