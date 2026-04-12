import logging
from typing import List

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, CPU-friendly


class Embedder:
    """
    Wraps SentenceTransformer for batch text embedding.
    Model is loaded once on first instantiation.
    """

    def __init__(self):
        logger.info("embedder_loading model=%s", MODEL_NAME)
        self._model = SentenceTransformer(MODEL_NAME)
        logger.info("embedder_ready")

    def embed_batch(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """
        Encodes a list of texts into 384-dim float vectors.
        Returns List[List[float]] aligned to input order.
        """
        if not texts:
            return []

        logger.info("embed_batch texts=%d batch_size=%d", len(texts), batch_size)
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()
