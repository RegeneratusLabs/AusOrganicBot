import logging
from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL, EMBEDDING_FAMILY

logger = logging.getLogger("ausorganicbot")

MINICPM_QUERY_INSTRUCTION = "Query: "
MINICPM_TRUST_REMOTE_CODE = True


class EmbeddingService:
    """Service for generating embeddings using sentence-transformers.

    Supports two model families:
    - all-minilm: symmetric, no query instruction (default)
    - minicpm (OpenBMB MiniCPM-Embedding-Light): asymmetric — queries get a
      "Query: " instruction prefix, documents are encoded without one.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        import torch

        model_kwargs = {}
        if EMBEDDING_FAMILY == "minicpm" and torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16

        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            model_kwargs=model_kwargs,
        )
        self.dimension = self.model.get_embedding_dimension()
        self.family = EMBEDDING_FAMILY
        logger.info("Embedding model loaded: %s (family=%s, dim=%d)", model_name, self.family, self.dimension)

    def embed(self, text: str) -> list[float]:
        """Generate an embedding for a single query."""
        kwargs = {}
        if self.family == "minicpm":
            kwargs["prompt"] = MINICPM_QUERY_INSTRUCTION
        embedding = self.model.encode(text, convert_to_numpy=True, **kwargs)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for documents (no query instruction)."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
