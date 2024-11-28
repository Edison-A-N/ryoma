from typing import Literal

from ryoma.core.embedding.base import BaseEmbedding
from ryoma.core.embedding.backend.fasttext import FastTextEmbedding


def create_embedding(
    embedding_type: Literal["fasttext"], model_path: str
) -> BaseEmbedding:
    """Create embedding model instance based on type.

    Args:
        embedding_type: Type of embedding model, currently supports 'fasttext'
        model_path: Path to the model file

    Returns:
        BaseEmbedding instance
    """
    if embedding_type.lower() == "fasttext":
        return FastTextEmbedding(model_path)
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")


__all__ = ["BaseEmbedding", "create_embedding"]
