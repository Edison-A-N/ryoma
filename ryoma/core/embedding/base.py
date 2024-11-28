from abc import ABC, abstractmethod
from typing import List, Union


class BaseEmbedding(ABC):
    """Base class for text embedding models."""

    def __init__(self, **kwargs):
        """Initialize embedding model.

        Args:
            **kwargs: Keyword arguments for specific embedding models
        """

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Embed a single text into vector.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts into vectors.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors
        """
        pass

    async def embed(
        self, text: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """Unified interface for both single and batch embedding.

        Args:
            text: Single text string or list of texts

        Returns:
            Single embedding vector or list of vectors
        """
        if isinstance(text, str):
            return await self.embed_text(text)
        return await self.embed_batch(text)

    @abstractmethod
    async def close(self) -> None:
        """Cleanup resources."""
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
