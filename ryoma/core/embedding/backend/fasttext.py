import fasttext
from typing import List
from ryoma.core.embedding.base import BaseEmbedding


class FastTextEmbedding(BaseEmbedding):
    """FastText embedding model implementation."""

    def __init__(self, model_path: str):
        """Initialize FastText model.

        Args:
            model_path: Path to the pretrained FastText model file
        """
        self.model = fasttext.load_model(model_path)
        # FastText models typically have 300 dimensions
        super().__init__(model_name="fasttext", dimension=300)

    async def embed_text(self, text: str) -> List[float]:
        """Embed single text using FastText.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats
        """
        # Get sentence vector directly from FastText
        return self.model.get_sentence_vector(text).tolist()

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts using FastText.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors
        """
        return [self.model.get_sentence_vector(text).tolist() for text in texts]

    async def close(self) -> None:
        """Cleanup resources - nothing to cleanup for FastText."""
        pass
