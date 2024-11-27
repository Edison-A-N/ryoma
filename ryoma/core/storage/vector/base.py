from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class VectorDatabase(ABC):
    @abstractmethod
    def __init__(self, db_type: str = "inmemory") -> None:
        """Initialize database connection"""
        pass

    @abstractmethod
    def insert(
        self, vectors: List[List[float]], metadata: List[Dict[str, Any]]
    ) -> List[str]:
        """Insert vectors and metadata

        Args:
            vectors: List of vectors to insert
            metadata: List of metadata dicts corresponding to vectors

        Returns:
            List of vector IDs for inserted vectors
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        distance_metric: str = "cosine",
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for most similar vectors

        Args:
            query_vector: Vector to search for
            top_k: Number of results to return
            distance_metric: Distance metric to use
            metadata_filter: Optional filter on metadata

        Returns:
            List of dicts containing vector matches and scores
        """
        pass

    @abstractmethod
    def update(
        self,
        vector_ids: List[str],
        new_vectors: Optional[List[List[float]]] = None,
        new_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Update vectors and/or metadata"""
        pass

    @abstractmethod
    def delete(self, vector_ids: List[str]) -> None:
        """Delete vectors by ID"""
        pass

    @abstractmethod
    def get_metadata(self, vector_ids: List[str]) -> List[Dict[str, Any]]:
        """Get metadata for vectors"""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close database connection"""
        pass
