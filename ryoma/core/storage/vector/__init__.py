from typing import Literal

from ryoma.core.storage.vector.base import VectorDatabase
from ryoma.core.storage.vector.backend.milvus import MilvusVectorDB


def create_vector_database(db_type: Literal["milvus"]) -> VectorDatabase:
    """Create vector database instance based on type.

    Args:
        db_type: Type of vector database, currently supports 'milvus'

    Returns:
        VectorDatabase instance
    """
    if db_type.lower() == "milvus":
        return MilvusVectorDB()
    else:
        raise ValueError(f"Unsupported vector database type: {db_type}")


__all__ = ["VectorDatabase", "create_vector_database"]
