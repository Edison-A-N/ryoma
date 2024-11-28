from typing import List, Dict, Any, Optional
import uuid

from pymilvus import (
    connections,
    Collection,
    utility,
    CollectionSchema,
    FieldSchema,
    DataType,
)

from ryoma.core.storage.vector.base import VectorDatabase

from ryoma.core.config import settings


class MilvusVectorDB(VectorDatabase):
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
        dim: Optional[int] = None,
    ) -> None:
        """Initialize Milvus connection

        Args:
            host: Milvus server host
            port: Milvus server port
            collection_name: Name of collection to use
            dim: Vector dimension
        """
        self.host = host or settings.MILVUS_HOST
        self.port = port or settings.MILVUS_PORT
        self.collection_name = collection_name or settings.MILVUS_COLLECTION_NAME
        self.dim = dim or settings.MILVUS_DIM

        # Connect to Milvus
        connections.connect(host=self.host, port=self.port)

        # Create collection if it doesn't exist
        if not utility.has_collection(self.collection_name):
            self._create_collection()

        self.collection = Collection(self.collection_name)
        self.collection.load()

    def _create_collection(self) -> None:
        """Create Milvus collection with schema"""
        fields = [
            FieldSchema(
                name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36
            ),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]
        schema = CollectionSchema(fields=fields)
        collection = Collection(self.collection_name, schema)

        # Create IVF_FLAT index for vector field
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        }
        collection.create_index(field_name="vector", index_params=index_params)

    def insert(
        self, vectors: List[List[float]], metadata: List[Dict[str, Any]]
    ) -> List[str]:
        ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        entities = [ids, vectors, metadata]
        self.collection.insert(entities)
        self.collection.flush()
        return ids

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        distance_metric: str = "cosine",
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        search_params = {
            "metric_type": distance_metric,
            "params": {"nprobe": 10},
        }

        expr = None
        if metadata_filter:
            # Convert metadata filter to Milvus expression
            conditions = []
            for key, value in metadata_filter.items():
                conditions.append(f'metadata["{key}"] == {repr(value)}')
            expr = " && ".join(conditions)

        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["metadata"],
        )

        matches = []
        for hit in results[0]:
            matches.append(
                {
                    "id": hit.id,
                    "score": hit.score,
                    "metadata": hit.entity.get("metadata", {}),
                }
            )
        return matches

    def update(
        self,
        vector_ids: List[str],
        new_vectors: Optional[List[List[float]]] = None,
        new_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if new_vectors:
            self.collection.upsert(
                [vector_ids, new_vectors, new_metadata or [{}] * len(vector_ids)]
            )
        elif new_metadata:
            # Only update metadata
            for vid, meta in zip(vector_ids, new_metadata):
                self.collection.update(expr=f'id == "{vid}"', data={"metadata": meta})
        self.collection.flush()

    def delete(self, vector_ids: List[str]) -> None:
        expr = " || ".join([f'id == "{vid}"' for vid in vector_ids])
        self.collection.delete(expr)
        self.collection.flush()

    def get_metadata(self, vector_ids: List[str]) -> List[Dict[str, Any]]:
        expr = " || ".join([f'id == "{vid}"' for vid in vector_ids])
        results = self.collection.query(expr=expr, output_fields=["metadata"])
        return [r.get("metadata", {}) for r in results]

    def close(self) -> None:
        connections.disconnect(alias=self.host)
