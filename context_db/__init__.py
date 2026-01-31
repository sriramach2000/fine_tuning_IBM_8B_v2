"""
Vectorized Knowledge Graph Context Database.

Persists compressed /compact contexts in a Neo4j graph with HNSW vector indexes
for semantic retrieval and graph-based traversal.
"""

from context_db.embeddings import LocalEmbeddingService
from context_db.graph_client import Neo4jGraphClient
from context_db.context_manager import ContextManager
from context_db.topic_extractor import TopicExtractor

__all__ = [
    "LocalEmbeddingService",
    "Neo4jGraphClient",
    "ContextManager",
    "TopicExtractor",
]
