"""High-level context manager orchestrating storage, retrieval, and linking."""

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from context_db.embeddings import LocalEmbeddingService
from context_db.graph_client import Neo4jGraphClient
from context_db.topic_extractor import TopicExtractor

logger = logging.getLogger(__name__)


class ContextManager:
    """Orchestrates embedding, topic extraction, graph storage, and retrieval."""

    def __init__(
        self,
        graph_client: Optional[Neo4jGraphClient] = None,
        embedding_service: Optional[LocalEmbeddingService] = None,
        topic_extractor: Optional[TopicExtractor] = None,
    ):
        self.graph = graph_client or Neo4jGraphClient()
        self.embeddings = embedding_service or LocalEmbeddingService()
        self.extractor = topic_extractor or TopicExtractor()
        self.max_contexts = int(os.getenv("CONTEXT_DB_MAX_CONTEXTS", "10000"))
        self.similarity_threshold = float(
            os.getenv("CONTEXT_DB_SIMILARITY_THRESHOLD", "0.75")
        )
        self.default_top_k = int(os.getenv("CONTEXT_DB_TOP_K", "10"))

    def close(self):
        self.graph.close()

    def init_schema(self):
        """Initialize the graph schema and indexes."""
        self.graph.init_schema()

    # ── Store ────────────────────────────────────────────────────────

    def store_compact_context(
        self,
        summary: str,
        full_text: str,
        session_id: str,
        project: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Store a compressed /compact context with auto-extracted topics and code refs.

        Returns dict with context_id, topics_extracted, code_refs_found.
        """
        context_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        token_count = len(full_text.split())

        # Embed the full text
        embedding = self.embeddings.embed_text(full_text)

        # Create context node
        self.graph.create_context_node(
            id=context_id,
            summary=summary,
            full_text=full_text,
            timestamp=timestamp,
            session_id=session_id,
            token_count=token_count,
            embedding=embedding,
        )

        # Ensure session exists
        self.graph.create_session_node(
            id=session_id,
            project=project or os.getenv("PROJECT_NAME", "unknown"),
            start_time=timestamp,
        )
        self.graph.create_relationship(
            "CompactContext", context_id, "Session", session_id, "BELONGS_TO"
        )

        # Link to previous context in session (FOLLOWS edge)
        prev_id = self.graph.get_latest_context_in_session(session_id)
        if prev_id and prev_id != context_id:
            self.graph.create_relationship(
                "CompactContext", context_id, "CompactContext", prev_id, "FOLLOWS"
            )

        # Extract and link topics
        topics = self.extractor.extract_topics(full_text)
        topic_names = []
        for topic in topics:
            topic_id = str(uuid.uuid4())
            topic_embedding = self.embeddings.embed_text(topic.name)
            self.graph.create_topic_node(
                id=topic_id,
                name=topic.name,
                description=topic.description,
                embedding=topic_embedding,
            )
            self.graph.create_relationship(
                "CompactContext", context_id, "Topic", topic_id, "CONTAINS_TOPIC"
            )
            topic_names.append(topic.name)

        # Extract and link code entities
        code_entities = self.extractor.extract_code_entities(full_text)
        code_refs = []
        for entity in code_entities:
            entity_id = str(uuid.uuid4())
            entity_embedding = self.embeddings.embed_text(
                f"{entity.entity_type} {entity.name} {entity.file_path}"
            )
            self.graph.create_code_entity_node(
                id=entity_id,
                file_path=entity.file_path,
                entity_type=entity.entity_type,
                name=entity.name,
                embedding=entity_embedding,
            )
            self.graph.create_relationship(
                "CompactContext", context_id,
                "CodeEntity", entity_id,
                "REFERENCES_CODE",
            )
            code_refs.append(entity.file_path or entity.name)

        logger.info(
            "Stored context %s: %d topics, %d code refs",
            context_id, len(topic_names), len(code_refs),
        )

        return {
            "context_id": context_id,
            "topics_extracted": topic_names,
            "code_refs_found": code_refs,
            "token_count": token_count,
        }

    # ── Retrieve ─────────────────────────────────────────────────────

    def retrieve_relevant_contexts(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Semantic vector search with 1-hop graph expansion."""
        k = top_k or self.default_top_k
        query_embedding = self.embeddings.embed_text(query)

        raw_results = self.graph.vector_search(
            "context_embedding", query_embedding, k
        )

        results = []
        for row in raw_results:
            node = row.get("node", {})
            score = row.get("score", 0.0)
            if score < self.similarity_threshold:
                continue

            context_id = node.get("id", "")
            neighbors = self.graph.get_context_with_neighbors(context_id)

            results.append({
                "context_id": context_id,
                "summary": neighbors.get("summary", ""),
                "full_text": neighbors.get("full_text", ""),
                "timestamp": str(neighbors.get("timestamp", "")),
                "token_count": neighbors.get("token_count", 0),
                "similarity_score": score,
                "topics": neighbors.get("topics", []),
                "code_refs": neighbors.get("code_refs", []),
            })

        return results

    def get_topic_contexts(self, topic_name: str) -> List[Dict[str, Any]]:
        """Get all contexts linked to a topic via graph traversal."""
        return self.graph.get_contexts_by_topic(topic_name)

    def get_file_history(self, file_path: str) -> List[Dict[str, Any]]:
        """Get all contexts that reference a specific file."""
        return self.graph.get_contexts_by_file(file_path)

    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get temporal chain of contexts for a session."""
        return self.graph.get_session_history(session_id)

    def list_topics(self) -> List[Dict[str, Any]]:
        """List all topics with context counts."""
        return self.graph.get_all_topics()
