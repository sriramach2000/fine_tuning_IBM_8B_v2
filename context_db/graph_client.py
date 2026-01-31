"""Neo4j graph client with vector search and graph traversal."""

import logging
import os
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase

from context_db.graph_schema import ALL_SCHEMA_QUERIES

logger = logging.getLogger(__name__)


class Neo4jGraphClient:
    """Wraps the Neo4j Python driver with connection pooling and typed operations."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        logger.info("Connected to Neo4j at %s", self.uri)

    def close(self):
        self.driver.close()

    def execute_query(
        self, cypher: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results as list of dicts."""
        with self.driver.session() as session:
            result = session.run(cypher, params or {})
            return [record.data() for record in result]

    def init_schema(self):
        """Create all constraints, vector indexes, and composite indexes."""
        for query in ALL_SCHEMA_QUERIES:
            try:
                self.execute_query(query)
            except Exception as e:
                logger.warning("Schema query skipped (may already exist): %s", e)
        logger.info("Graph schema initialized")

    # ── Node CRUD ────────────────────────────────────────────────────

    def create_context_node(
        self,
        id: str,
        summary: str,
        full_text: str,
        timestamp: str,
        session_id: str,
        token_count: int,
        embedding: List[float],
    ) -> Dict[str, Any]:
        cypher = """
        CREATE (c:CompactContext {
            id: $id, summary: $summary, full_text: $full_text,
            timestamp: datetime($timestamp), session_id: $session_id,
            token_count: $token_count, embedding: $embedding
        })
        RETURN c.id AS id
        """
        results = self.execute_query(cypher, {
            "id": id, "summary": summary, "full_text": full_text,
            "timestamp": timestamp, "session_id": session_id,
            "token_count": token_count, "embedding": embedding,
        })
        return results[0] if results else {}

    def create_topic_node(
        self, id: str, name: str, description: str, embedding: List[float]
    ) -> Dict[str, Any]:
        cypher = """
        MERGE (t:Topic {name: $name})
        ON CREATE SET t.id = $id, t.description = $description, t.embedding = $embedding
        ON MATCH SET t.description = $description, t.embedding = $embedding
        RETURN t.id AS id, t.name AS name
        """
        results = self.execute_query(cypher, {
            "id": id, "name": name, "description": description, "embedding": embedding,
        })
        return results[0] if results else {}

    def create_code_entity_node(
        self,
        id: str,
        file_path: str,
        entity_type: str,
        name: str,
        embedding: List[float],
    ) -> Dict[str, Any]:
        cypher = """
        MERGE (e:CodeEntity {file_path: $file_path, name: $name})
        ON CREATE SET e.id = $id, e.entity_type = $entity_type, e.embedding = $embedding
        ON MATCH SET e.entity_type = $entity_type, e.embedding = $embedding
        RETURN e.id AS id, e.file_path AS file_path
        """
        results = self.execute_query(cypher, {
            "id": id, "file_path": file_path, "entity_type": entity_type,
            "name": name, "embedding": embedding,
        })
        return results[0] if results else {}

    def create_session_node(self, id: str, project: str, start_time: str) -> Dict[str, Any]:
        cypher = """
        MERGE (s:Session {id: $id})
        ON CREATE SET s.project = $project, s.start_time = datetime($start_time)
        RETURN s.id AS id
        """
        results = self.execute_query(cypher, {
            "id": id, "project": project, "start_time": start_time,
        })
        return results[0] if results else {}

    # ── Relationships ────────────────────────────────────────────────

    def create_relationship(
        self,
        from_label: str,
        from_id: str,
        to_label: str,
        to_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ):
        props_str = ""
        params: Dict[str, Any] = {"from_id": from_id, "to_id": to_id}
        if properties:
            prop_parts = []
            for k, v in properties.items():
                param_key = f"prop_{k}"
                prop_parts.append(f"{k}: ${param_key}")
                params[param_key] = v
            props_str = " {" + ", ".join(prop_parts) + "}"

        cypher = f"""
        MATCH (a:{from_label} {{id: $from_id}})
        MATCH (b:{to_label} {{id: $to_id}})
        MERGE (a)-[r:{rel_type}{props_str}]->(b)
        RETURN type(r) AS rel_type
        """
        self.execute_query(cypher, params)

    # ── Vector Search ────────────────────────────────────────────────

    def vector_search(
        self, index_name: str, embedding: List[float], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Query a vector index for nearest neighbors."""
        cypher = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
        YIELD node, score
        RETURN node, score
        ORDER BY score DESC
        """
        return self.execute_query(cypher, {
            "index_name": index_name, "top_k": top_k, "embedding": embedding,
        })

    # ── Graph Traversals ─────────────────────────────────────────────

    def get_context_with_neighbors(self, context_id: str) -> Dict[str, Any]:
        """Get a context node with its connected topics and code entities."""
        cypher = """
        MATCH (c:CompactContext {id: $id})
        OPTIONAL MATCH (c)-[:CONTAINS_TOPIC]->(t:Topic)
        OPTIONAL MATCH (c)-[:REFERENCES_CODE]->(e:CodeEntity)
        RETURN c.id AS id, c.summary AS summary, c.full_text AS full_text,
               c.timestamp AS timestamp, c.token_count AS token_count,
               collect(DISTINCT t.name) AS topics,
               collect(DISTINCT e.file_path) AS code_refs
        """
        results = self.execute_query(cypher, {"id": context_id})
        return results[0] if results else {}

    def get_contexts_by_topic(self, topic_name: str) -> List[Dict[str, Any]]:
        """Traverse from a topic to all linked contexts."""
        cypher = """
        MATCH (t:Topic {name: $name})<-[:CONTAINS_TOPIC]-(c:CompactContext)
        RETURN c.id AS id, c.summary AS summary, c.timestamp AS timestamp
        ORDER BY c.timestamp DESC
        """
        return self.execute_query(cypher, {"name": topic_name})

    def get_contexts_by_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Traverse from a code entity to all linked contexts."""
        cypher = """
        MATCH (e:CodeEntity {file_path: $file_path})<-[:REFERENCES_CODE]-(c:CompactContext)
        RETURN c.id AS id, c.summary AS summary, c.timestamp AS timestamp
        ORDER BY c.timestamp DESC
        """
        return self.execute_query(cypher, {"file_path": file_path})

    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get temporal chain of contexts for a session."""
        cypher = """
        MATCH (c:CompactContext {session_id: $session_id})
        RETURN c.id AS id, c.summary AS summary, c.timestamp AS timestamp,
               c.token_count AS token_count
        ORDER BY c.timestamp ASC
        """
        return self.execute_query(cypher, {"session_id": session_id})

    def get_latest_context_in_session(self, session_id: str) -> Optional[str]:
        """Get the most recent context ID in a session (for FOLLOWS linking)."""
        cypher = """
        MATCH (c:CompactContext {session_id: $session_id})
        RETURN c.id AS id
        ORDER BY c.timestamp DESC
        LIMIT 1
        """
        results = self.execute_query(cypher, {"session_id": session_id})
        return results[0]["id"] if results else None

    def get_all_topics(self) -> List[Dict[str, Any]]:
        """List all topics with their context counts."""
        cypher = """
        MATCH (t:Topic)
        OPTIONAL MATCH (t)<-[:CONTAINS_TOPIC]-(c:CompactContext)
        RETURN t.name AS name, t.description AS description, count(c) AS context_count
        ORDER BY context_count DESC
        """
        return self.execute_query(cypher)
