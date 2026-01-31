#!/usr/bin/env python3
"""Setup script for the vectorized knowledge graph context database.

Run on the target machine after installing Neo4j:
    python scripts/setup_context_db.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def check_neo4j_connection():
    """Verify Neo4j is running and accessible."""
    logger.info("Checking Neo4j connection...")
    from context_db.graph_client import Neo4jGraphClient

    client = Neo4jGraphClient()
    try:
        result = client.execute_query("RETURN 1 AS ok")
        assert result[0]["ok"] == 1
        logger.info("  Neo4j connection OK")
        return client
    except Exception as e:
        logger.error("  Failed to connect to Neo4j: %s", e)
        logger.error(
            "  Make sure Neo4j is running: sudo systemctl status neo4j"
        )
        logger.error("  Check NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD in .env")
        sys.exit(1)


def create_schema(client):
    """Create graph constraints, vector indexes, and composite indexes."""
    logger.info("Creating graph schema...")
    client.init_schema()
    logger.info("  Schema created")


def verify_vector_indexes(client):
    """Check that vector indexes are online."""
    logger.info("Verifying vector indexes...")
    result = client.execute_query("SHOW INDEXES YIELD name, state, type")
    vector_indexes = [r for r in result if r.get("type") == "VECTOR"]

    for idx in vector_indexes:
        name = idx["name"]
        state = idx["state"]
        status = "OK" if state == "ONLINE" else f"WARNING: {state}"
        logger.info("  %s: %s", name, status)

    if not vector_indexes:
        logger.warning("  No vector indexes found. Neo4j 5.11+ is required.")
    return vector_indexes


def smoke_test(client):
    """Store a test context and retrieve it."""
    logger.info("Running smoke test...")
    from context_db.embeddings import LocalEmbeddingService

    embed = LocalEmbeddingService()

    # Create a test node
    test_embedding = embed.embed_text("smoke test context")
    client.execute_query(
        """
        CREATE (c:CompactContext {
            id: 'smoke-test',
            summary: 'Smoke test',
            full_text: 'This is a smoke test context for validation.',
            timestamp: datetime(),
            session_id: 'test-session',
            token_count: 8,
            embedding: $embedding
        })
        """,
        {"embedding": test_embedding},
    )

    # Retrieve it
    results = client.vector_search("context_embedding", test_embedding, top_k=1)
    if results:
        logger.info("  Vector search returned %d result(s) - OK", len(results))
    else:
        logger.warning("  Vector search returned no results (index may be populating)")

    # Clean up
    client.execute_query("MATCH (c:CompactContext {id: 'smoke-test'}) DETACH DELETE c")
    logger.info("  Smoke test cleaned up")


def print_summary():
    """Print configuration summary."""
    logger.info("")
    logger.info("=== Context DB Configuration ===")
    logger.info("  NEO4J_URI:           %s", os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    logger.info("  EMBEDDING_MODEL:     %s", os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"))
    logger.info("  EMBEDDING_DEVICE:    %s", os.getenv("EMBEDDING_DEVICE", "auto"))
    logger.info("  EMBEDDING_DIMENSIONS:%s", os.getenv("EMBEDDING_DIMENSIONS", "1024"))
    logger.info("  MAX_CONTEXTS:        %s", os.getenv("CONTEXT_DB_MAX_CONTEXTS", "10000"))
    logger.info("  SIMILARITY_THRESHOLD:%s", os.getenv("CONTEXT_DB_SIMILARITY_THRESHOLD", "0.75"))
    logger.info("  TOP_K:               %s", os.getenv("CONTEXT_DB_TOP_K", "10"))
    logger.info("")
    logger.info("MCP server command:")
    logger.info("  python -m context_db.mcp_server")
    logger.info("")
    logger.info("Add to Claude Code settings (~/.claude/settings.json):")
    logger.info('  "mcpServers": {')
    logger.info('    "context-db": {')
    logger.info('      "command": "python",')
    logger.info('      "args": ["-m", "context_db.mcp_server"],')
    logger.info('      "cwd": "%s"', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logger.info("    }")
    logger.info("  }")


def main():
    logger.info("Setting up Vectorized Knowledge Graph Context Database")
    logger.info("=" * 55)

    client = check_neo4j_connection()
    create_schema(client)
    verify_vector_indexes(client)
    smoke_test(client)
    client.close()

    print_summary()
    logger.info("Setup complete.")


if __name__ == "__main__":
    main()
