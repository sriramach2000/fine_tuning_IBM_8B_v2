"""MCP server exposing the context database as tools for Claude Code."""

import json
import logging
import os
from typing import Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

mcp = FastMCP("context-db")

# Lazy-initialized singleton
_manager = None


def _get_manager():
    global _manager
    if _manager is None:
        from context_db.context_manager import ContextManager
        _manager = ContextManager()
        _manager.init_schema()
        logger.info("ContextManager initialized")
    return _manager


@mcp.tool()
def store_context(summary: str, full_text: str, session_id: str, project: Optional[str] = None) -> str:
    """Store a compressed /compact context with auto-extracted topics and code references.

    Args:
        summary: Short summary of the context
        full_text: Full compressed context text
        session_id: ID of the current session
        project: Optional project name (defaults to env PROJECT_NAME)

    Returns:
        JSON with context_id, topics_extracted, code_refs_found, token_count
    """
    result = _get_manager().store_compact_context(
        summary=summary,
        full_text=full_text,
        session_id=session_id,
        project=project,
    )
    return json.dumps(result, indent=2)


@mcp.tool()
def search_contexts(query: str, top_k: int = 10) -> str:
    """Semantic vector search for relevant past contexts.

    Args:
        query: Natural language search query
        top_k: Maximum number of results to return

    Returns:
        JSON list of matching contexts ranked by similarity, with graph neighbors
    """
    results = _get_manager().retrieve_relevant_contexts(query=query, top_k=top_k)
    return json.dumps(results, indent=2, default=str)


@mcp.tool()
def get_topic_contexts(topic_name: str) -> str:
    """Get all contexts linked to a topic via knowledge graph traversal.

    Args:
        topic_name: Name of the topic to search for

    Returns:
        JSON list of contexts connected to this topic
    """
    results = _get_manager().get_topic_contexts(topic_name)
    return json.dumps(results, indent=2, default=str)


@mcp.tool()
def get_file_history(file_path: str) -> str:
    """Get all contexts that reference a specific file.

    Args:
        file_path: Path to the file to search for

    Returns:
        JSON list of contexts that referenced this file
    """
    results = _get_manager().get_file_history(file_path)
    return json.dumps(results, indent=2, default=str)


@mcp.tool()
def get_session_history(session_id: str) -> str:
    """Reconstruct the temporal chain of contexts for a session.

    Args:
        session_id: ID of the session

    Returns:
        JSON list of contexts in chronological order
    """
    results = _get_manager().get_session_history(session_id)
    return json.dumps(results, indent=2, default=str)


@mcp.tool()
def list_topics() -> str:
    """List all known topics with their context counts.

    Returns:
        JSON list of topics with name, description, and context_count
    """
    results = _get_manager().list_topics()
    return json.dumps(results, indent=2, default=str)


if __name__ == "__main__":
    mcp.run()
