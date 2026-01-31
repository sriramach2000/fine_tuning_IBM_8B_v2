"""Tests for the vectorized knowledge graph context database."""

import json
import uuid
from unittest.mock import MagicMock, patch

import pytest

from context_db.topic_extractor import TopicExtractor, ExtractedTopic, ExtractedCodeEntity


# ── TopicExtractor tests (no external deps) ─────────────────────────


class TestTopicExtractor:
    def setup_method(self):
        self.extractor = TopicExtractor(use_keybert=False)

    def test_extract_topics_returns_topics(self):
        text = "TSN protocol handles time-sensitive networking with AVB bridging"
        topics = self.extractor.extract_topics(text, top_n=3)
        assert len(topics) > 0
        assert all(isinstance(t, ExtractedTopic) for t in topics)

    def test_extract_topics_deduplicates(self):
        text = "training training training model model"
        topics = self.extractor.extract_topics(text, top_n=5)
        names = [t.name for t in topics]
        assert len(names) == len(set(names))

    def test_extract_code_entities_finds_files(self):
        text = "Modified training/train_granite_qlora.py and scripts/setup.py"
        entities = self.extractor.extract_code_entities(text)
        paths = [e.file_path for e in entities if e.entity_type == "file"]
        assert "training/train_granite_qlora.py" in paths
        assert "scripts/setup.py" in paths

    def test_extract_code_entities_finds_functions(self):
        text = "def train_epoch(self, data):\n    pass\ndef evaluate(model):\n    pass"
        entities = self.extractor.extract_code_entities(text)
        func_names = [e.name for e in entities if e.entity_type == "function"]
        assert "train_epoch" in func_names
        assert "evaluate" in func_names

    def test_extract_code_entities_finds_classes(self):
        text = "class IterativeDistillationTrainer:\n    pass"
        entities = self.extractor.extract_code_entities(text)
        class_names = [e.name for e in entities if e.entity_type == "class"]
        assert "IterativeDistillationTrainer" in class_names

    def test_extract_code_entities_no_duplicates(self):
        text = "file.py file.py def foo(): def foo():"
        entities = self.extractor.extract_code_entities(text)
        keys = [(e.entity_type, e.name) for e in entities]
        assert len(keys) == len(set(keys))


# ── Graph Client tests (mocked) ─────────────────────────────────────


class TestNeo4jGraphClientMocked:
    def setup_method(self):
        with patch("context_db.graph_client.GraphDatabase") as mock_gdb:
            self.mock_driver = MagicMock()
            mock_gdb.driver.return_value = self.mock_driver
            from context_db.graph_client import Neo4jGraphClient
            self.client = Neo4jGraphClient(
                uri="bolt://localhost:7687", user="neo4j", password="test"
            )

    def test_execute_query(self):
        mock_session = MagicMock()
        mock_record = MagicMock()
        mock_record.data.return_value = {"ok": 1}
        mock_session.run.return_value = [mock_record]
        self.mock_driver.session.return_value.__enter__ = lambda s: mock_session
        self.mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        result = self.client.execute_query("RETURN 1 AS ok")
        assert result == [{"ok": 1}]

    def test_close(self):
        self.client.close()
        self.mock_driver.close.assert_called_once()


# ── Embedding Service tests (mocked) ────────────────────────────────


class TestLocalEmbeddingServiceMocked:
    @patch("context_db.embeddings.SentenceTransformer")
    def test_embed_text(self, mock_st_class):
        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(1024).astype(np.float32)
        mock_st_class.return_value = mock_model

        from context_db.embeddings import LocalEmbeddingService
        svc = LocalEmbeddingService(model_name="test-model", device="cpu")
        result = svc.embed_text("hello world")

        assert isinstance(result, list)
        assert len(result) == 1024
        mock_model.encode.assert_called_once()

    @patch("context_db.embeddings.SentenceTransformer")
    def test_embed_batch(self, mock_st_class):
        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(3, 1024).astype(np.float32)
        mock_st_class.return_value = mock_model

        from context_db.embeddings import LocalEmbeddingService
        svc = LocalEmbeddingService(model_name="test-model", device="cpu")
        result = svc.embed_batch(["a", "b", "c"])

        assert len(result) == 3
        assert all(len(v) == 1024 for v in result)


# ── Context Manager tests (mocked) ──────────────────────────────────


class TestContextManagerMocked:
    def setup_method(self):
        self.mock_graph = MagicMock()
        self.mock_embeddings = MagicMock()
        self.mock_embeddings.embed_text.return_value = [0.1] * 1024
        self.mock_extractor = MagicMock()
        self.mock_extractor.extract_topics.return_value = [
            ExtractedTopic(name="TSN", description="test"),
        ]
        self.mock_extractor.extract_code_entities.return_value = [
            ExtractedCodeEntity(
                file_path="train.py", entity_type="file", name="train"
            ),
        ]
        self.mock_graph.get_latest_context_in_session.return_value = None
        self.mock_graph.create_context_node.return_value = {"id": "test-id"}
        self.mock_graph.create_topic_node.return_value = {"id": "t1"}
        self.mock_graph.create_code_entity_node.return_value = {"id": "e1"}
        self.mock_graph.create_session_node.return_value = {"id": "s1"}

        from context_db.context_manager import ContextManager
        self.manager = ContextManager(
            graph_client=self.mock_graph,
            embedding_service=self.mock_embeddings,
            topic_extractor=self.mock_extractor,
        )

    def test_store_compact_context(self):
        result = self.manager.store_compact_context(
            summary="Test summary",
            full_text="Test context about TSN and train.py",
            session_id="session-1",
        )
        assert "context_id" in result
        assert "TSN" in result["topics_extracted"]
        assert "train.py" in result["code_refs_found"]
        self.mock_graph.create_context_node.assert_called_once()
        self.mock_graph.create_topic_node.assert_called_once()
        self.mock_graph.create_code_entity_node.assert_called_once()

    def test_store_creates_follows_edge(self):
        self.mock_graph.get_latest_context_in_session.return_value = "prev-id"
        self.manager.store_compact_context(
            summary="Second context",
            full_text="Following up",
            session_id="session-1",
        )
        # Should create FOLLOWS relationship
        calls = self.mock_graph.create_relationship.call_args_list
        follows_calls = [c for c in calls if c[0][4] == "FOLLOWS"]
        assert len(follows_calls) == 1

    def test_retrieve_relevant_contexts(self):
        self.mock_graph.vector_search.return_value = [
            {"node": {"id": "c1"}, "score": 0.9},
        ]
        self.mock_graph.get_context_with_neighbors.return_value = {
            "summary": "test", "full_text": "text", "timestamp": "2024-01-01",
            "token_count": 10, "topics": ["TSN"], "code_refs": ["file.py"],
        }
        results = self.manager.retrieve_relevant_contexts("TSN timing")
        assert len(results) == 1
        assert results[0]["similarity_score"] == 0.9

    def test_retrieve_filters_by_threshold(self):
        self.mock_graph.vector_search.return_value = [
            {"node": {"id": "c1"}, "score": 0.5},  # Below default 0.75
        ]
        results = self.manager.retrieve_relevant_contexts("something")
        assert len(results) == 0


# ── MCP Server tests ────────────────────────────────────────────────


class TestMCPServerTools:
    @patch("context_db.mcp_server._get_manager")
    def test_store_context_tool(self, mock_get_mgr):
        mock_mgr = MagicMock()
        mock_mgr.store_compact_context.return_value = {
            "context_id": "abc", "topics_extracted": ["TSN"],
            "code_refs_found": [], "token_count": 10,
        }
        mock_get_mgr.return_value = mock_mgr

        from context_db.mcp_server import store_context
        result = store_context("summary", "full text", "session-1")
        data = json.loads(result)
        assert data["context_id"] == "abc"

    @patch("context_db.mcp_server._get_manager")
    def test_search_contexts_tool(self, mock_get_mgr):
        mock_mgr = MagicMock()
        mock_mgr.retrieve_relevant_contexts.return_value = [
            {"context_id": "c1", "summary": "test", "similarity_score": 0.9}
        ]
        mock_get_mgr.return_value = mock_mgr

        from context_db.mcp_server import search_contexts
        result = search_contexts("TSN protocol")
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["similarity_score"] == 0.9

    @patch("context_db.mcp_server._get_manager")
    def test_list_topics_tool(self, mock_get_mgr):
        mock_mgr = MagicMock()
        mock_mgr.list_topics.return_value = [
            {"name": "TSN", "description": "", "context_count": 5}
        ]
        mock_get_mgr.return_value = mock_mgr

        from context_db.mcp_server import list_topics
        result = list_topics()
        data = json.loads(result)
        assert data[0]["name"] == "TSN"
