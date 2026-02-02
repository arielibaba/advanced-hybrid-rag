"""Tests for hybrid_retriever.py — retrieval cache, fusion, context manager, serialization."""

import json
import sqlite3
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from cognidoc.hybrid_retriever import (
    RetrievalCache,
    HybridRetrievalResult,
    HybridRetriever,
    QueryAnalysis,
    fuse_results,
    compute_answer_confidence,
)
from cognidoc.query_orchestrator import QueryType, RetrievalMode, RoutingDecision
from cognidoc.graph_retrieval import GraphRetrievalResult
from cognidoc.utils.rag_utils import Document, NodeWithScore


# ===========================================================================
# HybridRetrievalResult serialization
# ===========================================================================


class TestHybridRetrievalResultSerialization:
    """Tests for to_dict / from_dict JSON serialization."""

    def _make_result(self):
        analysis = QueryAnalysis(
            query="test query",
            query_type=QueryType.FACTUAL,
            vector_weight=0.7,
            graph_weight=0.3,
        )
        vector_results = [
            NodeWithScore(
                node=Document(text="chunk 1", metadata={"source": "doc1.pdf"}),
                score=0.95,
            ),
            NodeWithScore(
                node=Document(text="chunk 2", metadata={"parent": "p1"}),
                score=0.80,
            ),
        ]
        graph_results = GraphRetrievalResult(
            query="test query",
            retrieval_type="entity",
            context="Entity A is related to B",
            confidence=0.7,
        )
        return HybridRetrievalResult(
            query="test query",
            query_analysis=analysis,
            vector_results=vector_results,
            graph_results=graph_results,
            fused_context="fused text",
            source_chunks=["chunk_a", "chunk_b"],
            metadata={"vector_count": 2, "from_cache": False},
        )

    def test_roundtrip(self):
        """to_dict + from_dict preserves all fields."""
        original = self._make_result()
        data = original.to_dict()
        restored = HybridRetrievalResult.from_dict(data)

        assert restored.query == original.query
        assert restored.query_analysis.query_type == QueryType.FACTUAL
        assert restored.query_analysis.vector_weight == 0.7
        assert len(restored.vector_results) == 2
        assert restored.vector_results[0].node.text == "chunk 1"
        assert restored.vector_results[0].score == 0.95
        assert restored.graph_results is not None
        assert restored.graph_results.context == "Entity A is related to B"
        assert restored.fused_context == "fused text"
        assert restored.source_chunks == ["chunk_a", "chunk_b"]
        assert restored.metadata["vector_count"] == 2

    def test_roundtrip_no_graph(self):
        """Roundtrip with no graph results."""
        result = self._make_result()
        result.graph_results = None
        data = result.to_dict()
        restored = HybridRetrievalResult.from_dict(data)
        assert restored.graph_results is None

    def test_json_serializable(self):
        """to_dict output is fully JSON-serializable."""
        result = self._make_result()
        data = result.to_dict()
        serialized = json.dumps(data)
        deserialized = json.loads(serialized)
        restored = HybridRetrievalResult.from_dict(deserialized)
        assert restored.query == "test query"

    def test_empty_vector_results(self):
        """Roundtrip with empty vector results."""
        result = self._make_result()
        result.vector_results = []
        data = result.to_dict()
        restored = HybridRetrievalResult.from_dict(data)
        assert restored.vector_results == []


# ===========================================================================
# RetrievalCache (JSON-based, no pickle)
# ===========================================================================


class TestRetrievalCache:
    """Tests for RetrievalCache with JSON serialization."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton between tests."""
        RetrievalCache._instance = None
        RetrievalCache._initialized = False
        yield
        RetrievalCache._instance = None
        RetrievalCache._initialized = False

    def _make_cache(self, tmp_path):
        db_path = str(tmp_path / "test_cache.db")
        return RetrievalCache(db_path=db_path, max_size=5, ttl_seconds=60)

    def _make_result(self, query="q"):
        analysis = QueryAnalysis(
            query=query, query_type=QueryType.FACTUAL, vector_weight=0.7, graph_weight=0.3
        )
        return HybridRetrievalResult(
            query=query,
            query_analysis=analysis,
            fused_context="context",
            metadata={"from_cache": False},
        )

    def test_put_and_get_exact(self, tmp_path):
        """Cache hit on exact same query."""
        cache = self._make_cache(tmp_path)
        result = self._make_result("test query")
        cache.put("test query", 10, True, result)
        cached = cache.get("test query", 10, True)
        assert cached is not None
        assert cached.query == "test query"
        assert cached.fused_context == "context"

    def test_miss_on_unknown(self, tmp_path):
        """Cache miss for unseen query."""
        cache = self._make_cache(tmp_path)
        assert cache.get("unknown", 10, True) is None

    def test_miss_different_params(self, tmp_path):
        """Different top_k produces different cache key."""
        cache = self._make_cache(tmp_path)
        result = self._make_result()
        cache.put("q", 10, True, result)
        assert cache.get("q", 5, True) is None

    def test_clear(self, tmp_path):
        """clear() empties the cache."""
        cache = self._make_cache(tmp_path)
        cache.put("q", 10, True, self._make_result())
        cache.clear()
        assert cache.get("q", 10, True) is None

    def test_stats(self, tmp_path):
        """stats() returns correct hit/miss counts."""
        cache = self._make_cache(tmp_path)
        cache.put("q", 10, True, self._make_result())
        cache.get("q", 10, True)  # hit
        cache.get("other", 10, True)  # miss
        s = cache.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["size"] == 1

    def test_eviction(self, tmp_path):
        """Oldest entries are evicted when max_size is exceeded."""
        cache = self._make_cache(tmp_path)  # max_size=5
        for i in range(7):
            cache.put(f"q{i}", 10, True, self._make_result(f"q{i}"))
        s = cache.stats()
        assert s["size"] <= 5

    def test_no_pickle_in_db(self, tmp_path):
        """Verify stored data is JSON, not pickle."""
        cache = self._make_cache(tmp_path)
        cache.put("q", 10, True, self._make_result())
        db_path = str(tmp_path / "test_cache.db")
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT result FROM retrieval_cache LIMIT 1").fetchone()
        # Should be valid JSON string
        data = json.loads(row[0])
        assert data["query"] == "q"


# ===========================================================================
# fuse_results
# ===========================================================================


class TestFuseResults:
    """Tests for result fusion logic."""

    def test_vector_only_fusion(self):
        """Fusion with no graph results."""
        analysis = QueryAnalysis(
            query="q", query_type=QueryType.FACTUAL, vector_weight=1.0, graph_weight=0.0
        )
        vec = [
            NodeWithScore(
                node=Document(text="text1", metadata={"name": "c1", "source": "d"}), score=0.9
            )
        ]
        graph = GraphRetrievalResult(query="q", retrieval_type="none")
        context, chunks = fuse_results("q", vec, graph, analysis)
        assert "text1" in context
        assert "c1" in chunks

    def test_graph_only_fusion(self):
        """Fusion with no vector results."""
        analysis = QueryAnalysis(
            query="q", query_type=QueryType.EXPLORATORY, vector_weight=0.0, graph_weight=1.0
        )
        graph = GraphRetrievalResult(query="q", retrieval_type="entity", context="Graph info")
        context, chunks = fuse_results("q", [], graph, analysis)
        assert "Graph info" in context


# ===========================================================================
# compute_answer_confidence
# ===========================================================================


class TestComputeAnswerConfidence:
    """Tests for confidence scoring."""

    def test_high_confidence(self):
        analysis = QueryAnalysis(query="q", query_type=QueryType.FACTUAL)
        result = HybridRetrievalResult(
            query="q",
            query_analysis=analysis,
            metadata={
                "vector_confidence": 0.9,
                "graph_confidence": 0.8,
                "vector_count": 5,
                "vector_weight": 0.7,
                "graph_weight": 0.3,
            },
        )
        conf = compute_answer_confidence(result)
        assert 0.5 < conf <= 1.0

    def test_zero_confidence(self):
        analysis = QueryAnalysis(query="q", query_type=QueryType.FACTUAL)
        result = HybridRetrievalResult(
            query="q",
            query_analysis=analysis,
            metadata={
                "vector_confidence": 0.0,
                "graph_confidence": 0.0,
                "vector_count": 0,
                "vector_weight": 0.5,
                "graph_weight": 0.5,
            },
        )
        conf = compute_answer_confidence(result)
        assert conf == 0.0


# ===========================================================================
# HybridRetriever context manager
# ===========================================================================


class TestHybridRetrieverContextManager:
    """Tests for __enter__/__exit__ on HybridRetriever."""

    def test_context_manager_returns_self(self):
        with patch("cognidoc.hybrid_retriever.get_graph_config"):
            retriever = HybridRetriever.__new__(HybridRetriever)
            retriever._vector_index = None
            retriever._keyword_index = None
            retriever._bm25_index = None
            retriever._graph_retriever = None
            with retriever as r:
                assert r is retriever

    def test_context_manager_calls_close(self):
        with patch("cognidoc.hybrid_retriever.get_graph_config"):
            retriever = HybridRetriever.__new__(HybridRetriever)
            retriever._vector_index = MagicMock()
            retriever._keyword_index = None
            retriever._bm25_index = None
            retriever._graph_retriever = None
            vi = retriever._vector_index
            with retriever:
                pass
            vi.close.assert_called_once()
            assert retriever._vector_index is None
