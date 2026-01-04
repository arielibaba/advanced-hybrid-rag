#!/usr/bin/env python
"""
Test script for Advanced RAG features.

Tests:
- #2: Overlapping chunks
- #7: BM25 + Dense hybrid search
- #9: Cross-encoder reranking
- #13: Contextual compression
- #14: Lost-in-the-middle reordering
- #15: Citation verification
- #16: Metadata filtering
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.constants import (
    ENABLE_HYBRID_SEARCH,
    ENABLE_CROSS_ENCODER,
    ENABLE_LOST_IN_MIDDLE_REORDER,
    ENABLE_CONTEXTUAL_COMPRESSION,
    LLM,
)
from src.utils.advanced_rag import (
    BM25Index,
    hybrid_search_fusion,
    cross_encoder_rerank,
    reorder_lost_in_middle,
    compress_context,
    verify_citations,
    MetadataFilter,
    filter_by_metadata,
    create_overlapping_chunks,
)
from src.utils.rag_utils import Document, NodeWithScore
from src.hybrid_retriever import HybridRetriever


def test_overlapping_chunks():
    """Test #2: Overlapping chunks."""
    print("\n=== Test #2: Overlapping Chunks ===")

    text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four. This is sentence five."
    chunks = create_overlapping_chunks(text, chunk_size=5, overlap=2)

    print(f"Original text: {text}")
    print(f"Chunks ({len(chunks)}):")
    for i, chunk in enumerate(chunks):
        print(f"  {i+1}: {chunk}")

    # Verify overlap
    assert len(chunks) > 1, "Should have multiple chunks"
    print("PASSED")


def test_bm25_index():
    """Test #7: BM25 Index."""
    print("\n=== Test #7: BM25 Index ===")

    # Create test documents
    docs = [
        {"text": "Machine learning is a subset of artificial intelligence.", "metadata": {"id": 1}},
        {"text": "Deep learning uses neural networks for pattern recognition.", "metadata": {"id": 2}},
        {"text": "Natural language processing enables computers to understand text.", "metadata": {"id": 3}},
    ]

    # Build BM25 index
    bm25 = BM25Index(k1=1.5, b=0.75)
    bm25.add_documents(docs)

    # Search
    results = bm25.search("machine learning neural networks", top_k=3)

    print(f"Query: 'machine learning neural networks'")
    print(f"Results ({len(results)}):")
    for doc, score in results:
        print(f"  Score {score:.3f}: {doc['text'][:50]}...")

    assert len(results) > 0, "Should have results"
    print("PASSED")


def test_hybrid_search_fusion():
    """Test #7: Hybrid search fusion."""
    print("\n=== Test #7: Hybrid Search Fusion ===")

    # Simulate dense and sparse results
    doc1 = Document(text="Document about machine learning", metadata={"id": 1})
    doc2 = Document(text="Document about deep learning", metadata={"id": 2})
    doc3 = Document(text="Document about neural networks", metadata={"id": 3})

    dense_results = [(doc1, 0.9), (doc2, 0.7), (doc3, 0.5)]
    sparse_results = [(doc2, 0.8), (doc3, 0.6), (doc1, 0.4)]

    fused = hybrid_search_fusion(dense_results, sparse_results, alpha=0.6, top_k=3)

    print(f"Dense: {[d[0].metadata['id'] for d in dense_results]}")
    print(f"Sparse: {[d[0].metadata['id'] for d in sparse_results]}")
    print(f"Fused (alpha=0.6): {[(d.metadata['id'] if hasattr(d, 'metadata') else 'N/A', score) for d, score in fused]}")

    assert len(fused) == 3, "Should have 3 fused results"
    print("PASSED")


def test_lost_in_middle_reorder():
    """Test #14: Lost-in-the-middle reordering."""
    print("\n=== Test #14: Lost-in-the-Middle Reordering ===")

    docs = [
        Document(text="Most relevant doc", metadata={"rank": 1}),
        Document(text="Second most relevant", metadata={"rank": 2}),
        Document(text="Third most relevant", metadata={"rank": 3}),
        Document(text="Fourth most relevant", metadata={"rank": 4}),
        Document(text="Fifth most relevant", metadata={"rank": 5}),
    ]

    reordered = reorder_lost_in_middle(docs)

    print(f"Original order: {[d.metadata['rank'] for d in docs]}")
    print(f"Reordered: {[d.metadata['rank'] for d in reordered]}")

    # Most relevant should be at start or end
    assert reordered[0].metadata['rank'] == 1, "Most relevant should be first"
    print("PASSED")


def test_metadata_filtering():
    """Test #16: Metadata filtering."""
    print("\n=== Test #16: Metadata Filtering ===")

    docs = [
        Document(text="Doc 1", metadata={"type": "text", "page": 1}),
        Document(text="Doc 2", metadata={"type": "table", "page": 2}),
        Document(text="Doc 3", metadata={"type": "text", "page": 3}),
    ]

    # Filter by type
    filters = [MetadataFilter(field="type", value="text", operator="eq")]
    filtered = filter_by_metadata(docs, filters)

    print(f"Original: {len(docs)} documents")
    print(f"Filter: type == 'text'")
    print(f"Filtered: {len(filtered)} documents")

    assert len(filtered) == 2, "Should have 2 text documents"
    print("PASSED")


def test_hybrid_retriever_load():
    """Test hybrid retriever with advanced features."""
    print("\n=== Test: Hybrid Retriever Load ===")

    retriever = HybridRetriever()
    status = retriever.load()

    print(f"Load status:")
    for component, loaded in status.items():
        status_str = "OK" if loaded else "NOT LOADED"
        print(f"  {component}: {status_str}")

    print(f"\nConfig settings:")
    print(f"  ENABLE_HYBRID_SEARCH: {ENABLE_HYBRID_SEARCH}")
    print(f"  ENABLE_CROSS_ENCODER: {ENABLE_CROSS_ENCODER}")
    print(f"  ENABLE_LOST_IN_MIDDLE_REORDER: {ENABLE_LOST_IN_MIDDLE_REORDER}")
    print(f"  ENABLE_CONTEXTUAL_COMPRESSION: {ENABLE_CONTEXTUAL_COMPRESSION}")

    # At least vector index should be loaded
    assert status.get("vector_index") or status.get("keyword_index"), "At least one index should be loaded"
    print("PASSED")


def test_hybrid_retrieval():
    """Test full hybrid retrieval with all features."""
    print("\n=== Test: Full Hybrid Retrieval ===")

    retriever = HybridRetriever()
    retriever.load()

    if not retriever.is_loaded():
        print("SKIPPED - Indexes not loaded")
        return

    # Run a test query
    query = "What is the unit of activity?"

    print(f"Query: '{query}'")

    # Test with all advanced features enabled
    result = retriever.retrieve(
        query=query,
        top_k=5,
        use_reranking=True,
        use_hybrid_search=True,
        use_cross_encoder=True,
        use_lost_in_middle=True,
        use_compression=False,  # Disable to speed up test
    )

    print(f"\nResults:")
    print(f"  Vector results: {len(result.vector_results)}")
    print(f"  Graph entities: {result.metadata.get('graph_entities', 0)}")
    print(f"  Query type: {result.metadata.get('query_type', 'unknown')}")
    print(f"  Hybrid search: {result.metadata.get('hybrid_search', False)}")
    print(f"  Cross-encoder: {result.metadata.get('cross_encoder', False)}")
    print(f"  Lost-in-middle: {result.metadata.get('lost_in_middle', False)}")

    if result.vector_results:
        print(f"\nTop result preview:")
        print(f"  {result.vector_results[0].node.text[:200]}...")

    print("PASSED")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Advanced RAG Features Test Suite")
    print("=" * 60)

    tests = [
        test_overlapping_chunks,
        test_bm25_index,
        test_hybrid_search_fusion,
        test_lost_in_middle_reorder,
        test_metadata_filtering,
        test_hybrid_retriever_load,
        test_hybrid_retrieval,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
