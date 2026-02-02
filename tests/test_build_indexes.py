"""Tests for build_indexes.py — loading embeddings and building vector/keyword indexes."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from cognidoc.build_indexes import load_embeddings_with_documents


class TestLoadEmbeddingsWithDocuments:
    """Tests for load_embeddings_with_documents()."""

    def _setup_dirs(self, tmp_path):
        """Create embeddings, chunks, and docs directories with sample data."""
        embeddings_dir = tmp_path / "embeddings"
        chunks_dir = tmp_path / "chunks"
        docs_dir = tmp_path / "processed"
        embeddings_dir.mkdir()
        chunks_dir.mkdir()
        docs_dir.mkdir()
        return embeddings_dir, chunks_dir, docs_dir

    def _write_embedding(self, embeddings_dir, chunks_dir, docs_dir, idx=1, parent_in_chunks=False):
        """Write a sample embedding file and its associated documents."""
        child_name = f"doc_page_{idx}_child_chunk_{idx}.txt"
        if parent_in_chunks:
            parent_name = f"doc_page_{idx}_parent_chunk_{idx}.txt"
        else:
            parent_name = f"doc_page_{idx}.txt"

        embedding_data = {
            "embedding": [0.1 * idx, 0.2 * idx, 0.3 * idx],
            "metadata": {
                "child": child_name,
                "parent": parent_name,
                "source": f"doc_{idx}.pdf",
            },
        }
        emb_file = embeddings_dir / f"emb_{idx}.json"
        emb_file.write_text(json.dumps(embedding_data), encoding="utf-8")

        # Write child chunk
        child_file = chunks_dir / child_name
        child_file.write_text(f"Child content {idx}", encoding="utf-8")

        # Write parent
        if parent_in_chunks:
            parent_file = chunks_dir / parent_name
        else:
            parent_file = docs_dir / parent_name
        parent_file.write_text(f"Parent content {idx}", encoding="utf-8")

        return embedding_data

    def test_loads_single_embedding(self, tmp_path):
        """Single embedding file loads correctly."""
        embeddings_dir, chunks_dir, docs_dir = self._setup_dirs(tmp_path)
        self._write_embedding(embeddings_dir, chunks_dir, docs_dir, idx=1)

        embeddings, child_docs, parent_docs = load_embeddings_with_documents(
            str(embeddings_dir), str(chunks_dir), str(docs_dir)
        )

        assert len(embeddings) == 1
        assert len(child_docs) == 1
        assert len(parent_docs) == 1
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert child_docs[0].text == "Child content 1"
        assert parent_docs[0].text == "Parent content 1"

    def test_loads_multiple_embeddings(self, tmp_path):
        """Multiple embedding files load correctly."""
        embeddings_dir, chunks_dir, docs_dir = self._setup_dirs(tmp_path)
        for i in range(1, 4):
            self._write_embedding(embeddings_dir, chunks_dir, docs_dir, idx=i)

        embeddings, child_docs, parent_docs = load_embeddings_with_documents(
            str(embeddings_dir), str(chunks_dir), str(docs_dir)
        )

        assert len(embeddings) == 3
        assert len(child_docs) == 3
        assert len(parent_docs) == 3

    def test_child_metadata_correct(self, tmp_path):
        """Child document metadata includes name, parent, and source."""
        embeddings_dir, chunks_dir, docs_dir = self._setup_dirs(tmp_path)
        self._write_embedding(embeddings_dir, chunks_dir, docs_dir, idx=1)

        _, child_docs, _ = load_embeddings_with_documents(
            str(embeddings_dir), str(chunks_dir), str(docs_dir)
        )

        meta = child_docs[0].metadata
        assert "name" in meta
        assert "parent" in meta
        assert "source" in meta
        assert meta["source"] == "doc_1.pdf"

    def test_parent_from_chunks_dir(self, tmp_path):
        """Parent with _parent_chunk in name is loaded from chunks_dir."""
        embeddings_dir, chunks_dir, docs_dir = self._setup_dirs(tmp_path)
        self._write_embedding(embeddings_dir, chunks_dir, docs_dir, idx=1, parent_in_chunks=True)

        _, _, parent_docs = load_embeddings_with_documents(
            str(embeddings_dir), str(chunks_dir), str(docs_dir)
        )

        assert parent_docs[0].text == "Parent content 1"

    def test_parent_from_docs_dir(self, tmp_path):
        """Parent without _parent_chunk is loaded from docs_dir."""
        embeddings_dir, chunks_dir, docs_dir = self._setup_dirs(tmp_path)
        self._write_embedding(embeddings_dir, chunks_dir, docs_dir, idx=1, parent_in_chunks=False)

        _, _, parent_docs = load_embeddings_with_documents(
            str(embeddings_dir), str(chunks_dir), str(docs_dir)
        )

        assert parent_docs[0].text == "Parent content 1"

    def test_empty_directory(self, tmp_path):
        """Empty embeddings directory returns empty lists."""
        embeddings_dir, chunks_dir, docs_dir = self._setup_dirs(tmp_path)

        embeddings, child_docs, parent_docs = load_embeddings_with_documents(
            str(embeddings_dir), str(chunks_dir), str(docs_dir)
        )

        assert embeddings == []
        assert child_docs == []
        assert parent_docs == []

    def test_sorted_loading_order(self, tmp_path):
        """Embedding files are loaded in sorted order."""
        embeddings_dir, chunks_dir, docs_dir = self._setup_dirs(tmp_path)
        # Write in reverse order
        for i in [3, 1, 2]:
            self._write_embedding(embeddings_dir, chunks_dir, docs_dir, idx=i)

        embeddings, child_docs, _ = load_embeddings_with_documents(
            str(embeddings_dir), str(chunks_dir), str(docs_dir)
        )

        # Should be sorted by filename (emb_1, emb_2, emb_3)
        assert child_docs[0].text == "Child content 1"
        assert child_docs[1].text == "Child content 2"
        assert child_docs[2].text == "Child content 3"


class TestBuildIndexes:
    """Tests for build_indexes() function."""

    def test_build_indexes_creates_directories(self, tmp_path):
        """build_indexes creates vector store and index directories."""
        vector_dir = tmp_path / "vector_store"
        index_dir = tmp_path / "indexes"

        with (
            patch("cognidoc.build_indexes.VECTOR_STORE_DIR", str(vector_dir)),
            patch("cognidoc.build_indexes.INDEX_DIR", str(index_dir)),
            patch("cognidoc.build_indexes.load_embeddings_with_documents") as mock_load,
            patch("cognidoc.build_indexes.VectorIndex") as mock_vi,
            patch("cognidoc.build_indexes.KeywordIndex") as mock_ki,
        ):

            mock_load.return_value = ([], [], [])
            mock_index = MagicMock()
            mock_vi.create.return_value = mock_index

            from cognidoc.build_indexes import build_indexes

            build_indexes(recreate=True)

            assert vector_dir.exists()
            assert index_dir.exists()

    def test_build_indexes_calls_vector_create(self, tmp_path):
        """build_indexes creates a VectorIndex with correct parameters."""
        vector_dir = tmp_path / "vector_store"
        index_dir = tmp_path / "indexes"

        with (
            patch("cognidoc.build_indexes.VECTOR_STORE_DIR", str(vector_dir)),
            patch("cognidoc.build_indexes.INDEX_DIR", str(index_dir)),
            patch("cognidoc.build_indexes.EMBEDDINGS_DIR", str(tmp_path / "emb")),
            patch("cognidoc.build_indexes.CHUNKS_DIR", str(tmp_path / "chunks")),
            patch("cognidoc.build_indexes.PROCESSED_DIR", str(tmp_path / "proc")),
            patch("cognidoc.build_indexes.load_embeddings_with_documents") as mock_load,
            patch("cognidoc.build_indexes.VectorIndex") as mock_vi,
            patch("cognidoc.build_indexes.KeywordIndex") as mock_ki,
        ):

            mock_load.return_value = ([[0.1, 0.2]], [MagicMock()], [MagicMock()])
            mock_index = MagicMock()
            mock_vi.create.return_value = mock_index

            from cognidoc.build_indexes import build_indexes

            build_indexes(recreate=True)

            mock_vi.create.assert_called_once()
            mock_index.add_documents.assert_called_once()
            mock_index.save.assert_called_once()
            mock_index.close.assert_called_once()

    def test_build_indexes_creates_keyword_index(self, tmp_path):
        """build_indexes creates and saves a KeywordIndex."""
        vector_dir = tmp_path / "vector_store"
        index_dir = tmp_path / "indexes"

        with (
            patch("cognidoc.build_indexes.VECTOR_STORE_DIR", str(vector_dir)),
            patch("cognidoc.build_indexes.INDEX_DIR", str(index_dir)),
            patch("cognidoc.build_indexes.EMBEDDINGS_DIR", str(tmp_path / "emb")),
            patch("cognidoc.build_indexes.CHUNKS_DIR", str(tmp_path / "chunks")),
            patch("cognidoc.build_indexes.PROCESSED_DIR", str(tmp_path / "proc")),
            patch("cognidoc.build_indexes.load_embeddings_with_documents") as mock_load,
            patch("cognidoc.build_indexes.VectorIndex") as mock_vi,
            patch("cognidoc.build_indexes.KeywordIndex") as mock_ki,
        ):

            parent_doc = MagicMock()
            mock_load.return_value = ([[0.1, 0.2]], [MagicMock()], [parent_doc])
            mock_vi.create.return_value = MagicMock()

            mock_ki_instance = MagicMock()
            mock_ki.return_value = mock_ki_instance

            from cognidoc.build_indexes import build_indexes

            build_indexes(recreate=True)

            mock_ki_instance.add_documents.assert_called_once_with([parent_doc])
            mock_ki_instance.save.assert_called_once()
