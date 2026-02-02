"""Tests for ingestion_manifest.py — incremental ingestion tracking."""

import json
import time
from pathlib import Path

import pytest

from cognidoc.ingestion_manifest import IngestionManifest, FileRecord, MANIFEST_VERSION


class TestFileRecord:
    """Tests for the FileRecord dataclass."""

    def test_create_file_record(self):
        record = FileRecord(
            path="doc.pdf",
            stem="doc",
            size=1024,
            mtime=1000.0,
            content_hash="abc123",
            ingested_at="2024-01-01T00:00:00Z",
        )
        assert record.path == "doc.pdf"
        assert record.stem == "doc"
        assert record.size == 1024


class TestIngestionManifestLoad:
    """Tests for IngestionManifest.load()."""

    def test_load_returns_none_when_missing(self, tmp_path):
        """load() returns None when manifest file doesn't exist."""
        result = IngestionManifest.load(tmp_path / "nonexistent.json")
        assert result is None

    def test_load_valid_manifest(self, tmp_path):
        """load() reads a valid manifest file correctly."""
        manifest_path = tmp_path / "manifest.json"
        data = {
            "version": MANIFEST_VERSION,
            "created_at": "2024-01-01T00:00:00Z",
            "last_updated": "2024-01-02T00:00:00Z",
            "files": {
                "doc.pdf": {
                    "path": "doc.pdf",
                    "stem": "doc",
                    "size": 100,
                    "mtime": 1000.0,
                    "content_hash": "abc",
                    "ingested_at": "2024-01-01T00:00:00Z",
                }
            },
        }
        manifest_path.write_text(json.dumps(data), encoding="utf-8")

        manifest = IngestionManifest.load(manifest_path)
        assert manifest is not None
        assert len(manifest.files) == 1
        assert manifest.files["doc.pdf"].stem == "doc"
        assert manifest.version == MANIFEST_VERSION

    def test_load_corrupted_returns_none(self, tmp_path):
        """load() returns None for corrupted JSON."""
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text("NOT VALID JSON{{{", encoding="utf-8")

        result = IngestionManifest.load(manifest_path)
        assert result is None

    def test_load_missing_fields_returns_none(self, tmp_path):
        """load() returns None when file records have missing fields."""
        manifest_path = tmp_path / "manifest.json"
        data = {"files": {"doc.pdf": {"path": "doc.pdf"}}}  # Missing required fields
        manifest_path.write_text(json.dumps(data), encoding="utf-8")

        result = IngestionManifest.load(manifest_path)
        assert result is None


class TestIngestionManifestSave:
    """Tests for IngestionManifest.save()."""

    def test_save_creates_file(self, tmp_path):
        """save() creates a valid JSON file."""
        manifest = IngestionManifest()
        manifest.files["doc.pdf"] = FileRecord(
            path="doc.pdf",
            stem="doc",
            size=100,
            mtime=1000.0,
            content_hash="abc",
            ingested_at="2024-01-01",
        )
        manifest_path = tmp_path / "manifest.json"
        manifest.save(manifest_path)

        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "files" in data
        assert "doc.pdf" in data["files"]

    def test_save_sets_timestamps(self, tmp_path):
        """save() sets created_at and last_updated."""
        manifest = IngestionManifest()
        manifest_path = tmp_path / "manifest.json"
        manifest.save(manifest_path)

        assert manifest.created_at != ""
        assert manifest.last_updated != ""

    def test_save_preserves_created_at(self, tmp_path):
        """save() preserves existing created_at."""
        manifest = IngestionManifest()
        manifest.created_at = "2024-01-01T00:00:00Z"
        manifest_path = tmp_path / "manifest.json"
        manifest.save(manifest_path)

        assert manifest.created_at == "2024-01-01T00:00:00Z"

    def test_save_creates_parent_directories(self, tmp_path):
        """save() creates parent directories if needed."""
        manifest = IngestionManifest()
        deep_path = tmp_path / "a" / "b" / "c" / "manifest.json"
        manifest.save(deep_path)
        assert deep_path.exists()

    def test_save_roundtrip(self, tmp_path):
        """save() + load() preserves all data."""
        manifest = IngestionManifest()
        manifest.files["doc.pdf"] = FileRecord(
            path="doc.pdf",
            stem="doc",
            size=200,
            mtime=2000.0,
            content_hash="def456",
            ingested_at="2024-06-01",
        )
        manifest_path = tmp_path / "manifest.json"
        manifest.save(manifest_path)

        loaded = IngestionManifest.load(manifest_path)
        assert loaded is not None
        assert len(loaded.files) == 1
        assert loaded.files["doc.pdf"].content_hash == "def456"
        assert loaded.files["doc.pdf"].size == 200


class TestGetNewAndModifiedFiles:
    """Tests for get_new_and_modified_files()."""

    def test_all_new_files(self, tmp_path):
        """Empty manifest means all files are new."""
        sources = tmp_path / "sources"
        sources.mkdir()
        (sources / "a.pdf").write_bytes(b"%PDF")
        (sources / "b.pdf").write_bytes(b"%PDF-2")

        manifest = IngestionManifest()
        new, modified, stems = manifest.get_new_and_modified_files(sources)

        assert len(new) == 2
        assert len(modified) == 0
        assert stems == {"a", "b"}

    def test_no_changes(self, tmp_path):
        """When all files match manifest, nothing is new or modified."""
        sources = tmp_path / "sources"
        sources.mkdir()
        f = sources / "doc.pdf"
        f.write_bytes(b"%PDF-test")

        manifest = IngestionManifest()
        manifest.record_file(f, sources, "doc")

        new, modified, stems = manifest.get_new_and_modified_files(sources)
        assert len(new) == 0
        assert len(modified) == 0

    def test_modified_file_detected(self, tmp_path):
        """Modified file (different content) is detected."""
        sources = tmp_path / "sources"
        sources.mkdir()
        f = sources / "doc.pdf"
        f.write_bytes(b"original content")

        manifest = IngestionManifest()
        manifest.record_file(f, sources, "doc")

        # Modify the file (different size to trigger fast-path size check on all platforms)
        f.write_bytes(b"modified content that is longer")

        new, modified, stems = manifest.get_new_and_modified_files(sources)
        assert len(new) == 0
        assert len(modified) == 1
        assert "doc" in stems

    def test_source_files_filter(self, tmp_path):
        """source_files parameter limits which files are checked."""
        sources = tmp_path / "sources"
        sources.mkdir()
        (sources / "a.pdf").write_bytes(b"A")
        (sources / "b.pdf").write_bytes(b"B")

        manifest = IngestionManifest()
        new, modified, stems = manifest.get_new_and_modified_files(
            sources, source_files=[str(sources / "a.pdf")]
        )

        assert len(new) == 1
        assert stems == {"a"}


class TestGetDeletedFiles:
    """Tests for get_deleted_files()."""

    def test_deleted_file_detected(self, tmp_path):
        """File in manifest but not on disk is detected."""
        sources = tmp_path / "sources"
        sources.mkdir()

        manifest = IngestionManifest()
        manifest.files["gone.pdf"] = FileRecord(
            path="gone.pdf",
            stem="gone",
            size=10,
            mtime=0,
            content_hash="abc",
            ingested_at="now",
        )

        deleted = manifest.get_deleted_files(sources)
        assert len(deleted) == 1
        assert deleted[0].stem == "gone"

    def test_no_deleted_files(self, tmp_path):
        """All files present means empty result."""
        sources = tmp_path / "sources"
        sources.mkdir()
        (sources / "present.pdf").write_bytes(b"data")

        manifest = IngestionManifest()
        manifest.files["present.pdf"] = FileRecord(
            path="present.pdf",
            stem="present",
            size=4,
            mtime=0,
            content_hash="x",
            ingested_at="now",
        )

        deleted = manifest.get_deleted_files(sources)
        assert len(deleted) == 0


class TestRecordFile:
    """Tests for record_file() and record_all_sources()."""

    def test_record_file_stores_hash(self, tmp_path):
        """record_file creates a record with SHA-256 hash."""
        sources = tmp_path / "sources"
        sources.mkdir()
        f = sources / "doc.pdf"
        f.write_bytes(b"hello world")

        manifest = IngestionManifest()
        manifest.record_file(f, sources, "doc")

        assert "doc.pdf" in manifest.files
        record = manifest.files["doc.pdf"]
        assert record.stem == "doc"
        assert record.size == 11
        assert len(record.content_hash) == 64  # SHA-256 hex length
        assert record.ingested_at != ""

    def test_record_file_outside_sources_dir(self, tmp_path):
        """record_file handles files outside sources_dir gracefully."""
        sources = tmp_path / "sources"
        sources.mkdir()
        other = tmp_path / "other"
        other.mkdir()
        f = other / "external.pdf"
        f.write_bytes(b"data")

        manifest = IngestionManifest()
        manifest.record_file(f, sources, "external")

        # Falls back to using the filename as key
        assert "external.pdf" in manifest.files

    def test_record_all_sources(self, tmp_path):
        """record_all_sources records all files in directory."""
        sources = tmp_path / "sources"
        sources.mkdir()
        (sources / "a.pdf").write_bytes(b"A")
        (sources / "b.pdf").write_bytes(b"B")
        sub = sources / "sub"
        sub.mkdir()
        (sub / "c.pdf").write_bytes(b"C")

        manifest = IngestionManifest()
        manifest.record_all_sources(sources)

        assert len(manifest.files) == 3


class TestComputeFileHash:
    """Tests for compute_file_hash()."""

    def test_deterministic_hash(self, tmp_path):
        """Same content produces same hash."""
        f = tmp_path / "test.txt"
        f.write_bytes(b"test content")

        hash1 = IngestionManifest.compute_file_hash(f)
        hash2 = IngestionManifest.compute_file_hash(f)
        assert hash1 == hash2

    def test_different_content_different_hash(self, tmp_path):
        """Different content produces different hashes."""
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_bytes(b"content A")
        f2.write_bytes(b"content B")

        assert IngestionManifest.compute_file_hash(f1) != IngestionManifest.compute_file_hash(f2)

    def test_hash_is_sha256_hex(self, tmp_path):
        """Hash is a 64-character hex string (SHA-256)."""
        f = tmp_path / "test.txt"
        f.write_bytes(b"data")

        h = IngestionManifest.compute_file_hash(f)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)
