"""
Pytest configuration and fixtures for CogniDoc tests.
"""

import sys
from pathlib import Path

import pytest

# Add src directory to Python path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


# Session-scoped CogniDoc instance to avoid Qdrant lock conflicts
# Qdrant embedded only allows one client per storage folder
_session_cognidoc = None


@pytest.fixture(scope="session")
def cognidoc_session():
    """
    Session-scoped CogniDoc instance shared across ALL test modules.

    This prevents Qdrant embedded lock conflicts when multiple tests
    try to access the vector store.
    """
    global _session_cognidoc

    project_root = Path(__file__).parent.parent
    indexes_dir = project_root / "data" / "indexes"

    if not (indexes_dir / "child_documents").exists():
        pytest.skip("Indexes not found. Run ingestion first.")

    if _session_cognidoc is None:
        from cognidoc import CogniDoc
        _session_cognidoc = CogniDoc(
            llm_provider="gemini",
            embedding_provider="ollama",
            use_yolo=False,
            use_graph=True,
        )

    return _session_cognidoc


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (run with --run-slow)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --run-slow is provided."""
    if config.getoption("--run-slow", default=False):
        return

    import pytest
    skip_slow = pytest.mark.skip(reason="Slow test - use --run-slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add --run-slow command line option."""
    try:
        parser.addoption(
            "--run-slow",
            action="store_true",
            default=False,
            help="Run slow E2E tests with GraphRAG"
        )
    except ValueError:
        # Option already added
        pass
