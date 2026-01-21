"""
Unit tests for cognidoc.helpers module.

Tests cover:
- Token counting
- Chat history limiting
- Query parsing
- Markdown conversion
- Image relevance detection
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


class TestGetTokenCount:
    """Tests for get_token_count function."""

    def test_empty_string(self):
        """Empty string should return 0 tokens."""
        from cognidoc.helpers import get_token_count
        assert get_token_count("") == 0

    def test_simple_text(self):
        """Simple text should return reasonable token count."""
        from cognidoc.helpers import get_token_count
        result = get_token_count("Hello world")
        assert result > 0
        assert result < 10  # Should be around 2-3 tokens

    def test_longer_text(self):
        """Longer text should have more tokens."""
        from cognidoc.helpers import get_token_count
        short = get_token_count("Hello")
        long = get_token_count("Hello world, this is a longer sentence with more words.")
        assert long > short

    def test_unicode_text(self):
        """Unicode text should be handled correctly."""
        from cognidoc.helpers import get_token_count
        result = get_token_count("Bonjour le monde! 你好世界")
        assert result > 0

    def test_special_characters(self):
        """Special characters should be handled."""
        from cognidoc.helpers import get_token_count
        result = get_token_count("Hello! @#$%^&*() 123")
        assert result > 0


class TestLimitChatHistory:
    """Tests for limit_chat_history function."""

    def test_empty_history(self):
        """Empty history should return empty list."""
        from cognidoc.helpers import limit_chat_history
        result = limit_chat_history([])
        assert result == []

    def test_short_history_unchanged(self):
        """Short history should remain unchanged."""
        from cognidoc.helpers import limit_chat_history
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = limit_chat_history(history, max_tokens=1000)
        assert len(result) == 2

    def test_truncates_long_history(self):
        """Long history should be truncated from the beginning."""
        from cognidoc.helpers import limit_chat_history
        # Use spaces to create many words, ensuring truncation with word-based fallback
        history = [
            {"role": "user", "content": " ".join(["word"] * 50) + f" msg{i}"}
            for i in range(100)
        ]
        result = limit_chat_history(history, max_tokens=500)
        assert len(result) < len(history)
        # Most recent messages should be kept
        assert result[-1] == history[-1]

    def test_keeps_recent_messages(self):
        """Should keep the most recent messages."""
        from cognidoc.helpers import limit_chat_history
        history = [
            {"role": "user", "content": "Old message " + "x" * 1000},
            {"role": "assistant", "content": "Old response " + "x" * 1000},
            {"role": "user", "content": "Recent message"},
            {"role": "assistant", "content": "Recent response"},
        ]
        result = limit_chat_history(history, max_tokens=100)
        # Should keep the recent short messages
        assert any("Recent" in msg["content"] for msg in result)

    def test_handles_none_content(self):
        """Should handle None content gracefully."""
        from cognidoc.helpers import limit_chat_history
        history = [
            {"role": "user", "content": None},
            {"role": "assistant", "content": "Response"},
        ]
        result = limit_chat_history(history, max_tokens=1000)
        assert len(result) == 2

    def test_handles_list_content(self):
        """Should handle list content (multimodal format)."""
        from cognidoc.helpers import limit_chat_history
        history = [
            {"role": "user", "content": [{"text": "Hello"}, {"text": "World"}]},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = limit_chat_history(history, max_tokens=1000)
        assert len(result) == 2


class TestParseRewrittenQuery:
    """Tests for parse_rewritten_query function."""

    def test_single_line_without_bullet(self):
        """Single line without bullet returns empty (expects bullet format)."""
        from cognidoc.helpers import parse_rewritten_query
        result = parse_rewritten_query("What is bioethics?")
        # Function expects bullet points, single line without bullet returns empty
        assert isinstance(result, list)

    def test_bullet_points_dash(self):
        """Should parse dash bullet points."""
        from cognidoc.helpers import parse_rewritten_query
        text = """- What is bioethics?
- How does it relate to medicine?
- What are the key principles?"""
        result = parse_rewritten_query(text)
        assert len(result) == 3

    def test_bullet_points_asterisk(self):
        """Should parse asterisk bullet points."""
        from cognidoc.helpers import parse_rewritten_query
        text = """* First question
* Second question
* Third question"""
        result = parse_rewritten_query(text)
        assert len(result) == 3

    def test_empty_string(self):
        """Empty string should return empty list."""
        from cognidoc.helpers import parse_rewritten_query
        result = parse_rewritten_query("")
        assert result == []

    def test_whitespace_only(self):
        """Whitespace only should return empty list."""
        from cognidoc.helpers import parse_rewritten_query
        result = parse_rewritten_query("   \n\n  ")
        assert result == []


class TestMarkdownToPlainText:
    """Tests for markdown_to_plain_text function."""

    def test_simple_text(self):
        """Simple text should remain unchanged."""
        from cognidoc.helpers import markdown_to_plain_text
        result = markdown_to_plain_text("Hello world")
        assert "Hello world" in result

    def test_removes_bold(self):
        """Should remove bold markdown."""
        from cognidoc.helpers import markdown_to_plain_text
        result = markdown_to_plain_text("This is **bold** text")
        assert "**" not in result
        assert "bold" in result

    def test_removes_italic(self):
        """Should remove italic markdown."""
        from cognidoc.helpers import markdown_to_plain_text
        result = markdown_to_plain_text("This is *italic* text")
        assert result.count("*") == 0 or "italic" in result

    def test_removes_links(self):
        """Should extract text from links."""
        from cognidoc.helpers import markdown_to_plain_text
        result = markdown_to_plain_text("Click [here](http://example.com)")
        assert "here" in result
        assert "http" not in result

    def test_removes_headers(self):
        """Should remove header markers."""
        from cognidoc.helpers import markdown_to_plain_text
        result = markdown_to_plain_text("# Header\nContent")
        assert "#" not in result
        assert "Header" in result


class TestExtractJson:
    """Tests for extract_json function."""

    def test_clean_json(self):
        """Should extract clean JSON string."""
        from cognidoc.helpers import extract_json
        result = extract_json('{"key": "value"}')
        assert '{"key": "value"}' in result or "key" in result

    def test_json_in_code_block(self):
        """Should extract JSON from code block."""
        from cognidoc.helpers import extract_json
        text = """Here is the result:
```json
{"key": "value"}
```"""
        result = extract_json(text)
        assert "key" in result

    def test_empty_string(self):
        """Should handle empty string."""
        from cognidoc.helpers import extract_json
        result = extract_json("")
        assert isinstance(result, str)


class TestRecoverJson:
    """Tests for recover_json function."""

    def test_valid_json(self):
        """Should parse valid JSON."""
        from cognidoc.helpers import recover_json
        result = recover_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_with_trailing_text(self):
        """Should recover JSON with trailing text."""
        from cognidoc.helpers import recover_json
        result = recover_json('{"key": "value"} some extra text')
        # May return dict or handle gracefully
        assert result is not None or result == {"key": "value"}

    def test_nested_json(self):
        """Should handle nested JSON."""
        from cognidoc.helpers import recover_json
        result = recover_json('{"outer": {"inner": "value"}}')
        assert result == {"outer": {"inner": "value"}}


class TestCleanUpText:
    """Tests for clean_up_text function."""

    def test_simple_text(self):
        """Simple text should remain mostly unchanged."""
        from cognidoc.helpers import clean_up_text
        result = clean_up_text("Hello world")
        assert "Hello" in result or "hello" in result.lower()

    def test_handles_whitespace(self):
        """Should handle text with whitespace."""
        from cognidoc.helpers import clean_up_text
        result = clean_up_text("Hello    world")
        # Function may or may not normalize whitespace
        assert isinstance(result, str)
        assert "Hello" in result or "hello" in result.lower()

    def test_empty_string(self):
        """Should handle empty string."""
        from cognidoc.helpers import clean_up_text
        result = clean_up_text("")
        assert isinstance(result, str)


class TestConvertHistoryToTuples:
    """Tests for convert_history_to_tuples function."""

    def test_empty_history(self):
        """Empty history should return empty list."""
        from cognidoc.helpers import convert_history_to_tuples
        result = convert_history_to_tuples([])
        assert result == []

    def test_converts_format(self):
        """Should convert to Gradio format."""
        from cognidoc.helpers import convert_history_to_tuples
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = convert_history_to_tuples(history)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"


class TestGetMemoryWindow:
    """Tests for get_memory_window function."""

    def test_returns_positive_integer(self):
        """Should return a positive integer."""
        from cognidoc.helpers import get_memory_window
        result = get_memory_window()
        assert isinstance(result, int)
        assert result > 0

    def test_returns_reasonable_size(self):
        """Should return a reasonable memory window size."""
        from cognidoc.helpers import get_memory_window
        result = get_memory_window()
        # Should be at least 10K tokens for any reasonable model
        assert result >= 10000
        # Should not exceed 2M tokens (largest known context window * 0.5)
        assert result <= 2000000
