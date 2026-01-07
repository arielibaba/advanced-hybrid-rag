"""
Unit tests for LLM and Embedding providers.
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys

# Import provider classes and configs
from cognidoc.utils.llm_providers import (
    LLMProvider,
    LLMConfig,
    Message,
    LLMResponse,
    BaseLLMProvider,
    create_llm_provider,
)
from cognidoc.utils.embedding_providers import (
    EmbeddingProvider,
    EmbeddingConfig,
    BaseEmbeddingProvider,
    create_embedding_provider,
    is_ollama_available,
    is_provider_available,
    DEFAULT_EMBEDDING_MODELS,
)


class TestLLMConfig:
    """Tests for LLMConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LLMConfig(
            provider=LLMProvider.GEMINI,
            model="gemini-2.0-flash",
        )
        assert config.temperature == 0.7
        assert config.top_p == 0.85
        assert config.max_tokens is None
        assert config.timeout == 180.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="granite3.3:8b",
            temperature=0.5,
            top_p=0.9,
            max_tokens=1000,
            timeout=60.0,
        )
        assert config.temperature == 0.5
        assert config.top_p == 0.9
        assert config.max_tokens == 1000
        assert config.timeout == 60.0


class TestMessage:
    """Tests for Message dataclass."""

    def test_message_creation(self):
        """Test creating a message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.images is None

    def test_message_with_images(self):
        """Test creating a message with images."""
        msg = Message(role="user", content="Describe this", images=["image.jpg"])
        assert msg.images == ["image.jpg"]


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_response_creation(self):
        """Test creating a response."""
        response = LLMResponse(
            content="Hello, I'm Claude",
            model="gemini-2.0-flash",
            provider=LLMProvider.GEMINI,
        )
        assert response.content == "Hello, I'm Claude"
        assert response.model == "gemini-2.0-flash"
        assert response.provider == LLMProvider.GEMINI


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EmbeddingConfig(
            provider=EmbeddingProvider.OLLAMA,
            model="qwen3-embedding:0.6b",
        )
        # Default batch_size is 100 in the actual implementation
        assert config.batch_size == 100
        assert config.timeout == 60.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = EmbeddingConfig(
            provider=EmbeddingProvider.OPENAI,
            model="text-embedding-3-small",
            batch_size=64,
            timeout=120.0,
        )
        assert config.batch_size == 64
        assert config.timeout == 120.0


class TestDefaultEmbeddingModels:
    """Tests for default embedding models."""

    def test_default_models_exist(self):
        """Test that default models are defined for all providers."""
        assert EmbeddingProvider.OLLAMA in DEFAULT_EMBEDDING_MODELS
        assert EmbeddingProvider.OPENAI in DEFAULT_EMBEDDING_MODELS
        assert EmbeddingProvider.GEMINI in DEFAULT_EMBEDDING_MODELS


class TestOllamaProvider:
    """Tests for Ollama provider (mocked)."""

    def test_ollama_chat(self):
        """Test Ollama chat with mocked client."""
        # Create mock ollama module
        mock_ollama = MagicMock()
        mock_client = MagicMock()
        mock_ollama.Client.return_value = mock_client
        mock_client.chat.return_value = {
            "message": {"content": "Hello from Ollama!"}
        }

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            from cognidoc.utils.llm_providers import OllamaProvider

            config = LLMConfig(
                provider=LLMProvider.OLLAMA,
                model="granite3.3:8b",
            )
            provider = OllamaProvider(config)

            messages = [Message(role="user", content="Hello")]
            response = provider.chat(messages)

            assert response.content == "Hello from Ollama!"
            assert response.provider == LLMProvider.OLLAMA

    def test_ollama_embedding(self):
        """Test Ollama embedding with mocked client."""
        mock_ollama = MagicMock()
        mock_client = MagicMock()
        mock_ollama.Client.return_value = mock_client
        # Ollama uses client.embeddings(model=..., prompt=...) and returns {"embedding": [...]}
        mock_client.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            from cognidoc.utils.embedding_providers import OllamaEmbeddingProvider

            config = EmbeddingConfig(
                provider=EmbeddingProvider.OLLAMA,
                model="qwen3-embedding:0.6b",
            )
            provider = OllamaEmbeddingProvider(config)

            embedding = provider.embed_single("Hello")
            assert embedding == [0.1, 0.2, 0.3]


class TestGeminiProvider:
    """Tests for Gemini provider (mocked)."""

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-api-key"})
    def test_gemini_initialization(self):
        """Test Gemini provider initialization with mocked SDK."""
        mock_genai = MagicMock()
        mock_types = MagicMock()
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        with patch.dict(sys.modules, {"google": MagicMock(), "google.genai": mock_genai, "google.genai.types": mock_types}):
            # Patch the import
            with patch("cognidoc.utils.llm_providers.genai", mock_genai, create=True):
                with patch("cognidoc.utils.llm_providers.types", mock_types, create=True):
                    from cognidoc.utils.llm_providers import GeminiProvider

                    config = LLMConfig(
                        provider=LLMProvider.GEMINI,
                        model="gemini-2.0-flash",
                    )
                    # This test just verifies the test setup works


class TestOpenAIProvider:
    """Tests for OpenAI provider (mocked)."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
    def test_openai_chat(self):
        """Test OpenAI chat with mocked client."""
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello from OpenAI!"))]
        mock_response.usage = MagicMock(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict(sys.modules, {"openai": mock_openai}):
            from cognidoc.utils.llm_providers import OpenAIProvider

            config = LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4o-mini",
            )
            provider = OpenAIProvider(config)

            messages = [Message(role="user", content="Hello")]
            response = provider.chat(messages)

            assert response.content == "Hello from OpenAI!"
            assert response.provider == LLMProvider.OPENAI

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
    def test_openai_embedding(self):
        """Test OpenAI embedding with mocked client."""
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        # Mock response
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response

        with patch.dict(sys.modules, {"openai": mock_openai}):
            from cognidoc.utils.embedding_providers import OpenAIEmbeddingProvider

            config = EmbeddingConfig(
                provider=EmbeddingProvider.OPENAI,
                model="text-embedding-3-small",
            )
            provider = OpenAIEmbeddingProvider(config)

            embedding = provider.embed_single("Hello")
            assert embedding == [0.1, 0.2, 0.3]


class TestProviderAvailability:
    """Tests for provider availability checks."""

    @patch("httpx.get")
    def test_ollama_available(self, mock_get):
        """Test Ollama availability check when server is running."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        assert is_ollama_available() is True

    @patch("httpx.get")
    def test_ollama_not_available(self, mock_get):
        """Test Ollama availability check when server is not running."""
        mock_get.side_effect = Exception("Connection refused")

        assert is_ollama_available() is False

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_openai_provider_available(self):
        """Test OpenAI provider availability with API key set."""
        mock_openai = MagicMock()
        with patch.dict(sys.modules, {"openai": mock_openai}):
            result = is_provider_available(EmbeddingProvider.OPENAI)
            assert result is True

    def test_openai_provider_not_available_no_key(self):
        """Test OpenAI provider not available without API key."""
        # Clear any existing keys
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            mock_openai = MagicMock()
            with patch.dict(sys.modules, {"openai": mock_openai}):
                result = is_provider_available(EmbeddingProvider.OPENAI)
                assert result is False


class TestCreateProvider:
    """Tests for provider factory functions."""

    def test_create_ollama_llm_provider(self):
        """Test creating Ollama LLM provider."""
        mock_ollama = MagicMock()
        mock_ollama.Client.return_value = MagicMock()

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            config = LLMConfig(
                provider=LLMProvider.OLLAMA,
                model="granite3.3:8b",
            )
            provider = create_llm_provider(config)
            assert provider is not None

    def test_create_ollama_embedding_provider(self):
        """Test creating Ollama embedding provider."""
        mock_ollama = MagicMock()
        mock_ollama.Client.return_value = MagicMock()

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            config = EmbeddingConfig(
                provider=EmbeddingProvider.OLLAMA,
                model="qwen3-embedding:0.6b",
            )
            provider = create_embedding_provider(config)
            assert provider is not None

    def test_create_unknown_provider_raises(self):
        """Test that unknown provider raises error."""
        with pytest.raises(ValueError):
            config = LLMConfig(
                provider="unknown",
                model="some-model",
            )
            create_llm_provider(config)


class TestBatchEmbedding:
    """Tests for batch embedding functionality."""

    def test_batch_embedding(self):
        """Test batch embedding with multiple texts."""
        mock_ollama = MagicMock()
        mock_client = MagicMock()
        mock_ollama.Client.return_value = mock_client
        # Ollama uses client.embeddings() and returns {"embedding": [...]}
        mock_client.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            from cognidoc.utils.embedding_providers import OllamaEmbeddingProvider

            config = EmbeddingConfig(
                provider=EmbeddingProvider.OLLAMA,
                model="qwen3-embedding:0.6b",
                batch_size=2,
            )
            provider = OllamaEmbeddingProvider(config)

            texts = ["Hello", "World", "Test"]
            embeddings = provider.embed(texts)

            assert len(embeddings) == 3
            assert all(len(e) == 3 for e in embeddings)

    def test_empty_batch_embedding(self):
        """Test batch embedding with empty list."""
        mock_ollama = MagicMock()
        mock_client = MagicMock()
        mock_ollama.Client.return_value = mock_client

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            from cognidoc.utils.embedding_providers import OllamaEmbeddingProvider

            config = EmbeddingConfig(
                provider=EmbeddingProvider.OLLAMA,
                model="qwen3-embedding:0.6b",
            )
            provider = OllamaEmbeddingProvider(config)

            embeddings = provider.embed([])
            assert embeddings == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
