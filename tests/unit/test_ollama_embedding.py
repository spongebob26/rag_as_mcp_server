"""Unit tests for Ollama Embedding provider implementation.

This test suite validates the Ollama Embedding implementation using
mocked HTTP responses to ensure reliable, fast, and offline testing.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import Mock, patch, MagicMock

import pytest

# Mock ollama module before importing the module under test
sys.modules["ollama"] = MagicMock()

from src.libs.embedding.ollama_embedding import OllamaEmbedding, OllamaEmbeddingError

@pytest.fixture
def mock_settings_ollama():
    """Mock settings for Ollama embedding."""
    settings = Mock()
    settings.embedding = Mock()
    settings.embedding.provider = "ollama"
    settings.embedding.model = "nomic-embed-text"
    settings.embedding.dimensions = 768
    return settings

def test_embed_single_text(mock_settings_ollama):
    """Test embedding a single text using the Ollama client."""
    # Mock the embed function to return the correct response format
    mock_response = {"embeddings": [[0.1, 0.2, 0.3]]}
    mock_embed_function = Mock(return_value=mock_response)
    embedding = OllamaEmbedding(mock_settings_ollama, embed_function=mock_embed_function)

    # Call the embed method
    result = embedding.embed(["hello world"])

    # Assertions
    assert len(result) == 1
    assert result[0] == [0.1, 0.2, 0.3]
    mock_embed_function.assert_called_once_with(input=["hello world"], model="nomic-embed-text")

def test_embed_multiple_texts(mock_settings_ollama):
    """Test embedding multiple texts using the Ollama client."""
    # Mock the embed function to return the correct batch response format
    mock_response = {
        "embeddings": [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
    }
    mock_embed_function = Mock(return_value=mock_response)
    embedding = OllamaEmbedding(mock_settings_ollama, embed_function=mock_embed_function)

    # Call the embed method
    result = embedding.embed(["hello world", "test embedding"])

    # Assertions
    assert len(result) == 2
    assert result[0] == [0.1, 0.2, 0.3]
    assert result[1] == [0.4, 0.5, 0.6]
    # With batch processing, should be called once with the list of texts
    assert mock_embed_function.call_count == 1
    mock_embed_function.assert_called_once_with(
        input=["hello world", "test embedding"],
        model="nomic-embed-text"
    )

def test_embed_empty_list(mock_settings_ollama):
    """Test that embedding an empty list raises a ValueError."""
    embedding = OllamaEmbedding(mock_settings_ollama)

    with pytest.raises(ValueError, match="Texts list cannot be empty"):
        embedding.embed([])

def test_embed_invalid_input(mock_settings_ollama):
    """Test that embedding invalid input raises a ValueError."""
    embedding = OllamaEmbedding(mock_settings_ollama)

    with pytest.raises(ValueError, match="not a string"):
        embedding.embed([123])  # Non-string input

def test_embed_client_error(mock_settings_ollama):
    """Test handling of client errors during embedding."""
    # Mock the Ollama embed function to raise an exception
    mock_embed_function = Mock(side_effect=Exception("Client error"))
    embedding = OllamaEmbedding(mock_settings_ollama, embed_function=mock_embed_function)

    with pytest.raises(OllamaEmbeddingError, match="Failed to generate embeddings using Ollama"):
        embedding.embed(["hello world"])

def test_embed_missing_embeddings_key(mock_settings_ollama):
    """Test handling of malformed response without embeddings key."""
    # Mock response without 'embeddings' key
    mock_response = {"other_key": "some_value"}
    mock_embed_function = Mock(return_value=mock_response)
    embedding = OllamaEmbedding(mock_settings_ollama, embed_function=mock_embed_function)

    with pytest.raises(OllamaEmbeddingError, match="Failed to generate embeddings using Ollama"):
        embedding.embed(["hello world"])
