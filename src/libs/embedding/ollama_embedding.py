"""Ollama Embedding implementation using the official Ollama Python library.

This module provides the Ollama Embedding implementation that works with
locally running Ollama instances using the official Python library.
"""

from __future__ import annotations

from typing import Any, List, Optional

import ollama

from httpx import HTTPTransport
transport = HTTPTransport(http1=True)
from src.libs.embedding.base_embedding import BaseEmbedding
from src.observability.logger import get_logger

logger = get_logger(__name__)

class OllamaEmbeddingError(RuntimeError):
    """Raised when Ollama Embeddings API call fails."""


class OllamaEmbedding(BaseEmbedding):
    """Ollama Embedding provider implementation using the official Python library.
    
    This class implements the BaseEmbedding interface for Ollama's embeddings API,
    enabling local embedding generation without cloud dependencies.
    
    Attributes:
        model: The model identifier to use (e.g., 'nomic-embed-text', 'mxbai-embed-large').
        dimension: The dimensionality of embeddings produced by this model.
    
    Example:
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> embedding = OllamaEmbedding(settings)
        >>> vectors = embedding.embed(["hello world", "test"])
    """

    DEFAULT_DIMENSION = 768  # Common dimension for local embedding models

    def __init__(
        self,
        settings: Any,
        embed_function: Optional[callable] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Ollama Embedding provider.
        
        Args:
            settings: Application settings containing Embedding configuration.
            embed_function: Optional callable to use for embedding (defaults to ollama.embed).
            **kwargs: Additional configuration overrides.
        
        Raises:
            ValueError: If required configuration is missing.
        """
        self.model = settings.embedding.model
        self.dimension = getattr(settings.embedding, 'dimensions', self.DEFAULT_DIMENSION)
        self.base_url = settings.embedding.base_url
        self.client = ollama.Client(host=self.base_url, transport=transport)

    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """Generate embeddings for a batch of texts using Ollama Python library.
        
        Args:
            texts: List of text strings to embed. Must not be empty.
            trace: Optional TraceContext for observability (reserved for Stage F).
            **kwargs: Additional parameters (currently unused, reserved for future).
        
        Returns:
            List of embedding vectors, where each vector is a list of floats.
        
        Raises:
            ValueError: If texts list is empty or contains invalid entries.
            OllamaEmbeddingError: If embedding generation fails.
        
        Example:
            >>> embeddings = embedding.embed(["hello", "world"])
            >>> len(embeddings)  # 2 vectors
            >>> len(embeddings[0])  # dimension (e.g., 768)
        """
        # Validate input
        self.validate_texts(texts)

        try:
            # Ollama supports batch input for better performance
            # Response format: {'embeddings': [[...], [...], ...]}
            logger.info(f"Embedding texts: {texts}")
            response = self.client.embed(input=texts, model=self.model)
            embeddings = response['embeddings']
            logger.info(f"Ollama response: {response}")
            logger.info(f"Generated {len(embeddings)} embeddings using Ollama model '{self.model}'.")
            return embeddings
        except Exception as e:
            raise OllamaEmbeddingError(
                f"Failed to generate embeddings using Ollama: {str(e)}"
            ) from e

    def get_dimension(self) -> int:
        """Get the dimensionality of embeddings produced by this provider.
        
        Returns:
            The vector dimension configured for this instance.
        
        Note:
            The actual dimension may vary by model. Common dimensions:
            - nomic-embed-text: 768
            - mxbai-embed-large: 1024
            Configure via settings.embedding.dimensions or accepts default 768.
        """
        return self.dimension
