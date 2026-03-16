"""Ollama Vision LLM implementation.

This module provides Ollama Vision LLM implementation for multimodal
interactions (text + image). Supports local deployment for image understanding
tasks like image captioning, visual question answering, and document analysis.
"""

from __future__ import annotations

import base64
from typing import Any, Optional

import ollama
from httpx import HTTPTransport

from src.libs.llm.base_llm import ChatResponse, Message
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput


class OllamaVisionLLMError(RuntimeError):
    """Raised when Ollama Vision API call fails."""


class OllamaVisionLLM(BaseVisionLLM):
    """Ollama Vision LLM provider implementation.
    
    This class implements the BaseVisionLLM interface for Ollama's local Vision
    Service. It handles authentication, endpoint configuration, and image preprocessing.
    
    Attributes:
        base_url: The base URL for the Ollama Vision API.
        max_image_size: Maximum image dimension in pixels (default 2048).
    
    Example:
        >>> vision_llm = OllamaVisionLLM(
        ...     base_url='http://localhost:11434',
        ... )
        >>> image = ImageInput(path="diagram.png")
        >>> response = vision_llm.chat_with_image(
        ...     text="Describe this diagram",
        ...     image=image
        ... )
    """

    DEFAULT_MAX_IMAGE_SIZE = 2048  # pixels

    def __init__(
        self,
        settings: Any,
        max_image_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Ollama Vision LLM provider.

        Args:
            settings: Application settings containing vision_llm configuration.
            max_image_size: Maximum image dimension in pixels for auto-compression.
            **kwargs: Additional configuration overrides.

        Raises:
            ValueError: If required configuration is missing.
        """
        self.base_url = getattr(settings.vision_llm, 'base_url', 'http://localhost:11434')
        if not self.base_url:
            raise ValueError("Ollama Vision API base URL must be provided.")

        self.model = getattr(settings.vision_llm, 'model', 'qwen3-vl:4b')
        self.max_image_size = max_image_size or self.DEFAULT_MAX_IMAGE_SIZE

        # Force HTTP/1.1 for compatibility with Ollama server
        transport = HTTPTransport(http1=True)
        self.client = ollama.Client(host=self.base_url, transport=transport)

        self._extra_config = kwargs

    def chat_with_image(
        self,
        text: str,
        image: ImageInput,
        messages: Optional[list[Message]] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Generate a response based on text prompt and image input.

        This method sends the text and image to Ollama Vision API and returns the
        generated response.

        Args:
            text: The text prompt or question about the image.
            image: The image input (path, bytes, or base64).
            messages: Optional conversation history for context.
            trace: Optional TraceContext for observability.
            **kwargs: Override parameters (temperature, max_tokens, etc.).

        Returns:
            ChatResponse containing the generated text and metadata.

        Raises:
            ValueError: If text or image input is invalid.
            OllamaVisionLLMError: If API call fails.

        Example:
            >>> image = ImageInput(path="chart.png")
            >>> response = vision_llm.chat_with_image(
            ...     text="What does this chart show?",
            ...     image=image
            ... )
        """
        # Validate inputs
        self.validate_text(text)
        self.validate_image(image)

        # Preprocess image (compress if needed)
        processed_image = self.preprocess_image(
            image,
            max_size=(self.max_image_size, self.max_image_size)
        )

        # Convert image to base64 if needed
        image_base64 = self._get_image_base64(processed_image)

        # Prepare messages in Ollama format
        ollama_messages = []

        # Add conversation history if provided
        if messages:
            for msg in messages:
                ollama_messages.append({
                    'role': msg.role,
                    'content': msg.content,
                })

        # Add current message with image
        ollama_messages.append({
            'role': 'user',
            'content': text,
            'images': [image_base64],
        })

        # Send request to Ollama using client.chat()
        try:
            response = self.client.chat(
                model=self.model,
                messages=ollama_messages,
                **kwargs,
            )

            # Convert ollama.ChatResponse to base_llm.ChatResponse
            return ChatResponse(
                content=response.message.content,
                model=response.model or self.model,
                usage=None,  # Ollama doesn't always provide usage stats
                raw_response=response,
            )
        except Exception as e:
            raise OllamaVisionLLMError(f"Ollama Vision API call failed: {e}") from e

    def _get_image_base64(self, image: ImageInput) -> str:
        """Convert image to base64-encoded string."""
        if image.data:
            return base64.b64encode(image.data).decode("utf-8")
        elif image.path:
            with open(image.path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        else:
            raise ValueError("Invalid image input: must provide path or data.")