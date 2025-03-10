"""HuggingFace embedding provider implementation."""

import os
import asyncio
from functools import lru_cache
from typing import Any, List, cast

from loguru import logger

from nerve.memory.base import EmbeddingProvider
from nerve.memory.config import HuggingFaceEmbeddingConfig


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """Provider for HuggingFace embeddings."""
    
    def __init__(self, config: HuggingFaceEmbeddingConfig):
        """
        Initialize the HuggingFace embedding provider.
        
        Args:
            config: HuggingFace embedding configuration
        """
        self.config = config
        self.model_name = config.model_name
        self.device = config.device
        
        # Models are loaded lazily
        self._model: Any = None
        self._tokenizer: Any = None
        self._dimension: int | None = None
    
    @lru_cache(maxsize=8)
    def _ensure_model_loaded(self) -> tuple[Any, Any]:
        """
        Ensure the model and tokenizer are loaded.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            # Import here to avoid dependency if not used
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading HuggingFace model: {self.model_name}")
            model = SentenceTransformer(self.model_name, device=self.device)
            
            return model, None
            
        except ImportError:
            logger.error("sentence-transformers package not installed. Run: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Error loading HuggingFace model: {e}")
            raise
    
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts using HuggingFace.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            # Load model if not already loaded
            model, _ = self._ensure_model_loaded()
            
            # Process embeddings in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Use process_pool for CPU-intensive tasks
            embeddings = await loop.run_in_executor(
                None,  # Uses default executor
                lambda: model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
            )
            
            # Convert numpy arrays to Python lists
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Failed to generate HuggingFace embeddings: {e}")
            
            # Return zero vectors as fallback
            if self._dimension is None:
                # Use default dimension if we don't know the actual dimension
                self._dimension = await self.get_embedding_dimension()
            
            return [[0.0] * self._dimension for _ in texts]
    
    async def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings produced by the current model.
        
        Returns:
            Embedding dimension as an integer
        """
        if self._dimension is not None:
            return self._dimension
        
        try:
            # Load model if not already loaded
            model, _ = self._ensure_model_loaded()
            
            # Get dimension from model
            self._dimension = model.get_sentence_embedding_dimension()
            return self._dimension
            
        except Exception as e:
            logger.error(f"Error determining embedding dimension: {e}")
            # Default fallback - common for sentence-transformer models
            self._dimension = 384  
            return self._dimension