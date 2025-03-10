"""OpenAI embedding provider implementation."""

import asyncio
import os
from typing import cast

from loguru import logger
import openai
from openai import AsyncOpenAI

from nerve.memory.base import EmbeddingProvider
from nerve.memory.config import OpenAIEmbeddingConfig


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Provider for OpenAI embeddings."""
    
    def __init__(self, config: OpenAIEmbeddingConfig):
        """
        Initialize the OpenAI embedding provider.
        
        Args:
            config: OpenAI embedding configuration
        """
        self.config = config
        
        # Set API key from config or environment
        api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required. Set it in the config or OPENAI_API_KEY environment variable.")
        
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = config.model
        
        # Cache the embedding dimension
        self._dimension: int | None = None
    
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts using OpenAI.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            # Process in batches of 100 (OpenAI limit)
            batch_size = 100
            all_embeddings: list[list[float]] = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Retry logic for API reliability
                max_retries = 3
                retry_delay = 1
                
                for attempt in range(max_retries):
                    try:
                        response = await self.client.embeddings.create(
                            model=self.model,
                            input=batch,
                        )
                        
                        # Extract embeddings from response
                        batch_embeddings = [
                            cast(list[float], item.embedding) 
                            for item in response.data
                        ]
                        all_embeddings.extend(batch_embeddings)
                        break
                        
                    except openai.RateLimitError:
                        if attempt < max_retries - 1:
                            logger.warning(f"Rate limit exceeded, retrying in {retry_delay} seconds...")
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2
                        else:
                            raise
                    
                    except Exception as e:
                        logger.error(f"Error generating OpenAI embeddings: {e}")
                        raise
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            # Return zero vectors as fallback
            if self._dimension is None:
                # Use default dimension if we don't know the actual dimension
                dim = 1536  # Default for text-embedding-ada-002
            else:
                dim = self._dimension
            
            return [[0.0] * dim for _ in texts]
    
    async def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings produced by the current model.
        
        Returns:
            Embedding dimension as an integer
        """
        if self._dimension is not None:
            return self._dimension
        
        # Model dimension mapping
        dimension_map = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        
        # Check if we know the dimension for this model
        if self.model in dimension_map:
            self._dimension = dimension_map[self.model]
            return self._dimension
        
        # If not, query a simple embedding to find out
        try:
            embeddings = await self.embed(["Test"])
            if embeddings and embeddings[0]:
                self._dimension = len(embeddings[0])
                return self._dimension
            
            # Fallback
            logger.warning(f"Couldn't determine embedding dimension for {self.model}, using default")
            self._dimension = 1536
            return self._dimension
            
        except Exception as e:
            logger.error(f"Error determining embedding dimension: {e}")
            self._dimension = 1536  # Default fallback
            return self._dimension