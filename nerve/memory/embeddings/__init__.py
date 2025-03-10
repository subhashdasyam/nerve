"""Embedding providers for the memory system."""

from nerve.memory.base import EmbeddingProvider
from nerve.memory.config import EmbeddingProviderType, MemoryConfig


async def get_embedding_provider(config: MemoryConfig) -> EmbeddingProvider:
    """
    Get the configured embedding provider.
    
    Args:
        config: Memory configuration
        
    Returns:
        An initialized embedding provider
    """
    if config.embedding == EmbeddingProviderType.OPENAI:
        from nerve.memory.embeddings.openai import OpenAIEmbeddingProvider
        return OpenAIEmbeddingProvider(config.openai)
    
    elif config.embedding == EmbeddingProviderType.HUGGINGFACE:
        from nerve.memory.embeddings.huggingface import HuggingFaceEmbeddingProvider
        return HuggingFaceEmbeddingProvider(config.huggingface)
    
    else:
        raise ValueError(f"Unsupported embedding provider: {config.embedding}")