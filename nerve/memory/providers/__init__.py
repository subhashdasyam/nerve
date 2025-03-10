"""Memory providers for the memory system."""

from nerve.memory.base import EmbeddingProvider, MemoryProvider
from nerve.memory.config import MemoryConfig, MemoryProviderType


async def get_memory_provider(config: MemoryConfig, embedding_provider: EmbeddingProvider) -> MemoryProvider:
    """
    Get the configured memory provider.
    
    Args:
        config: Memory configuration
        embedding_provider: Provider for generating embeddings
        
    Returns:
        An initialized memory provider
    """
    if config.provider == MemoryProviderType.CHROMA:
        from nerve.memory.providers.chroma import ChromaMemoryProvider
        return ChromaMemoryProvider(config.chroma, embedding_provider)
    
    elif config.provider == MemoryProviderType.PGVECTOR:
        from nerve.memory.providers.pgvector import PGVectorMemoryProvider
        return PGVectorMemoryProvider(config.pgvector, embedding_provider)
    
    else:
        raise ValueError(f"Unsupported memory provider: {config.provider}")