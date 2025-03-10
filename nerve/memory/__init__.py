# In nerve/memory/__init__.py
from nerve.memory.base import MemoryEntry, MemoryManager, MemoryType

"""Memory system for Nerve agents."""

from nerve.memory.base import MemoryEntry, MemoryManager, MemoryType

# Function has to be defined here since it's imported directly from this module
async def get_memory_manager(config):
    """
    Create a memory manager with the configured providers.
    
    Args:
        config: Memory configuration
        
    Returns:
        An initialized MemoryManager
    """
    from nerve.memory.embeddings import get_embedding_provider
    from nerve.memory.providers import get_memory_provider
    
    if not config.enabled:
        raise ValueError("Memory system is disabled in configuration")
    
    # Create embedding provider
    embedding_provider = await get_embedding_provider(config)
    
    # Create memory provider
    memory_provider = await get_memory_provider(config, embedding_provider)
    
    # Create memory manager
    manager = MemoryManager(memory_provider, embedding_provider)
    await manager.initialize()
    
    return manager