"""
Memory storage and retrieval capabilities.
"""

import asyncio
import typing as t
from typing import Annotated

from loguru import logger

import nerve.runtime.state as state
from nerve.memory.base import MemoryManager, MemoryType
from nerve.memory.config import MemoryConfig

# Global memory manager instance
_memory_manager: MemoryManager | None = None


async def _get_memory_manager() -> MemoryManager:
    """
    Get or initialize the memory manager.
    
    Returns:
        An initialized memory manager
    """
    global _memory_manager
    
    if _memory_manager is None:
        # Lazy import to avoid circular dependencies
        from nerve.memory import get_memory_manager
        
        # Load memory configuration
        memory_config_data = state.get_variable("memory", {})
        if not memory_config_data:
            # Default memory configuration
            memory_config = MemoryConfig()
        else:
            memory_config = MemoryConfig(**memory_config_data)
        
        logger.debug(f"Initializing memory manager with config: {memory_config}")
        _memory_manager = await get_memory_manager(memory_config)
    
    return _memory_manager


async def store_memory(
    content: Annotated[str, "The text content to store in memory"],
    memory_type: Annotated[str, "The type of memory (episodic, semantic, working)"] = "episodic",
    metadata: Annotated[str, "Optional JSON metadata as a string"] = "{}",
) -> str:
    """
    Store a new memory.
    
    Stores the provided content in the memory system with optional metadata.
    The memory will be searchable using semantic retrieval later.
    """
    try:
        # Parse memory type
        try:
            mem_type = MemoryType(memory_type.lower())
        except ValueError:
            logger.warning(f"Invalid memory type: {memory_type}, using episodic")
            mem_type = MemoryType.EPISODIC
        
        # Parse metadata
        try:
            import json
            meta = json.loads(metadata)
            if not isinstance(meta, dict):
                logger.warning(f"Metadata is not a valid JSON object, using empty metadata")
                meta = {}
        except Exception:
            logger.warning(f"Failed to parse metadata as JSON, using empty metadata")
            meta = {}
        
        # Store memory
        manager = await _get_memory_manager()
        entry_id = await manager.store(content, mem_type, meta)
        
        return f"Successfully stored memory with ID: {entry_id}"
    
    except Exception as e:
        logger.error(f"Error storing memory: {e}")
        return f"Failed to store memory: {e}"


async def retrieve_memory(
    query: Annotated[str, "The query to search for relevant memories"],
    limit: Annotated[int, "Maximum number of memories to retrieve"] = 5,
    memory_type: Annotated[str, "Optional filter by memory type (episodic, semantic, working)"] = "",
    metadata_filter: Annotated[str, "Optional JSON metadata filter as a string"] = "{}",
) -> str:
    """
    Retrieve relevant memories based on a query.
    
    Searches for memories semantically similar to the query text,
    optionally filtered by memory type and metadata.
    """
    try:
        # Parse memory type
        mem_type = None
        if memory_type:
            try:
                mem_type = MemoryType(memory_type.lower())
            except ValueError:
                logger.warning(f"Invalid memory type: {memory_type}, using no type filter")
        
        # Parse metadata filter
        try:
            import json
            meta_filter = json.loads(metadata_filter)
            if not isinstance(meta_filter, dict):
                logger.warning(f"Metadata filter is not a valid JSON object, using no metadata filter")
                meta_filter = {}
        except Exception:
            logger.warning(f"Failed to parse metadata filter as JSON, using no metadata filter")
            meta_filter = {}
        
        # Retrieve memories
        manager = await _get_memory_manager()
        entries = await manager.retrieve(
            query=query,
            limit=limit,
            memory_type=mem_type,
            metadata_filter=meta_filter,
        )
        
        if not entries:
            return "No relevant memories found."
        
        # Format results
        result = f"Found {len(entries)} relevant memories:\n\n"
        
        for i, entry in enumerate(entries):
            result += f"Memory #{i+1} (ID: {entry.id}, Type: {entry.memory_type.value}):\n"
            result += f"Created: {entry.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            # Add metadata if present
            if entry.metadata:
                meta_str = ", ".join(f"{k}: {v}" for k, v in entry.metadata.items())
                result += f"Metadata: {meta_str}\n"
            
            # Add content
            result += f"Content: {entry.content}\n\n"
        
        return result
    
    except Exception as e:
        logger.error(f"Error retrieving memories: {e}")
        return f"Failed to retrieve memories: {e}"


async def reflect(
    query: Annotated[str, "The topic or question to reflect on"],
    limit: Annotated[int, "Maximum number of memories to consider"] = 10,
) -> str:
    """
    Analyze and reason about past memories related to a topic.
    
    Retrieves relevant memories and generates insights or conclusions based on them.
    """
    try:
        # Retrieve relevant memories
        manager = await _get_memory_manager()
        entries = await manager.retrieve(query=query, limit=limit)
        
        if not entries:
            return "No relevant memories found to reflect upon."
        
        # Format memory context
        memory_context = "Based on my memory, I can reflect on the following relevant information:\n\n"
        
        for i, entry in enumerate(entries):
            memory_context += f"Memory #{i+1} (Type: {entry.memory_type.value}):\n"
            
            # Add metadata if present
            if entry.metadata:
                meta_str = ", ".join(f"{k}: {v}" for k, v in entry.metadata.items())
                memory_context += f"Context: {meta_str}\n"
            
            # Add content
            memory_context += f"{entry.content}\n\n"
        
        # Store reflection result
        reflection = f"Memory reflection on '{query}':\n\n{memory_context}"
        
        return reflection
    
    except Exception as e:
        logger.error(f"Error during reflection: {e}")
        return f"Failed to reflect on memories: {e}"


async def clear_memories(
    memory_type: Annotated[str, "The type of memories to clear (episodic, semantic, working, or 'all')"] = "all",
) -> str:
    """
    Clear all memories of a specific type or all memories.
    
    This permanently deletes memories from storage.
    """
    try:
        manager = await _get_memory_manager()
        
        if memory_type.lower() == "all":
            await manager.clear()
            return "Successfully cleared all memories."
        else:
            try:
                mem_type = MemoryType(memory_type.lower())
                await manager.clear(mem_type)
                return f"Successfully cleared all {memory_type} memories."
            except ValueError:
                return f"Invalid memory type: {memory_type}. Valid types are: episodic, semantic, working, or all."
    
    except Exception as e:
        logger.error(f"Error clearing memories: {e}")
        return f"Failed to clear memories: {e}"