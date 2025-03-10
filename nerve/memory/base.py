"""Base classes and interfaces for the memory system."""

import abc
import typing as t
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Types of memory entries."""
    
    EPISODIC = "episodic"  # Memories of events and interactions
    SEMANTIC = "semantic"  # Factual knowledge
    WORKING = "working"    # Temporary, high-relevance memory


class MemoryEntry(BaseModel):
    """A single memory entry stored in the vector database."""
    
    id: str = Field(default="")
    content: str = Field(...)
    metadata: dict[str, t.Any] = Field(default_factory=dict)
    embedding: list[float] | None = Field(default=None)
    memory_type: MemoryType = Field(default=MemoryType.EPISODIC)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True


class MemoryProvider(abc.ABC):
    """Base interface for memory storage providers."""
    
    @abc.abstractmethod
    async def initialize(self) -> None:
        """Initialize the memory provider, creating necessary resources."""
        pass
    
    @abc.abstractmethod
    async def store(self, entry: MemoryEntry) -> str:
        """
        Store a memory entry in the database.
        
        Args:
            entry: The memory entry to store
            
        Returns:
            The ID of the stored entry
        """
        pass
    
    @abc.abstractmethod
    async def retrieve(
        self, 
        query: str, 
        limit: int = 5,
        memory_type: MemoryType | None = None,
        metadata_filter: dict[str, t.Any] | None = None,
    ) -> list[MemoryEntry]:
        """
        Retrieve relevant memories based on a query.
        
        Args:
            query: The query text to search for
            limit: Maximum number of results to return
            memory_type: Optional filter by memory type
            metadata_filter: Optional filter by metadata fields
            
        Returns:
            A list of memory entries sorted by relevance
        """
        pass
    
    @abc.abstractmethod
    async def update(self, entry_id: str, content: str | None = None, metadata: dict[str, t.Any] | None = None) -> None:
        """
        Update an existing memory entry.
        
        Args:
            entry_id: The ID of the entry to update
            content: New content (if provided)
            metadata: New or updated metadata (if provided)
        """
        pass
    
    @abc.abstractmethod
    async def delete(self, entry_id: str) -> None:
        """
        Delete a memory entry.
        
        Args:
            entry_id: The ID of the entry to delete
        """
        pass
    
    @abc.abstractmethod
    async def clear(self, memory_type: MemoryType | None = None) -> None:
        """
        Clear all memories or memories of a specific type.
        
        Args:
            memory_type: Optional type to clear (if None, clears all)
        """
        pass
    
    @abc.abstractmethod
    async def close(self) -> None:
        """Close the memory provider and release resources."""
        pass


class EmbeddingProvider(abc.ABC):
    """Base interface for embedding providers."""
    
    @abc.abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abc.abstractmethod
    async def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings produced by this provider.
        
        Returns:
            Embedding dimension as an integer
        """
        pass


class MemoryManager:
    """Manager class for memory operations."""
    
    def __init__(
        self, 
        provider: MemoryProvider,
        embedding_provider: EmbeddingProvider,
    ):
        self.provider = provider
        self.embedding_provider = embedding_provider
    
    async def initialize(self) -> None:
        """Initialize the memory system."""
        await self.provider.initialize()
    
    async def store(
        self, 
        content: str, 
        memory_type: MemoryType = MemoryType.EPISODIC,
        metadata: dict[str, t.Any] | None = None,
    ) -> str:
        """
        Store a new memory.
        
        Args:
            content: The text content to store
            memory_type: The type of memory
            metadata: Additional metadata for the memory
            
        Returns:
            The ID of the stored memory
        """
        metadata = metadata or {}
        
        # Generate embedding
        embeddings = await self.embedding_provider.embed([content])
        embedding = embeddings[0] if embeddings else None
        
        # Create memory entry
        entry = MemoryEntry(
            content=content,
            metadata=metadata,
            embedding=embedding,
            memory_type=memory_type,
        )
        
        # Store in provider
        entry_id = await self.provider.store(entry)
        return entry_id
    
    async def retrieve(
        self, 
        query: str, 
        limit: int = 5,
        memory_type: MemoryType | None = None,
        metadata_filter: dict[str, t.Any] | None = None,
    ) -> list[MemoryEntry]:
        """
        Retrieve relevant memories based on a query.
        
        Args:
            query: The query text to search for
            limit: Maximum number of results to return
            memory_type: Optional filter by memory type
            metadata_filter: Optional filter by metadata fields
            
        Returns:
            A list of memory entries sorted by relevance
        """
        return await self.provider.retrieve(
            query=query,
            limit=limit,
            memory_type=memory_type,
            metadata_filter=metadata_filter,
        )
    
    async def update(
        self, 
        entry_id: str, 
        content: str | None = None, 
        metadata: dict[str, t.Any] | None = None
    ) -> None:
        """
        Update an existing memory entry.
        
        Args:
            entry_id: The ID of the entry to update
            content: New content (if provided)
            metadata: New or updated metadata (if provided)
        """
        await self.provider.update(entry_id, content, metadata)
    
    async def delete(self, entry_id: str) -> None:
        """
        Delete a memory entry.
        
        Args:
            entry_id: The ID of the entry to delete
        """
        await self.provider.delete(entry_id)
    
    async def clear(self, memory_type: MemoryType | None = None) -> None:
        """
        Clear all memories or memories of a specific type.
        
        Args:
            memory_type: Optional type to clear (if None, clears all)
        """
        await self.provider.clear(memory_type)
    
    async def close(self) -> None:
        """Close the memory manager and release resources."""
        await self.provider.close()