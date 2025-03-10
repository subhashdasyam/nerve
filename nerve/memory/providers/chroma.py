"""ChromaDB memory provider implementation."""

import json
import os
import pathlib
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

from loguru import logger

from nerve.memory.base import EmbeddingProvider, MemoryEntry, MemoryProvider, MemoryType
from nerve.memory.config import ChromaConfig


class ChromaMemoryProvider(MemoryProvider):
    """ChromaDB implementation of the memory provider."""
    
    def __init__(
        self, 
        config: ChromaConfig,
        embedding_provider: EmbeddingProvider,
    ):
        """
        Initialize the ChromaDB memory provider.
        
        Args:
            config: ChromaDB configuration
            embedding_provider: Provider for generating embeddings
        """
        self.config = config
        self.embedding_provider = embedding_provider
        self.client = None
        self.collection = None
        self._dimension = None
    
    async def initialize(self) -> None:
        """Initialize the ChromaDB provider, creating necessary resources."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Ensure directory exists
            path = pathlib.Path(self.config.path)
            path.mkdir(parents=True, exist_ok=True)
            
            # Create persistent client
            self.client = chromadb.PersistentClient(
                path=str(path),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.config.collection_name,
                )
                logger.info(f"Using existing ChromaDB collection: {self.config.collection_name}")
            except ValueError:
                # Get embedding dimension for collection schema
                self._dimension = await self.embedding_provider.get_embedding_dimension()
                
                self.collection = self.client.create_collection(
                    name=self.config.collection_name,
                    metadata={"dimension": self._dimension},
                )
                logger.info(f"Created new ChromaDB collection: {self.config.collection_name}")
            
            logger.info(f"ChromaDB memory provider initialized at {path}")
            
        except ImportError:
            logger.error("chromadb package not installed. Run: pip install chromadb")
            raise
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise
    
    async def store(self, entry: MemoryEntry) -> str:
        """
        Store a memory entry in ChromaDB.
        
        Args:
            entry: The memory entry to store
            
        Returns:
            The ID of the stored entry
        """
        if not self.collection:
            raise RuntimeError("ChromaDB provider not initialized. Call initialize() first.")
        
        # Generate ID if not provided
        if not entry.id:
            entry.id = str(uuid.uuid4())
        
        # Generate embedding if not provided
        if not entry.embedding:
            embeddings = await self.embedding_provider.embed([entry.content])
            entry.embedding = embeddings[0] if embeddings else None
        
        # Convert metadata to JSON-compatible format
        metadata = {**entry.metadata}
        metadata["memory_type"] = entry.memory_type.value
        metadata["created_at"] = entry.created_at.isoformat()
        metadata["updated_at"] = entry.updated_at.isoformat()
        
        # Store in ChromaDB
        self.collection.add(
            ids=[entry.id],
            documents=[entry.content],
            embeddings=[entry.embedding] if entry.embedding else None,
            metadatas=[metadata],
        )
        
        return entry.id
    
    async def retrieve(
        self, 
        query: str, 
        limit: int = 5,
        memory_type: MemoryType | None = None,
        metadata_filter: dict[str, Any] | None = None,
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
        if not self.collection:
            raise RuntimeError("ChromaDB provider not initialized. Call initialize() first.")
        
        # Generate query embedding
        query_embedding = await self.embedding_provider.embed([query])
        
        # Build where clause for filtering
        where = {}
        if memory_type:
            where["memory_type"] = memory_type.value
        
        if metadata_filter:
            where.update(metadata_filter)
            
        # Execute query
        try:
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=limit,
                where=where or None,
                include=["documents", "metadatas", "embeddings", "distances"],
            )
            
            # Convert results to MemoryEntry objects
            entries = []
            if results and results.get("ids"):
                for i, id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                    
                    # Extract memory type and timestamps from metadata
                    memory_type_str = metadata.pop("memory_type", MemoryType.EPISODIC.value)
                    created_at_str = metadata.pop("created_at", datetime.now().isoformat())
                    updated_at_str = metadata.pop("updated_at", datetime.now().isoformat())
                    
                    entry = MemoryEntry(
                        id=id,
                        content=results["documents"][0][i] if results.get("documents") else "",
                        metadata=metadata,
                        embedding=results["embeddings"][0][i] if results.get("embeddings") else None,
                        memory_type=MemoryType(memory_type_str),
                        created_at=datetime.fromisoformat(created_at_str),
                        updated_at=datetime.fromisoformat(updated_at_str),
                    )
                    entries.append(entry)
            
            return entries
            
        except Exception as e:
            logger.error(f"Error retrieving from ChromaDB: {e}")
            return []
    
    async def update(self, entry_id: str, content: str | None = None, metadata: dict[str, Any] | None = None) -> None:
        """
        Update an existing memory entry.
        
        Args:
            entry_id: The ID of the entry to update
            content: New content (if provided)
            metadata: New or updated metadata (if provided)
        """
        if not self.collection:
            raise RuntimeError("ChromaDB provider not initialized. Call initialize() first.")
        
        # Get existing entry
        try:
            result = self.collection.get(
                ids=[entry_id],
                include=["documents", "metadatas", "embeddings"],
            )
            
            if not result or not result["ids"]:
                raise ValueError(f"Memory entry with ID {entry_id} not found")
            
            existing_content = result["documents"][0] if result.get("documents") else ""
            existing_metadata = result["metadatas"][0] if result.get("metadatas") else {}
            existing_embedding = result["embeddings"][0] if result.get("embeddings") else None
            
            # Update content and regenerate embedding if needed
            new_content = content if content is not None else existing_content
            new_embedding = None
            
            if content is not None:
                embeddings = await self.embedding_provider.embed([new_content])
                new_embedding = embeddings[0] if embeddings else None
            else:
                new_embedding = existing_embedding
            
            # Update metadata
            new_metadata = dict(existing_metadata)
            if metadata:
                new_metadata.update(metadata)
            
            # Always update the timestamp
            new_metadata["updated_at"] = datetime.now().isoformat()
            
            # Update in ChromaDB
            self.collection.update(
                ids=[entry_id],
                documents=[new_content],
                embeddings=[new_embedding] if new_embedding else None,
                metadatas=[new_metadata],
            )
            
        except Exception as e:
            logger.error(f"Error updating ChromaDB entry: {e}")
            raise
    
    async def delete(self, entry_id: str) -> None:
        """
        Delete a memory entry.
        
        Args:
            entry_id: The ID of the entry to delete
        """
        if not self.collection:
            raise RuntimeError("ChromaDB provider not initialized. Call initialize() first.")
        
        try:
            self.collection.delete(ids=[entry_id])
        except Exception as e:
            logger.error(f"Error deleting ChromaDB entry: {e}")
            raise
    
    async def clear(self, memory_type: MemoryType | None = None) -> None:
        """
        Clear all memories or memories of a specific type.
        
        Args:
            memory_type: Optional type to clear (if None, clears all)
        """
        if not self.collection:
            raise RuntimeError("ChromaDB provider not initialized. Call initialize() first.")
        
        try:
            if memory_type:
                # Delete only entries with matching memory type
                where = {"memory_type": memory_type.value}
                self.collection.delete(where=where)
            else:
                # Delete all entries
                self.collection.delete()
        except Exception as e:
            logger.error(f"Error clearing ChromaDB collection: {e}")
            raise
    
    async def close(self) -> None:
        """Close the ChromaDB provider and release resources."""
        self.collection = None
        self.client = None