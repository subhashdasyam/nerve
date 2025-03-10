"""PostgreSQL with pgvector memory provider implementation."""

import json
import uuid
from datetime import datetime
from typing import Any, List, Optional

import asyncpg
from loguru import logger

from nerve.memory.base import EmbeddingProvider, MemoryEntry, MemoryProvider, MemoryType
from nerve.memory.config import PGVectorConfig


class PGVectorMemoryProvider(MemoryProvider):
    """PostgreSQL with pgvector implementation of the memory provider."""
    
    def __init__(
        self, 
        config: PGVectorConfig,
        embedding_provider: EmbeddingProvider,
    ):
        """
        Initialize the pgvector memory provider.
        
        Args:
            config: PostgreSQL configuration
            embedding_provider: Provider for generating embeddings
        """
        self.config = config
        self.embedding_provider = embedding_provider
        self.conn = None
        self._dimension = None
    
    async def initialize(self) -> None:
        """Initialize the pgvector provider, creating necessary resources."""
        try:
            # Get embedding dimension for schema
            self._dimension = await self.embedding_provider.get_embedding_dimension()
            
            # Connect to database
            self.conn = await asyncpg.connect(self.config.connection_string)
            
            # Check if pgvector extension is installed
            result = await self.conn.fetchval("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector');")
            if not result:
                await self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                logger.info("Created pgvector extension")
            
            # Create schema if not exists
            await self.conn.execute(f'CREATE SCHEMA IF NOT EXISTS {self.config.schema_name};')
            
            # Create table if not exists
            table_name = f"{self.config.schema_name}.{self.config.table_name}"
            await self.conn.execute(f'''
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL,
                    embedding vector({self._dimension}) NOT NULL,
                    memory_type TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    updated_at TIMESTAMP WITH TIME ZONE NOT NULL
                );
            ''')
            
            # Create index for vector similarity search if not exists
            index_exists = await self.conn.fetchval(f'''
                SELECT EXISTS (
                    SELECT 1 FROM pg_indexes
                    WHERE schemaname = '{self.config.schema_name}'
                    AND tablename = '{self.config.table_name}'
                    AND indexname = '{self.config.table_name}_embedding_idx'
                );
            ''')
            
            if not index_exists:
                # Create vector index
                await self.conn.execute(f'''
                    CREATE INDEX {self.config.table_name}_embedding_idx
                    ON {table_name} USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                ''')
                
                # Create index on memory_type
                await self.conn.execute(f'''
                    CREATE INDEX {self.config.table_name}_memory_type_idx
                    ON {table_name} (memory_type);
                ''')
                
                logger.info(f"Created pgvector indexes on {table_name}")
            
            logger.info(f"PGVector memory provider initialized with table {table_name}")
            
        except ImportError:
            logger.error("asyncpg package not installed. Run: pip install asyncpg")
            raise
        except Exception as e:
            logger.error(f"Error initializing pgvector: {e}")
            if self.conn:
                await self.conn.close()
                self.conn = None
            raise
    
    async def store(self, entry: MemoryEntry) -> str:
        """
        Store a memory entry in PostgreSQL.
        
        Args:
            entry: The memory entry to store
            
        Returns:
            The ID of the stored entry
        """
        if not self.conn:
            raise RuntimeError("PGVector provider not initialized. Call initialize() first.")
        
        # Generate ID if not provided
        if not entry.id:
            entry.id = str(uuid.uuid4())
        
        # Generate embedding if not provided
        if not entry.embedding:
            embeddings = await self.embedding_provider.embed([entry.content])
            entry.embedding = embeddings[0] if embeddings else None
        
        # Insert into database
        table_name = f"{self.config.schema_name}.{self.config.table_name}"
        
        try:
            await self.conn.execute(f'''
                INSERT INTO {table_name}
                (id, content, metadata, embedding, memory_type, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (id) DO UPDATE
                SET content = $2, metadata = $3, embedding = $4, memory_type = $5, updated_at = $7
            ''',
                entry.id,
                entry.content,
                json.dumps(entry.metadata),
                entry.embedding,
                entry.memory_type.value,
                entry.created_at,
                entry.updated_at
            )
            
            return entry.id
            
        except Exception as e:
            logger.error(f"Error storing pgvector entry: {e}")
            raise
    
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
        if not self.conn:
            raise RuntimeError("PGVector provider not initialized. Call initialize() first.")
        
        # Generate query embedding
        query_embeddings = await self.embedding_provider.embed([query])
        if not query_embeddings or not query_embeddings[0]:
            return []
        
        query_embedding = query_embeddings[0]
        
        # Build SQL query with optional filters
        table_name = f"{self.config.schema_name}.{self.config.table_name}"
        sql = f'''
            SELECT id, content, metadata, embedding, memory_type, created_at, updated_at, 
                   1 - (embedding <=> $1) as similarity
            FROM {table_name}
            WHERE 1=1
        '''
        
        params = [query_embedding]
        param_index = 2
        
        # Add memory_type filter
        if memory_type:
            sql += f" AND memory_type = ${param_index}"
            params.append(memory_type.value)
            param_index += 1
        
        # Add metadata filters
        if metadata_filter:
            for key, value in metadata_filter.items():
                sql += f" AND metadata ->> '{key}' = ${param_index}"
                params.append(str(value))
                param_index += 1
        
        # Add ordering and limit
        sql += f'''
            ORDER BY similarity DESC
            LIMIT {limit}
        '''
        
        try:
            # Execute query
            rows = await self.conn.fetch(sql, *params)
            
            # Convert rows to MemoryEntry objects
            entries = []
            for row in rows:
                entry = MemoryEntry(
                    id=row['id'],
                    content=row['content'],
                    metadata=json.loads(row['metadata']),
                    embedding=row['embedding'],
                    memory_type=MemoryType(row['memory_type']),
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                )
                entries.append(entry)
            
            return entries
            
        except Exception as e:
            logger.error(f"Error retrieving from pgvector: {e}")
            return []
    
    async def update(self, entry_id: str, content: str | None = None, metadata: dict[str, Any] | None = None) -> None:
        """
        Update an existing memory entry.
        
        Args:
            entry_id: The ID of the entry to update
            content: New content (if provided)
            metadata: New or updated metadata (if provided)
        """
        if not self.conn:
            raise RuntimeError("PGVector provider not initialized. Call initialize() first.")
        
        table_name = f"{self.config.schema_name}.{self.config.table_name}"
        
        try:
            # Get existing entry
            row = await self.conn.fetchrow(f'''
                SELECT content, metadata, embedding
                FROM {table_name}
                WHERE id = $1
            ''', entry_id)
            
            if not row:
                raise ValueError(f"Memory entry with ID {entry_id} not found")
            
            existing_content = row['content']
            existing_metadata = json.loads(row['metadata'])
            existing_embedding = row['embedding']
            
            # Update content and regenerate embedding if needed
            new_content = content if content is not None else existing_content
            new_embedding = existing_embedding
            
            if content is not None:
                embeddings = await self.embedding_provider.embed([new_content])
                new_embedding = embeddings[0] if embeddings else existing_embedding
            
            # Update metadata
            new_metadata = dict(existing_metadata)
            if metadata:
                new_metadata.update(metadata)
            
            # Update in database
            await self.conn.execute(f'''
                UPDATE {table_name}
                SET content = $1, metadata = $2, embedding = $3, updated_at = $4
                WHERE id = $5
            ''',
                new_content,
                json.dumps(new_metadata),
                new_embedding,
                datetime.now(),
                entry_id
            )
            
        except Exception as e:
            logger.error(f"Error updating pgvector entry: {e}")
            raise
    
    async def delete(self, entry_id: str) -> None:
        """
        Delete a memory entry.
        
        Args:
            entry_id: The ID of the entry to delete
        """
        if not self.conn:
            raise RuntimeError("PGVector provider not initialized. Call initialize() first.")
        
        table_name = f"{self.config.schema_name}.{self.config.table_name}"
        
        try:
            await self.conn.execute(f'''
                DELETE FROM {table_name}
                WHERE id = $1
            ''', entry_id)
        except Exception as e:
            logger.error(f"Error deleting pgvector entry: {e}")
            raise
    
    async def clear(self, memory_type: MemoryType | None = None) -> None:
        """
        Clear all memories or memories of a specific type.
        
        Args:
            memory_type: Optional type to clear (if None, clears all)
        """
        if not self.conn:
            raise RuntimeError("PGVector provider not initialized. Call initialize() first.")
        
        table_name = f"{self.config.schema_name}.{self.config.table_name}"
        
        try:
            if memory_type:
                # Delete only entries with matching memory type
                await self.conn.execute(f'''
                    DELETE FROM {table_name}
                    WHERE memory_type = $1
                ''', memory_type.value)
            else:
                # Delete all entries
                await self.conn.execute(f'''
                    DELETE FROM {table_name}
                ''')
        except Exception as e:
            logger.error(f"Error clearing pgvector table: {e}")
            raise
    
    async def close(self) -> None:
        """Close the pgvector provider and release resources."""
        if self.conn:
            await self.conn.close()
            self.conn = None