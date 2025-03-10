"""Configuration models for the memory system."""

import os
import pathlib
from enum import Enum

from pydantic import BaseModel, Field

from nerve.cli.defaults import DEFAULT_NERVE_HOME


class MemoryProviderType(str, Enum):
    """Types of supported memory providers."""
    
    CHROMA = "chroma"
    PGVECTOR = "pgvector"


class EmbeddingProviderType(str, Enum):
    """Types of supported embedding providers."""
    
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


class ChromaConfig(BaseModel):
    """Configuration for ChromaDB provider."""
    
    path: str = Field(
        default=str(DEFAULT_NERVE_HOME / "memory"),
        description="Path to the ChromaDB persistence directory",
    )
    collection_name: str = Field(
        default="nerve_memory",
        description="Name of the collection to use",
    )


class PGVectorConfig(BaseModel):
    """Configuration for pgvector provider."""
    
    connection_string: str = Field(
        default=os.environ.get("NERVE_PGVECTOR_CONNECTION", ""),
        description="PostgreSQL connection string",
    )
    table_name: str = Field(
        default="nerve_memory",
        description="Name of the table to use",
    )
    schema_name: str = Field(
        default="public",
        description="PostgreSQL schema name",
    )


class OpenAIEmbeddingConfig(BaseModel):
    """Configuration for OpenAI embeddings."""
    
    api_key: str = Field(
        default=os.environ.get("OPENAI_API_KEY", ""),
        description="OpenAI API key",
    )
    model: str = Field(
        default="text-embedding-ada-002",
        description="OpenAI embedding model to use",
    )


class HuggingFaceEmbeddingConfig(BaseModel):
    """Configuration for HuggingFace embeddings."""
    
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model name",
    )
    device: str = Field(
        default="cpu",
        description="Device to use (cpu or cuda)",
    )


class MemoryConfig(BaseModel):
    """Configuration for the memory system."""
    
    enabled: bool = Field(
        default=True,
        description="Whether memory is enabled",
    )
    provider: MemoryProviderType = Field(
        default=MemoryProviderType.CHROMA,
        description="Memory provider to use",
    )
    embedding: EmbeddingProviderType = Field(
        default=EmbeddingProviderType.OPENAI,
        description="Embedding provider to use",
    )
    persist: bool = Field(
        default=True,
        description="Whether to persist memory between runs",
    )
    auto_store_conversations: bool = Field(
        default=True,
        description="Whether to automatically store conversations",
    )
    auto_retrieve: bool = Field(
        default=True,
        description="Whether to automatically retrieve relevant memories",
    )
    auto_retrieve_limit: int = Field(
        default=5,
        description="Maximum number of memories to auto-retrieve",
    )
    
    # Provider-specific configurations
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    pgvector: PGVectorConfig = Field(default_factory=PGVectorConfig)
    
    # Embedding-specific configurations
    openai: OpenAIEmbeddingConfig = Field(default_factory=OpenAIEmbeddingConfig)
    huggingface: HuggingFaceEmbeddingConfig = Field(default_factory=HuggingFaceEmbeddingConfig)