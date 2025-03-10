# Nerve Memory System

The Memory System enables Nerve agents to store, retrieve, and reason with historical data and knowledge, allowing for more contextual and informed responses across conversations.

## Overview

The memory system provides persistent storage for:

- **Episodic Memories**: Conversations and experiences
- **Semantic Memories**: Facts and knowledge
- **Working Memories**: Short-term, high-relevance context

## Supported Vector Databases

- **ChromaDB**: An open-source embedding database
- **pgvector**: PostgreSQL extension for vector similarity search

## Supported Embedding Models

- **OpenAI**: Uses OpenAI embedding models
- **HuggingFace**: Uses local embedding models via sentence-transformers

## Configuration

Add memory support to your agent YAML file:

```yaml
agent: You are a helpful assistant.
task: Answer questions about our previous conversations.

using:
- memory  # Adds memory namespace

memory:
  provider: chroma  # or pgvector
  persist: true
  enabled: true
  embedding: openai  # or huggingface
  auto_store_conversations: true
  auto_retrieve: true
  auto_retrieve_limit: 5

  # Provider-specific configurations
  chroma:
    path: ~/.nerve/memory
    collection_name: nerve_memory

  # OpenAI embedding configuration
  openai:
    model: text-embedding-ada-002
```

## Usage in Agent Tools

With memory enabled, your agent gains access to these tools:

- `store_memory`: Store new information
- `retrieve_memory`: Search for relevant memories
- `reflect`: Analyze and reason about memories
- `clear_memories`: Delete memories

## Example: Storing a Memory

```yaml
agent: You are a helpful assistant.
task: Save this fact for future reference.

using:
- memory
- task

# Agent will use these tools to store information
```

## Example: Retrieving Memories

```yaml
agent: You are a helpful assistant.
task: What have we discussed previously about machine learning?

using:
- memory
- task

# Agent will automatically retrieve relevant memories about machine learning
```

## Installation Requirements

To use the memory system, install these dependencies:

```bash
# For ChromaDB
pip install chromadb

# For pgvector
pip install asyncpg psycopg 

# For OpenAI embeddings
pip install openai

# For HuggingFace embeddings
pip install sentence-transformers
```

## PostgreSQL Setup for pgvector

If using pgvector, install the extension on your PostgreSQL server:

```sql
CREATE EXTENSION vector;
```

## Environment Variables

You can use these environment variables for configuration:

- `OPENAI_API_KEY`: OpenAI API key for embeddings
- `NERVE_PGVECTOR_CONNECTION`: PostgreSQL connection string