"""Utility functions for the memory system."""

import re
import typing as t
from datetime import datetime

from nerve.memory.base import MemoryEntry, MemoryManager, MemoryType


def extract_key_information(content: str) -> dict[str, t.Any]:
    """
    Extract key information from text content to use as metadata.
    
    Args:
        content: Text content to analyze
        
    Returns:
        Dictionary of extracted metadata
    """
    metadata = {}
    
    # Extract dates
    date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{2,4}\b'
    dates = re.findall(date_pattern, content, re.IGNORECASE)
    if dates:
        metadata['dates'] = dates
    
    # Extract potential entities (people, organizations, locations)
    entity_pattern = r'\b[A-Z][a-z]+ (?:[A-Z][a-z]+ )*[A-Z][a-z]+\b'
    entities = re.findall(entity_pattern, content)
    if entities:
        metadata['entities'] = entities[:5]  # Limit to top 5
    
    # Extract topics based on keyword frequency
    words = re.findall(r'\b[a-z]{4,}\b', content.lower())
    word_freq = {}
    for word in words:
        if word not in ['that', 'this', 'with', 'from', 'have', 'were', 'they', 'their', 'would', 'about', 'there']:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Get top 3 most frequent words as topics
    topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:3]
    if topics:
        metadata['topics'] = [topic[0] for topic in topics]
    
    return metadata


def format_memories_as_context(entries: list[MemoryEntry], query: str = "") -> str:
    """
    Format memory entries as a context block for the agent.
    
    Args:
        entries: List of memory entries to format
        query: Optional query that triggered the retrieval
        
    Returns:
        Formatted context as a string
    """
    if not entries:
        return "No relevant memories found."
    
    context = f"Relevant memories"
    if query:
        context += f" related to '{query}'"
    context += ":\n\n"
    
    for i, entry in enumerate(entries):
        context += f"Memory #{i+1} ({entry.memory_type.value})\n"
        
        # Add timestamp
        context += f"From: {entry.created_at.strftime('%Y-%m-%d %H:%M')}\n"
        
        # Add metadata if present
        if entry.metadata:
            meta_str = ", ".join(f"{k}: {v}" for k, v in entry.metadata.items())
            if len(meta_str) > 100:
                meta_str = meta_str[:97] + "..."
            context += f"Context: {meta_str}\n"
        
        # Add content
        context += f"Content: {entry.content}\n\n"
    
    return context


async def store_conversation_memory(
    manager: MemoryManager,
    user_message: str,
    assistant_message: str,
    conversation_id: str,
    tool_calls: list[dict[str, t.Any]] | None = None,
) -> tuple[str, str]:
    """
    Store conversation messages as memory entries.
    
    Args:
        manager: Memory manager instance
        user_message: The user's message
        assistant_message: The assistant's response
        conversation_id: Unique identifier for the conversation
        tool_calls: Optional list of tool calls made by the assistant
        
    Returns:
        Tuple of (user_entry_id, assistant_entry_id)
    """
    # Generate metadata
    timestamp = datetime.now().isoformat()
    metadata = {
        "conversation_id": conversation_id,
        "timestamp": timestamp,
    }
    
    # Extract additional metadata from content
    user_metadata = {**metadata, **extract_key_information(user_message), "role": "user"}
    assistant_metadata = {**metadata, **extract_key_information(assistant_message), "role": "assistant"}
    
    # Add tool call information if available
    if tool_calls:
        tool_names = [call.get("name", "unknown") for call in tool_calls]
        assistant_metadata["tool_calls"] = tool_names
    
    # Store user message
    user_entry_id = await manager.store(
        content=user_message,
        memory_type=MemoryType.EPISODIC,
        metadata=user_metadata,
    )
    
    # Store assistant message
    assistant_entry_id = await manager.store(
        content=assistant_message,
        memory_type=MemoryType.EPISODIC,
        metadata=assistant_metadata,
    )
    
    return user_entry_id, assistant_entry_id


async def retrieve_relevant_context(
    manager: MemoryManager, 
    query: str, 
    limit: int = 5,
    include_semantic: bool = True,
) -> str:
    """
    Retrieve and format relevant context for a query.
    
    Args:
        manager: Memory manager instance
        query: Query text to search for
        limit: Maximum number of entries to retrieve
        include_semantic: Whether to include semantic memories
        
    Returns:
        Formatted context string
    """
    # Retrieve episodic memories (conversations)
    episodic_entries = await manager.retrieve(
        query=query,
        limit=limit,
        memory_type=MemoryType.EPISODIC,
    )
    
    # Optionally retrieve semantic memories (facts, knowledge)
    semantic_entries = []
    if include_semantic:
        semantic_entries = await manager.retrieve(
            query=query,
            limit=limit // 2,  # Use fewer semantic entries
            memory_type=MemoryType.SEMANTIC,
        )
    
    # Combine and format
    all_entries = episodic_entries + semantic_entries
    all_entries.sort(key=lambda x: x.created_at, reverse=True)
    
    return format_memories_as_context(all_entries[:limit], query)