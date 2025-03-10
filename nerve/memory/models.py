"""Data models for memory entries."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from nerve.memory.base import MemoryEntry, MemoryType


class ConversationMemory(MemoryEntry):
    """Memory entry for a conversation message."""
    
    role: str = Field(...)  # "user" or "assistant"
    conversation_id: str = Field(...)
    message_index: int = Field(default=0)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    
    def __init__(self, **data: Any):
        """Initialize with default memory type and metadata extraction."""
        if "memory_type" not in data:
            data["memory_type"] = MemoryType.EPISODIC
        
        if "metadata" not in data:
            data["metadata"] = {}
        
        # Move specific fields to metadata
        metadata = data.get("metadata", {})
        for field in ["role", "conversation_id", "message_index", "tool_calls"]:
            if field in data and field != "metadata":
                metadata[field] = data[field]
        
        data["metadata"] = metadata
        super().__init__(**data)
    
    @property
    def role(self) -> str:
        """Get the role from metadata."""
        return self.metadata.get("role", "unknown")
    
    @property
    def conversation_id(self) -> str:
        """Get the conversation ID from metadata."""
        return self.metadata.get("conversation_id", "")
    
    @property
    def message_index(self) -> int:
        """Get the message index from metadata."""
        return self.metadata.get("message_index", 0)
    
    @property
    def tool_calls(self) -> List[Dict[str, Any]]:
        """Get the tool calls from metadata."""
        return self.metadata.get("tool_calls", [])


class FactMemory(MemoryEntry):
    """Memory entry for a semantic fact."""
    
    subject: str = Field(...)
    predicate: str = Field(...)
    object: str = Field(...)
    confidence: float = Field(default=1.0)
    source: str = Field(default="")
    
    def __init__(self, **data: Any):
        """Initialize with default memory type and metadata extraction."""
        if "memory_type" not in data:
            data["memory_type"] = MemoryType.SEMANTIC
        
        if "metadata" not in data:
            data["metadata"] = {}
        
        # Format content as triple if not provided
        if "content" not in data and all(f in data for f in ["subject", "predicate", "object"]):
            data["content"] = f"{data['subject']} {data['predicate']} {data['object']}"
        
        # Move specific fields to metadata
        metadata = data.get("metadata", {})
        for field in ["subject", "predicate", "object", "confidence", "source"]:
            if field in data and field != "metadata":
                metadata[field] = data[field]
        
        data["metadata"] = metadata
        super().__init__(**data)
    
    @property
    def subject(self) -> str:
        """Get the subject from metadata."""
        return self.metadata.get("subject", "")
    
    @property
    def predicate(self) -> str:
        """Get the predicate from metadata."""
        return self.metadata.get("predicate", "")
    
    @property
    def object(self) -> str:
        """Get the object from metadata."""
        return self.metadata.get("object", "")
    
    @property
    def confidence(self) -> float:
        """Get the confidence from metadata."""
        return self.metadata.get("confidence", 1.0)
    
    @property
    def source(self) -> str:
        """Get the source from metadata."""
        return self.metadata.get("source", "")


class ReflectionMemory(MemoryEntry):
    """Memory entry for a reflection or insight."""
    
    topic: str = Field(...)
    related_memory_ids: List[str] = Field(default_factory=list)
    
    def __init__(self, **data: Any):
        """Initialize with default memory type and metadata extraction."""
        if "memory_type" not in data:
            data["memory_type"] = MemoryType.SEMANTIC
        
        if "metadata" not in data:
            data["metadata"] = {}
        
        # Move specific fields to metadata
        metadata = data.get("metadata", {})
        for field in ["topic", "related_memory_ids"]:
            if field in data and field != "metadata":
                metadata[field] = data[field]
        
        data["metadata"] = metadata
        super().__init__(**data)
    
    @property
    def topic(self) -> str:
        """Get the topic from metadata."""
        return self.metadata.get("topic", "")
    
    @property
    def related_memory_ids(self) -> List[str]:
        """Get the related memory IDs from metadata."""
        return self.metadata.get("related_memory_ids", [])