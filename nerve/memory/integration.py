"""Integration between memory system and Nerve agent system."""

import asyncio
import uuid
from typing import Any, Dict, List, Optional

from loguru import logger

import nerve.runtime.state as state
from nerve.memory import MemoryManager, get_memory_manager
from nerve.memory.base import MemoryType
from nerve.memory.config import MemoryConfig
from nerve.memory.utils import retrieve_relevant_context, store_conversation_memory


class MemoryIntegration:
    """Integration between memory system and Nerve agent system."""
    
    def __init__(self, config: MemoryConfig):
        """
        Initialize the memory integration.
        
        Args:
            config: Memory configuration
        """
        self.config = config
        self.manager: Optional[MemoryManager] = None
        self.conversation_id = str(uuid.uuid4())
        self.message_count = 0
    
    async def initialize(self) -> None:
        """Initialize the memory integration."""
        if not self.config.enabled:
            logger.info("Memory system is disabled")
            return
        
        try:
            self.manager = await get_memory_manager(self.config)
            logger.info(f"Memory system initialized with {self.config.provider} provider")
        except Exception as e:
            logger.error(f"Failed to initialize memory system: {e}")
            # Disable memory system
            self.config.enabled = False
    
    async def before_step(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Add relevant memories to the prompt before agent step.
        
        Args:
            system_prompt: Current system prompt
            user_prompt: Current user prompt
            
        Returns:
            Updated knowledge
        """
        if not self.config.enabled or not self.manager or not self.config.auto_retrieve:
            return {}
        
        try:
            # Retrieve relevant memories based on user prompt
            context = await retrieve_relevant_context(
                self.manager,
                user_prompt,
                limit=self.config.auto_retrieve_limit,
            )
            
            # Add to knowledge if we found relevant memories
            if context and context != "No relevant memories found.":
                return {"memory_context": context}
            
            return {}
            
        except Exception as e:
            logger.error(f"Error retrieving memory context: {e}")
            return {}
    
    async def after_step(
        self, 
        user_prompt: str, 
        assistant_response: str,
        tool_calls: List[Dict[str, Any]] | None = None,
    ) -> None:
        """
        Store conversation in memory after agent step.
        
        Args:
            user_prompt: User prompt
            assistant_response: Assistant response
            tool_calls: Optional list of tool calls
        """
        if not self.config.enabled or not self.manager or not self.config.auto_store_conversations:
            return
        
        try:
            await store_conversation_memory(
                self.manager,
                user_prompt,
                assistant_response,
                self.conversation_id,
                tool_calls,
            )
            
            self.message_count += 1
            
        except Exception as e:
            logger.error(f"Error storing conversation in memory: {e}")
    
    async def close(self) -> None:
        """Close the memory integration."""
        if self.manager:
            await self.manager.close()
            self.manager = None


def update_configuration_models() -> None:
    """Update Nerve's configuration models to include memory support."""
    from nerve.models import Configuration
    
    # Add memory field to Configuration if it doesn't exist already
    if not hasattr(Configuration, "memory"):
        # Use setattr to add the field
        setattr(Configuration, "memory", MemoryConfig())
        
        # Update the model schema
        Configuration.model_rebuild()
        
        logger.debug("Updated Configuration model with memory field")


def register_memory_hooks() -> None:
    """Register memory hooks with the agent system."""
    # Register event handlers when we support memory
    pass


# Initialize memory support
def init_memory_support() -> None:
    """Initialize memory support in Nerve."""
    update_configuration_models()
    register_memory_hooks()
    
    logger.info("Memory system integration initialized")