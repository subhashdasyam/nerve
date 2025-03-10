import os
import pathlib

DEFAULT_GENERATOR: str = os.getenv("NERVE_GENERATOR", "openai/gpt-4o-mini")
DEFAULT_MAX_STEPS: int = int(os.getenv("NERVE_MAX_STEPS", 100))
DEFAULT_TIMEOUT: int | None = int(os.getenv("NERVE_TIMEOUT", 0)) or None
DEFAULT_CONVERSATION_STRATEGY: str = os.getenv("NERVE_CONVERSATION_STRATEGY", "full")

DEFAULT_NERVE_HOME: pathlib.Path = pathlib.Path.home() / ".nerve"

DEFAULT_AGENTS_LOAD_PATH: pathlib.Path = DEFAULT_NERVE_HOME / "agents"
DEFAULT_PROMPTS_LOAD_PATH: pathlib.Path = DEFAULT_NERVE_HOME / "prompts"

DEFAULT_AGENT_PATH: pathlib.Path = pathlib.Path("agent.yml")
DEFAULT_AGENT_SYSTEM_PROMPT: str = "You are a helpful assistant."
DEFAULT_AGENT_TASK: str = "Make an HTTP request to {{{{ url }}}}"
DEFAULT_AGENT_TOOLS: list[str] = ["shell", "task"]
