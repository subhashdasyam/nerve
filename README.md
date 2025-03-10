<div align="center">

# `nerve`

<i>The Simple Agent Development Kit</i>

[![Release](https://img.shields.io/github/release/evilsocket/nerve.svg?style=flat-square)](https://github.com/evilsocket/nerve/releases/latest)
[![Package](https://img.shields.io/pypi/v/nerve-adk.svg)](https://pypi.org/project/nerve-adk)
[![Docker](https://img.shields.io/docker/v/evilsocket/nerve?logo=docker)](https://hub.docker.com/r/evilsocket/nerve)
[![CI](https://img.shields.io/github/actions/workflow/status/evilsocket/nerve/ci.yml)](https://github.com/evilsocket/nerve/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-GPL3-brightgreen.svg?style=flat-square)](https://github.com/evilsocket/nerve/blob/master/LICENSE.md)

**[Documentation](https://github.com/evilsocket/nerve/blob/main/docs/index.md) - [Examples](https://github.com/evilsocket/nerve/blob/main/examples)**

</div>

Nerve is an ADK ( _Agent Development Kit_ ) designed to be a simple yet powerful platform for creating and executing LLM-based agents.

\U0001f5a5\ufe0f Install with:

```bash
pip install nerve-adk
```

\U0001f4a1 Create an agent with the guided procedure:

```bash
nerve create new-agent
```

\U0001f916 Agents are simple YAML files that can use a set of built-in tools such as a bash shell, file system primitives [and more](https://github.com/evilsocket/nerve/blob/main/docs/namespaces.md):

```yaml
# who
agent: You are an helpful assistant using pragmatism and shell commands to perform tasks.
# what
task: Find which running process is using more RAM.
# how
using: [task, shell]
```

\U0001f680 Execute the agent with:

```bash
nerve run new-agent
```

\U0001f9e0 Memory capabilities with vector database integration:

```yaml
using:
- memory  # Adds memory namespace

memory:
  provider: chroma  # or pgvector
  embedding: openai  # or huggingface
  auto_store_conversations: true
  auto_retrieve: true
```

\U0001f6e0\ufe0f The agent capabilities can be extended directly via YAML (the [android-agent](https://github.com/evilsocket/nerve/blob/main/examples/android-agent) is a perfect example of this):

```yaml
tools:
  - name: get_weather
    description: Get the current weather in a given place.
    arguments: 
      - name: place
        description: The place to get the weather of.
        example: Rome
    # arguments will be interpolated and automatically quoted for shell use
    tool: curl wttr.in/{{ place }}
```

\U0001f40d Or in Python, by adding a `tools.py` file, for more complex features (check this [webcam agent example](https://github.com/evilsocket/nerve/blob/main/examples/webcam)):

```python
import typing as t

# This annotated function will be available as a tool to the agent.
def read_webcam_image(foo: t.Annotated[str, "Describe arguments to the model like this."]) -> dict[str, str]:
    """Reads an image from the webcam."""

    base64_image = '...'
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
    }
```

\U0001f468\u200d\U0001f4bb Alternatively, you can use Nerve as a Python package and leverage its abstractions to create an entirely custom agent loop (see [the ADK examples](https://github.com/evilsocket/nerve/blob/main/examples/adk/)).

## Features

### Memory System

Nerve includes a powerful memory system enabling agents to remember information across conversations:

- **Vector Database Support**: Store and retrieve memories using ChromaDB or pgvector
- **Memory Types**: Episodic (conversations), Semantic (facts), and Working memories
- **Embedding Options**: OpenAI or HuggingFace embedding models
- **Automatic Features**: Conversation history tracking and relevant context retrieval
- **Memory Tools**: Store, retrieve, and reflect on memories with dedicated tools

To use memory capabilities:
```bash
# Install requirements
pip install chromadb openai  # For ChromaDB with OpenAI embeddings
# or
pip install asyncpg openai  # For pgvector with OpenAI embeddings
# or
pip install chromadb sentence-transformers  # For local embeddings
```

See the [memory-agent example](https://github.com/evilsocket/nerve/blob/main/examples/memory-agent) for full usage.

## Usage

Please refer to the [documentation](https://github.com/evilsocket/nerve/blob/main/docs/index.md) and the [examples](https://github.com/evilsocket/nerve/tree/main/examples).

## License

Nerve is released under the GPL 3 license.

[![Star History Chart](https://api.star-history.com/svg?repos=evilsocket/nerve&type=Date)](https://star-history.com/#evilsocket/nerve&Date)