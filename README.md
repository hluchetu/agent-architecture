# Agent Architecture

Study notes and code sketches for agent architecture patterns.

The project is intentionally structured like a small production service:

```text
agent_architecture/
  config.py                 # Pydantic settings
  llm/
    ollama.py               # Local model client
  memory/
    items.py                # Typed memory item models
    short_term.py           # Short-term memory context
  examples/
    short_term_memory.py    # Runnable example
```

## Run the short-term memory example

From this folder:

```bash
uv run short-term-memory
```

Or run the project entry file:

```bash
uv run python main.py
```

Or run the module directly:

```bash
uv run python -m agent_architecture.examples.short_term_memory
```

You can override the local model with environment variables:

```bash
AGENT_ARCH_OLLAMA_MODEL=gemma3:4b uv run python main.py
```

You can also create a local `.env` file using `.env.example` as a guide.

The current example shows a LiveKit-inspired short-term memory pattern:

- `ChatContext` stores ordered context items.
- `MessageItem` stores LLM-visible messages.
- `ToolCallItem` stores tool calls.
- `ToolResultItem` stores tool outputs.
- `EventItem` stores useful session events.
- `SummaryItem` stores compacted older context.
- `messages_for_llm()` builds the model-facing view.
- `OllamaClient` sends that view to your local Ollama model.
