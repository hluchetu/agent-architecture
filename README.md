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
  observability/
    events.py               # Structured event model
    log.py                  # Human-readable event log
  examples/
    ollama_chat_demo.py     # Runnable Ollama demo
```

## Run the Ollama chat demo

From this folder:

```bash
uv run ollama-chat-demo
```

Or run the project entry file:

```bash
uv run python main.py
```

Or run the module directly:

```bash
uv run python -m agent_architecture.examples.ollama_chat_demo
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
- `EventLog` shows memory writes, tool calls, tool results, context builds, and LLM requests/responses.
