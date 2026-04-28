# MCP Client — Local Agent Harness (Learning Project)

> **This is a personal learning project.** Not production-ready. Not intended for public use. Security vulnerabilities exist by design of the learning scope — see [Security](#security) below.

A local agentic harness built around the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/). Connects to one or more MCP servers, exposes their tools to an LLM, and lets you chat with the agent in a browser — with per-tool-call approval controls.

Built to understand how LLM agent loops, tool orchestration, task planning, and context management actually work at the code level.

---

## What It Does

- **MCP server integration** — connect to any MCP server (filesystem, web search, custom tools) via `servers_config.json`
- **Agentic loop** — LLM decides which tools to call, executes them, feeds results back, loops until a final answer
- **Tool approval gate** — every tool call pauses for Allow / Deny / Always-allow before execution
- **Task planning** — agent can break work into tasks (`plan_tasks`, `update_task`, `add_task`); live sidebar shows the plan
- **Planner pass** — optional upfront LLM call restricted to planning only, before the main loop starts
- **File tools** — built-in `list_files`, `read_file`, `edit_file`, `replace_lines`, `bash` scoped to a configured base directory
- **Bash tool** — runs commands via WSL (`wsl.exe bash -ls`, passed via stdin); Windows paths auto-mapped to `/mnt/...`
- **Skills system** — extend the LLM with specialised instructions or callable Python functions via `mcp_chatbot/skills/`
- **Multimodal input** — accepts text files and images alongside chat messages
- **Mission brief** — static `plan.md` in workspace root injected into every system prompt
- **Token budget tracker** — per-category token estimates (system / tools / history); triggers rolling summarization at 70% context
- **Rolling summarization** — compresses old turns into a `[Summary]:` block to keep context within limits
- **Token usage display** — segmented progress bar in UI showing system / tools / history usage with color thresholds
- **Gradio UI** — browser-based chat with sidebar for server selection, task list, and settings (optimized for personal use only)

---

## Stack

- Python 3.10+, `uv` for package management
- [Gradio](https://gradio.app/) for the web UI
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) for server connections
- `httpx` for LLM API calls (OpenAI-compatible endpoint)
- LM Studio (local) or Groq (cloud) as the LLM backend (tested with Gemma 4 E4B Q6_K)

---

## Running

```bash
# Install dependencies
uv sync

# Start the app
uv run mcp_chatbot/frontend/app.py
```

Opens at `http://localhost:7860`.

### LLM Backend

Configured in `mcp_chatbot/core/config.py`:

| `local_model` | Endpoint | Model |
|---|---|---|
| `True` | `http://localhost:1234/v1` (LM Studio) | `model-identifier` |
| `False` | Groq API | `meta-llama/llama-4-scout-17b-16e-instruct` |

Set `LLM_API_KEY` in `mcp_chatbot/.env` for cloud mode.

### MCP Servers

Edit `mcp_chatbot/servers_config.json`:

```json
{
  "mcpServers": {
    "my-server": {
      "command": "uvx",
      "args": ["my-mcp-server"],
      "timeout": 30.0
    }
  }
}
```

Servers must be checked in the UI sidebar to activate them.

---

## Project Layout

```
mcp_chatbot/
  core/
    config.py          # Configuration
    context_manager.py # ContextManager — token budget tracker + compression trigger
    llm_client.py      # LLMClient (httpx, streaming + non-streaming)
    server.py          # Server, Tool (MCP lifecycle)
    session.py         # ChatSession (agentic loop)
    task_manager.py    # TaskManager (plan/track tasks)
  frontend/
    app.py             # Gradio UI
    multimodal.py      # Image/file input handling
    schemas.py         # Pydantic models
  tools/
    file_manager.py    # FileManager (list/read/edit/replace_lines/bash)
    skills_manager.py  # SkillsManager
  skills/              # Skill definitions (SKILL.md + optional scripts.py)
  memory/              # Placeholder — Persistent memory
  settings.py          # settings.json load/save
```

---

## Skills

Skills extend the agent with specialised knowledge or callable Python functions. Drop a folder under `mcp_chatbot/skills/` with:

- `SKILL.md` — YAML frontmatter (`name`, `description`) + markdown instructions
- `scripts.py` (optional) — Python functions + `DISPATCH` dict; type hints and docstrings become the tool schema

The LLM sees skill names and descriptions in its system prompt, calls `read_skill()` to load a skill, then calls skill functions as normal tool calls.

---

## Security

**This project has known security issues that are intentional to the learning scope.**

- **Bash tool runs unsandboxed WSL commands.** Any prompt injection or malicious tool result could execute arbitrary shell commands. 
- **File tool has no authentication.** Anyone with access to the Gradio URL can read/edit files within `file_root`.
- **Tool approval gate is the only guardrail.** "Always-allow"(Auto-tool) mode bypasses it entirely.
- **No input sanitisation** on MCP tool results before they enter the LLM context.
- **Gradio runs without auth** — do not expose to public networks.

Do not run this against sensitive directories or on a shared/public machine.

---


## Future Plans

### Persistent Memory
SQLite episodic store, `remember_fact` / `search_memory` agent tools, optional semantic search via vector DB.

### Agent Loop Hardening
Auto-compaction in loop, surface-agnostic event model (typed events instead of `o|o` strings), state persistence across restarts, optional thinking phase.

### CLI & API Surface
`python -m mcp_chatbot.cli` entry point, minimal FastAPI endpoint (`POST /chat`, streaming).

### Multi-Agent Patterns
Hierarchical orchestrator/worker, parallel fan-out via `asyncio.gather`, A2A compatibility.

---

#### Gradio ui:
![ui](img/ui.png)