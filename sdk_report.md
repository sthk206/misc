# Claude Agent SDK (Python) — Architecture & Customization Report

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [How It Calls Claude](#how-it-calls-claude)
3. [Entry Points](#entry-points)
4. [Main Features & Functionalities](#main-features--functionalities)
5. [Customization & Extensibility Interfaces](#customization--extensibility-interfaces)
6. [Subagents](#subagents)
7. [Memory System](#memory-system)
8. [Using Your Own Memory System](#using-your-own-memory-system)
9. [Limitations & Boundaries](#limitations--boundaries)

---

## Architecture Overview

The Claude Agent SDK is **not** a direct wrapper around the Anthropic API. It is an orchestration layer on top of the **Claude Code CLI binary**, which is bundled inside the Python package at build time.

```
Your Python code
    → Claude Agent SDK (python)
        → Spawns Claude Code CLI as a subprocess (bundled binary)
            → Claude Code CLI calls the Anthropic API internally
            → CLI manages the agentic loop, tool execution, permissions, memory
        ← SDK reads NDJSON messages from CLI stdout
        → SDK writes JSON control messages to CLI stdin
```

All communication between the SDK and the CLI happens over **stdin/stdout using NDJSON** (newline-delimited JSON). The SDK never imports `anthropic`, never calls `messages.create()`, and never touches the HTTP API directly.

### The Bundled CLI Binary

- At **build time**, `scripts/build_wheel.py` downloads a pinned version of the Claude Code CLI (currently v2.1.210 per `src/claude_agent_sdk/_cli_version.py`) and places it in `src/claude_agent_sdk/_bundled/claude`.
- The published wheel ships with this binary baked in — **no npm install or runtime download occurs**.
- At **runtime**, the SDK resolves the CLI path in this order (`subprocess_cli.py:142-188`):
  1. Bundled binary inside the installed package (`_bundled/claude`)
  2. System-wide `claude` via `shutil.which()`
  3. Common install paths (`~/.npm-global/bin/claude`, `/usr/local/bin/claude`, etc.)
  4. Explicit path via `ClaudeAgentOptions(cli_path="/path/to/claude")`

---

## How It Calls Claude

The call chain is:

1. `query()` or `ClaudeSDKClient.connect()` → `InternalClient._process_query_inner()` (`_internal/client.py:92`)
2. Creates `SubprocessCLITransport` → `anyio.open_process(cmd, stdin=PIPE, stdout=PIPE)` (`subprocess_cli.py:535`)
3. `Query` class manages the bidirectional control protocol over stdin/stdout (`_internal/query.py`)
4. The CLI subprocess handles the agentic loop: calling the Anthropic API, executing tools, managing permissions, running subagents

The SDK's `Query` class routes three categories of messages:
- **Control responses** — replies to requests the SDK sent (initialize, permission mode changes, MCP status queries)
- **Control requests** — requests from the CLI to the SDK (tool permission prompts, hook callbacks, MCP tool calls)
- **Regular messages** — user/assistant/system/result messages forwarded to your code

---

## Entry Points

### `query()` — One-Shot

Defined in `src/claude_agent_sdk/query.py`. Stateless, fire-and-forget. Send a prompt (string or async iterable), iterate over response messages.

```python
async for message in query(prompt="What is 2+2?", options=options):
    print(message)
```

### `ClaudeSDKClient` — Stateful / Interactive

Defined in `src/claude_agent_sdk/client.py`. Bidirectional, multi-turn. Supports:
- Sending follow-up messages based on responses
- Interrupting the agent mid-execution
- Changing permission mode or model mid-session
- Querying MCP server status and context usage
- Stopping background tasks/subagents
- Rewinding file changes to checkpoints

```python
async with ClaudeSDKClient(options=options) as client:
    await client.query("Analyze this codebase")
    async for msg in client.receive_response():
        ...
    await client.query("Now fix the bug we discussed")
    async for msg in client.receive_response():
        ...
```

---

## Main Features & Functionalities

### Tool Management

- **`tools`** — Define the base set of available built-in tools (e.g., `["Bash", "Read", "Edit"]`), or use the `claude_code` preset for all defaults.
- **`allowed_tools`** — Tools that execute automatically without permission prompts.
- **`disallowed_tools`** — Tools removed from the model's context entirely.
- **MCP tools** — Custom tools via the `@tool` decorator and `create_sdk_mcp_server()`, running in-process. Also supports external MCP servers (stdio, SSE, HTTP).

### Permission Control

- **`permission_mode`** — Global mode: `"default"`, `"acceptEdits"`, `"plan"`, `"bypassPermissions"`, `"dontAsk"`, `"auto"`.
- **`can_use_tool`** — Async callback invoked when the CLI would prompt for permission. Can allow, deny, or modify tool inputs.
- **`set_permission_mode()`** — Change permission mode mid-session (ClaudeSDKClient only).

### Hooks (Event-Driven Middleware)

10 lifecycle events with async callbacks:

| Event | When It Fires | What You Can Do |
|-------|--------------|-----------------|
| `PreToolUse` | Before a tool executes | Allow/deny/defer, modify inputs, inject context |
| `PostToolUse` | After a tool executes | Inspect/replace output, inject context |
| `PostToolUseFailure` | After a tool fails | Add error context |
| `UserPromptSubmit` | Before each user prompt | Inject system context |
| `Stop` | Session ending | Custom shutdown logic |
| `SubagentStop` | Subagent ending | Custom cleanup |
| `SubagentStart` | Subagent spawned | Track subagent lifecycle |
| `PreCompact` | Before context compaction | Custom compaction instructions |
| `Notification` | CLI notification | Observe notifications |
| `PermissionRequest` | Permission prompt | Programmatic decisions |

Hooks support **pattern matching** (e.g., `matcher="Bash"` or `matcher="Write|Edit"`) and **concurrent dispatch** (multiple matchers on the same event run in parallel).

### Session Management

- **`resume`** / **`continue_conversation`** — Resume previous sessions.
- **`session_id`** — Use a specific session ID.
- **`fork_session`** — Fork a resumed session to a new ID.
- **`session_store`** — Mirror transcripts to external storage (see SessionStore below).
- **Session listing** — `list_sessions()`, `get_session_info()`, `get_session_messages()` for reading historical data.
- **Session mutations** — `rename_session()`, `tag_session()`, `delete_session()`, `fork_session()`.

### Model & Thinking Control

- **`model`** / **`fallback_model`** — Choose the Claude model.
- **`set_model()`** — Change model mid-session.
- **`thinking`** — Adaptive, enabled (with budget), or disabled.
- **`effort`** — `"low"` through `"max"` to control reasoning depth.

### Output & Structured Data

- **`output_format`** — JSON schema for structured responses.
- **`include_partial_messages`** — Streaming partial message events.
- **`include_hook_events`** — Hook lifecycle events in the message stream.

### Sandbox & Security

- **`sandbox`** — Filesystem and network isolation for bash commands.
- **`add_dirs`** — Additional directories Claude can access.

### Settings & Isolation

- **`setting_sources`** — Control which filesystem settings load: `"user"` (global `~/.claude/`), `"project"` (`.claude/` in project), `"local"` (`.claude-local/`). Pass `[]` to disable all.
- **`settings`** — Path to or JSON string of additional settings (highest priority).
- **`plugins`** — Load local plugins for custom commands, agents, skills, and hooks.
- **`skills`** — Enable specific skills or all discovered skills.

---

## Customization & Extensibility Interfaces

### 1. `Transport` (Abstract Base Class)

**File:** `src/claude_agent_sdk/_internal/transport/__init__.py`

The lowest-level extensibility point. An ABC with 5 abstract methods that define how the SDK communicates with the Claude process.

```python
class Transport(ABC):
    async def connect(self) -> None: ...
    async def write(self, data: str) -> None: ...
    def read_messages(self) -> AsyncIterator[dict[str, Any]]: ...
    async def close(self) -> None: ...
    def is_ready(self) -> bool: ...
    async def end_input(self) -> None: ...
```

The default implementation (`SubprocessCLITransport`) spawns a local CLI subprocess, but you can replace it to route communication to a remote Claude Code instance, add encryption/auth layers, or mock for testing.

```python
client = ClaudeSDKClient(options, transport=MyTransport())
# or:
async for msg in query(prompt="...", transport=MyTransport()):
    ...
```

**Caveat:** The docstring warns this is an internal API that may change between releases.

### 2. `SessionStore` (Protocol)

**File:** `src/claude_agent_sdk/types.py:1426`

A duck-typed Protocol for mirroring session transcripts to external storage.

**Required methods (2):**
- `append(key, entries)` — Mirror a batch of transcript entries
- `load(key)` — Load a full session for resume

**Optional methods (4):**
- `list_sessions(project_key)` — List sessions with modification times
- `list_session_summaries(project_key)` — Return incrementally-maintained summaries
- `delete(key)` — Delete a session (with cascade to subkeys)
- `list_subkeys(key)` — List subagent transcripts under a session

The SDK ships `InMemorySessionStore` as a reference implementation and a **conformance test suite** (`claude_agent_sdk.testing.run_session_store_conformance`) that validates custom stores against 14 behavioral contracts.

Example implementations exist for Redis, S3, and Postgres in `examples/session_stores/`.

```python
class MyRedisStore:  # No subclass needed — it's a Protocol
    async def append(self, key, entries): ...
    async def load(self, key): ...

options = ClaudeAgentOptions(session_store=MyRedisStore())
```

### 3. `can_use_tool` / `CanUseTool` (Callback)

**File:** `src/claude_agent_sdk/types.py:254`

An async callback that replaces the interactive permission prompt. Invoked when the CLI's permission rules evaluate to "ask."

```python
async def my_handler(tool_name, input_data, context):
    if tool_name == "Bash" and "rm" in input_data.get("command", ""):
        return PermissionResultDeny(message="Blocked")
    return PermissionResultAllow()

options = ClaudeAgentOptions(can_use_tool=my_handler)
```

Capabilities:
- Allow or deny any tool call programmatically
- Modify tool inputs before execution via `PermissionResultAllow(updated_input={...})`
- Update permission rules dynamically via `PermissionResultAllow(updated_permissions=[...])`
- Access context: `tool_use_id`, `agent_id`, `blocked_path`, `decision_reason`, `title`, `description`

### 4. Hooks System (`HookCallback` / `HookMatcher`)

**File:** `src/claude_agent_sdk/types.py:574-599`

The most feature-rich customization point. See the [Hooks section above](#hooks-event-driven-middleware) for the full event list.

```python
options = ClaudeAgentOptions(
    hooks={
        "PreToolUse": [
            HookMatcher(matcher="Bash", hooks=[check_bash_command]),
            HookMatcher(matcher="Write|Edit", hooks=[audit_file_writes]),
        ],
        "PostToolUse": [HookMatcher(matcher=None, hooks=[log_all_results])],
    }
)
```

### 5. MCP Servers (`@tool` + `create_sdk_mcp_server`)

**File:** `src/claude_agent_sdk/__init__.py:160-524`

Define custom tools running in-process (no subprocess/IPC overhead).

```python
@tool("query_db", "Query our database", {"sql": str})
async def query_db(args):
    result = await db.execute(args["sql"])
    return {"content": [{"type": "text", "text": str(result)}]}

server = create_sdk_mcp_server("my-tools", tools=[query_db])
options = ClaudeAgentOptions(
    mcp_servers={"db": server},
    allowed_tools=["mcp__db__query_db"],
)
```

Supports `dict` schemas, `TypedDict` schemas, raw JSON Schema, and `Annotated` types for parameter descriptions. Also supports external MCP servers (stdio, SSE, HTTP) via config.

### 6. `AgentDefinition` (Dataclass)

**File:** `src/claude_agent_sdk/types.py:83-102`

Declarative configuration for named subagents. See [Subagents section](#subagents) below.

### Summary Table

| Interface | Type | Analogy (LangChain) | What You Replace |
|-----------|------|---------------------|------------------|
| `Transport` | ABC | `Runnable` | The entire I/O layer |
| `SessionStore` | Protocol | `BaseMemory` | Transcript persistence backend |
| `can_use_tool` | Callback | `CallbackHandler` | Permission prompt logic |
| Hooks | Callbacks + Matchers | Chain middleware | Event-driven interception |
| MCP Tools | Decorator + Factory | `@tool` decorator | Claude's available tools |
| `AgentDefinition` | Dataclass | `AgentExecutor` config | Subagent definitions |

---

## Subagents

Subagents are defined in Python but **execute entirely inside the CLI subprocess**. The SDK observes their lifecycle through typed system messages.

### Defining Subagents

```python
options = ClaudeAgentOptions(
    agents={
        "code-reviewer": AgentDefinition(
            description="Reviews code for best practices",
            prompt="You are a code reviewer...",
            tools=["Read", "Grep"],          # Scoped tool access
            disallowedTools=["Write"],        # Blocked tools
            model="sonnet",                   # Can use a different model
            maxTurns=5,                       # Turn limit
            memory="project",                 # Memory scope: "user", "project", "local"
            mcpServers=["my-server"],          # MCP servers available to subagent
            skills=["code-review"],            # Skills available
            permissionMode="dontAsk",          # Independent permission mode
            background=True,                   # Run in background
            effort="high",                     # Effort level
        ),
    },
)
```

### How They Execute

1. At initialization, the SDK sends agent definitions to the CLI via the `initialize` control request.
2. The main Claude agent can spawn subagents by calling the built-in `Agent` tool.
3. The CLI creates a new Claude session for the subagent with the defined configuration.
4. Subagents can be run in foreground (blocking) or background (concurrent).

### Observing Subagent Lifecycle

The SDK emits typed messages into your message stream:

- `TaskStartedMessage` — subagent spawned (contains `task_id`, `description`, `session_id`)
- `TaskProgressMessage` — periodic updates (token usage, last tool used)
- `TaskNotificationMessage` — subagent finished/failed/stopped (contains `summary`, `output_file`)
- `TaskUpdatedMessage` — granular state transitions (`pending` → `running` → `completed`/`failed`/`killed`)

### Controlling Subagents

- `client.stop_task(task_id)` — kill a running subagent
- `SubagentStart` hook — callback when a subagent spawns
- `SubagentStop` hook — callback when a subagent finishes
- Hook callbacks receive `agent_id` and `agent_type` in their input for attribution

### Filesystem-Based Agents

Agents can also be loaded from `.claude/agents/*.md` files when `setting_sources` includes `"project"`. These coexist with programmatically defined agents.

### Limitation

You cannot intercept communication between the main agent and a subagent from Python. The subagent runs entirely inside the CLI. You only see lifecycle events and the final result.

---

## Memory System

Claude Code has a file-based memory system that lives entirely inside the CLI. The SDK does not have its own memory abstraction.

### How Built-In Memory Works

1. **CLAUDE.md files** — Static instruction files loaded at session start. Located at project root (`CLAUDE.md`) and globally (`~/.claude/CLAUDE.md`). Controlled by `setting_sources`.

2. **Auto-memory** — Files at `~/.claude/projects/<project>/memory/*.md`. The CLI loads these into the system prompt at session start. The agent reads/writes them using the standard `Write` and `Edit` tools (no dedicated memory tool).

3. **Per-agent memory scope** — `AgentDefinition.memory` controls which memory scope a subagent uses: `"user"`, `"project"`, or `"local"`.

4. **Observing loaded memory** — `client.get_context_usage()` returns `memoryFiles` with path, type, and token counts for each loaded memory file.

### Disabling Auto-Memory

Auto-memory can be disabled cleanly without affecting other features:

```python
# Option 1: Environment variable
options = ClaudeAgentOptions(
    env={"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"},
)

# Option 2: Settings JSON
options = ClaudeAgentOptions(
    settings='{"autoMemoryEnabled": false}',
)
```

Both pass through to the CLI subprocess. The env var is set on the child process environment (`subprocess_cli.py:496`). The settings JSON is injected via the `--settings` CLI flag (`subprocess_cli.py:359`).

This is distinct from `setting_sources=[]`, which is a much blunter instrument that also disables CLAUDE.md loading, slash commands, filesystem-based agents, and other project settings.

---

## Using Your Own Memory System

You **cannot** replace Claude Code's internal memory file format or read/write mechanism — that's hardcoded in the CLI binary. But you can effectively substitute your own memory system by disabling auto-memory and injecting context via hooks.

### Recommended Pattern

```python
async def inject_memory(input_data, tool_use_id, context):
    """Query your own memory store on every user prompt."""
    prompt = input_data.get("prompt", "")
    memories = await my_vector_db.search(prompt)
    return {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": f"Relevant memories:\n{memories}",
        }
    }

async def save_memory(input_data, tool_use_id, context):
    """Capture tool outputs to build your memory store."""
    tool_response = input_data.get("tool_response", "")
    await my_vector_db.store(str(tool_response))
    return {}

options = ClaudeAgentOptions(
    env={"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"},  # Disable built-in memory
    hooks={
        "UserPromptSubmit": [
            HookMatcher(matcher=None, hooks=[inject_memory]),
        ],
        "PostToolUse": [
            HookMatcher(matcher=None, hooks=[save_memory]),
        ],
    },
)
```

### What Each Approach Gives You

| Approach | How | Replaces Built-In? |
|----------|-----|-------------------|
| Disable auto-memory | `env={"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}` | Stops reading/writing `~/.claude/projects/*/memory/` |
| Inject memory per-prompt | `UserPromptSubmit` hook → `additionalContext` | Your retrieval replaces auto-memory's context injection |
| Inject memory into system prompt | `system_prompt="..."` or preset with `append` | Static instructions, not per-query retrieval |
| Inject memory before tool calls | `PreToolUse` hook → `additionalContext` | Per-tool-call context |
| Capture outputs for memory | `PostToolUse` hook → write to your store | Build your own memory from agent activity |
| Store conversation transcripts | `SessionStore` protocol | Full transcript persistence to any backend |
| Disable all filesystem settings | `setting_sources=[]` | Also kills CLAUDE.md, slash commands, agents — too broad for memory-only |

### SessionStore vs Memory

`SessionStore` is **not** a memory system — it's a transcript persistence layer. It stores raw JSONL conversation logs (every message, tool call, and result). It's designed for session resume, audit, and compliance, not for semantic retrieval. However, you could build a memory system on top of it by indexing/embedding the transcripts it stores.

---

## Limitations & Boundaries

### What You Can Customize

- Which tools are available, allowed, or denied
- Permission logic for every tool call
- Event-driven middleware at 10 lifecycle points
- Custom in-process tools via MCP
- Subagent definitions (tools, model, prompt, permissions, memory scope)
- Session transcript storage backend
- System prompt (custom, preset, file-based, or appended)
- Model, thinking depth, effort level (including mid-session changes)
- Sandbox/isolation settings
- Which filesystem settings load
- The entire transport layer (with caveats)

### What You Cannot Customize

| Limitation | Reason |
|-----------|--------|
| The agentic loop (tool calling, multi-turn) | Lives inside the CLI binary, not the Python SDK |
| Direct Anthropic API calls | The SDK talks to the CLI, which talks to the API |
| Intercept main↔subagent communication | Subagents run entirely inside the CLI |
| Replace the CLI's memory file format | Hardcoded in the CLI (but you can disable it and inject your own) |
| Replace the CLI's tool execution engine | Tools are executed by the CLI, not the SDK |
| Use without Claude Code CLI | The SDK requires the CLI binary; there is no "API-only" mode |
| Custom Transport without the control protocol | A custom Transport must speak the CLI's NDJSON control protocol (initialize handshake, control_request/response, hooks, MCP routing) |

### Key Architectural Insight

The Claude Agent SDK is best understood as a **Claude Code session orchestrator**, not an LLM framework. Unlike LangChain where `Runnable` lets you swap any component in the pipeline, here the core pipeline (prompt → API call → tool execution → next turn) is a sealed unit inside the CLI binary. The SDK gives you extensive customization *around* that unit — intercepting inputs/outputs, controlling permissions, defining agents, persisting data — but not the ability to replace the unit itself.

If you need full control over the agentic loop (custom tool dispatch, custom API calls, custom reasoning), use the **Anthropic Python SDK** (`anthropic`) directly instead.
