# Task: Migrate the CLI agent to a web-callable service (POC v1)

## Context

We have a working Claude Agent SDK agent that currently runs in a terminal via a CLI
entrypoint. It calls the SDK's `query()` from its `run()` method, imports plugins, and
handles tool approval by blocking on stdin.

Your job is to make the same agent callable from our web backend, with streaming output
and a tool-approval round-trip through the UI. **The agent's own logic — plugins, tools,
prompts, hooks — should not change.** This is a transport migration.

Existing code:

- `src/agents/for_migration/` — the working CLI agent
- `src/routes/agent.py` — agent routes (currently stubs/partial)
- `src/routes/agent_config.py` — config routes (currently stubs/partial)

## The core idea

In the CLI, tool approval blocked on `input()`. On the web, it blocks on an
`asyncio.Future` that a **separate HTTP request** resolves:

```
Browser ──POST /chat──────────────────────► Session (owns the agent task)
   ▲                                              │
   │  ◄─ SSE: deltas, tool events,  ──────────────┤
   │      {"type":"approval_request","id":"r1"}   │  ← can_use_tool awaits a Future
   │                                              │
   └──POST /approvals/r1 ────────────► fut.set_result(...) ─► callback returns
```

The agent task and the SSE response are decoupled by an `asyncio.Queue`, so a client
disconnect does not kill a running agent, and a reconnecting client can recover.

---

## Non-negotiable constraints

These are the things that are easy to get wrong and expensive to debug. Follow them exactly.

**SDK usage**

1. Use `query()`, **not** `ClaudeSDKClient`. One subprocess per turn, torn down at turn end.
2. The prompt **must** be an `AsyncIterable[dict]`, not a `str`. Streaming input mode is
   required for `can_use_tool` to work at all.
3. `permission_mode="default"`, and `Bash` must **not** appear in `allowed_tools`.
   `can_use_tool` only fires on an "ask" decision — if the tool is pre-allowed, the
   callback silently never runs.
4. Set `session_store_flush="eager"` so the transcript is on disk if we crash.
5. Wrap the prompt generator body in try/except and log loudly. In the Python SDK a
   generator exception is logged at debug level and the session stalls with no output —
   it looks like a hang, not an error.
6. Be explicit about `setting_sources` rather than relying on the default (it has changed
   across versions). Verify plugins/skills actually load before moving on.

**Async correctness**

7. Every `await` on an approval Future must be wrapped in `asyncio.wait_for`. No exceptions.
8. Always keep a reference to tasks created with `create_task` — store them on the session
   object. Untracked tasks can be garbage-collected mid-flight.
9. Attach a done-callback that logs exceptions on every background task. Exceptions in
   fire-and-forget tasks are swallowed silently.
10. Guard `if fut and not fut.done()` before `set_result` — double-clicks in the UI will
    otherwise raise `InvalidStateError`.
11. Use `time.monotonic()` for any elapsed-time math, never `time.time()`.
12. Never make a blocking call (`time.sleep`, `requests`, sync DB driver) on the event
    loop. Use `asyncio.to_thread`.

**Deployment**

13. Run with `uvicorn --workers 1` and add a comment explaining why: with multiple workers
    the approval POST can land in a process that does not hold the Future, and it fails
    silently.

**Security**

14. Every endpoint that accepts a session id must verify the session belongs to the calling
    user before doing anything. Without this, any user can resume another user's
    conversation by supplying their session id.

## Explicitly out of scope

Do not build these. If you think one is needed, stop and say so instead of building it.

- Warm session pooling, idle reapers, LRU eviction
- `ClaudeSDKClient`
- Redis, pub/sub, sticky routing, multi-worker or multi-pod support
- Auth/identity (assume an existing `current_user` dependency)
- Frontend code
- Retry logic, circuit breakers, rate limiting

---

## Phases

Work one phase at a time. **Make one commit per phase.** After each phase, stop and report
what you did and how to verify it. Do not start the next phase until told to continue.

### Phase 0 — Recon (no production code)

Read the existing agent and routes. Produce `MIGRATION_NOTES.md` answering:

- How is `query()` currently invoked? What `ClaudeAgentOptions` are set (`allowed_tools`,
  `permission_mode`, `setting_sources`, `cwd`, `env`, plugin/MCP config)?
- How does the CLI approval callback work today — a `PreToolUse` hook, `can_use_tool`, or
  something else? Quote the code.
- **Grep the plugins for `@tool` and `create_sdk_mcp_server`.** List every in-process MCP
  tool. These handlers will run on the web server's event loop, so flag any that do
  blocking I/O (sync DB calls, `requests`, file reads).
- What do `src/routes/agent.py` and `agent_config.py` contain now? What is already wired?
- Which SDK and `claude` binary versions are installed?
- What is `current_user` / how does auth work in this codebase?
- Where would a `sessions` table live — what ORM/migration tooling is in use?

Commit: `docs: migration recon notes`

### Phase 1 — Session store and ownership

- Migration for a `sessions` table: `session_id` (PK), `user_id`, `title`, `status`,
  `created_at`, `updated_at`.
- `GET /sessions` — list the current user's sessions, most recent first.
- A reusable dependency/helper `get_owned_session(session_id, current_user)` that 404s (not
  403 — don't confirm existence) if the session isn't theirs.
- Do **not** derive the session list from the SDK's transcript files on disk. Those have no
  user scoping, no titles, and vanish when the container restarts.

Verify: create rows manually, confirm user A cannot see or fetch user B's sessions.

Commit: `feat: session store with ownership checks`

### Phase 2 — SSE transport (no agent yet)

- `Session` runtime object: `out: asyncio.Queue`, `pending: dict[str, asyncio.Future]`,
  `state` enum (`IDLE` / `RUNNING` / `AWAITING_APPROVAL`), `task` handle.
- In-memory `SessionRegistry` keyed by session id.
- `GET /sessions/{sid}/stream` — SSE endpoint that drains `out`. Ownership-checked.
  - Emit `: heartbeat\n\n` every 20s (proxies kill idle connections).
  - Set `X-Accel-Buffering: no` (nginx buffers SSE by default).
  - Events accumulate in the queue whether or not a client is attached, and the endpoint
    drains any backlog on connect. This is what makes reconnect work — do not skip it.
- A temporary `POST /sessions/{sid}/_test-emit` that pushes a dummy event into the queue.

Verify: `curl -N` the stream, hit `_test-emit`, see the event arrive. Disconnect, emit
twice, reconnect, confirm both buffered events arrive.

Commit: `feat: SSE event stream with heartbeat and buffering`

### Phase 3 — Agent turn

- `POST /sessions/{sid}/chat` — kicks off the turn as a background task, returns `202`
  immediately. Does **not** stream the response itself.
- The turn runs `query()` in streaming input mode, serializes each SDK message into an
  event, and puts it on `out`. Emits a terminal `{"type":"done"}`.
- **Session id capture:** turn 1 passes no `resume`. Read `session_id` off the messages and
  persist it. Turn 2+ passes `resume=<stored id>`. Re-read and update it on every turn —
  the id can change on resume depending on version.
- Set `include_partial_messages=True` so the UI gets token deltas rather than paragraph
  dumps.
- For this phase only, restrict `allowed_tools` to read-only tools so nothing needs
  approval yet.
- Remove `_test-emit`.

Verify: send a message, watch text stream over SSE, send a second message, confirm the
agent remembers the first (proves resume works).

Commit: `feat: agent turn over streaming input with session resume`

### Phase 4 — Approval round-trip

This is the heart of the migration.

- Implement `can_use_tool`:
  - Generate a `request_id`, create a Future, store it in `session.pending`.
  - Emit `{"type": "approval_request", "id", "tool", "input"}` onto `out`.
  - `await asyncio.wait_for(fut, timeout=300)`.
  - On `TimeoutError`: set `session.abandoned = True` and return `PermissionResultDeny`.
  - Always `pop` from `pending` in a `finally`.
- **Abandoned-session short-circuit:** if `session.abandoned` is set, return deny
  *immediately* without waiting. Otherwise a denial just makes the model try a different
  tool, which hangs for another full timeout — three tools would mean 15 minutes of leak
  instead of 5.
- `POST /sessions/{sid}/approve` and `/reject` — look up the Future by request id,
  ownership-check, `set_result`. Approve accepts an optional edited tool input, returned as
  `updated_input` so the UI can let users modify a bash command before approving.
- `GET /sessions/{sid}/pending-approvals` — returns outstanding requests so a reconnecting
  client can recover ones it missed.
- Re-enable `Bash` (by removing it from `allowed_tools`, not by adding it).

Verify: ask the agent to run a shell command; confirm the approval appears over SSE, the
agent is blocked, approving lets it continue, and rejecting produces a clean turn end.
Then trigger an approval and **close the browser tab** — after the timeout, confirm the
turn ends and `pgrep -c claude` drops back down.

Commit: `feat: tool approval round-trip via futures`

### Phase 5 — Interrupt and turn bounds

- `POST /sessions/{sid}/interrupt`: resolve every pending Future with a denial
  (`"Cancelled by user"`). This covers the common case — a user hitting stop while an
  approval is on screen — and ends the turn cleanly through the existing path.
- If no approval is pending, cancel the turn task. Add a comment noting this kills the
  subprocess mid-tool and may leave a dangling `tool_use` in the transcript, which can
  break resume. Accepted for v1.
- Wrap the whole turn in `asyncio.wait_for(..., timeout=1800)` as an outer bound.
- Set `max_turns` in the options so a runaway loop can't burn tokens.

Verify: interrupt while awaiting approval → clean end, resume still works. Interrupt
mid-`Bash` → subprocess dies, no orphan.

Commit: `feat: interrupt and turn-level timeouts`

### Phase 6 — Workspace isolation

The agent creates artifacts per user. Right now the target path comes from a global `.env`
var plus a `CLAUDE.md` instruction. Both are inadequate: a process-global env var is the
same value for every concurrent session, and a `CLAUDE.md` instruction is model compliance,
not enforcement — the model drops absolute prefixes and writes relative paths into the
shared cwd.

- Derive a per-session artifact dir **from the session id** so it is stable across turns
  (`ARTIFACT_ROOT / user_id / session_id`). If you change `cwd`, it must be stable per
  session — transcripts live under `~/.claude/projects/<encoded-cwd>/`, so a per-turn temp
  dir breaks `resume`.
- Pass the path per-subprocess via `ClaudeAgentOptions(env=...)` and/or the system prompt.
  Keep the `CLAUDE.md` instruction as a hint.
- Add a `PreToolUse` hook that **denies** any `Write`/`Edit`/`NotebookEdit` whose resolved
  path falls outside the session's artifact dir. Resolve symlinks; reject `..` traversal.
  This is the actual enforcement.
- Keep `cwd` at the project root unless recon showed a reason to move it — moving it loses
  project settings and skills, which only load from `<cwd>/.claude/` with no parent fallback.

Verify: prompt the agent to write to a relative filename and to an absolute path outside
its workspace. Both must be denied. Run two sessions concurrently writing the same
filename; confirm no collision.

Commit: `feat: per-session workspace isolation with path guard`

### Phase 7 — Config endpoints

- `GET /agent-config/skills` — registered skills
- `GET /agent-config/tools` — MCP tool connectors from config
- `GET /agent-config/bundle?path=<node>` — effective config bundle for the current user

Note: the intended semantics of `/bundle` are unclear. **Do not guess.** Read the existing
code and any specs in the repo; if it remains ambiguous, implement `/skills` and `/tools`,
leave `/bundle` as a documented stub, and report what you found. Its output determines
which `allowed_tools` get passed, which determines whether `can_use_tool` fires — so
getting it wrong is worse than leaving it unfinished.

Commit: `feat: agent config endpoints`

---

## Observability (fold into each phase, not a separate commit)

- Log `session_id`, `user_id`, and `request_id` on every approval event.
- Log the registry size and live subprocess count on a timer.
- Log at INFO on: turn start/end, approval requested/resolved/timed out, subprocess spawn
  and exit.

Leak check, used during Phase 4 verification:

```bash
watch -n2 'pgrep -c claude'
```

Open several tabs, trigger approvals, close the tabs without answering. The count should
rise and then fall after the timeout window. If it rises and stays, something still holds a
reference to a Future or a task.

## Definition of done for v1

- Two users can hold concurrent conversations without interference
- Approvals round-trip through the UI and the agent continues correctly
- Closing a tab mid-approval does not leak a subprocess past the timeout
- Sessions resume across turns and appear in `GET /sessions`
- Writes outside the session workspace are denied by the hook, not just discouraged by a prompt