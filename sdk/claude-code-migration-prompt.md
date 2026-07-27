# Task: Migrate CLI agent to web-callable service (POC v1)

You are migrating the existing CLI agent (Claude Agent SDK, Python) so it can be driven from our web backend. Work in phases. **Make exactly one git commit per phase**, message format `poc-v1 phase N: <summary>`. Run the phase's smoke script before committing. If the code contradicts these instructions, stop and ask me — do not silently deviate.

## Existing code — read all of this in Phase 0 before writing anything

- `src/agents/for_migration/` — the working CLI agent. `graph.py` builds options and iterates `query(prompt=user_input, options=options)` (string prompt, `resume=self.session_id`, `permission_mode="acceptEdits"`, `cwd=PROJECT_DIR`, `setting_sources=["project"]`). Approval is `bash_approval_hook` (PreToolUse) which delegates via `await asyncio.to_thread(callback.request_approval, command)` to an abstract `AgentCallback`; the CLI implementation (`src/cli/callbacks.py`, `CLICallback`) uses questionary. `main.py` also starts a LiteLLM proxy (port 4000), a logging proxy (port 4001), Phoenix OTEL tracing, and a SQLite **ConversationStore** for session metadata.
- `src/routes/agent.py` — route stubs under `/api/v1/agent/`: `/chat` (returns a `StreamingResponse` directly), `/interrupt`, `/approve`, `/reject`, `/pending-approvals`, plus an existing `GET /slash-commands` (keep it working, no changes needed).
- Schemas/DTOs — **the frontend already uses these contracts; do not rename anything.** `AgentChatRequestDTO` uses `workspace_id` + `message`. Approval schemas use `call_id` + `justification`. `workspace_id` is a path param for `/interrupt` and `/pending-approvals`, a body field for `/chat`, `/approve`, `/reject`. The persistent identifier is **workspace_id** throughout — never introduce `session_id`, `prompt`, `request_id`, or `updated_input`.
- `src/routes/project.py` — the existing workspace routes. This is where "list previous conversations" lives; modify these, do NOT add a new `GET /sessions`.
- `ConversationStore` — existing store with `thread_id`, `user_id`, `title`, `created_at`, `updated_at`. This is the session/workspace source of truth for v1 (no filesystem listing, no new tables).
- Auth: `get_current_user` (`dependencies.py`, SSO/JWT + DB). **Assume it works; do not modify it and do not build a dev-mode bypass — out of scope.**
- `src/routes/agent_config.py` (`/skills`, `/tools`, `/bundle`) — **out of scope, do not modify.**
- Installed SDK: check the actual package in the venv. `include_partial_messages` exists — see `.venv/lib/python3.12/site-packages/claude_agent_sdk/types.py:976-1017`. Verify hook/option signatures there rather than guessing.

## Fixed design decisions (do not revisit)

1. Per-turn `query()` with a **string prompt** and `resume=<thread_id>`. No streaming input mode, no `ClaudeSDKClient`, and **no `can_use_tool` anywhere** — approval stays in the existing PreToolUse hook design.
2. The web approval mechanism is a new **`AgentCallback` subclass** (`WebCallback`): same interface the CLI callback implements, but `request_approval` round-trips to the browser via an `asyncio.Future` instead of questionary. The hook itself and `permission_mode="acceptEdits"` stay as they are.
3. `cwd` stays `PROJECT_DIR` — the agent's `.claude/`, `CLAUDE.md` (risk-analyst persona, render-html skill, etc.) resolve from it. Do NOT move cwd per user.
4. Artifact isolation: per-user output directory **appended on top of `TEMP_OUTPUT_DIR`** (e.g. `{TEMP_OUTPUT_DIR}/{user_id}/`), passed **per subprocess/turn** — never by mutating the process-global `.env` value, which is shared by every concurrent user by construction. The `CLAUDE.md` "## Output Location" instruction stays as the soft nudge; a small PreToolUse path guard is the enforcement backstop.
5. **Ownership checks are mandatory.** Because cwd is shared, all transcripts live in one project dir — nothing structural stops user A resuming user B's `thread_id`. Every endpoint that accepts a `workspace_id` must verify the ConversationStore row's `user_id` matches `current_user`, else 404 (not 403 — don't leak existence). This is the IDOR fix on a risk-intelligence platform; it is not optional.
6. Approval round-trip via `asyncio.Future`, keyed **per approval request** (`dict[call_id, Future]` on the workspace runtime) — parallel tool calls can have multiple approvals outstanding.
7. Timeouts are in scope NOW: 300s per approval, 1800s outer bound per turn, `max_turns` set.
8. **One-route SSE**: `POST /chat` returns the `StreamingResponse` directly (matching the existing stub) — the SSE stream IS the chat response. No separate `GET /stream`. Internally, events still flow through a per-workspace queue so `/pending-approvals` can recover state after a dropped stream.
9. v1 interrupt handles the approval-blocked case cleanly; mid-tool interrupt is a dirty `task.cancel()` and that's accepted.
10. Single worker (`uvicorn --workers 1`) — futures and queues are per-process.
11. Workspace status (`running`/`interrupted`) is in-memory runtime state only; ConversationStore holds the durable metadata.
12. LiteLLM is required infrastructure (Phase 1), not an external dependency to assume away.

## Known failure modes to respect

- The `.env`-global output dir is the classic collision: two users, `Write("report.csv")`, one file. Hence decision 4.
- A hook denial does not end the turn — the model may try another tool and fire another approval. That's what the `abandoned` flag is for (Phase 2): first timeout arms it, every subsequent approval fails instantly, so an abandoned workspace costs one 300s window, not one per tool.
- After resolution or timeout: `finally: pending.pop(call_id, None)`.
- Double-click guard on `/approve` and `/reject`: `if fut and not fut.done()` before `set_result`, else you get `InvalidStateError`.
- Capture the SDK session id (`thread_id`) from **every** message that carries one and upsert the ConversationStore row — resume can produce a new id; always store the latest.
- The existing hook wraps `callback.request_approval` in `asyncio.to_thread` because `CLICallback` is sync (questionary). `WebCallback` must NOT go through a worker thread — it's pure async on the server loop. The sanctioned fix is a dispatch branch in `bash_approval_hook`:
  ```python
  if inspect.iscoroutinefunction(callback.request_approval):
      result = await callback.request_approval(command)
  else:
      result = await asyncio.to_thread(callback.request_approval, command)
  ```
  `CLICallback` stays sync and unchanged; `WebCallback.request_approval` is `async def` and awaits its Future natively — no `run_coroutine_threadsafe`, no parked executor threads.
- SSE heartbeat `: ping` every ~20s or proxies will kill the stream mid-turn.

---

## Phase 0 — Recon + verification (no product code)

1. Read everything under "Existing code" above: `graph.py`, the callback abstraction, both route files, `project.py`, ConversationStore, all DTOs. 
2. Confirm in the installed SDK: `include_partial_messages` (types.py:976-1017), the exact PreToolUse hook signature, and `ClaudeAgentOptions` fields for `env`/`cwd`/`resume`/`max_turns`.
3. Trace how `main.py` starts the LiteLLM proxy (`_start_litellm_proxy`), the logging proxy, and what `_fix_additional_properties` does — Phase 1 replicates these.
4. Write `docs/poc-v1-findings.md`: the confirmed DTO field names, hook signature, callback interface, how the sync/async bridge in the hook works today, and the LiteLLM/logging-proxy startup shape.

**Commit** (findings doc only).

## Phase 1 — LiteLLM + model-routing route

**Goal: the web service can route model calls the same way the CLI did.**

1. Start the LiteLLM proxy from the web service lifecycle (lifespan/startup), replicating `_start_litellm_proxy`'s logic.
2. Implement the `openai/chat/completions` route to replace the old logging proxy: auth + forwarding to LiteLLM, **plus** request/response logging, **plus** the `_fix_additional_properties` schema fix carried over from the old proxy. Point the agent's SDK base URL at this route.
3. Phoenix tracing: wire it if it's a small lift in the web context; otherwise note it in findings as deferred and move on. Ask me if unsure.

**Smoke** `scripts/smoke_phase1.py`: hit the completions route with a trivial request; assert a valid completion returns, the request/response got logged, and a payload that needs the `_fix_additional_properties` fix passes through correctly.

**Commit.**

## Phase 2 — WorkspaceRuntime + WebCallback

**Goal: the approval round-trip, testable without HTTP or the binary.**

1. In-memory registry `runtimes: dict[workspace_id, WorkspaceRuntime]` with:
   - `out: asyncio.Queue` (events accumulate regardless of listeners — this is what makes `/pending-approvals` recovery possible)
   - `pending: dict[call_id, asyncio.Future]` + `pending_payloads: dict[call_id, dict]`
   - `abandoned: bool`, `task: asyncio.Task | None`, `status: str`
2. `WebCallback(AgentCallback)` — same interface as `CLICallback`. `request_approval(command)`:
   ```
   if runtime.abandoned: return deny  # instant
   call_id = uuid4(); fut = <future on the server loop>
   pending[call_id] = fut; pending_payloads[call_id] = approval event
   put approval event on out
   try:
       decision = wait for fut, timeout=300
   except timeout:
       runtime.abandoned = True; runtime.status = "interrupted"
       return deny("Approval timed out — workspace interrupted.")
   finally:
       pending.pop / pending_payloads.pop
   ```
   Approve resolutions carry allow; reject/interrupt carry deny (+ `justification` per the existing schema). `request_approval` is `async def`, awaiting `asyncio.wait_for(fut, timeout=300)` directly on the server loop — plain single-threaded async, no bridging.
3. Apply the ONE sanctioned edit to `bash_approval_hook`: the `inspect.iscoroutinefunction` dispatch branch from "Known failure modes" above. Nothing else in the hook changes; `CLICallback` and `permission_mode="acceptEdits"` stay untouched.

**Smoke** `scripts/smoke_phase2.py` (no binary needed): drive the hook's dispatch with both callback types — a dummy sync callback (asserts the `to_thread` branch still works, i.e. CLI path unbroken) and the `WebCallback` (await `request_approval` as a task, resolve the future from another task, assert allow flows back). Timeout case: never resolve, timeout overridden to ~5s via parameter; assert deny returns, `abandoned` flips, and a follow-up `request_approval` denies instantly.

**Commit.**

## Phase 3 — Turn runner

**Goal: `run_turn(runtime, user, message)` streams events to the queue, resumes correctly, cannot leak.**

1. Build the agent exactly as `graph.py` does (`cwd=PROJECT_DIR`, `permission_mode="acceptEdits"`, `setting_sources=["project"]`, plugins), with three injections: the `WebCallback`, `include_partial_messages=True`, and the per-user output dir from decision 4 (create the dir on first use; pass it per-turn via options/env override, not global `.env` mutation).
2. String prompt; `resume=thread_id` when the workspace has one.
3. Iterate `query()`: forward events to `out` (partial text, tool use, results, final); capture `thread_id` from every message carrying one and upsert ConversationStore (`thread_id`, `user_id`, `title` = first ~60 chars of the first message, timestamps).
4. Add a second PreToolUse matcher (path guard): deny writes resolving outside the caller's output dir. Backstop only; existing bash approval hook stays.
5. `max_turns`; outer `asyncio.wait_for(..., 1800)` → error event on timeout. `finally`: update `status`, clear `runtime.task`.

**Smoke** `scripts/smoke_phase3.py` (needs binary + Phase 1 running): direct `run_turn` with a no-tool message; assert events stream, a ConversationStore row appears with the thread_id, and turn 2 with `resume` sees turn-1 context. Assert the artifact output dir for the user was created.

**Commit.**

## Phase 4 — Workspace listing + ownership

1. `get_workspace_authorized(workspace_id, current_user)` — loads the ConversationStore row; 404 on missing or `user_id` mismatch. Every workspace-taking endpoint from here on goes through it (decision 5).
2. Modify the existing `src/routes/project.py` routes so listing returns the current user's workspaces from ConversationStore ordered by `updated_at` desc (`workspace_id`/`thread_id`, `title`, timestamps) — matching whatever response shape those routes already promise the frontend.

**Smoke** `scripts/smoke_phase4.py` (httpx): seed rows for users A and B; list as A → only A's, ordered; A hitting B's workspace_id → 404.

**Commit.**

## Phase 5 — POST /chat (one-route SSE)

1. Wire the existing stub: validate DTO (`workspace_id?`, `message`), auth; existing workspace → ownership check + 409 if `runtime.task` is live; no `workspace_id` → new workspace (row created once the thread_id is captured).
2. Spawn the turn task (keep the handle), return the `StreamingResponse` whose generator drains `runtime.out` as SSE events, heartbeat every 20s, and closes after the final result event. For new workspaces, emit the `workspace_id`/`thread_id` as an early SSE event so the client learns it.
3. Client disconnect does NOT cancel the turn — it keeps running; the Phase 2 timeout is the reclaim path, and `/pending-approvals` + `/approve` (Phase 6) are the recovery path. Confirm `/slash-commands` still works untouched.

**Smoke** `scripts/smoke_phase5.py` (httpx): stream a no-tool chat to completion (new workspace, capture id from the early event); second POST on the same workspace → resumed context; concurrent second POST while running → 409; someone else's workspace_id → 404.

**Commit.**

## Phase 6 — /approve, /reject, /pending-approvals, /interrupt

All through `get_workspace_authorized`, all using the **existing** schema shapes (path-param vs body, `call_id`, `justification`).

1. `/approve` / `/reject`: `fut = runtime.pending.get(call_id)`; `if fut and not fut.done(): set_result(...)`; else 409/410.
2. `/pending-approvals` (path param): return `pending_payloads` values — this is the recovery mechanism when the chat stream died mid-approval.
3. `/interrupt` (path param): resolve every undone pending future with deny("Cancelled by user") — clean turn end through the SDK, transcript stays resumable; do NOT set `abandoned` (the user is still here). Else if `runtime.task` is live: `task.cancel()` (dirty, accepted). Set `status="interrupted"`, emit an event.

**Smoke** `scripts/smoke_phase6.py` (httpx): chat with a Bash-triggering message; read the stream to the approval event; approve with a `call_id` + `justification`; stream completes. Run 2: interrupt while approval-blocked; then chat the same workspace again and assert resume works. Run 3: kill the stream connection mid-approval, fetch `/pending-approvals`, approve, verify via a fresh turn that the prior turn completed.

**Commit.**

## Phase 7 — Hardening, observability, docs

1. Run script/Makefile: `uvicorn --workers 1`, loud comment on why multi-worker breaks approvals.
2. Every 30s log: `len(runtimes)`, total pending futures, claude subprocess count (psutil or `pgrep -c claude`).
3. Grep plugins for `@tool` / `create_sdk_mcp_server`; flag blocking handlers in the README (fixing = out of scope).
4. `README-poc-v1.md`: architecture sketch, endpoint list, limitations:
   - one-route SSE → no re-attach to a live stream (recovery = `/pending-approvals` + `/approve`; final output lands in the transcript/artifacts)
   - dirty mid-tool cancel can leave a dangling tool_use and hurt resume for that workspace
   - in-memory runtimes lost on restart (ConversationStore rows survive)
   - shared cwd → all transcripts in one project dir; isolation rests entirely on the ownership check + per-user output dirs
   - single worker only
   - leak check: `watch -n2 'pgrep -c claude'` while triggering approvals and killing streams unanswered — count climbs, then drops within the timeout window.

**Smoke:** rerun `smoke_phase6.py`; confirm subprocess count returns to baseline.

**Commit.**

---

## Ground rules

- Stick to existing schemas and route shapes everywhere — the frontend contract is fixed.
- No new dependencies beyond httpx/psutil (or repo equivalents) without asking.
- No changes to `agent_config.py`, `get_current_user`, `/slash-commands`, or core agent logic in `for_migration/` beyond the hook dispatch branch (Phase 2) and the three injections (Phase 3) — ask first if you think you need more.
- If the installed SDK differs from what's described, adapt to the installed version and note it in the commit message.
