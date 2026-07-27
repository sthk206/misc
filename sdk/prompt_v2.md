# Task: Migrate CLI agent to web-callable service (POC v1)

You are migrating an existing CLI agent built on the Claude Agent SDK (Python) so it can be driven from our web backend. Work in phases. **Make exactly one git commit per phase**, with the commit message format `poc-v1 phase N: <summary>`. Run the phase's smoke script before committing. If anything in the existing code contradicts these instructions, stop and ask me before proceeding — do not silently deviate.

## Existing code — read this first, before writing anything

- `src/agents/for_migration/` — the working CLI agent. It calls the SDK's `query()` inside a run function, loads plugins, and currently uses a **PreToolUse hook that blocks on terminal input** for tool approval. Do NOT rewrite the agent logic; reuse its config/plugin loading. Your job is the web wrapper around it.
- `src/routes/agent.py` — route stubs for the agent endpoints: `/chat`, `/interrupt`, `/approve`, `/reject`, `/pending-approvals`.
- `src/routes/agent_config.py` — `/skills`, `/tools`, `/bundle`. **All three are OUT OF SCOPE. Do not modify this file.**
- Auth: an existing `get_current_user` dependency (bearer header or `?token=` query param; dev mode accepts the master API key and returns a synthetic user stub). **Reuse it on every endpoint; do not modify it.**
- Use whatever web framework the existing routes use (uvicorn-served ASGI).
- Terminology: Bash is a **tool**, not a skill. Don't conflate SDK tools with the `/skills` concept elsewhere in the repo.

## Fixed design decisions (do not revisit)

1. Use bare `query()` per turn with `resume=<session_id>` — NOT a persistent `ClaudeSDKClient`.
2. Streaming input mode (async-generator prompt) so `can_use_tool` works.
3. Routes stay **flat** (`/chat`, `/approve`, ...) with `session_id` in the request body (or query param for GETs/SSE). No `/sessions/{sid}/...` nesting.
4. **No database. The filesystem is the session store.** Each user gets their own working directory (`/workspaces/{user_id}`), which is passed as `cwd` to the SDK. Transcripts therefore land in `~/.claude/projects/<encoded-user-cwd>/<session-id>.jsonl`, partitioned per user by construction:
   - `GET /sessions` = list that directory, sorted by mtime desc.
   - Authorization is structural: resuming someone else's session id fails because the SDK resolves it under the caller's own project dir. (This assumption is explicitly verified in Phase 0 before anything is built on it.)
   - Transcript exists ⇔ session exists. No rows to drift out of sync.
5. Because session ids from the client now land in filesystem paths, **input validation is load-bearing**: `re.fullmatch(r"[0-9a-fA-F-]{36}", session_id)` on every endpoint that accepts one, 400 otherwise. `user_id` is always server-derived from `get_current_user`, never accepted from the client.
6. Approval round-trip via `asyncio.Future`, keyed **per approval request** (`dict[request_id, Future]` on the session runtime object), never per session — parallel tool calls mean multiple approvals can be outstanding.
7. Approval timeout is in scope NOW (not deferred): 300s per approval, plus a 1800s outer bound per turn, plus `max_turns`.
8. Both permission mechanisms, for different jobs:
   - **PreToolUse hook** = deterministic policy (path guard, now a backstop since relative writes land in the user's own cwd). No human involved.
   - **`can_use_tool`** = the human approval round-trip. Supports `updated_input` so the UI can edit a command before approving.
9. v1 interrupt handles the approval-blocked case cleanly; mid-tool interrupt is a dirty `task.cancel()` and that's accepted for v1.
10. Server must run single-worker (`uvicorn --workers 1`) — futures and queues live in process memory.
11. Session status (`running` / `interrupted`) is in-memory runtime state only, not persisted. A restart loses live turns but not transcripts; that's accepted for v1.

## SDK gotchas to respect (these are known failure modes, not suggestions)

- `can_use_tool` only fires on "ask": if `Bash` is in `allowed_tools` or `permission_mode != "default"`, the callback silently never runs. Verify the options you build actually leave Bash in ask state.
- Capture the SDK session id from **every** message that carries one and keep the latest — resume can produce a new id, so always update what you report/track.
- A denial does not end the turn — the model may try another tool and trigger another approval. That's what the `abandoned` flag is for (see Phase 3).
- After a successful approval resolution or timeout, `finally: pending.pop(request_id, None)`.
- Guard `/approve` / `/reject` against double-clicks: `if fut and not fut.done()` before `set_result`, or you get `InvalidStateError`.
- Moving `cwd` off the project root means project settings/skills won't load — they resolve from `<cwd>/.claude/` with no parent fallback. Each user workspace needs a `.claude` symlink back to the project root's `.claude` (Phase 1).
- `include_partial_messages=True` for token-level streaming to the UI.
- Consult the installed SDK's actual types/signatures (`claude_agent_sdk` package) rather than guessing — verify `PermissionResultAllow/Deny`, hook signatures, and `ClaudeAgentOptions` fields against the installed version.

---

## Phase 0 — Recon + verify the isolation assumption

**Goal: confirm the load-bearing assumption before building on it.** The filesystem design rests on `resume` being scoped to the current `cwd`'s project dir. That follows from where transcripts are stored, but it must be tested, not inferred.

1. Read `src/agents/for_migration/`, both route files, and note the installed SDK version and relevant type signatures.
2. Write `scripts/verify_resume_scoping.py`:
   - Run a trivial turn with `cwd=/tmp/scope_test_a`, capture the session id.
   - Attempt `resume=<that id>` with `cwd=/tmp/scope_test_b`.
   - Assert it **fails** (does not resume A's conversation). Print clearly PASS/FAIL.
   - Also record the exact directory-encoding scheme the SDK uses for `~/.claude/projects/<encoded-cwd>/` (needed by Phase 2's listing code) — print the observed path.
3. **If cross-cwd resume succeeds, STOP and report to me** — the isolation design changes and we'd need an explicit ownership check instead.

**Commit:** the script + a short `docs/poc-v1-findings.md` with the result and the observed encoding scheme.

## Phase 1 — Per-user workspaces + validation helpers

**Goal: the path structure that everything else hangs off.**

1. `workspace_for(user) -> Path`: returns `/workspaces/{user_id}` (base dir configurable), creating it on first use, and creating the symlink `workspace/.claude -> PROJECT_ROOT/.claude` if missing. This also means relative file writes land in the user's own directory rather than the shared project root.
2. `validate_session_id(s)`: the UUID regex check from design decision 5, raising 400. Used by every session-taking endpoint from here on.
3. `project_dir_for(user) -> Path`: maps the user's workspace to its `~/.claude/projects/<encoded>` transcript directory using the encoding scheme observed in Phase 0.

**Smoke** `scripts/smoke_phase1.py`: call `workspace_for` for two fake users; assert separate dirs, symlink resolves, and `validate_session_id` rejects `../../etc`, accepts a uuid4.

**Commit.**

## Phase 2 — GET /sessions (list from disk)

**Goal: previous conversations visible on load, no schema.**

1. `GET /sessions` (auth required): list `project_dir_for(current_user)/*.jsonl`, sorted by mtime desc. Return `[{session_id, updated_at}]` — id from filename, timestamp as the display title for v1 (no jsonl parsing). Empty dir / missing dir → `[]`.

**Smoke** `scripts/smoke_phase2.py` (httpx): drop two dummy `.jsonl` files into user A's project dir with different mtimes; list as A → ordered correctly; list as B (different token) → empty.

**Commit.**

## Phase 3 — Turn runner plumbing (no approvals, no HTTP)

**Goal: a `run_turn(runtime, prompt)` coroutine that streams events to a queue, with resume working.**

1. In-memory registry: `runtime: dict[session_id, SessionRuntime]` where `SessionRuntime` holds:
   - `out: asyncio.Queue` — event queue (events accumulate whether or not anyone is listening; this is what makes SSE ordering safe and `/pending-approvals` possible)
   - `pending: dict[request_id, asyncio.Future]` and `pending_payloads: dict[request_id, dict]` (used from Phase 4)
   - `abandoned: bool`, `task: asyncio.Task | None`, `status: str` (`idle`/`running`/`interrupted`/`error` — in-memory only)
2. Build `ClaudeAgentOptions` from the existing CLI agent's config/plugin loading: `cwd=workspace_for(user)`, streaming input mode, `permission_mode="default"`, Bash NOT pre-allowed, `include_partial_messages=True`, `max_turns` set, `resume=session_id` when continuing an existing session.
3. `run_turn`: iterate `query()`, forward events to `out` (partial text, tool use, results, final result), and capture the SDK session id from every message that carries one. For a brand-new session, signal the captured id to the caller as soon as it's known (an `asyncio.Event`/future the HTTP layer will await in Phase 5) and re-key the runtime under it.
4. Outer bound: `await asyncio.wait_for(<the loop>, timeout=1800)`; on timeout push an error event. Always update `status` and clear `runtime.task` in a `finally`.

**Smoke** `scripts/smoke_phase3.py`: drive `run_turn` directly (no HTTP) with a no-tool prompt; print events; assert a transcript file appeared in the user's project dir; run a second turn with `resume` and assert the model sees turn-1 context. (Header note: requires the Claude Code binary + API access.)

**Commit.**

## Phase 4 — Approval round-trip + path guard

**Goal: the Future-based human-in-the-loop, with no leak path.**

1. `can_use_tool` callback:
   ```
   if runtime.abandoned: return Deny("Session abandoned")   # instant, no wait
   rid = uuid4(); fut = loop.create_future()
   runtime.pending[rid] = fut; runtime.pending_payloads[rid] = request_event
   await out.put(approval_request event)
   try:
       decision = await asyncio.wait_for(fut, timeout=300)
   except TimeoutError:
       runtime.abandoned = True; runtime.status = "interrupted"
       return Deny("Approval timed out — session interrupted.")
   finally:
       runtime.pending.pop(rid, None); runtime.pending_payloads.pop(rid, None)
   ```
   An approve resolution carries allow (+ `updated_input` if provided); reject/interrupt resolutions carry deny.
2. PreToolUse hook: ~10-line deterministic path guard — deny any file write whose resolved path falls outside the caller's workspace dir. This is a backstop (relative writes already land in the user cwd thanks to Phase 1), keep it anyway. Keep the existing CLAUDE.md instruction as-is.

**Smoke** `scripts/smoke_phase4.py`: prompt that triggers a Bash approval; mode A auto-approves via the future after printing the request; mode B never resolves and (with the timeout overridden to ~10s via parameter) asserts the deny lands, `abandoned` flips, and the coroutine returns. Also assert a write aimed at `/tmp/outside.txt` gets denied by the hook.

**Commit.**

## Phase 5 — HTTP: /chat + /stream (SSE)

**Goal: start a turn and watch it over HTTP.** Every endpoint: auth + `validate_session_id` where applicable.

1. `POST /chat` — body `{session_id?, prompt}`:
   - Existing session: 409 if `runtime.task` is live; otherwise spawn the turn task (keep the handle), return **202** `{session_id}` immediately.
   - New session (no id): spawn the turn, await the new-session-id signal from Phase 3 (cap the wait at ~15s → 500 on failure), then return 202 `{session_id}` so the client can open the stream. Any events emitted before the stream attaches sit safely in the queue.
2. `GET /stream?session_id=` — SSE. Opened when the user **enters** a session, not when an approval appears. Drains `runtime.out` (create an empty runtime lazily if absent, e.g. after a restart); emits a `: ping` heartbeat every 20s (proxies kill idle connections). **Do not cancel anything on disconnect** — reconnect-and-recover is a feature; the Phase 4 timeout is the cleanup mechanism.

**Smoke** `scripts/smoke_phase5.py` (httpx): POST /chat with a no-tool prompt (new session), open the stream with the returned id, read to completion; POST /chat again on the same session and confirm resumed context; confirm a bogus session_id format gets 400.

**Commit.**

## Phase 6 — HTTP: /approve, /reject, /pending-approvals, /interrupt

**Goal: close the loop.**

1. `POST /approve` / `POST /reject` — body `{session_id, request_id, updated_input?}`: `fut = runtime.pending.get(request_id)`; `if fut and not fut.done(): fut.set_result(...)`; else 409/410.
2. `GET /pending-approvals?session_id=` — returns `pending_payloads` values (SSE-reconnect recovery).
3. `POST /interrupt` — body `{session_id}`:
   - If any pending futures: resolve each undone one with deny "Cancelled by user" (clean path — turn ends through the SDK, transcript stays valid/resumable). Do NOT set `abandoned` — the user is still here.
   - Else if `runtime.task` is live: `task.cancel()` (dirty mid-tool cancel, accepted for v1).
   - Mark `status = "interrupted"`, emit an `interrupted` event on the stream.

**Smoke** `scripts/smoke_phase6.py` (httpx): full loop — chat with a Bash-triggering prompt, stream until `approval_request`, approve, stream to completion; second run: interrupt while blocked on approval, then chat again on the same session and confirm resume works.

**Commit.**

## Phase 7 — Hardening, observability, docs

1. Run script / Makefile target starting `uvicorn --workers 1`, with a loud comment explaining why multi-worker breaks approvals (futures/queues are per-process).
2. Background task logging every 30s: `len(runtime)`, total pending futures, and claude subprocess count (psutil or `pgrep -c claude`).
3. Grep the loaded plugins for `@tool` / `create_sdk_mcp_server`; if any handler makes blocking calls, list them in the README as event-loop risks (fixing them is out of scope — just flag).
4. `README-poc-v1.md`: architecture sketch, endpoint list, and known v1 limitations:
   - dirty mid-tool cancel can leave a dangling tool_use and break resume for that session
   - in-memory runtime lost on restart (transcripts survive only if the volume does)
   - single worker only
   - in dev mode, all master-key tokens map to one synthetic user → one shared workspace
   - no disconnect grace timer (timeout is the only reclaim path)
   - leak-check procedure: `watch -n2 'pgrep -c claude'` while triggering approvals and closing tabs unanswered — count should climb, then drop within the timeout window.

**Smoke:** rerun `smoke_phase6.py`; confirm the subprocess count log returns to baseline afterward.

**Commit.**

---

## Ground rules

- Ask me before adding dependencies beyond httpx/psutil/sse-starlette (or equivalents already in the repo).
- No changes to `agent_config.py`, `get_current_user`, or the core agent logic in `src/agents/for_migration/` beyond what's needed to inject options/callbacks.
- If the installed SDK's API differs from what's described above (names, signatures), adapt to the installed version and note the difference in the commit message.
- Phase 0's result gates everything: do not proceed past it on a FAIL without checking with me.