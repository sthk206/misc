# Migrate stress-pnl to per-desk plugins + an installed core package

## Context
I build a risk agent with the Claude Agent SDK. Today there is one plugin, `stress-pnl`, in <PLUGIN_REPO>, containing:
- 3 skills, one per desk: gsp-ig-ca, gsp-muni, fxo
- `knowledge/` — markdown that skills inject with `!` lines
- `src/` — shared Python code the skills' scripts import

The calling agent lives in <CALLING_AGENT_REPO>. It currently makes the shared markdown reachable through `add_dirs`.

## Goal
1. **One plugin per desk.** A user only gets the plugins for desks they're entitled to, and a desk plugin contains nothing about other desks. Desk isolation is the reason for this migration, so treat any cross-desk leak as a bug.
2. **Desk-neutral code and markdown become one installable Python package, `risk-core`** (import name `risk_core`), installed into the calling agent's venv. Skills read its markdown through a `risk-ref` command the package provides, and scripts import `risk_core` normally. Every user's session has this package, so it must contain **nothing desk-specific**.
3. **Desk-specific markdown lives inside that desk's skill**, in `skills/<skill>/references/`, and is read with `${CLAUDE_SKILL_DIR}`.


## Target layout
```
<PLUGIN_REPO>/
  pyproject.toml              # dev-only project, never published (see step 6)
  risk-core/
    pyproject.toml            # distribution "risk-core"
    src/risk_core/
      __init__.py
      ref.py                  # the risk-ref command
      <desk-neutral modules from today's src/>
      knowledge/              # desk-NEUTRAL markdown only
  gsp-ig-ca/
    .claude-plugin/plugin.json   # {"name": "gsp-ig-ca", ...}
    skills/stress-pnl/
      SKILL.md
      references/             # this desk's markdown for this skill
    scripts/
  gsp-muni/   (same shape)
  fxo-plugin/        (same shape)
  tests/
```
Skills become `/fxo:stress-pnl`, `/gsp-muni:stress-pnl`, and so on. 

## Facts already verified in Claude Code v2.1.294 (do not work around these)
- **Permissions:** `!` commands never prompt. Any command not allowed (via skill `allowed-tools` or the session's `allowed_tools`) aborts the whole skill.
- **`cat` with `${CLAUDE_SKILL_DIR}`:** `` !`cat "${CLAUDE_SKILL_DIR}/references/x.md"` `` works when the plugin's folder is in the SDK's `add_dirs` (no allow rule needed). Without that, it's blocked, because `cat` may only read inside the session's working directories.
- **Path variables:** `${CLAUDE_SKILL_DIR}` and `${CLAUDE_PLUGIN_ROOT}` are substituted before the permission check. For local-path plugins they point at the plugin's own folder. **Never use `${CLAUDE_PLUGIN_ROOT}/..`**: nothing outside the plugin folder is guaranteed to be there.
- **No shell variables:** `$FOO` inside a `!` line is refused ("can't be checked before it runs").
- **`PATH`:** `risk-ref` and `python` are found through the `PATH` Claude Code inherits from the backend. If the backend is started as `.venv/bin/python app.py`, the venv is NOT on `PATH` and skills fail with "command not found". So the backend must pass `PATH` explicitly (step 5).
- **Failures abort:** if an injected command exits non-zero (e.g. a missing file), the whole skill invocation aborts. That's desired.

## Steps

### 1. Inventory, then stop and report
Before moving anything:
- **Shared-code imports:** list every file in the current plugin, plus every import of the shared `src/` code.
- **References:** list every `!` line, `CLAUDE_PROJECT_DIR`, `CLAUDE_PLUGIN_ROOT`, `CLAUDE_SKILL_DIR` and `add_dirs`, and every script the skills tell Claude to run.
- **Knowledge files:** classify each one as desk-neutral (→ `risk-core/src/risk_core/knowledge/`) or desk-specific (→ that desk's `skills/stress-pnl/references/`). Do the same for each module in `src/`: desk-neutral → `risk_core`, desk-specific → that desk's `scripts/`.
- **Dependencies:** list the third-party packages the shared code and scripts import.

Show me the classification and **wait for my confirmation**. If anything mixes desks, flag it rather than guessing. Honestly, the structure should be mostly similar to how it's organized currently.

### 2. Create the `risk-core` package
`risk-core/pyproject.toml`:
```toml
[project]
name = "risk-core"
version = "0.1.0"
requires-python = "<match the calling agent>"
dependencies = [<every third-party lib risk_core AND the desk scripts import at runtime>]

[project.scripts]
risk-ref = "risk_core.ref:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/risk_core"]
```

`src/risk_core/ref.py`:
```python
import sys
from importlib.resources import files
from pathlib import PurePosixPath

def main():
    if len(sys.argv) != 2:
        sys.exit("usage: risk-ref <path under knowledge/>, e.g. stress-pnl/method.md")
    rel = PurePosixPath(sys.argv[1])
    if rel.is_absolute() or ".." in rel.parts:
        sys.exit(f"risk-ref: invalid name {sys.argv[1]!r}")
    print(files("risk_core").joinpath("knowledge", *rel.parts).read_text())
```

Move the desk-neutral modules from today's `src/` into `src/risk_core/` and fix their internal imports.

### 3. Create the three desk plugins
- **Skills:** move each skill to `<desk>/skills/stress-pnl/SKILL.md`.
- **Plugin-local files:** move that desk's markdown to `<desk>/skills/stress-pnl/references/` and its scripts to `<desk>/scripts/`.
- **Manifest:** give each plugin a `.claude-plugin/plugin.json` with its desk name.

### 4. Rewrite the skills
- **Frontmatter:** add `allowed-tools: Bash(risk-ref *)`, plus whatever the skill needs to run its scripts.
- **Shared markdown:** `` !`risk-ref stress-pnl/method.md` ``
- **Desk markdown that must always be in context:** `` !`cat "${CLAUDE_SKILL_DIR}/references/taxonomy.md"` ``

- **Scripts:** run them as `python "${CLAUDE_SKILL_DIR}/scripts/<x>.py"`. Inside scripts, import `risk_core...`, never `shared...` or `src...`. If a script needs files next to it, resolve them from `Path(__file__)`, not the working directory.
- **Content:** keep the instructions themselves unchanged. This is a plumbing migration, not a rewrite.

### 5. Update the calling agent
- **Dependency:** in `pyproject.toml`, add `risk-core`, with this dev source:
```toml
  [tool.uv.sources]
  risk-core = { path = "<relative path to PLUGIN_REPO>/risk-core", editable = true }
```
- **Plugins and `add_dirs`:** build `plugins` from the user's desk entitlements (only entitled desk folders). Set `add_dirs` to **exactly the same folders**: it's what lets `!cat` read their references, and it must never include a desk the user isn't entitled to. Set the `skills` allowlist to the matching `"<desk>:stress-pnl"` names.
- **`PATH`:** pass it explicitly so it doesn't depend on how the backend is launched:
```python
  venv_bin = str(Path(sys.executable).parent)   # do NOT .resolve() — it would follow the symlink out of the venv
  env = {"PATH": venv_bin + os.pathsep + os.environ.get("PATH", "")}
```
  Also add `"Bash(risk-ref *)"` to `allowed_tools`.
- **Startup check:** fail at startup if `shutil.which("risk-ref", path=env["PATH"])` is `None`.
- **Cleanup:** remove `load_reference.sh` and any `add_dirs` entry for the old shared folder.

### 6. Dev environment at the plugin-repo root
Create a dev-only `<PLUGIN_REPO>/pyproject.toml` that depends on `risk-core` (editable, `path = "risk-core"`), with a `dev` dependency group for pytest. Runtime libraries belong in `risk-core/pyproject.toml`, never here.

### 7. Tests and CI checks (add them to the repo)
- **Unit tests:** for `risk_core`.
- **Script tests:** run each desk script via `subprocess` from a different working directory (e.g. a temp dir), the way a skill would.
- **Reference check:** every `risk-ref` line in every `SKILL.md` resolves to an existing file in the package, and every `${CLAUDE_SKILL_DIR}/...` reference (injected or linked) resolves to an existing file inside that same skill's folder.
- **Leak scan:** each desk plugin contains no other desk's name or identifiers, and `risk_core` (code and knowledge) contains no desk-specific names. Ask me for the identifier list if needed.
- **Validation:** `claude plugin validate <desk>` passes for all three.
- **Packaging:** `uv build` in `risk-core/` produces a wheel that contains the knowledge markdown. Install it into a throwaway venv and run `risk-ref` there to confirm.

### 8. End-to-end verification
For each desk, from the dev venv, outside the repo directory:
```bash
uv run --project <PLUGIN_REPO> claude -p "/<desk>:stress-pnl" \
  --plugin-dir <PLUGIN_REPO>/<desk> --add-dir <PLUGIN_REPO>/<desk> \
  --permission-mode default --output-format stream-json --verbose
```
- **Clean run:** confirm no message contains `local-command-stderr`.
- **Same content:** confirm the injected reference content matches what the old skill produced. Write a small script that expands the `!` lines of the old and new skills and diffs the reslts.
- **On-demand links:** confirm Claude can read a linked `references/` file without a permission prompt.
- **SDK run:** run the calling agent once through the SDK with a single-desk user, and confirm in the `init` message that only that desk's plugin and skill loaded.

## Constraints
- Work on a new git branch, and don't push.
- Don't delete the old `stress-pnl` plugin until step 8 passes. Then remove it in a separate commit.
- **No workarounds:** no `${CLAUDE_PLUGIN_ROOT}/..`, no `$VARS` in `!` lines, `cat` only for files inside the skill's own folder, no `sys.path.insert`/`PYTHONPATH`, no `uv run --project` inside skills, no symlinks between plugins.
- Never put desk-specific content in `risk-core`, even if it's convenient.
- If something in this plan doesn't fit what you find in the code, stop and tell me instead of improvising.

## Report back
Summarize the final tree, every file you changed in the calling agent, the test/CI results, the step-8 output, and anything you weren't sure about.