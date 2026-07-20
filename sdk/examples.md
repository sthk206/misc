# CLAUDE.md — .claude/CLAUDE.md

```
# Project
This is a FastAPI backend.
## Commands
pytest
ruff check
uv run app.py

## Architecture
- API
- Services
- Database

## Conventions
- Never use print()
- Always use dependency injection
- Use SQLAlchemy 2.0
```
Always loaded, every session, no conditions. This is the "onboarding doc" — the stuff a new engineer would need on day one.

# Rules — .claude/rules/database.md
```
When editing files in /db or /services/**/repository.py:
- Always use SQLAlchemy 2.0 async syntax (select(), not .query()).
- Never write raw SQL strings — use SQLAlchemy Core constructs.
- All queries must go through a Repository class, never called directly from routes.
- New tables need a corresponding Alembic migration in the same commit.

Only injected when Claude actually opens a matching file — keeps CLAUDE.md from bloating with things you don't need 90% of the time.
```

# Skills — .claude/skills/add-endpoint/SKILL.md

```
# Adding a New API Endpoint

Use this when the user asks to add a new route/endpoint to the FastAPI app.

## Steps
1. Define the Pydantic request/response schema in /schemas.
2. Create the route handler in /api, using APIRouter.
3. Inject the relevant service via Depends() — never instantiate services directly.
4. Add the business logic to the corresponding file in /services.
5. If the endpoint touches the DB, add a method to the Repository class in /db.
6. Write a test in /tests/api mirroring the route path.
7. Run `pytest` and `ruff check` before finishing.
8. Update the OpenAPI docstring on the route.

## Notes
- Follow the existing pattern in /api/users.py as a reference.
- Never skip step 6 — untested routes get flagged in review.
```

A full, reusable procedure — Claude pulls this in whenever "add an endpoint" comes up, without you re-explaining the steps each time.

# settings.json — .claude/settings.json
```json
{
  "model": "claude-sonnet-5",
  "permissions": {
    "allow": ["Bash(pytest*)", "Bash(ruff*)", "Bash(uv run*)"],
    "deny": ["Bash(rm -rf *)", "Bash(git push --force*)"]
  },
  "env": {
    "PYTHONDONTWRITEBYTECODE": "1"
  }
}
```
Not a .md — it's config, not guidance. This is what Claude can do, not what it should know.

# Commands (legacy) — .claude/commands/migrate.md

```
Run `alembic revision --autogenerate -m "$ARGUMENTS"`, then show me the generated migration file before applying it.
```

Superseded by skills, but this is the old-style single-shot /migrate shortcut — no multi-step logic, no "when to use this" reasoning, just a canned instruction.

# Subagents — .claude/agents/code-reviewer.md

```
---
name: code-reviewer
description: Reviews Python code for this FastAPI project before merge
---

You are a strict senior backend reviewer for a FastAPI + SQLAlchemy 2.0 project.

Review the diff for:
- Missing dependency injection (services instantiated directly instead of via Depends())
- Raw SQL or ORM .query() usage instead of async select()
- Missing or weak Pydantic validation on request bodies
- Endpoints without a corresponding test in /tests/api
- Blocking (sync) calls inside async route handlers

Output findings as a numbered list, ordered by severity. Do not comment on style Ruff would already catch.
```

Runs in its own fresh context — won't clutter or bias your main conversation, and it's a separate "persona" whose whole job is review, not building.

# Hooks — inside .claude/settings.json
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit",
        "hooks": [{ "type": "command", "command": "ruff check --fix $CLAUDE_FILE_PATH" }]
      }
    ]
  }
}
```

Fires automatically, every time — not something Claude decides to do, something that just happens.

# MCP servers — .mcp.json

```json
{
  "mcpServers": {
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"]
    },
    "github": {
      "url": "https://mcp.github.com/sse"
    }
  }
}
```
Not a .md either — gives Claude live access to your actual database and GitHub, not just instructions about them.

# Output styles — .claude/output-styles/terse-senior-eng.md
```
---
name: terse-senior-eng
---

Respond like a senior backend engineer in a code review comment thread:
- No preamble, no "Sure! Here's..."
- Lead with the answer, not the explanation.
- Assume the reader knows FastAPI and SQLAlchemy — skip basic explanations.
- Use code blocks over prose wherever possible.
```

Pure tone/persona — has zero project facts in it. Could be reused verbatim on a totally different project.

# Workflows — .claude/workflows/release.js
```js
module.exports = {
  name: "release",
  steps: [
    "Run pytest and confirm all tests pass",
    "Bump version in pyproject.toml",
    "Run `alembic upgrade head` against staging DB",
    "Generate changelog from merged PR titles since last tag",
    "Create and push a git tag",
    "Trigger deploy via `gh workflow run deploy.yml`"
  ]
};
```

A fixed, ordered pipeline you trigger on demand — distinct from a skill because it's not "load when relevant," it's "run this exact sequence when I say /release."