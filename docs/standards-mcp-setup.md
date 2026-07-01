# dgroup-standards MCP setup

This repo is wired to the shared **`dgroup-standards`** MCP server (a remote HTTP
service on Cloud Run). It gives Claude Code / agents direct access to the org's
Flutter engineering standards, ADRs, PRDs, task records, rules, and templates —
the same source of truth used by the DGroup app — so Primordis can be built to
house standards from day one.

## Why it's here

- `get_coding_standards("flutter")` → the AI-enforced Flutter standards
  (Riverpod, Freezed, Retrofit/Dio, GoRouter, `package:lint`, testing,
  accessibility, versioning).
- `list_adrs` / `get_adr` → reusable Flutter/web ADRs (state management,
  routing, web rendering, URL strategy, etc.).
- `get_template` → the ADR / task / PR templates used to author `docs/`.

Prefer these tools over re-deriving conventions from scratch.

## One-time local setup (required — the key is NOT committed)

Primordis is a **public** repo, so the MCP API key must never be committed. The
committed `.mcp.json` references the key via an environment variable; you provide
the actual key in a **gitignored** local settings file.

```bash
cp .claude/settings.local.json.example .claude/settings.local.json
# then edit .claude/settings.local.json and paste the real key into
# env.DGROUP_STANDARDS_API_KEY  (ask a maintainer for it)
```

Restart Claude Code (or reconnect MCP) so the server picks up the key, then
verify with `/mcp` — `dgroup-standards` should show as connected.

## How it works

| File | Committed? | Purpose |
| --- | --- | --- |
| `.mcp.json` | ✅ yes | Declares the `dgroup-standards` HTTP server; key is `${DGROUP_STANDARDS_API_KEY}` (no secret) |
| `.claude/settings.json` | ✅ yes | Enables the project server and allowlists standards lookups + the task read/write and git/gh/flutter workflow tools used by the `mcp-autotask` / `land` skills |
| `.claude/settings.local.json.example` | ✅ yes | Template to copy |
| `.claude/settings.local.json` | 🚫 gitignored | Your real key, injected as `env.DGROUP_STANDARDS_API_KEY` and expanded into `.mcp.json` |

The `env` block in `.claude/settings.local.json` is applied before MCP servers
connect, so `${DGROUP_STANDARDS_API_KEY}` in `.mcp.json` resolves to your key.

> Note: the Primordis scopes now exist in the MCP server — **`PRIMORDIS`** (product) and
> **`PRIMORDIS_WEB`** (project). **Pending tasks are hosted in the server** under
> `PRIMORDIS_WEB` (create/read via `create_task` / `list_tasks` / `get_task_context` with
> `scope: "PRIMORDIS_WEB"`). The PRD, ADRs, research, and the completed early tasks
> (001–005) remain as files under `docs/`. The server is also used for read access to the
> org-wide Flutter standards and reusable ADRs.
