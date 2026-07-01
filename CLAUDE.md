# Primordis — Claude-Specific Instructions

This file is supplemental guidance for Claude-based workflows in this repository.

## Source of Truth

For architecture, read order, coding standards, quality gates, versioning, and PR/merge
workflow, use:

- [AGENTS.md](AGENTS.md)

Do not duplicate or diverge from `AGENTS.md` for shared policy.

## MCP Server: dgroup-standards

The `dgroup-standards` MCP server is available and **preferred** over reading standards
files directly. Primordis's project scope is **`PRIMORDIS_WEB`** (product **`PRIMORDIS`**).
Local key setup is in [docs/standards-mcp-setup.md](docs/standards-mcp-setup.md) — the key
is gitignored because this is a public repo.

Use MCP as the source of truth for **active task instructions and status** (scope
`PRIMORDIS_WEB`) and for **org Flutter standards / reusable ADRs**. Primordis-specific ADRs
and the PRD remain file-based under `docs/adr/` and `docs/prd/`.

Key tools: `get_task_context(number)` / `list_tasks` / `create_task` / `update_task`,
`get_coding_standards("flutter")`, `list_adrs` / `get_adr(number)`, `search_guidelines`,
`get_template`, `remember`, `log_agent_run`.

## Skills

- **`mcp-autotask`** — execute a `PRIMORDIS_WEB` task end-to-end: resolve the MCP task →
  branch/worktree → implement + tests → PR → Copilot review loop → wait for CI → log the run.
  Invoke with a task number.
- **`land`** — wrap up a session: quality gates → push → update MCP task status → clean up
  worktrees/branches → hand off. Use when finishing work.

## Claude-Specific Workflow Notes

- Treat `AGENTS.md` as authoritative when there is overlap.
- Before coding, fetch the assigned task from the `dgroup-standards` MCP server
  (`PRIMORDIS_WEB` scope), then applicable Flutter standards and ADRs.
- Reach compute only through the `SimBackend` interface — keep `dart:js_interop` and
  `dart:ffi` out of `lib/features/**` (they belong under `lib/sim/backends/**`).
- Never hand-edit generated `*.g.dart` / `*.freezed.dart`; run
  `dart run build_runner build --delete-conflicting-outputs`.
- Keep Claude worktree branches (`claude/*`, under `.claude/worktrees/`) temporary and
  local; do not push them as feature branches. Ship on `feat/task-<n>-<slug>`.
- In session handoff/PR notes, summarize what changed and any follow-up tasks filed in
  `PRIMORDIS_WEB`.

## Repository Context

Single Flutter package (not a monorepo):

- `lib/` — Flutter app (`app/`, `features/`, `shared/`) and the sim seam (`sim/`, with the
  shared WGSL kernel and per-platform backends under `sim/backends/`).
- `test/` — unit/widget tests.
- `web/`, `macos/` — platform hosts.
- `Primordis.py` — the Python reference sim (parity source of truth).
- `docs/` — PRD, ADRs, research, historical tasks.
