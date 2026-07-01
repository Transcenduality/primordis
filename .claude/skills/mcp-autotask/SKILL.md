---
name: mcp-autotask
description: Execute a task end-to-end using the MCP task record and repo workflow, saving key decisions to memory. Use when given a numeric task ID to implement. Handles full workflow from task resolution through PR creation.
---

Execute this task fully with no feedback loop unless blocked by missing credentials/permissions.

Inputs:
- `task_number`: `$ARGUMENTS`

> Primordis tasks live in the `dgroup-standards` MCP server under the **`PRIMORDIS_WEB`**
> scope. This is a single Flutter package (Flutter Web WASM + native macOS) porting the
> Python reference sim `Primordis.py`. See [AGENTS.md](../../../AGENTS.md) for the
> authoritative architecture and workflow.

## 1. Task Resolution (required before implementation)

- Resolve the task in the `dgroup-standards` MCP service, scope `PRIMORDIS_WEB`.
- Try `get_task_context(task_number)` first.
- If scope is ambiguous or no task is returned, use `list_tasks(scope: "PRIMORDIS_WEB")`, `search_tasks`, `get_task(number, scope: "PRIMORDIS_WEB")` to pick the single best match.
- If no task can be resolved, stop and report blocked input with the exact MCP lookup(s) attempted.
- Once resolved, set:
  - `task_id` = fully-qualified task identifier (e.g. `PRIMORDIS_WEB-TASK-796`)
  - `task_name` = resolved task title
  - `taskslug` = `task_name` lowercased, non-alphanumeric replaced with `-`, leading/trailing `-` trimmed
- Build `task_snippet` from the resolved MCP task record before implementation: title, status, dependencies, acceptance criteria, and any explicit constraints. Use it as the implementation source context.
- Update the task status to `In Progress` via `update_task(task_number, status: "In Progress", scope: "PRIMORDIS_WEB")`.

## 2. Context Gathering (standards from MCP; Primordis docs from repo)

- Fetch coding standards from MCP: `get_coding_standards("flutter")`.
- Fetch reusable org ADRs from MCP: `list_adrs` to enumerate, then `get_adr(number)` for applicable ones (state management, routing, web rendering, URL strategy, etc.).
- `search_guidelines(query)` when the task involves unfamiliar patterns.
- Read the **Primordis-specific** ADRs and PRD from the repo (these are file-based for now): `docs/adr/PRIMORDIS-ADR-0NN-*.md`, `docs/prd/PRIMORDIS-PRD-001-*.md`, and `docs/research/`. The task's own "Related" section names the ones that apply.

## 3. Hard Requirements

- Follow [AGENTS.md](../../../AGENTS.md) in the repo root exactly.
- If the harness has not already put you in a worktree, create a feature branch + worktree in one step:
  `git worktree add -b feat/task-<n>-<taskslug> .claude/worktrees/<taskslug>`
- Implement code + tests for all meaningful changes.
- Respect Primordis conventions:
  - **Feature/UI/domain code** (`lib/features/**`, `lib/app/**`, `lib/shared/**`): Riverpod annotation style with plain `Ref` (no `setState` for business state), Freezed models, GoRouter, Material 3 + GoogleFonts, `package:lint` zero-warning. Never hand-edit generated `*.g.dart` / `*.freezed.dart`.
  - **Simulation backends** (`lib/sim/**`): everything reaches compute through the single `SimBackend` interface. The GPU/JS-interop and WGSL/FFI code is quarantined here **by design** — `dart:js_interop` (web WebGPU) and `dart:ffi` (macOS Dawn/Metal) live under `lib/sim/backends/**` and must never leak into `lib/features/**`. The shared WGSL kernel (`lib/sim/kernel/primordis.wgsl`) is reused on web (browser WebGPU) and native (Dawn/wgpu over Metal).
  - **Parity work** validates against the Python reference `Primordis.py` — "faithful" means visual/statistical, never bit-exact.
- Apply version bumps when Flutter/native code changed: bump `pubspec.yaml` `version:` and the mirrored `PrimordisConfig.version` constant (semver).

## 4. Task Status Updates (via MCP throughout)

- Set `In Progress` at the start (done in step 1).
- If blocked, set `Blocked` and note the reason.
- Set `Done` before opening/updating the PR: `update_task(task_number, status: "Done", scope: "PRIMORDIS_WEB")`.
- Attach the PR link once created: `update_task(task_number, prLink: "<PR URL>", scope: "PRIMORDIS_WEB")`.

## 5. Quality Gates

Mirror CI (`.github/workflows/ci.yml`, job `analyze-test`):

```bash
flutter pub get
dart run build_runner build --delete-conflicting-outputs
flutter analyze     # zero warnings
flutter test
```

When the change touches build/present paths, also smoke `flutter build web --wasm` and/or `flutter build macos --debug`.

## 6. Commit and Landing Workflow

- Commit with clear, descriptive messages.
- Landing: `git pull --rebase origin main`, `git push`, verify `git status` is up to date with origin.
- Open a PR against `main` on `babernethy/primordis`. If a PR template exists (`.github/pull_request_template.md` or `.github/PULL_REQUEST_TEMPLATE/`), follow it; otherwise include summary, changed files, scope, and exact test commands/results. You may fetch the org template via MCP `get_template("pull-request")`.

## 6a. Request Copilot Code Review and Address Feedback

Immediately after `gh pr create` returns the PR URL, add Copilot as a reviewer and loop on its feedback. Do NOT skip this step.

Extract `<owner>`, `<repo>`, and `<num>` from the PR URL (owner/repo = `babernethy/primordis`), then:

```bash
gh api --method POST /repos/<owner>/<repo>/pulls/<num>/requested_reviewers \
  -f "reviewers[]=copilot-pull-request-reviewer[bot]"
```

Notes:
- The literal reviewer slug is `copilot-pull-request-reviewer[bot]` (the `[bot]` suffix is required; `Copilot` alone fails with 422 "not a collaborator"). The response `requested_reviewers[].login` echoes back as `Copilot` — same bot.
- `gh pr edit --add-reviewer` does NOT work for this bot.

Poll for the review with `ScheduleWakeup` (or `/loop`) at 180-second intervals, staying inside the 5-minute prompt cache window:

```bash
gh pr view <num> --repo <owner>/<repo> --json reviews \
  --jq '.reviews[] | select(.author.login=="copilot-pull-request-reviewer" or .author.login=="copilot-pull-request-reviewer[bot]") | {state, submittedAt, body}'
```

When a review arrives:
- If Copilot reports "generated no comments" (clean), record that in the handoff and move on.
- If Copilot leaves inline comments, fetch them with `gh api /repos/<owner>/<repo>/pulls/<num>/comments`, evaluate each one, apply the actionable fixes, and push a follow-up commit on the same branch. Reply to each thread via `gh api --method POST /repos/<owner>/<repo>/pulls/<num>/comments/<comment-id>/replies -f body='...'` so reviewers know how each point was handled.
- If a suggestion is wrong, skip it and note the rationale in the reply or handoff.

Give up after 15 minutes of polling and hand off with a note that Copilot didn't respond.

**Merge-speed caveat:** before pushing any Copilot-response commit, re-check `gh pr view <num> --repo <owner>/<repo> --json state,mergeCommit` — if `MERGED`, file a follow-up task with the commit SHA instead of pushing to a stranded branch.

## 6b. Wait for CI Checks to Clear

After Copilot feedback is addressed and any follow-up commits are pushed, wait for CI before ending the session. The MCP task status was already set to `Done` in §4 — this is about not handing back while checks are red or in flight.

```bash
gh pr checks <num> --repo <owner>/<repo> --json name,state,conclusion,link
```

The primary check is `analyze-test`. Decision tree:

- Any check `pending|queued|in_progress` → `ScheduleWakeup` at 180-second intervals and re-check.
- All checks terminal with `conclusion=success` → done; record the green status in the handoff.
- Any check terminal with `conclusion=failure|timed_out|cancelled` → a real failure to fix, not a blocker to report. Fetch logs:
  ```bash
  gh run view <run-id> --repo <owner>/<repo> --log-failed
  ```
  Diagnose, commit the fix on the same branch, push, then restart the poll loop (don't re-request Copilot review just for a CI fix unless the change is substantive).

Give up after 30 minutes of total wait and hand off with the failing check name, the run URL, and diagnosis.

## 7. Agent Run Logging (via MCP)

At the end of execution, call `log_agent_run(...)` to record a structured execution trace.

| Field | How to populate | Required? |
|-------|----------------|-----------|
| `taskNumber` | From the resolved task | Yes |
| `taskId` | From the resolved task (e.g., `PRIMORDIS_WEB-TASK-796`) | Yes |
| `agentModel` | The model powering this session (e.g., `claude-opus-4-8`) | Yes |
| `outcome` | `success`, `failure`, or `blocked` | Yes |
| `startTime` / `endTime` / `durationMs` | ISO-8601 timestamps; record `startTime` as the first action of the session | Best-effort |
| `filesModified` | `git diff --name-only origin/main...HEAD` | Best-effort |
| `toolCallsSummary` | Map of MCP tool → approximate call count | Best-effort |
| `testsRun` | Map of exact test command → `"pass"`/`"fail"` | Best-effort |
| `prLink` | PR URL if created | Best-effort |
| `blockerReason` | If outcome is `blocked` | When blocked |
| `memoryIds` | Memory IDs created via `remember(...)` | Best-effort |
| `tokenInput` / `tokenOutput` / `tokenCacheRead` | Omit if the runtime does not expose them — do not guess | Omit if unavailable |

**Best-effort:** if `log_agent_run` fails, skip it gracefully — do not treat it as a blocker.

## 8. Memory and Follow-Up Tasks (via MCP)

- Save key decisions, blockers, or novel patterns as memory via `remember(content, tags, importance)`.
- For out-of-scope issues that should be done later:
  - `get_next_task_number()`, then `create_task(title, description, priority, scope: "PRIMORDIS_WEB")`.
  - Set the new task's status to `Proposed` and reference the originating task for traceability.

## 9. Execution Rules

- Do not stop for confirmation; only stop if truly blocked, and report the exact blocker + command that failed.
- Prefer MCP for org standards/ADRs/templates; read Primordis-specific ADRs/PRD from `docs/` (they are file-based).
- Do not create local task files — all task tracking goes through MCP under `PRIMORDIS_WEB`.
- Do not leave the task `In Progress` once the PR is opened — mark it `Done` and attach the PR link.
- Keep `dart:js_interop` / `dart:ffi` out of `lib/features/**`; reach compute only through `SimBackend`.
