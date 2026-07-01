---
name: land
description: Complete a work session by running quality gates, pushing all changes to remote, updating task/issue status, and handing off context. Use when finishing work, wrapping up, or ending a session.
---

Land the current session. Work is NOT complete until `git push` succeeds.

> Primordis is a single Flutter package (Flutter Web + native macOS) with a Python
> reference sim (`Primordis.py`). Active task tracking lives in the `dgroup-standards`
> MCP server under the **`PRIMORDIS_WEB`** scope — not in `docs/tasks/`. See
> [AGENTS.md](../../../AGENTS.md) for the authoritative workflow.

## 1. File Issues for Remaining Work

- Review uncommitted ideas, TODOs, or out-of-scope findings from this session.
- Create follow-up tasks for anything that needs it via `create_task(title, description, priority, scope: "PRIMORDIS_WEB")` on the `dgroup-standards` MCP server.
- Set new tasks to `Proposed` and reference the originating task for traceability.

## 2. Run Quality Gates (if code changed)

Mirror the CI gate (`.github/workflows/ci.yml`, job `analyze-test`) locally before pushing:

```bash
flutter pub get
dart run build_runner build --delete-conflicting-outputs   # regenerate Freezed/Riverpod (*.g.dart, *.freezed.dart are gitignored)
flutter analyze                                             # zero-warning policy (package:lint)
flutter test
```

When the change touches the web or macOS build/present paths, also smoke the build that applies:

```bash
flutter build web --wasm       # Skwasm/WasmGC target (see PRIMORDIS-ADR-007)
flutter build macos --debug    # native macOS shell
```

Fix any failures before proceeding. Do not skip this step.

## 3. Update Task Status (via MCP)

- Mark completed tasks `Done` via `update_task(number, status: "Done", scope: "PRIMORDIS_WEB")`.
- Update in-progress items with current state.
- If a PR was opened or updated, attach the PR URL: `update_task(number, prLink: "<PR URL>", scope: "PRIMORDIS_WEB")`.

## 4. Push to Remote (MANDATORY)

```bash
git pull --rebase origin main
git push
git status  # MUST show "up to date with origin"
```

- If push fails, resolve conflicts and retry until it succeeds.
- NEVER stop before pushing — that leaves work stranded locally.
- NEVER say "ready to push when you are" — YOU must push.

## 5. Clean Up (do not skip — this is what keeps `.claude/worktrees/` and stale branches from piling up)

Order matters. Worktree teardown is LAST because once the worktree is gone you can no longer run MCP calls or `remember(...)` from it.

### 5a. Stashes

```bash
git stash list
# git stash drop stash@{N}   # for any you no longer need
```

### 5b. Identify the branches in play

A Claude session typically has two branches:
- **Worktree branch** — `claude/<slug>`, created by the harness under `.claude/worktrees/`; throwaway per [CLAUDE.md](../../../CLAUDE.md).
- **Feature branch** — e.g. `feat/task-<n>-<slug>`, pushed to open the PR.

```bash
WORKTREE_PATH="$(git rev-parse --show-toplevel)"
WORKTREE_BRANCH="$(git branch --show-current)"
MAIN_REPO="$(git worktree list --porcelain | awk '/^worktree/{print $2; exit}')"
```

### 5c. Delete merged feature branches (local + remote)

For each `feat/...` branch you touched, verify the PR actually merged before deleting:

```bash
gh pr list --repo babernethy/primordis --head "<feature-branch>" --state all \
  --json number,state,mergedAt,headRefName --limit 1
```

Only proceed when `state` is `MERGED`. Then:

```bash
git -C "$MAIN_REPO" branch -d "<feature-branch>"                  # safe; refuses if unmerged
git push origin --delete "<feature-branch>" 2>/dev/null || true   # GitHub auto-delete-branch may already have run
```

If the PR is `OPEN` or `CLOSED` (not merged), leave the branch alone and call it out in the hand-off. Never `git branch -D` to force-delete unless the user explicitly says the work is abandoned.

```bash
git -C "$MAIN_REPO" remote prune origin   # tidy stale remote-tracking refs
```

### 5d. Tear down the worktree (must be last; skip if you weren't in one)

You cannot remove a worktree from inside itself — `cd` to the main repo first. Skip this entire sub-step if `WORKTREE_PATH == MAIN_REPO`.

```bash
cd "$MAIN_REPO"
git worktree remove "$WORKTREE_PATH"   # fails if the worktree is dirty — that's the safety net
git branch -D "$WORKTREE_BRANCH"       # claude/* branches are throwaway per CLAUDE.md
git worktree prune                     # tidy stale worktree admin files
```

If `git worktree remove` reports the worktree is dirty, STOP and surface what's outstanding — do not pass `--force`.

## 6. Verify

- Confirm all changes are committed AND pushed.
- `git status` must show a clean working tree and up-to-date with origin.
- `git log --oneline -5` to confirm pushed commits.

## 7. Hand Off

Provide context for the next session:
- Summary of what was done and what remains.
- Note any unresolved follow-ups (and the `PRIMORDIS_WEB` task numbers filed for them).
- Save key decisions or novel patterns as memory via `remember(content, tags, importance)` on the MCP server.
