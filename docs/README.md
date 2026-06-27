# Primordis — Flutter Web + macOS Port Documentation

This folder holds the research, product requirements, architecture decisions, and
task breakdown for porting **Primordis** (a GPU particle-life simulator, currently
a single Python file `../Primordis.py` built on pygame + moderngl + numpy) to a
single **Flutter** app that runs on **Flutter Web (WASM)** and **native Flutter macOS**.

These documents follow the DGROUP house format (PRD / ADR / task templates from the
`dgroup-standards` MCP server). Primordis is its own standalone repo and is not yet a
project in the MCP server, so the documents are committed here as files. They use the
local scope prefix `PRIMORDIS` and local numbering from `001`.

## Start here

1. **[Research summary](research/PRIMORDIS-research-summary.md)** — the verified
   feasibility analysis (web + macOS), the core constraint, particle-count tiers,
   approaches, risks, and effort. Read this first for the "why."
2. **[PRIMORDIS-PRD-001: Flutter Web + macOS Port](prd/PRIMORDIS-PRD-001-flutter-web-and-macos-port.md)**
   — goals, non-goals, the architecture, user stories, success metrics, open
   questions, and the phased delivery plan.
3. The **ADRs** below for the technology decisions, and the **tasks** for execution.

## The one-paragraph version

Flutter has **no first-party Dart GPU-compute API on any platform** (verified, mid-2026):
`FragmentProgram` is fragment-only; `flutter_gpu` is render-only and not on web. So
Primordis becomes a Flutter app that **hosts a GPU canvas**, with all compute living
*outside* Flutter behind a single Dart `SimBackend` interface. On the **web**, compute
runs on **browser WebGPU** (WGSL) via `dart:js_interop` (full 24k particles), falling
back to a pure **Dart→WASM CPU** sim (~3–4k) where WebGPU is absent. On **native macOS**,
the *same* WGSL kernel runs via **Dawn/wgpu-over-Metal through `dart:ffi`** (full 24k+,
100k+ on Apple Silicon), with a hand-written **Metal (MSL)** plugin as a robust fallback.
The UI, sliders, and sim parameters are shared across all platforms; only the compute
dispatch and the present/compositing path are platform-specific.

## Architecture Decision Records

Status: all **Proposed** (2026-06-27).

| ADR | Decision |
| --- | --- |
| [ADR-001](adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md) | Cross-platform architecture — one Flutter app + a Dart `SimBackend` interface with swappable per-platform compute backends |
| [ADR-002](adr/PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md) | Web GPU compute via **browser WebGPU (WGSL)** through `dart:js_interop` on an owned canvas; reject `FragmentProgram`/`flutter_gpu` |
| [ADR-003](adr/PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md) | **One shared WGSL kernel** reused on web (browser WebGPU) and native (Dawn/wgpu over Metal) — write once, two backends |
| [ADR-004](adr/PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md) | Native macOS GPU via **Dawn/wgpu-over-Metal (FFI)** primary, hand-written **Metal (MSL)** plugin fallback; OpenGL impossible (frozen at 4.1) |
| [ADR-005](adr/PRIMORDIS-ADR-005-rendering-and-compositing.md) | Compositing — web: stacked WebGPU canvas behind a transparent Flutter glass-pane; macOS: IOSurface-backed Metal texture via the `Texture` widget |
| [ADR-006](adr/PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md) | Capability detection + graceful-degradation tiers and particle-count policy |
| [ADR-007](adr/PRIMORDIS-ADR-007-web-build-and-cross-origin-isolation.md) | Web build (`--wasm`/Skwasm + CanvasKit fallback) and COOP/COEP cross-origin isolation |

## Particle-count tiers

| Tier | Backend | Realistic ceiling @ 60fps |
| --- | --- | --- |
| Native GPU | macOS Dawn/wgpu-over-Metal (WGSL) or MSL plugin | **24k and well beyond — 100k–500k+ on Apple Silicon** |
| Web GPU | Browser WebGPU (same WGSL) | **Full 24k+** |
| Native CPU | Isolates + FFI shared buffer | **~10–14k solid (estimate — benchmark)** |
| Web CPU | Dart→WASM single-thread | **~3–4k only (hard web ceiling)** |

## Tasks

Status: all **Todo**. Suggested execution order follows the phases in the PRD. Each
task file has full acceptance criteria, implementation notes, testing steps, and
dependency links.

| Phase | Task | Pri | Depends on |
| --- | --- | --- | --- |
| Web 0 | [TASK-001 Project scaffold & build config](tasks/PRIMORDIS-TASK-001-project-scaffold-and-build-config.md) | High | — |
| Web 0 | [TASK-002 SimBackend interface & shared sim model](tasks/PRIMORDIS-TASK-002-simbackend-interface-and-shared-sim-model.md) | High | 001 |
| Web 1 | [TASK-003 Port simulation to WGSL compute kernel](tasks/PRIMORDIS-TASK-003-port-simulation-to-wgsl-compute-kernel.md) | Critical | 002 |
| Web 2 | [TASK-004 Web WebGPU backend (js_interop)](tasks/PRIMORDIS-TASK-004-web-webgpu-backend-js-interop.md) | Critical | 003 |
| Web 2 | [TASK-005 Web canvas compositing & pointer routing](tasks/PRIMORDIS-TASK-005-web-canvas-compositing-and-pointer-routing.md) | High | 004 |
| Web 2 | [TASK-006 Sliders → uniforms & UI chrome](tasks/PRIMORDIS-TASK-006-sliders-to-uniforms-and-ui-chrome.md) | High | 002, 004 |
| Web 3 | [TASK-008 Dart-WASM CPU fallback backend](tasks/PRIMORDIS-TASK-008-dart-wasm-cpu-fallback-backend.md) | High | 002 |
| Web 3 | [TASK-007 WebGPU feature detection & fallback switch](tasks/PRIMORDIS-TASK-007-webgpu-feature-detection-and-fallback-switch.md) | High | 004, 008 |
| Web 3 | [TASK-010 Web build, hosting & cross-origin isolation](tasks/PRIMORDIS-TASK-010-web-build-hosting-and-cross-origin-isolation.md) | Medium | 005 |
| Web 4 | [TASK-009 Parity test harness vs Python reference](tasks/PRIMORDIS-TASK-009-parity-test-harness-vs-python-reference.md) | High | 003, 008 |
| Web 4 | [TASK-018 Test coverage & accessibility](tasks/PRIMORDIS-TASK-018-test-coverage-and-accessibility.md) | High | 006, 008 |
| macOS M1 | [TASK-011 macOS target — Dawn/wgpu FFI backend](tasks/PRIMORDIS-TASK-011-macos-target-dawn-wgpu-ffi-backend.md) | High | 003, 004 |
| macOS M1 | [TASK-017 Atomics parity validation — Dawn vs browser](tasks/PRIMORDIS-TASK-017-atomics-parity-validation-dawn-vs-browser.md) | High | 011 |
| macOS M2 | [TASK-012 macOS Metal texture present path](tasks/PRIMORDIS-TASK-012-macos-metal-texture-present-path.md) | High | 011 |
| macOS M3 | [TASK-014 Native CPU isolate fallback backend](tasks/PRIMORDIS-TASK-014-native-cpu-isolate-fallback-backend.md) | Medium | 002, 008 |
| macOS M3 | [TASK-015 Cross-platform backend selection & reduced-mode UX](tasks/PRIMORDIS-TASK-015-cross-platform-backend-selection-and-reduced-mode-ux.md) | High | 007, 011, 014 |
| macOS M4 | [TASK-016 macOS packaging, signing & GPU gating](tasks/PRIMORDIS-TASK-016-macos-packaging-signing-and-gpu-gating.md) | Medium | 012 |
| Fallback | [TASK-013 macOS Metal (MSL) compute plugin (de-risking)](tasks/PRIMORDIS-TASK-013-macos-metal-msl-compute-plugin-fallback.md) | Medium | 011 |

## Effort (from research)

- **Web WebGPU path:** ~5–8 person-weeks. The shader port is the easy part; the
  DOM/Flutter compositing, pointer routing, DPR/resize, feature-detect, and CPU
  fallback are where the time goes.
- **macOS GPU target:** ~1–2 weeks incremental on top of the web build (the WGSL
  kernel is already written; cost is the Dawn native-asset build, the IOSurface
  texture bridge + frame pacing, `std430`→Metal alignment, and signing/notarization).

## Recommended first step

Begin with **[TASK-001](tasks/PRIMORDIS-TASK-001-project-scaffold-and-build-config.md)**
(scaffold + `--wasm` build config) and **[TASK-002](tasks/PRIMORDIS-TASK-002-simbackend-interface-and-shared-sim-model.md)**
(the `SimBackend` interface + shared Freezed sim model), then de-risk the foundation
with a WebGPU `js_interop` spike inside **[TASK-004](tasks/PRIMORDIS-TASK-004-web-webgpu-backend-js-interop.md)**
before committing to the full kernel port in **[TASK-003](tasks/PRIMORDIS-TASK-003-port-simulation-to-wgsl-compute-kernel.md)**.

## Conventions

- **Scope prefix:** `PRIMORDIS`. **Numbering:** local, from `001`.
- **Statuses:** PRD `Draft` · ADRs `Proposed` · tasks `Todo`. Update in place as work
  proceeds (e.g. ADRs → `Accepted` once ratified, tasks → `In Progress`/`Done`).
- If/when a Primordis project is added to the `dgroup-standards` MCP server, these
  files can be imported as the seed content with their existing numbering.
