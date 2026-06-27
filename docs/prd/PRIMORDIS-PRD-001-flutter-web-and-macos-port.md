# PRIMORDIS-PRD-001: Flutter Web + macOS Port of Primordis

**Status:** Draft
**Author:** Bruce Abernethy (Director of Solution Architecture)
**Date:** 2026-06-27
**Last Updated:** 2026-06-27

---

## Problem Statement

Primordis is a GPU particle-life ("clusters") simulator that currently exists as a single ~350-line Python file (`Primordis.py`) built on pygame + moderngl (OpenGL 4.3) + numpy. It simulates 24,000 particles across 32 types in a 1080x720 toroidal world, driving the *entire* physics through GPU compute shaders (GLSL `#version 430`, `std430` SSBOs, `atomicAdd`). At 24k particles the simulation performs roughly 67M particle-pair tests per frame — trivial on a GPU, but only ~1-2.5 fps single-threaded on a CPU.

DGROUP is a Flutter-first shop. We want Primordis to live where our users and our tooling are: as a Flutter app that runs on the **web** and on **native macOS**, while honoring the org's standards (Riverpod, Freezed, GoRouter, `package:lint`, tests, accessibility). Primordis is its **own standalone repo** (not the DGroup monorepo), so template/repo paths are adapted accordingly.

The hard obstacle is verified and structural: **Flutter has no first-party Dart GPU-compute API on any platform as of mid-2026.**

- `dart:ui` `FragmentProgram` is fragment-shader only — no compute, no SSBO/UBO, no atomics — and historically fragile on web.
- `flutter_gpu` is render-only (no `ComputePass` in the Dart API) and does not run on web; even on macOS it exposes no compute. Impeller's Metal backend has internal compute, but it is engine-internal and unexposed, and Impeller is still opt-in (behind a flag) on macOS mid-2026 and not on web at all (web = CanvasKit + Skwasm, both Skia/WebGL).

The consequence shapes the entire project and must be stated plainly: **this is a Flutter app *hosting* a GPU canvas, not a Skia-rendered simulation.** All GPU compute must live *outside* Flutter — browser WebGPU via JS interop on web; Dawn/wgpu-over-Metal via `dart:ffi` (or a native Metal plugin) on desktop. We must **not** architect around "`flutter_gpu` compute will ship soon."

A second honesty constraint: the original GPU binning is already nondeterministic (single-buffered, with a known scatter race), so a faithful port means **visually and statistically equivalent**, never **bit-exact**. Bit-exact reproduction across CPU, browser WebGPU, and Dawn-over-Metal backends is impossible and is explicitly out of scope.

## Goals

1. Ship **one Flutter app** that runs Primordis on **Flutter web (WASM)** and **native Flutter macOS**, sharing the entire Flutter UI, sliders, and simulation parameters.
2. Reproduce the simulation's *behavior*: 24,000 particles, 32 types, asymmetric 32x32 force/min-distance/radius matrices, toroidal world, 3-pass GPU compute (clear bins -> atomic scatter-bin -> interaction+integrate), point rendering, and the 3 live sliders (Attraction K, Repulsion K, Drift/friction).
3. Establish a single Dart **`SimBackend` interface** with swappable compute implementations behind it, so the UI never knows which backend is live.
4. Reuse **one WGSL compute kernel** across web (browser WebGPU) and native (Dawn/wgpu-over-Metal via FFI) — "write the kernel once, two backends."
5. On macOS, run the **full 24k+ GPU simulation** (and reach for 100k-500k+ on Apple Silicon) — the payoff a native build provides over web.
6. Provide **graceful capability-based degradation** with a clear "reduced mode" UX when GPU compute is unavailable.
7. Meet DGROUP house standards in the UI layer: Riverpod (plain `Ref`, no `setState` for business logic), Freezed models, GoRouter for any routing, Material 3 + GoogleFonts, `package:lint` zero-warnings, tests for new code, and accessibility — including **reduced-motion** compliance, since the entire screen is motion.

## Non-Goals

- **Bit-exact reproduction** of `Primordis.py`. The reference is already nondeterministic; parity is visual/statistical, not bitwise.
- **A Skia/CanvasKit-rendered simulation.** Primordis renders into a GPU canvas/texture we own, composited with Flutter — not through Flutter's painting layer.
- **Relying on a first-party Flutter compute API** (`FragmentProgram`, `flutter_gpu`). Re-verify `flutter_gpu` against the pinned Flutter version before sign-off, but do not depend on it shipping.
- **WebGL2 GPGPU** as a primary or required path. It is a *rewrite*, not a port (no compute/atomics/scatter; binning becomes a prefix-sum counting sort over textures), and costs more than WebGPU. Skipped unless full 24k on non-WebGPU browsers becomes a hard requirement.
- **Guaranteeing 24k @ 60fps on CPU fallbacks.** CPU tiers are graceful degradation, not the target experience.
- **Full iOS / Windows / Linux delivery in this PRD.** These fall out of the same shared WGSL backend "for little extra," but are not committed scope here.

## Proposed Solution

### One Flutter app, one `SimBackend`, swappable compute backends

The architecture's load-bearing decision (see [PRIMORDIS-ADR-001](../adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md)) is: **one Flutter app, one Dart `SimBackend` interface, swappable compute implementations behind it.**

**Shared across all platforms:**

- The entire Flutter UI: the slider chrome (Attraction K, Repulsion K, Drift), reset/seed controls, and reduced-mode UX.
- The simulation parameters as Freezed models: the asymmetric 32x32 `forces`, `min_distances`, and `radii` matrices, plus the random per-type colors and the world/grid constants.
- Particle seeding, the frame loop, and parameter marshalling into the live backend.
- Under the recommended native approach, **the WGSL kernel source itself** ([PRIMORDIS-ADR-003](../adr/PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md)).

**Platform-specific, hidden behind `SimBackend`:**

- GPU device/adapter/pipeline creation.
- Compute dispatch (browser WebGPU vs Dawn-over-Metal FFI vs hand-written MSL).
- The present/composite path (web stacked-canvas vs macOS `Texture` widget).

The UI layer therefore stays fully standards-compliant (Riverpod/Freezed/Material 3). The compute/FFI/shader/JS-interop code necessarily lives **outside** the standard feature/data/domain layers — **this is expected and explicitly sanctioned**, provided it stays behind `SimBackend`.

### Architecture overview

```
            ┌──────────────────────────────────────────────────┐
            │            Flutter UI (shared, standards-compliant)│
            │  Material 3 · GoogleFonts · Riverpod · Freezed     │
            │  Sliders (Attraction K / Repulsion K / Drift)      │
            │  Seed/Reset · Reduced-mode indicator · a11y        │
            └───────────────┬──────────────────────────────────┘
                            │ params (Freezed) · frame loop
                            ▼
            ┌──────────────────────────────────────────────────┐
            │             SimBackend interface (Dart)            │
            │  init() · seed() · setParams() · step() · present()│
            └───┬───────────────┬───────────────┬───────────────┘
                │               │               │
   ┌────────────▼───┐  ┌────────▼─────────┐  ┌──▼──────────────┐
   │ Web WebGPU     │  │ Native Dawn/wgpu │  │ CPU fallbacks   │
   │ (WGSL, shared) │  │ over Metal (FFI, │  │ web: dart2wasm  │
   │ js_interop +   │  │ WGSL, shared)    │  │ native: isolates│
   │ package:web    │  │  ── or ──        │  │ + FFI shared buf│
   │ owned <canvas> │  │ MSL plugin (b)   │  │                 │
   └────────────────┘  └──────────────────┘  └─────────────────┘
        present:            present:                present:
   stacked canvas      IOSurface Metal          drawRawPoints /
   behind glass-pane   texture → Texture        drawVertices
```

### Web compute backend (primary path)

Browser **WebGPU** (WGSL compute) reached from Dart via `dart:js_interop` + `package:web`, running on a `<canvas>` **we own** — not Flutter's CanvasKit/Skwasm context ([PRIMORDIS-ADR-002](../adr/PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md)). The GLSL-430 -> WGSL port is near-1:1:

- SSBO -> `var<storage, read_write>`
- `atomicAdd(bins[i], 1)` -> `atomicAdd(&bins[i], 1u)` (returns the old value = scatter offset)
- `local_size_x` -> `@workgroup_size`
- `gl_GlobalInvocationID` -> `@builtin(global_invocation_id)`
- atomics must be `atomic<u32>` consistently and read via `atomicLoad`.

This delivers the full 24k+ at 60fps where WebGPU is present.

### Native macOS compute backend (the Mac payoff)

macOS OpenGL is frozen at 4.1 — **no compute shaders** (the original needs 4.3) — so the original GLSL compute cannot run via OpenGL on a Mac. **Metal is mandatory and sufficient** ([PRIMORDIS-ADR-004](../adr/PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md)).

- **Recommended — Approach (a):** run the **same single WGSL kernel** via Dawn/wgpu-over-Metal through `dart:ffi` (e.g. the experimental `minigpu` package). One kernel, two backends.
- **Fallback — Approach (b):** hand-written **Metal Shading Language (MSL)** compute kernels in a native macOS plugin (Swift/Obj-C++), a near-1:1 GLSL->MSL port (SSBO -> `device` buffer; `atomicAdd` -> `atomic_fetch_add_explicit` on a `device atomic_uint`; `local_size_x` -> threadgroup). Most robust and ship-ready, but a **second kernel** to maintain and Apple-only. Kept warm as the de-risking escape hatch.

This is where a Mac build earns its keep: macOS can run the full 24k+ (and 100k-500k+) GPU sim, whereas web tops out at the CPU fallback unless WebGPU is present.

### Compositing

- **Web** ([PRIMORDIS-ADR-005](../adr/PRIMORDIS-ADR-005-rendering-and-compositing.md)): stack the WebGPU `<canvas>` as a sibling DOM element **behind** a transparent Flutter glass-pane and overlay the Flutter UI. Do **not** wrap the canvas in `HtmlElementView` (it forces overlay/canvas-splitting, overlay-count limits, jank). Route pointer events explicitly; sync canvas size to `MediaQuery` device-pixel-ratio on resize.
- **macOS** ([PRIMORDIS-ADR-005](../adr/PRIMORDIS-ADR-005-rendering-and-compositing.md)): render the sim into an **IOSurface-backed Metal texture** (BGRA8 / CVPixelBuffer) and composite *under* the shared Flutter UI via the external-texture `Texture` widget. The macOS embedder supports this (`FlutterPluginRegistrar.textures` / `FlutterTextureRegistry`; engine PR `flutter/engine#24523`). The contract is `CVPixelBuffer`/`IOSurface` (`copyPixelBuffer`); watch frame pacing (`textureFrameAvailable` vs Metal completion) and confirm **no CPU readback** at 24k/60fps. This is cleaner than the web overlay.

### CPU fallbacks

- **Web fallback:** pure Dart -> WASM (`dart2wasm`) CPU sim ([PRIMORDIS-ADR-006](../adr/PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md)). Single-thread is viable only at ~3-4k particles @ 60fps. The web has **no real isolates** (web "isolates" are web workers that *copy* data); `SharedArrayBuffer` needs COOP/COEP cross-origin-isolation headers (which excludes plain static hosts). Render via **one** `Canvas.drawRawPoints`/`drawVertices` from a `Float32List` (never per-point draws). Use a **sequential counting-sort** binning (deterministic) instead of atomic scatter.
- **Native fallback:** native Dart has **real isolates**. Share one particle buffer across isolates by allocating it as native memory via `dart:ffi` (`calloc` `Pointer`; share the integer **address**, not the `Pointer` object) plus a `package:native_synchronization` mutex/barrier — true multi-core shared memory, unlike web. Realistic ceiling ~10-14k @ 60fps (low-confidence/extrapolated; Dart AOT SIMD is currently broken, so no vectorization). This does **not** reliably hit 24k@60fps; GPU is the real win. Used only as graceful degradation when GPU init fails.

### Particle-count tiers

| Tier | Backend | Compute path | Realistic particle ceiling @ 60fps |
| --- | --- | --- | --- |
| Native GPU | macOS Dawn/wgpu-over-Metal (WGSL) **or** MSL plugin | GPU compute | **24k and well beyond — 100k-500k+ on Apple Silicon** |
| Web GPU | Browser WebGPU (same WGSL) | GPU compute | **Full 24k+** |
| Native CPU | Isolates + FFI shared buffer | Multi-core CPU | **~10-14k solid; ~16-40fps at 24k (estimate — benchmark before promising)** |
| Web CPU | Dart -> WASM (`dart2wasm`), single-thread | CPU | **~3-4k only (hard web ceiling)** |

## User Stories

- **As a casual web visitor on a modern browser**, I open the Primordis URL and the full 24k-particle simulation runs smoothly at 60fps, so I can watch clusters form and drift without installing anything.
- **As a web visitor without WebGPU** (e.g. Firefox on Linux, an Intel Mac, pre-26 Safari), I still see a working simulation at a reduced particle count, with a clear indicator that I'm in reduced mode and why — rather than a blank screen or a crash.
- **As a curious tinkerer**, I drag the Attraction K, Repulsion K, and Drift sliders and watch the system reorganize in real time, so I can explore the parameter space interactively.
- **As a macOS desktop user**, I run the native app and get the full GPU simulation — and can push the particle count well beyond 24k on my Apple Silicon machine — for a noticeably richer experience than the web build's fallback.
- **As a user sensitive to motion**, the app respects my OS reduced-motion preference: it offers a pause/static state and does not force full-screen continuous motion on me.
- **As a keyboard/screen-reader user**, the sliders and controls are semantic, labeled, and tooltipped, so the UI chrome is usable even though the canvas itself is a visual artifact.
- **As a DGROUP engineer**, I work in a codebase where the UI layer follows house standards (Riverpod/Freezed/GoRouter/Material 3) and all the unavoidable non-standard GPU/FFI/JS-interop code is quarantined behind a single `SimBackend` interface, so the standard layers stay clean.

## Technical Considerations

### No first-party compute API (the governing constraint)

Verified mid-2026: Flutter exposes **no Dart GPU-compute API** on any platform. `FragmentProgram` is fragment-only; `flutter_gpu` is render-only and web-absent; Impeller's Metal compute is engine-internal and unexposed; Impeller is opt-in on macOS and not on web. Therefore **all compute is external** (browser WebGPU on web; Dawn/wgpu-over-Metal FFI or a native Metal plugin on desktop). Re-verify `flutter_gpu` against the pinned Flutter version before sign-off, but do not plan around it.

### Browser support and feature detection

WebGPU availability mid-2026 (verified): Chrome/Edge 113+ (2023), Safari 26 (macOS Tahoe 26 / iOS 26, GA Sep 2025), Firefox 141+ (Windows) / 145+ (Apple-Silicon mac). **Gaps:** Firefox on Linux/Android, Intel Macs, pre-26 Safari/iOS. A **hard `navigator.gpu` feature-detect with fallback is required** ([PRIMORDIS-ADR-006](../adr/PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md)). Do not assume coverage beyond what is listed here.

### Compositing details

- Web: own `<canvas>` stacked behind a transparent Flutter glass-pane; explicit pointer routing; DPR/resize sync. Avoid `HtmlElementView`.
- macOS: IOSurface-backed Metal texture via `FlutterTextureRegistry` + `Texture` widget; mind frame pacing and avoid CPU readback at 24k/60fps.

### Build and hosting

Target `flutter build web --wasm` (Skwasm renderer; Skwasm needs WasmGC: Chrome 119+ / FF 120+ / Safari 18.2+) with automatic CanvasKit/dart2js fallback ([PRIMORDIS-ADR-007](../adr/PRIMORDIS-ADR-007-web-build-and-cross-origin-isolation.md)). Multi-threaded Skwasm **and** any `SharedArrayBuffer` CPU path require `COOP: same-origin` + `COEP: require-corp` headers (cross-origin isolation) — plain static hosts without header control cannot do it. `--wasm` **forbids legacy interop** (`dart:html` / `dart:js_util`) anywhere in the dependency tree; use `dart:js_interop` + `package:web` only.

This relates to `DGROUP_WEB-ADR-020` (Flutter Web Rendering and Compilation), which chose CanvasKit + dart2js for the DGroup app and noted a future `--wasm` switch. Primordis needs `--wasm` **plus an own WebGPU canvas** — a deliberately different posture worth recording ([PRIMORDIS-ADR-002](../adr/PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md), [PRIMORDIS-ADR-007](../adr/PRIMORDIS-ADR-007-web-build-and-cross-origin-isolation.md)).

### Accessibility / reduced-motion

The entire screen is motion. Per the org accessibility standard we must respect OS reduced-motion (offer a pause/static state), keep all controls semantic and labeled with tooltips, and not gate functionality behind continuous animation. Accessibility is a top-level goal, not an afterthought.

### Determinism and parity

The original GPU binning is already nondeterministic and single-buffered (a known race), so "faithful" means visually/statistically equivalent, never bit-exact. A parity harness validates cluster formation and drift against `Primordis.py` across backends. Separately, atomics parity between Dawn (Tint) and browser/wgpu (Naga) WGSL translators must be validated — both have a history of `atomicAdd` bugs, so the binning pass must be confirmed to produce identical results on both.

### Where standards bend (and why that's fine)

The compute/FFI/shader/JS-interop layers cannot follow the standard feature/data/domain structure — they are device pipelines, raw buffers, and interop bindings. This is **expected**. The mitigation is the `SimBackend` boundary: everything non-standard sits behind it, and the UI/state layers above remain fully Riverpod/Freezed/Material 3 compliant and fully tested.

## Success Metrics

- **Web (WebGPU present):** full 24,000 particles, 32 types, sustained 60fps in a supported browser.
- **Web (no WebGPU):** automatic, crash-free fallback to the Dart-WASM CPU backend at ~3-4k particles with a visible reduced-mode indicator and correct feature-detect across the known gaps (Firefox Linux/Android, Intel Macs, pre-26 Safari/iOS).
- **macOS (GPU):** full 24,000 particles at 60fps, with headroom demonstrated well beyond 24k (target 100k+) on Apple Silicon.
- **macOS (CPU fallback):** graceful degradation via isolates+FFI shared buffer; ~10-14k solid (benchmark-confirmed before any promise; do not commit 24k@60fps on CPU).
- **Parity:** visual/statistical equivalence to `Primordis.py` (cluster formation, drift) across backends; atomic scatter-binning confirmed identical on Dawn(Tint)-Metal vs browser(Naga) WebGPU at 24k/32-types.
- **Standards/quality:** `flutter analyze` zero warnings; `package:lint` clean; unit/widget tests for all new Dart; reduced-motion and semantic-control accessibility verified.
- **One-kernel discipline:** under Approach (a), a single WGSL source runs on both web and native (no kernel drift).

## Open Questions

1. **FFI WebGPU maturity / bus-factor.** `minigpu` is experimental and effectively solo-maintained. Will the spike (the exact 3-pass atomic-binning kernel at 24k/32-types) hold up, or do we commit to the hand-written Metal plugin? (Top risk — spike first, keep Metal warm.)
2. **Atomics parity.** Do Dawn (Tint) and browser/wgpu (Naga) produce identical scatter-binning results at 24k/32-types, or will translator-specific `atomicAdd` behavior force per-backend handling?
3. **macOS compute->texture handoff.** Can we confirm zero CPU readback and acceptable frame pacing (`textureFrameAvailable` vs Metal completion) at 24k/60fps through the IOSurface/`Texture` path?
4. **`flutter_gpu` status at sign-off.** Has anything changed against the pinned Flutter version that would alter the no-first-party-compute premise? (Re-verify before sign-off.)
5. **Approach (b) drift.** If we adopt the MSL plugin, how do we prevent three-way kernel drift (GLSL original + WGSL + MSL)? (Proposed: one source-of-truth spec.)
6. **Native CPU ceiling.** Is ~10-14k @ 60fps real on target hardware? (Low-confidence/extrapolated; Dart AOT SIMD currently broken — benchmark before promising.)
7. **Hosting headers.** Does the chosen host allow `COOP`/`COEP` so we can enable threaded Skwasm and any `SharedArrayBuffer` CPU path, or must we constrain to single-thread?
8. **Reduced-motion default.** What is the right default static/paused state and re-entry UX for motion-sensitive users?

## Phased Delivery

Effort framing (from research): the Web WebGPU path is ~5-8 person-weeks — the **shader port is the easy part**; the DOM/Flutter compositing, pointer routing, DPR/resize, feature-detect, and CPU fallback are where the time goes. Adding the macOS GPU target on top is ~1-2 weeks incremental for Approach (a) (kernel already written; cost is the Dawn native-asset build, the IOSurface texture bridge + frame pacing, `std430`->Metal alignment re-validation, signing/notarization). Approach (b) adds a few days for the GLSL->MSL port + parity.

### Web

**Phase 0 — WebGPU `js_interop` spike.**
De-risk the foundation: minimal `dart:js_interop` + `package:web` bindings to `navigator.gpu` (adapter/device) on an owned `<canvas>`. Prove the interop path before building on it.
Tasks: [TASK-001](../tasks/PRIMORDIS-TASK-001-project-scaffold-and-build-config.md), [TASK-002](../tasks/PRIMORDIS-TASK-002-simbackend-interface-and-shared-sim-model.md), [TASK-004](../tasks/PRIMORDIS-TASK-004-web-webgpu-backend-js-interop.md) (spike scope).

**Phase 1 — WGSL kernel, standalone.**
Port the 3+1 passes (clear bins / atomic-`atomicAdd` scatter-bin / interaction / Euler integrate) to **one WGSL kernel** and validate it standalone at 24k/32-types, independent of Flutter compositing.
Tasks: [TASK-003](../tasks/PRIMORDIS-TASK-003-port-simulation-to-wgsl-compute-kernel.md).

**Phase 2 — Flutter web integration / compositing.**
Wire the WGSL kernel into the Web `SimBackend`; stack the owned WebGPU canvas behind the transparent Flutter glass-pane; route pointers; sync DPR/resize; wire sliders -> uniforms.
Tasks: [TASK-004](../tasks/PRIMORDIS-TASK-004-web-webgpu-backend-js-interop.md), [TASK-005](../tasks/PRIMORDIS-TASK-005-web-canvas-compositing-and-pointer-routing.md), [TASK-006](../tasks/PRIMORDIS-TASK-006-sliders-to-uniforms-and-ui-chrome.md).

**Phase 3 — CPU-WASM fallback + feature-detect + `--wasm`/COOP-COEP.**
Build the Dart->WASM CPU backend (Float32List SoA, sequential counting-sort binning, single `drawRawPoints`); add the hard `navigator.gpu` feature-detect and backend switch; establish the `--wasm`/Skwasm build with CanvasKit/dart2js fallback and COOP/COEP cross-origin isolation.
Tasks: [TASK-008](../tasks/PRIMORDIS-TASK-008-dart-wasm-cpu-fallback-backend.md), [TASK-007](../tasks/PRIMORDIS-TASK-007-webgpu-feature-detection-and-fallback-switch.md), [TASK-010](../tasks/PRIMORDIS-TASK-010-web-build-hosting-and-cross-origin-isolation.md).

**Phase 4 — Web polish.**
Parity harness vs `Primordis.py` (visual/statistical, not bit-exact); reduced-mode UX; accessibility (reduced-motion/pause, semantic controls, tooltips); widget/unit tests; `flutter analyze` zero-warnings.
Tasks: [TASK-009](../tasks/PRIMORDIS-TASK-009-parity-test-harness-vs-python-reference.md), [TASK-015](../tasks/PRIMORDIS-TASK-015-cross-platform-backend-selection-and-reduced-mode-ux.md), [TASK-018](../tasks/PRIMORDIS-TASK-018-test-coverage-and-accessibility.md).

### macOS

**Phase M1 — Dawn/wgpu FFI backend, same WGSL.**
Enable the macOS target; implement the desktop `SimBackend` running the **same WGSL kernel** via Dawn/wgpu-over-Metal through `dart:ffi` (spike `minigpu` first). Validate the std430->Metal alignment and the Dawn native-asset build.
Tasks: [TASK-011](../tasks/PRIMORDIS-TASK-011-macos-target-dawn-wgpu-ffi-backend.md), [TASK-017](../tasks/PRIMORDIS-TASK-017-atomics-parity-validation-dawn-vs-browser.md).

**Phase M2 — Metal texture present path.**
Render into an IOSurface-backed Metal texture; register with `FlutterTextureRegistry`; composite under the Flutter UI via the `Texture` widget; tune frame pacing; confirm no CPU readback at 24k/60fps.
Tasks: [TASK-012](../tasks/PRIMORDIS-TASK-012-macos-metal-texture-present-path.md).

**Phase M3 — Native CPU isolate fallback.**
Multi-core CPU `SimBackend`: real isolates over an FFI `calloc`'d shared buffer (share the integer address) + `package:native_synchronization`; ~10-14k. Wire into cross-platform backend selection.
Tasks: [TASK-014](../tasks/PRIMORDIS-TASK-014-native-cpu-isolate-fallback-backend.md), [TASK-015](../tasks/PRIMORDIS-TASK-015-cross-platform-backend-selection-and-reduced-mode-ux.md).

**Phase M4 — Packaging / signing.**
macOS packaging: signing/notarization, GPU-family gating for old Intel Macs, CI.
Tasks: [TASK-016](../tasks/PRIMORDIS-TASK-016-macos-packaging-signing-and-gpu-gating.md).

**De-risking fallback (parallel, kept warm):** the hand-written **Metal (MSL) compute plugin** — a near-1:1 GLSL->MSL port — stands ready as the escape hatch if the Dawn/wgpu FFI layer (Approach a) proves too immature. It is Apple-only and a second kernel to maintain, so it is the fallback, not the default.
Task: [TASK-013](../tasks/PRIMORDIS-TASK-013-macos-metal-msl-compute-plugin-fallback.md).

## References

### Research

- [PRIMORDIS-research-summary](../research/PRIMORDIS-research-summary.md) — synthesis of the feasibility analysis (web + macOS), verified claims, particle tiers, approaches, risks, effort.

### Architecture Decision Records

- [PRIMORDIS-ADR-001: Cross-platform architecture — `SimBackend`](../adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md)
- [PRIMORDIS-ADR-002: Web GPU compute — WebGPU via `js_interop`](../adr/PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md)
- [PRIMORDIS-ADR-003: Shared WGSL compute kernel](../adr/PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md)
- [PRIMORDIS-ADR-004: Native macOS GPU — Dawn/wgpu over Metal via FFI](../adr/PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md)
- [PRIMORDIS-ADR-005: Rendering and compositing](../adr/PRIMORDIS-ADR-005-rendering-and-compositing.md)
- [PRIMORDIS-ADR-006: CPU fallback tiers and feature detection](../adr/PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md)
- [PRIMORDIS-ADR-007: Web build and cross-origin isolation](../adr/PRIMORDIS-ADR-007-web-build-and-cross-origin-isolation.md)

### Tasks

- [PRIMORDIS-TASK-001: Project scaffold and build config](../tasks/PRIMORDIS-TASK-001-project-scaffold-and-build-config.md)
- [PRIMORDIS-TASK-002: `SimBackend` interface and shared sim model](../tasks/PRIMORDIS-TASK-002-simbackend-interface-and-shared-sim-model.md)
- [PRIMORDIS-TASK-003: Port simulation to WGSL compute kernel](../tasks/PRIMORDIS-TASK-003-port-simulation-to-wgsl-compute-kernel.md)
- [PRIMORDIS-TASK-004: Web WebGPU backend (`js_interop`)](../tasks/PRIMORDIS-TASK-004-web-webgpu-backend-js-interop.md)
- [PRIMORDIS-TASK-005: Web canvas compositing and pointer routing](../tasks/PRIMORDIS-TASK-005-web-canvas-compositing-and-pointer-routing.md)
- [PRIMORDIS-TASK-006: Sliders to uniforms and UI chrome](../tasks/PRIMORDIS-TASK-006-sliders-to-uniforms-and-ui-chrome.md)
- [PRIMORDIS-TASK-007: WebGPU feature detection and fallback switch](../tasks/PRIMORDIS-TASK-007-webgpu-feature-detection-and-fallback-switch.md)
- [PRIMORDIS-TASK-008: Dart-WASM CPU fallback backend](../tasks/PRIMORDIS-TASK-008-dart-wasm-cpu-fallback-backend.md)
- [PRIMORDIS-TASK-009: Parity test harness vs Python reference](../tasks/PRIMORDIS-TASK-009-parity-test-harness-vs-python-reference.md)
- [PRIMORDIS-TASK-010: Web build, hosting, and cross-origin isolation](../tasks/PRIMORDIS-TASK-010-web-build-hosting-and-cross-origin-isolation.md)
- [PRIMORDIS-TASK-011: macOS target — Dawn/wgpu FFI backend](../tasks/PRIMORDIS-TASK-011-macos-target-dawn-wgpu-ffi-backend.md)
- [PRIMORDIS-TASK-012: macOS Metal texture present path](../tasks/PRIMORDIS-TASK-012-macos-metal-texture-present-path.md)
- [PRIMORDIS-TASK-013: macOS Metal MSL compute plugin (fallback)](../tasks/PRIMORDIS-TASK-013-macos-metal-msl-compute-plugin-fallback.md)
- [PRIMORDIS-TASK-014: Native CPU isolate fallback backend](../tasks/PRIMORDIS-TASK-014-native-cpu-isolate-fallback-backend.md)
- [PRIMORDIS-TASK-015: Cross-platform backend selection and reduced-mode UX](../tasks/PRIMORDIS-TASK-015-cross-platform-backend-selection-and-reduced-mode-ux.md)
- [PRIMORDIS-TASK-016: macOS packaging, signing, and GPU gating](../tasks/PRIMORDIS-TASK-016-macos-packaging-signing-and-gpu-gating.md)
- [PRIMORDIS-TASK-017: Atomics parity validation — Dawn vs browser](../tasks/PRIMORDIS-TASK-017-atomics-parity-validation-dawn-vs-browser.md)
- [PRIMORDIS-TASK-018: Test coverage and accessibility](../tasks/PRIMORDIS-TASK-018-test-coverage-and-accessibility.md)

### Related external decisions

- `DGROUP_WEB-ADR-020` — Flutter Web Rendering and Compilation (chose CanvasKit + dart2js for the DGroup app; noted a future `--wasm` switch). Primordis adopts a different posture: `--wasm` + an own WebGPU canvas (see [PRIMORDIS-ADR-002](../adr/PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md), [PRIMORDIS-ADR-007](../adr/PRIMORDIS-ADR-007-web-build-and-cross-origin-isolation.md)).
