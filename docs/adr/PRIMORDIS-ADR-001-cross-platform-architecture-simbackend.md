<!-- Filename convention: <SCOPE>-ADR-NNN-short-title.md -->

# PRIMORDIS-ADR-001: Cross-platform architecture — one Flutter app with a Dart SimBackend interface

**Status:** Proposed
**Date:** 2026-06-27
**Deciders:** Bruce Abernethy
**Review date:** 2026-09-27
**Supersedes:** N/A
**Superseded by:** N/A
**Compliance/Security:** None (foundational structural decision; accessibility/reduced-motion obligations are tracked in [PRIMORDIS-ADR-006](./PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md))

## Context

Primordis is a GPU particle-life ("clusters") simulator currently implemented as a single ~350-line Python file (`Primordis.py`) using pygame + moderngl (OpenGL 4.3) + numpy. The simulation runs 24,000 particles across 32 types in a 1080x720 toroidal world. Per-type-pair behaviour is encoded in three asymmetric 32x32 float32 matrices (forces, min_distances, radii; i->j != j->i) plus random per-type colors. The entire physics runs in GPU compute shaders (GLSL `#version 430`, std430 SSBOs, `atomicAdd`): three compute passes per frame — clear bin counts; bin particles into a uniform spatial grid (11x7 = 77 bins, bin size = MAX_RADIUS = 96, MAX_BIN_PARTICLES = 512) via `atomicAdd` scatter; and an interaction pass that scans the 3x3 toroidal neighbor bins, computes minimum-image distance, applies short-range repulsion vs. linear-falloff attraction, and Euler-integrates — followed by a point render (`gl_PointSize = 2`). The workload is ~67M particle-pair tests/frame: trivial on a GPU, but ~1–2.5 fps single-threaded on a CPU.

The goal is to port Primordis to **Flutter Web (WASM)** and **native Flutter macOS** from a Flutter-first shop (org "DGROUP"; standards mandate Riverpod + Freezed + Retrofit/Dio + GoRouter + `package:lint` + tests + accessibility). Primordis is its **own standalone repo**, not the DGroup monorepo; the Flutter app lives at the repo root (or `apps/web`), and the version lives in `pubspec.yaml` plus an `AppConfig`/`PrimordisConfig` constant.

The single hardest constraint — verified for mid-2026 — shapes the entire architecture: **Flutter has no first-party Dart GPU-compute API on any platform.**

- `dart:ui` `FragmentProgram` is fragment-shader only (no compute, no SSBO/UBO, no atomics) and historically fragile on web.
- `flutter_gpu` is render-only (no `ComputePass` in the Dart API) and does not run on web; even on macOS it exposes no compute. Impeller's Metal backend has internal compute, but it is engine-internal and unexposed. Impeller is still opt-in (behind a flag) on macOS mid-2026 and is not on web at all (web = CanvasKit + Skwasm, both Skia/WebGL).

Consequently, **all GPU compute must live outside Flutter**, behind a Dart seam: browser WebGPU via JS interop on web; Dawn/wgpu-over-Metal via `dart:ffi` (or a native Metal plugin) on desktop. The architecture must not be designed around any assumption that "`flutter_gpu` compute will ship soon."

The platform-specific layers diverge sharply. On web the GPU path is browser WebGPU (WGSL compute) reached from Dart via `dart:js_interop` + `package:web`, running on a `<canvas>` the app owns (not Flutter's CanvasKit/Skwasm context). On native macOS, OpenGL is frozen at 4.1 (no compute shaders — the original requires 4.3), so Metal — directly or via Dawn/wgpu-over-Metal — is mandatory and sufficient; macOS can run the full 24k+ sim (and 100k–500k+ on Apple Silicon), whereas web tops out at a CPU fallback unless WebGPU is present. Each platform also has a distinct compositing path (web stacked-canvas vs. macOS external `Texture` widget) and distinct degradation tiers. Despite this divergence, the Flutter UI, the simulation parameters, particle seeding, the frame loop, param marshalling, and — under the recommended native approach — the WGSL kernel source itself are all identical across platforms.

This ADR records the foundational decision about how the codebase is organized so that this shared surface stays shared and the platform-specific compute/present layers stay swappable and isolated. It precedes and frames all other Primordis ADRs.

## Decision

Build **one Flutter application** that defines a single Dart **`SimBackend`** interface, with **swappable per-platform compute implementations** selected at runtime behind that interface. The UI never knows which backend is live.

**Target platforms:** Flutter Web (WASM) and native Flutter macOS. iOS, Windows, and Linux are not in initial scope but fall out of the same backend design for little extra cost (the native GPU path via Dawn/wgpu over the platform graphics API and the WGSL kernel are reusable).

**Shared across all platforms (one implementation, standards-compliant):**

- The entire Flutter UI: the 3 sliders (Attraction K, Repulsion K, Drift/friction), chrome, and reset/seed controls.
- The simulation parameters as Freezed models: the three asymmetric 32x32 float32 matrices (forces, min_distances, radii) and per-type colors.
- Particle seeding.
- The frame loop and parameter marshalling.
- Under the recommended native approach, the **WGSL kernel source itself**.

**Platform-specific, hidden behind `SimBackend`:**

- GPU device/adapter/pipeline creation.
- Compute dispatch — browser WebGPU vs. Dawn/wgpu-over-Metal (FFI) vs. hand-written Metal (MSL).
- The present/composite path — web stacked-canvas vs. macOS external `Texture` widget.

**Layering and standards.** The Flutter UI layer uses the DGroup-mandated stack: Riverpod for state (plain `Ref`, no `setState` for business logic), Freezed for the sim-parameter and color models, GoRouter for any routing, Material 3 + GoogleFonts, `package:lint` at zero warnings, and tests for new Dart. The compute/FFI/shader/JS-interop code necessarily lives **outside** the standard feature/data/domain layers — this is expected and explicitly sanctioned by this ADR — and is confined behind the `SimBackend` interface so that the UI layer above the seam remains fully standards-compliant and testable against a fake backend. Backend selection (capability detection and graceful degradation) is itself a Riverpod-provided concern; the concrete backends are wired in per platform.

The concrete backends, kernel, compositing, fallback tiers, and build posture are specified in the related ADRs listed under [References](#references); this ADR fixes only the top-level shape.

## Consequences

### Positive

- **One UI, one set of sim params, one frame loop** serve web and macOS, eliminating duplicated business logic and keeping the standards-compliant surface (Riverpod/Freezed/Material 3/tests) in a single place.
- **The hard "no first-party compute" constraint is contained.** All GPU compute and platform graphics code sits behind one seam; the UI is insulated from WebGPU/FFI/Metal details and stays portable, lint-clean, and unit-testable against a fake `SimBackend`.
- **"Write the kernel once, two backends"** becomes structurally natural: the same WGSL kernel runs on browser WebGPU (web) and Dawn/wgpu-over-Metal (native), so iOS/Windows/Linux fall out of the same backend for little extra effort (see [PRIMORDIS-ADR-003](./PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md), [PRIMORDIS-ADR-004](./PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md)).
- **Graceful degradation is a first-class shape, not a bolt-on.** Because backends are swappable, capability detection can select GPU 24k+ tiers or CPU fallback tiers per platform without touching the UI (see [PRIMORDIS-ADR-006](./PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md)).
- **macOS unlocks the real payoff:** the full 24k+ (and 100k–500k+ on Apple Silicon) GPU sim runs natively, where web caps at a CPU fallback unless browser WebGPU is present.

### Negative

- **GPU compute necessarily lives outside Flutter.** There is no in-engine Dart path; the project must own WebGPU JS-interop bindings on web and an FFI/native Metal bridge on desktop — code outside the standard feature/data/domain layers, with its own maturity and bus-factor exposure (the experimental, solo-maintained FFI WebGPU layer is a known top risk; the hand-written Metal plugin is kept as a warm fallback).
- **Maintenance model spans heterogeneous runtimes.** The team maintains Dart UI, WGSL, JS-interop/`package:web` bindings, FFI/native-asset builds, and a macOS texture bridge — a broader surface than a typical Flutter app, with kernel-translator parity (Tint vs. Naga) and a compute->display handoff to validate.
- **The `SimBackend` interface is a load-bearing contract.** It must abstract device creation, dispatch, parameter upload, and present across genuinely divergent models (owned `<canvas>` overlay on web vs. external `Texture`/IOSurface on macOS) without leaking platform specifics; getting this seam wrong forces rework across every backend.
- **Faithfulness is statistical, not bit-exact.** The original GPU binning is already nondeterministic (single-buffered race), so "faithful" means visually/statistically equivalent across backends, never bit-exact — parity must be asserted by a harness rather than assumed.

### Neutral

- iOS, Windows, and Linux are reachable from the same backend but remain out of initial scope; the architecture neither commits to nor precludes them.
- The exact native GPU path (Dawn/wgpu-over-Metal FFI as primary vs. hand-written MSL plugin as fallback) and the exact web/native compositing mechanics are deferred to downstream ADRs; this ADR only guarantees they sit behind `SimBackend`.
- The standalone-repo posture (Flutter app at repo root or `apps/web`; version in `pubspec.yaml` + a config constant) differs from the DGroup monorepo conventions and is adopted deliberately.

## Alternatives Considered

### Single-platform, web-only

Ship only the Flutter Web build and never target native macOS. Rejected: web cannot run the full 24k sim unless browser WebGPU is present (Firefox Linux/Android, Intel Macs, and pre-26 Safari/iOS have no `navigator.gpu`), and the pure Dart->WASM CPU fallback caps at ~3–4k particles @ 60 fps (24k runs at ~1–2.5 fps). Native macOS is precisely where the full 24k+ (and 100k–500k+) GPU sim is guaranteed, so a web-only scope forfeits the payoff of the port for a large share of users. A web-only scope also leaves no shared-backend foundation from which iOS/Windows/Linux could later fall out.

### One bespoke renderer per platform, with no shared interface

Implement web and macOS as independent codebases, each with its own UI, sim model, frame loop, and compute path, and no common abstraction. Rejected: this duplicates the entire standards-compliant surface (Riverpod state, Freezed sim-parameter and color models, sliders, seeding, frame loop, param marshalling) and the kernel logic across platforms, multiplying maintenance and inviting drift between the GLSL-origin physics and each port. It also discards the "write the kernel once, two backends" benefit and makes cross-backend parity testing far harder. A single `SimBackend` interface delivers the same per-platform freedom in the compute/present layer while preserving one shared UI and one shared parameter/seed/frame-loop core.

### Wait for first-party `flutter_gpu` compute

Defer the architecture and adopt a first-party Dart compute API once Flutter ships one. Rejected: as verified for mid-2026, no such API exists — `flutter_gpu` is render-only (no `ComputePass` in the Dart API) and does not run on web; `dart:ui` `FragmentProgram` is fragment-only with no SSBO/UBO/atomics; and Impeller's internal Metal compute is engine-internal and unexposed (Impeller itself is still opt-in on macOS and absent on web). Designing around an unshipped API would block the project indefinitely. The decision is therefore to keep all GPU compute outside Flutter behind `SimBackend`, and to re-verify `flutter_gpu` against the pinned Flutter version before sign-off rather than depend on it.

## References

**Related PRIMORDIS documents**

- PRD: [PRIMORDIS-PRD-001 — Flutter Web and macOS port](../prd/PRIMORDIS-PRD-001-flutter-web-and-macos-port.md)
- Research summary: [PRIMORDIS-research-summary](../research/PRIMORDIS-research-summary.md)
- [PRIMORDIS-ADR-002 — Web GPU compute via browser WebGPU + JS interop](./PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md)
- [PRIMORDIS-ADR-003 — Shared WGSL compute kernel](./PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md)
- [PRIMORDIS-ADR-004 — Native macOS GPU compute via Dawn/wgpu-over-Metal (FFI)](./PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md)
- [PRIMORDIS-ADR-005 — Rendering and compositing](./PRIMORDIS-ADR-005-rendering-and-compositing.md)
- [PRIMORDIS-ADR-006 — CPU fallback tiers and feature detection](./PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md)
- [PRIMORDIS-ADR-007 — Web build and cross-origin isolation](./PRIMORDIS-ADR-007-web-build-and-cross-origin-isolation.md)

**Related tasks**

- [PRIMORDIS-TASK-001 — Project scaffold and build config](../tasks/PRIMORDIS-TASK-001-project-scaffold-and-build-config.md)
- [PRIMORDIS-TASK-002 — SimBackend interface and shared sim model](../tasks/PRIMORDIS-TASK-002-simbackend-interface-and-shared-sim-model.md)
- [PRIMORDIS-TASK-015 — Cross-platform backend selection and reduced-mode UX](../tasks/PRIMORDIS-TASK-015-cross-platform-backend-selection-and-reduced-mode-ux.md)

**External documentation**

- Flutter — `flutter_gpu` (render-only; no Dart `ComputePass`): https://docs.flutter.dev/
- Flutter — `dart:ui` `FragmentProgram` (fragment-shader only): https://api.flutter.dev/flutter/dart-ui/FragmentProgram-class.html
- Flutter — web renderers (CanvasKit / Skwasm) and `flutter build web --wasm`: https://docs.flutter.dev/platform-integration/web/renderers
- Flutter — `dart:js_interop` and `package:web` interop guidance: https://dart.dev/interop/js-interop
- Flutter — `dart:ffi` C interop: https://dart.dev/interop/c-interop
- WebGPU specification (`navigator.gpu`, WGSL compute, storage buffers, atomics): https://www.w3.org/TR/webgpu/
- WGSL specification (`@workgroup_size`, `@builtin(global_invocation_id)`, `var<storage, read_write>`, `atomic<u32>`): https://www.w3.org/TR/WGSL/
- Apple — Metal compute (compute pipelines, threadgroups, device atomics): https://developer.apple.com/documentation/metal
- Apple — `Texture`/external textures via IOSurface / `CVPixelBuffer` on the macOS embedder: https://developer.apple.com/documentation/iosurface
