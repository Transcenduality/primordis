<!-- Filename convention: <SCOPE>-ADR-NNN-short-title.md -->

# PRIMORDIS-ADR-004: Native macOS GPU compute via Dawn/wgpu-over-Metal (FFI), with a hand-written Metal (MSL) plugin fallback

**Status:** Proposed
**Date:** 2026-06-27
**Deciders:** Bruce Abernethy
**Review date:** 2026-09-27
**Supersedes:** N/A
**Superseded by:** N/A
**Compliance/Security:** GPU compute and FFI/native-plugin code live outside the standard Flutter feature/data/domain layers and behind the `SimBackend` interface (see [PRIMORDIS-ADR-001](./PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md)); the UI layer stays standards-compliant. macOS distribution requires code signing and notarization (tracked in [PRIMORDIS-TASK-016](../tasks/PRIMORDIS-TASK-016-macos-packaging-signing-and-gpu-gating.md)).

## Context

Primordis runs its entire physics in GPU compute shaders: three compute passes per frame (clear bin counts; bin particles into a uniform spatial grid via `atomicAdd` scatter; interaction + Euler integrate over a 3x3 toroidal neighbor scan), then a point render pass. At 24,000 particles / 32 types this is roughly 67M particle-pair tests per frame — trivial on a GPU, but ~1-2.5 fps single-threaded on a CPU. The original is GLSL `#version 430`, `std430` SSBOs, `atomicAdd`.

The payoff of a native macOS build is that the Mac can run the **full** 24k+ GPU simulation (and well beyond — 100k-500k+ on Apple Silicon), whereas the web target tops out at the CPU fallback (~3-4k) unless the browser exposes WebGPU. To get there on macOS we must resolve two hard, verified platform facts:

1. **Flutter exposes no first-party Dart GPU-compute API on any platform, including macOS.** `dart:ui` `FragmentProgram` is fragment-shader only (no compute, no SSBO/UBO, no atomics). `flutter_gpu` is **render-only**: there is no `ComputePass` class in its Dart API, it does not run on web, and even on macOS it exposes no compute. Impeller's Metal backend has internal compute, but it is engine-internal and unexposed, and Impeller remains opt-in (behind a flag) on macOS as of mid-2026. Therefore all GPU compute must live **outside** Flutter.

2. **macOS OpenGL is frozen at 4.1 — it has no compute shaders.** Compute shaders require OpenGL 4.3. The original GLSL-430 compute kernel therefore **cannot** run via OpenGL on a Mac. Metal — used directly, or via Dawn/wgpu over Metal — is **mandatory and sufficient** to run GPU compute on macOS.

This ADR decides how native macOS performs GPU compute. It builds directly on the shared-kernel decision in [PRIMORDIS-ADR-003](./PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md) (one WGSL kernel reused on web and native) and pairs with the present/composite decision in [PRIMORDIS-ADR-005](./PRIMORDIS-ADR-005-rendering-and-compositing.md) (the macOS IOSurface-backed Metal texture path). It does not address the web compute path (see [PRIMORDIS-ADR-002](./PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md)) or CPU fallback tiers (see [PRIMORDIS-ADR-006](./PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md)).

## Decision

On macOS, the GPU compute backend behind `SimBackend` will be:

- **PRIMARY — Approach (a): Dawn/wgpu over Metal via `dart:ffi`.** Run the **same single WGSL kernel** authored in [PRIMORDIS-ADR-003](./PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md) through a Dawn/wgpu-over-Metal layer reached from Dart via `dart:ffi` (spiking the experimental `minigpu` package). This realizes "write the kernel once, two backends": browser WebGPU on web ([PRIMORDIS-ADR-002](./PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md)) and Dawn/wgpu on native, with iOS/Windows/Linux falling out of the same native backend for little extra effort. This is the recommended path because it eliminates a third kernel and keeps the source of truth singular.

- **FALLBACK — Approach (b): a hand-written Metal Shading Language (MSL) compute plugin.** A native macOS plugin (Swift / Obj-C++) containing a near-1:1 GLSL->MSL port of the three compute passes. This is the most robust, ship-ready path, but it is a **second kernel to maintain** and is Apple-only. It is kept as a de-risking escape hatch, warm and ready, against the maturity/bus-factor risk of the FFI WebGPU layer.

Both approaches present results identically: the simulation renders into an IOSurface-backed Metal texture composited under the shared Flutter UI via the external-texture `Texture` widget (decided in [PRIMORDIS-ADR-005](./PRIMORDIS-ADR-005-rendering-and-compositing.md)). The UI never knows which backend is live.

Because Approach (a) is experimental and solo-maintained, we will **spike Approach (a) first** with the exact 3-pass atomic-binning kernel at 24k / 32-types ([PRIMORDIS-TASK-011](../tasks/PRIMORDIS-TASK-011-macos-target-dawn-wgpu-ffi-backend.md)) and **keep the Metal plugin warm** as the fallback ([PRIMORDIS-TASK-013](../tasks/PRIMORDIS-TASK-013-macos-metal-msl-compute-plugin-fallback.md)). We will explicitly **validate atomics parity** between Dawn (Tint) and browser/wgpu (Naga) translators on the binning pass ([PRIMORDIS-TASK-017](../tasks/PRIMORDIS-TASK-017-atomics-parity-validation-dawn-vs-browser.md)).

### GLSL -> MSL mapping (for the Approach (b) fallback kernel)

The fallback port is near-1:1. The load-bearing mappings:

| GLSL `#version 430` (`std430`) | Metal Shading Language (MSL) |
| --- | --- |
| SSBO (`buffer ... { ... }`, `std430`) | `device` buffer pointer argument (e.g. `device float* positions [[buffer(0)]]`) |
| `atomicAdd(bins[i], 1)` (returns old value = scatter offset) | `atomic_fetch_add_explicit(&bins[i], 1, memory_order_relaxed)` on a `device atomic_uint` (returns old value) |
| `layout(local_size_x = N) in;` | threadgroup size (dispatch `threadsPerThreadgroup`) |
| `gl_GlobalInvocationID.x` | `[[thread_position_in_grid]]` |
| `gl_LocalInvocationID` / `gl_WorkGroupID` | `[[thread_position_in_threadgroup]]` / `[[threadgroup_position_in_grid]]` |

Atomic counters must be a consistent `device atomic_uint` type, read back via the atomic load API, mirroring the WGSL constraint that atomics be `atomic<u32>` and read via `atomicLoad` (see [PRIMORDIS-ADR-003](./PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md)).

## Consequences

### Positive

- macOS runs the **full** 24k+ simulation — and 100k-500k+ on Apple Silicon — which is the entire reason to build a native Mac target.
- Approach (a) reuses the single WGSL source from [PRIMORDIS-ADR-003](./PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md), so the macOS GPU target is roughly **1-2 person-weeks incremental** on top of the web WebGPU work (the kernel already exists; cost is the Dawn native-asset build, the IOSurface texture bridge + frame pacing, `std430`->Metal alignment re-validation, and signing/notarization).
- The same Dawn/wgpu-over-Metal backend gives iOS/Windows/Linux GPU compute "for free" later.
- The MSL fallback is the most robust, ship-ready Metal path and directly de-risks the experimental FFI layer; the GLSL->MSL port adds only a few days.
- Metal is the correct and only viable native compute path on macOS given the OpenGL 4.1 freeze — there is no architectural dead-end here.

### Negative

- Approach (a) depends on an **experimental, solo-maintained** FFI WebGPU layer (`minigpu`); maturity and bus-factor are top risks and require an early spike with the real kernel ([PRIMORDIS-TASK-011](../tasks/PRIMORDIS-TASK-011-macos-target-dawn-wgpu-ffi-backend.md)).
- **Atomics parity risk:** Dawn (Tint) and browser/wgpu (Naga) are different WGSL translators with a history of `atomicAdd` bugs; the binning pass must be proven to produce identical results on both ([PRIMORDIS-TASK-017](../tasks/PRIMORDIS-TASK-017-atomics-parity-validation-dawn-vs-browser.md)).
- Choosing Approach (b) introduces **three-way kernel drift** (GLSL original + WGSL + MSL); mitigated by a single source-of-truth spec per [PRIMORDIS-ADR-003](./PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md).
- The compute->display texture handoff is the fiddliest integration risk: frame pacing (`textureFrameAvailable` vs Metal completion) and confirming **no CPU readback** at 24k/60fps (owned by [PRIMORDIS-ADR-005](./PRIMORDIS-ADR-005-rendering-and-compositing.md) / [PRIMORDIS-TASK-012](../tasks/PRIMORDIS-TASK-012-macos-metal-texture-present-path.md)).
- Approach (b) is Apple-only and a second kernel to maintain.

### Neutral

- The native CPU fallback tier (real isolates over an FFI `calloc`'d shared buffer + `native_synchronization`) is unaffected by this decision and is handled in [PRIMORDIS-ADR-006](./PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md); it is graceful degradation only (~10-14k solid, not a 24k path).
- Determinism is not a goal: the original GPU binning is already nondeterministic with a single-buffered race, so "faithful" means visually/statistically equivalent, never bit-exact.
- Because compute lives behind `SimBackend`, neither approach changes the Flutter UI, Riverpod providers, Freezed param models, or the shared frame loop.
- `flutter_gpu` must be **re-verified against the pinned Flutter version before sign-off**, since this decision rests on it exposing no compute API.

## Alternatives Considered

### flutter_gpu (rejected)
`flutter_gpu` is render-only: no `ComputePass` in its Dart API, no web support, and no exposed compute even on macOS where Impeller's Metal backend has internal-but-unexposed compute. It cannot run the three compute passes. Rejected; do **not** architect around "flutter_gpu compute will ship soon."

### MoltenVK / Vulkan (rejected)
Running the compute via Vulkan translated to Metal through MoltenVK would add a heavyweight translation dependency and a second graphics API surface, without the cross-platform kernel reuse that Dawn/wgpu-over-Metal already provides from the shared WGSL source. It does not improve on Approach (a) and is strictly more integration than the hand-written MSL fallback. Rejected.

### OpenGL (impossible)
macOS OpenGL is frozen at **4.1**, which has no compute shaders (compute requires 4.3). The original GLSL-430 compute kernel cannot run via OpenGL on a Mac at all. Not a viable option — this is the constraint that makes Metal mandatory.

## References

- [PRIMORDIS-ADR-001 — Cross-platform architecture and the SimBackend interface](./PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md)
- [PRIMORDIS-ADR-002 — Web GPU compute via browser WebGPU + JS interop](./PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md)
- [PRIMORDIS-ADR-003 — Shared WGSL compute kernel](./PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md)
- [PRIMORDIS-ADR-005 — Rendering and compositing](./PRIMORDIS-ADR-005-rendering-and-compositing.md)
- [PRIMORDIS-ADR-006 — CPU fallback tiers and feature detection](./PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md)
- [PRIMORDIS-PRD-001 — Flutter web and macOS port](../prd/PRIMORDIS-PRD-001-flutter-web-and-macos-port.md)
- [Research summary](../research/PRIMORDIS-research-summary.md)
- [PRIMORDIS-TASK-011 — macOS target: Dawn/wgpu FFI backend](../tasks/PRIMORDIS-TASK-011-macos-target-dawn-wgpu-ffi-backend.md)
- [PRIMORDIS-TASK-012 — macOS Metal texture present path](../tasks/PRIMORDIS-TASK-012-macos-metal-texture-present-path.md)
- [PRIMORDIS-TASK-013 — macOS Metal/MSL compute plugin fallback](../tasks/PRIMORDIS-TASK-013-macos-metal-msl-compute-plugin-fallback.md)
- [PRIMORDIS-TASK-016 — macOS packaging, signing, and GPU gating](../tasks/PRIMORDIS-TASK-016-macos-packaging-signing-and-gpu-gating.md)
- [PRIMORDIS-TASK-017 — Atomics parity validation: Dawn vs browser](../tasks/PRIMORDIS-TASK-017-atomics-parity-validation-dawn-vs-browser.md)
- Flutter — `flutter_gpu` (render-only; no `ComputePass`): https://github.com/flutter/engine/tree/main/lib/gpu
- Flutter — `dart:ui` `FragmentProgram` (fragment-only shaders): https://api.flutter.dev/flutter/dart-ui/FragmentProgram-class.html
- Flutter — Writing and using fragment shaders: https://docs.flutter.dev/ui/design/graphics/fragment-shaders
- Flutter — Impeller rendering engine: https://docs.flutter.dev/perf/impeller
- WebGPU specification (compute pipelines): https://www.w3.org/TR/webgpu/
- WGSL specification (atomics: `atomic<u32>`, `atomicAdd`, `atomicLoad`): https://www.w3.org/TR/WGSL/
- Dawn (WebGPU/Tint implementation): https://dawn.googlesource.com/dawn
- wgpu / Naga (WGSL shader translation): https://github.com/gfx-rs/wgpu
- Apple — Metal Shading Language Specification: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
- Apple — Performing Calculations on a GPU (Metal compute): https://developer.apple.com/documentation/metal/performing-calculations-on-a-gpu
- Apple — Atomic functions in Metal (`atomic_fetch_add_explicit`, `atomic_uint`): https://developer.apple.com/documentation/metal
