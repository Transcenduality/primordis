<!-- Filename convention: <SCOPE>-ADR-NNN-short-title.md -->

# PRIMORDIS-ADR-002: Web GPU compute via browser WebGPU (WGSL) through `dart:js_interop` on an owned canvas

**Status:** Proposed
**Date:** 2026-06-27
**Deciders:** Bruce Abernethy
**Review date:** 2026-09-27
**Supersedes:** N/A
**Superseded by:** N/A
**Compliance/Security:** Cross-origin isolation (COOP/COEP) and the `--wasm` legacy-interop ban are owned by [PRIMORDIS-ADR-007](./PRIMORDIS-ADR-007-web-build-and-cross-origin-isolation.md). This ADR introduces no PII, network, or auth surface; the WebGPU canvas runs entirely client-side. Accessibility/reduced-motion obligations apply to the composited UI (see [PRIMORDIS-ADR-005](./PRIMORDIS-ADR-005-rendering-and-compositing.md), [PRIMORDIS-ADR-006](./PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md)).

## Context

Primordis runs its *entire* physics step in GPU compute shaders: GLSL `#version 430` with `std430` SSBOs and `atomicAdd`, in three compute passes per frame — (1) clear bin counts; (2) scatter-bin 24,000 particles into a uniform spatial grid (11×7 = 77 bins, bin size = `MAX_RADIUS` = 96, `MAX_BIN_PARTICLES` = 512 cap) via `atomicAdd`; (3) interaction over the 3×3 toroidal neighbor bins (minimum-image distance, short-range repulsion vs. linear-falloff attraction) followed by Euler integration and wrap. The work is ~67M particle-pair tests per frame at 24k/32-types — trivial on a GPU, but only ~1–2.5 fps single-threaded on a CPU. Reproducing Primordis on the web therefore requires real GPU **compute** (storage buffers + atomics), not just fragment shading.

The governing project constraint (verified mid-2026, established in [PRIMORDIS-ADR-001](./PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md)) is that **Flutter exposes no first-party Dart GPU-compute API on any platform**:

- `dart:ui` `FragmentProgram` is **fragment-shader only** — no compute entry points, no SSBO/UBO, no atomics — and has historically been fragile on web.
- `flutter_gpu` is **render-only** (no `ComputePass` in the Dart API) and **does not run on web at all**. Impeller's Metal backend has internal compute, but it is engine-internal and unexposed; Impeller is opt-in on macOS mid-2026 and absent on web. On web, Flutter renders through CanvasKit + Skwasm (both Skia over WebGL).

Consequently the web compute path must live **outside** Flutter's rendering stack. The browser already ships a compute-capable API — **WebGPU** — that maps near-1:1 onto the existing GLSL-430 compute kernel. Reaching it from Dart requires JS interop. Because Primordis targets `flutter build web --wasm` (Skwasm), the toolchain **forbids legacy interop** (`dart:html` / `dart:js_util`) anywhere in the dependency tree, leaving `dart:js_interop` + `package:web` as the only sanctioned bridge ([PRIMORDIS-ADR-007](./PRIMORDIS-ADR-007-web-build-and-cross-origin-isolation.md)).

This is a deliberately different posture from `DGROUP_WEB-ADR-020` (Flutter Web Rendering and Compilation), which chose **CanvasKit + dart2js** for the DGroup app and noted a *future* `--wasm` switch. The DGroup app renders all of its content through Flutter/Skia. Primordis instead **hosts a GPU canvas it owns**: it needs `--wasm` (for the legacy-interop ban that makes clean WebGPU interop possible and for Skwasm) **plus** a separate WebGPU `<canvas>` that is *not* Flutter's CanvasKit/Skwasm surface. Recording that divergence is the purpose of this ADR.

WebGPU is not universally available mid-2026 (verified): Chrome/Edge 113+ (since 2023), Safari 26 (macOS Tahoe 26 / iOS 26, GA Sep 2025), Firefox 141+ (Windows) / 145+ (Apple-Silicon Mac). It is **absent** on Firefox Linux/Android, Intel Macs, and pre-26 Safari/iOS. A hard `navigator.gpu` feature-detect with a fallback path is therefore mandatory.

## Decision

On **web**, perform all GPU compute via **browser WebGPU using a WGSL compute kernel**, reached from Dart through **`dart:js_interop` + `package:web`**, running on a **`<canvas>` the app owns** as a sibling DOM element — explicitly *not* Flutter's CanvasKit/Skwasm rendering context.

Concretely:

- The Web `SimBackend` implementation (behind the interface from [PRIMORDIS-ADR-001](./PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md)) acquires `navigator.gpu` → adapter → device, creates storage buffers, compute pipelines, and a render pipeline, and dispatches the three compute passes plus a point-render pass per frame against the WGSL kernel.
- The WGSL kernel is the **same single source** reused on native via Dawn/wgpu-over-Metal ([PRIMORDIS-ADR-003](./PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md)). The GLSL-430 → WGSL port is near-1:1: SSBO → `var<storage, read_write>`; `atomicAdd(bins[i], 1)` → `atomicAdd(&bins[i], 1u)` (the returned old value is the scatter offset); `local_size_x` → `@workgroup_size`; `gl_GlobalInvocationID` → `@builtin(global_invocation_id)`; bin counters declared `atomic<u32>` consistently and read via `atomicLoad`.
- All JS interop uses `dart:js_interop` + `package:web` exclusively. No `dart:html` and no `dart:js_util` are introduced anywhere in the dependency tree, as required by `--wasm`.
- A **hard `navigator.gpu` feature-detect** gates this backend. When WebGPU is absent (Firefox Linux/Android, Intel Macs, pre-26 Safari/iOS) the app degrades to the Dart→WASM CPU fallback per [PRIMORDIS-ADR-006](./PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md).
- The owned WebGPU canvas is composited by stacking it behind a transparent Flutter glass-pane (not `HtmlElementView`); the present/compositing contract is owned by [PRIMORDIS-ADR-005](./PRIMORDIS-ADR-005-rendering-and-compositing.md).

We **reject** `dart:ui` `FragmentProgram` and `flutter_gpu` as the web compute mechanism, and we **defer** WebGL2 GPGPU as a non-default middle fallback (see Alternatives Considered). The interop/WebGPU/canvas code is non-standard relative to the DGROUP feature/data/domain structure; this is expected and explicitly sanctioned provided it stays quarantined behind `SimBackend`, leaving the UI layer fully Riverpod/Freezed/Material 3 compliant.

## Consequences

### Positive

- **Reproduces the real architecture.** WebGPU storage buffers + atomics let the original three-pass `atomicAdd` scatter-binning kernel run essentially as-is, delivering the **full 24k+ particles at 60fps** where WebGPU is present — not a downscaled imitation.
- **One kernel, two backends.** The WGSL written here is the same source consumed by the native Dawn/wgpu-over-Metal backend ([PRIMORDIS-ADR-003](./PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md)), avoiding a separate web shader and shrinking kernel drift to the web-vs-native translator-parity question.
- **Standards stay intact above the boundary.** Because all interop/WebGPU lives behind `SimBackend`, the Flutter UI/state layers remain fully standards-compliant; the unavoidable non-standard code is contained, not spread.
- **`--wasm`-clean interop.** Restricting to `dart:js_interop` + `package:web` is exactly what the `--wasm`/Skwasm build requires, so the WebGPU bridge does not block the target build mode ([PRIMORDIS-ADR-007](./PRIMORDIS-ADR-007-web-build-and-cross-origin-isolation.md)).

### Negative

- **No universal coverage.** WebGPU is absent on Firefox Linux/Android, Intel Macs, and pre-26 Safari/iOS, so a full CPU fallback tier (~3–4k particles) and a hard feature-detect are mandatory, not optional ([PRIMORDIS-ADR-006](./PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md)).
- **Owned-canvas compositing complexity.** A separate `<canvas>` outside Flutter's surface forces explicit DOM stacking, pointer-event routing, and DPR/resize syncing — the effort-heavy part of the web work, owned by [PRIMORDIS-ADR-005](./PRIMORDIS-ADR-005-rendering-and-compositing.md). (Effort framing: the web WebGPU path is ~5–8 person-weeks; the shader port is the easy part — compositing, pointer routing, DPR/resize, feature-detect, and the CPU fallback consume the time.)
- **Diverges from `DGROUP_WEB-ADR-020`.** Primordis adopts `--wasm` and an own WebGPU canvas where the DGroup app standardized on CanvasKit + dart2js with all-Flutter rendering — a divergence other DGroup engineers must be aware of when reading across repos.
- **Translator-parity risk surfaces here.** The browser path uses the Naga WGSL translator while native uses Tint; both have a history of `atomicAdd` bugs, so the scatter-binning pass must be validated identical across them ([PRIMORDIS-ADR-003](./PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md)).

### Neutral

- **Determinism is unchanged.** The original GPU binning is already nondeterministic (single-buffered, with a known scatter race), so "faithful" means visually/statistically equivalent across backends, never bit-exact. WebGPU does not change this contract.
- **The `flutter_gpu` premise must be re-checked at sign-off.** This decision assumes no first-party web compute API exists; re-verify `flutter_gpu` against the pinned Flutter version before final sign-off, but do not plan around it shipping ([PRIMORDIS-ADR-001](./PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md)).
- **iOS/Safari support follows the same code.** Because the path is browser WebGPU, Safari 26 / iOS 26 are covered by the identical backend with no extra work; older Safari/iOS simply route to the fallback.

## Alternatives Considered

### `dart:ui` `FragmentProgram` GPGPU (fragment-shader ping-pong)

Emulate compute in a fragment shader, encoding particle state in textures and ping-ponging render targets. **Rejected.** `FragmentProgram` is fragment-only with no SSBO/UBO and no atomics, so the `atomicAdd` scatter-binning that the spatial grid depends on cannot be expressed; binning would have to be re-architected into a multi-pass prefix-sum counting sort over textures — a rewrite, not a port — and `FragmentProgram` has historically been fragile on web. It would also couple the simulation to Flutter's painting layer, which the design explicitly avoids (Primordis hosts a GPU canvas, it does not render the sim through Skia).

### `flutter_gpu` compute

Use Flutter's lower-level GPU package for the compute passes. **Rejected.** `flutter_gpu` is render-only (no `ComputePass` in the Dart API) and **does not run on web at all**; Impeller's internal Metal compute is engine-internal and unexposed, and Impeller is not present on web (CanvasKit + Skwasm only). There is no web compute surface here to build on. This must not be architected around on the assumption that compute "will ship soon."

### WebGL2 GPGPU (middle fallback for non-WebGPU browsers)

Implement a WebGL2 GPGPU path so that browsers lacking WebGPU could still run a large simulation. **Deferred (not default).** WebGL2 has no compute shaders, no atomics, and no scatter; binning would become a prefix-sum counting sort over textures — a **rewrite, not a port** — costing more than the WebGPU path. It is skipped unless "full 24k on non-WebGPU browsers" becomes a hard requirement; until then, non-WebGPU browsers route to the Dart→WASM CPU fallback ([PRIMORDIS-ADR-006](./PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md)).

### Pure CPU simulation on web (no GPU path)

Skip web GPU entirely and run only the Dart→WASM CPU simulation. **Rejected as the primary path.** Single-threaded Dart→WASM tops out at ~3–4k particles @ 60fps (24k runs ~1–2.5 fps); the web has no real isolates (web "isolates" are workers that copy data) and `SharedArrayBuffer` needs COOP/COEP cross-origin isolation, which excludes plain static hosts. This cannot reproduce the intended 24k experience and is retained only as the capability-based fallback tier, not as the web default.

## References

### Primordis documents

- [PRIMORDIS-PRD-001: Flutter Web + macOS Port of Primordis](../prd/PRIMORDIS-PRD-001-flutter-web-and-macos-port.md)
- [PRIMORDIS research summary](../research/PRIMORDIS-research-summary.md)
- [PRIMORDIS-ADR-001: Cross-platform architecture — `SimBackend`](./PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md)
- [PRIMORDIS-ADR-003: Shared WGSL compute kernel](./PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md)
- [PRIMORDIS-ADR-004: Native macOS GPU — Dawn/wgpu over Metal via FFI](./PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md)
- [PRIMORDIS-ADR-005: Rendering and compositing](./PRIMORDIS-ADR-005-rendering-and-compositing.md)
- [PRIMORDIS-ADR-006: CPU fallback tiers and feature detection](./PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md)
- [PRIMORDIS-ADR-007: Web build and cross-origin isolation](./PRIMORDIS-ADR-007-web-build-and-cross-origin-isolation.md)

### Related tasks

- [PRIMORDIS-TASK-003: Port simulation to WGSL compute kernel](../tasks/PRIMORDIS-TASK-003-port-simulation-to-wgsl-compute-kernel.md)
- [PRIMORDIS-TASK-004: Web WebGPU backend (`js_interop`)](../tasks/PRIMORDIS-TASK-004-web-webgpu-backend-js-interop.md)
- [PRIMORDIS-TASK-005: Web canvas compositing and pointer routing](../tasks/PRIMORDIS-TASK-005-web-canvas-compositing-and-pointer-routing.md)
- [PRIMORDIS-TASK-006: Sliders to uniforms and UI chrome](../tasks/PRIMORDIS-TASK-006-sliders-to-uniforms-and-ui-chrome.md)
- [PRIMORDIS-TASK-007: WebGPU feature detection and fallback switch](../tasks/PRIMORDIS-TASK-007-webgpu-feature-detection-and-fallback-switch.md)
- [PRIMORDIS-TASK-008: Dart-WASM CPU fallback backend](../tasks/PRIMORDIS-TASK-008-dart-wasm-cpu-fallback-backend.md)
- [PRIMORDIS-TASK-010: Web build, hosting, and cross-origin isolation](../tasks/PRIMORDIS-TASK-010-web-build-hosting-and-cross-origin-isolation.md)
- [PRIMORDIS-TASK-017: Atomics parity validation — Dawn vs browser](../tasks/PRIMORDIS-TASK-017-atomics-parity-validation-dawn-vs-browser.md)

### Related external decisions

- `DGROUP_WEB-ADR-020` — Flutter Web Rendering and Compilation (chose CanvasKit + dart2js for the DGroup app; noted a future `--wasm` switch). Primordis adopts a different posture: `--wasm` + an own WebGPU canvas.

### External documentation topics

- WebGPU specification (W3C) — `GPUAdapter`, `GPUDevice`, `GPUBuffer` storage usage, `GPUComputePipeline`, compute pass encoding, and `navigator.gpu` adapter request.
- WGSL specification (W3C) — `var<storage, read_write>` address space, `atomic<u32>`, `atomicAdd` / `atomicLoad`, `@workgroup_size`, and `@builtin(global_invocation_id)`.
- flutter.dev — "Web renderers" (CanvasKit / Skwasm) and `flutter build web --wasm` / WasmGC requirements.
- flutter.dev / dart.dev — JavaScript interop with `dart:js_interop` and `package:web`; the `--wasm` legacy-interop (`dart:html` / `dart:js_util`) prohibition.
- dart.dev — `FragmentProgram` (`dart:ui`) capabilities and limitations (fragment-only; no compute/SSBO/atomics).
- flutter.dev — `flutter_gpu` package status (render-only; no `ComputePass`; no web support).
- Apple Developer — Metal compute / WebGPU on Apple platforms (Safari 26 / macOS Tahoe 26 / iOS 26 WebGPU availability), relevant to the browser support matrix.
