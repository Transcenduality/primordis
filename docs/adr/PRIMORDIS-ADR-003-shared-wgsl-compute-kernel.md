<!-- Filename convention: <SCOPE>-ADR-NNN-short-title.md -->

# PRIMORDIS-ADR-003: Single shared WGSL compute kernel for web and native

**Status:** Proposed
**Date:** 2026-06-27
**Deciders:** Bruce Abernethy
**Review date:** 2026-09-27
**Supersedes:** N/A
**Superseded by:** N/A
**Compliance/Security:** None

## Context

Primordis is a GPU particle-life simulator whose entire physics runs in GPU compute shaders. The reference implementation (`Primordis.py`, ~350 lines, pygame + moderngl + numpy) expresses the simulation as GLSL `#version 430` compute shaders against `std430` SSBOs using `atomicAdd`, run as three compute passes per frame:

1. **Clear** bin counts.
2. **Bin** 24,000 particles into a uniform 11x7=77-cell toroidal spatial grid (bin size = `MAX_RADIUS` = 96, `MAX_BIN_PARTICLES` = 512 cap) via an `atomicAdd` scatter that returns each particle's write offset.
3. **Interaction + integrate**: each particle scans its 3x3 neighbour bins (toroidal, minimum-image distance), applies short-range repulsion (`dist < min_dist`, 5x weighted, `abs(force)`) vs. linear-falloff signed attraction (`dist < radius`), then Euler-integrates (`v += f*dt; v *= friction; p += v*dt`) and wraps. This is ~67M particle-pair tests per frame at 24k particles / 32 types — trivial on a GPU, ~1-2.5 fps single-thread on a CPU.

The port targets Flutter Web (WASM) and native Flutter macOS. Two facts from the research summary (`../research/PRIMORDIS-research-summary.md`) constrain where this physics can run:

- **No first-party Dart GPU-compute API exists on any platform as of mid-2026.** `dart:ui` `FragmentProgram` is fragment-only (no compute / SSBO / atomics); `flutter_gpu` is render-only and does not run on web. All GPU compute must live *outside* Flutter — browser WebGPU on web, Dawn/wgpu-over-Metal (or a native Metal plugin) on macOS. See [`../adr/PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md`](./PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md) and [`../adr/PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md`](./PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md).
- **macOS OpenGL is frozen at 4.1** — no compute shaders (which need 4.3). The original GLSL-430 compute therefore cannot run via OpenGL on a Mac at all; Metal is mandatory on desktop.

This leaves a choice about *how many* compute kernels the project authors and maintains. The selected web backend (ADR-002) consumes WGSL via browser WebGPU. The selected native backend (ADR-004) has two paths: Approach (a) Dawn/wgpu-over-Metal, which also consumes WGSL, and Approach (b) a hand-written Metal Shading Language (MSL) plugin. A naive port could end up maintaining three distinct kernels — the original GLSL-430, a WGSL kernel for web, and an MSL kernel for macOS — with the attendant drift risk: a fix or tuning change to one must be hand-replicated, byte-for-byte in behaviour, across the other two.

The GLSL-430 -> WGSL translation is near 1:1, which makes a single shared source genuinely feasible rather than aspirational:

- `layout(std430) buffer` SSBO -> `var<storage, read_write>` storage buffer binding.
- `atomicAdd(bins[i], 1)` (returns old value, used as the scatter write offset) -> `atomicAdd(&bins[i], 1u)`, which likewise returns the previous value. WGSL requires the target be `atomic<u32>` and that it is read consistently via `atomicLoad`; atomic and non-atomic access to the same memory cannot be mixed.
- `layout(local_size_x = N)` -> `@workgroup_size(N)`.
- `gl_GlobalInvocationID` -> `@builtin(global_invocation_id)`; `gl_LocalInvocationID` -> `@builtin(local_invocation_id)`.

Because both the web WebGPU runtime and Dawn/wgpu consume WGSL, one kernel can drive two backends. The KEY ARCHITECTURE DECISION (ADR-001) already places the WGSL kernel *source* in the shared layer, alongside the Flutter UI, the 32x32 force/radius/min_distance + colour params, particle seeding, and the frame loop; only device/pipeline creation, dispatch, and the present path are platform-specific behind the `SimBackend` interface.

## Decision

**Author the Primordis simulation once as a single WGSL compute kernel and reuse that exact source on both web (browser WebGPU) and native (Dawn/wgpu-over-Metal via `dart:ffi`).** This collapses the would-be three-kernel set (GLSL original + WGSL + MSL) to one source of truth and eliminates cross-kernel drift on the primary paths.

Specifics:

- **One kernel source.** The 3+1 passes (clear bin counts / atomic-scatter binning / interaction / Euler integrate) are ported once from GLSL-430 to WGSL and committed as the canonical shader, living in the shared layer per ADR-001. The port is tracked by [`../tasks/PRIMORDIS-TASK-003-port-simulation-to-wgsl-compute-kernel.md`](../tasks/PRIMORDIS-TASK-003-port-simulation-to-wgsl-compute-kernel.md), which validates it standalone at 24k particles / 32 types before any backend integration.
- **GLSL -> WGSL mapping (canonical).** SSBO -> `var<storage, read_write>`; `atomicAdd(bins[i], 1)` (old-value = scatter offset) -> `atomicAdd(&bins[i], 1u)`; bin counters and any other atomically-touched storage declared `atomic<u32>` and read only via `atomicLoad`; `local_size_x` -> `@workgroup_size`; `gl_GlobalInvocationID` -> `@builtin(global_invocation_id)`. Uniforms (the three live sliders — Attraction K, Repulsion K, Drift/friction — plus dt, world dims, grid dims, counts) are marshalled identically into a uniform buffer regardless of backend.
- **Two backends consume the same source.** The web backend ([`../tasks/PRIMORDIS-TASK-004-web-webgpu-backend-js-interop.md`](../tasks/PRIMORDIS-TASK-004-web-webgpu-backend-js-interop.md)) hands the WGSL string to browser WebGPU via `dart:js_interop` + `package:web`; the macOS backend ([`../tasks/PRIMORDIS-TASK-011-macos-target-dawn-wgpu-ffi-backend.md`](../tasks/PRIMORDIS-TASK-011-macos-target-dawn-wgpu-ffi-backend.md)) hands the identical string to Dawn/wgpu-over-Metal through `dart:ffi` (spiking the experimental `minigpu` package). Buffer layouts, bind-group indices, and dispatch geometry are shared; only device/pipeline creation differs.
- **Single source of truth even if the MSL fallback ships.** The hand-written MSL compute plugin remains available as the native de-risking fallback (Approach (b), ADR-004, [`../tasks/PRIMORDIS-TASK-013-macos-metal-msl-compute-plugin-fallback.md`](../tasks/PRIMORDIS-TASK-013-macos-metal-msl-compute-plugin-fallback.md)), but is explicitly designated a *derived* artefact, not a peer. The WGSL kernel is the normative spec; the MSL kernel must be a faithful 1:1 transliteration of it (SSBO -> device buffer, `atomicAdd` -> `atomic_fetch_add_explicit` on a `device atomic_uint`, `local_size_x` -> threadgroup), validated against the WGSL behaviour. This keeps "one source of truth" intact even in the three-kernel fallback case.

This decision is conditional on, and depends on, the native FFI WebGPU layer being mature enough to run the kernel, and on atomics-translation parity between Dawn (Tint) and the browser/wgpu runtime (Naga). Both risks are owned by ADR-004 and validated by [`../tasks/PRIMORDIS-TASK-017-atomics-parity-validation-dawn-vs-browser.md`](../tasks/PRIMORDIS-TASK-017-atomics-parity-validation-dawn-vs-browser.md).

## Consequences

### Positive

- **No kernel drift on the primary paths.** A physics fix or tuning change is made once in WGSL and is live on web and macOS simultaneously. There is exactly one place where the binning and interaction logic exists.
- **The shader port is done once.** Per the effort estimate, the GLSL -> WGSL shader port is the *easy* part of the web work; reusing it on macOS makes the incremental cost of the native GPU target ~1-2 weeks (Dawn native-asset build, IOSurface texture bridge + frame pacing, `std430` -> Metal alignment re-validation, signing/notarization) rather than a second from-scratch physics implementation.
- **"Write the kernel once, two backends" extends for free.** Because Dawn/wgpu is a portable WebGPU implementation, iOS / Windows / Linux GPU targets fall out of the same WGSL source for little extra effort, consistent with the ADR-001 platform strategy.
- **A single normative spec governs parity testing.** The parity harness ([`../tasks/PRIMORDIS-TASK-009-parity-test-harness-vs-python-reference.md`](../tasks/PRIMORDIS-TASK-009-parity-test-harness-vs-python-reference.md)) and the atomics-parity check (TASK-017) compare backends against one canonical kernel rather than reconciling three independent implementations.

### Negative

- **Hard dependency on FFI WebGPU maturity.** The native path's use of the shared kernel rests on the experimental, solo-maintained `minigpu` (bus-factor / maturity risk). If it cannot run the exact 3-pass atomic-binning kernel at 24k / 32 types, the shared-kernel benefit on native is lost and the project falls back to the MSL plugin (a second, hand-maintained kernel). Mitigation: spike `minigpu` first with the real kernel and keep the Metal plugin warm. Owned by ADR-004 and [`../tasks/PRIMORDIS-TASK-011-macos-target-dawn-wgpu-ffi-backend.md`](../tasks/PRIMORDIS-TASK-011-macos-target-dawn-wgpu-ffi-backend.md).
- **Tint-vs-Naga atomics parity is not guaranteed.** Dawn uses the Tint WGSL translator; the browser / wgpu path uses Naga. The two have a documented history of `atomicAdd` translation bugs. The *same* WGSL source can therefore produce *different* scatter-binning results on the two backends, undermining the "one kernel, identical behaviour" premise. Mitigation: explicit equivalence validation of the binning pass on both translators at 24k / 32 types — [`../tasks/PRIMORDIS-TASK-017-atomics-parity-validation-dawn-vs-browser.md`](../tasks/PRIMORDIS-TASK-017-atomics-parity-validation-dawn-vs-browser.md).
- **WGSL constrains the source to its atomics rules.** Maintaining one kernel across translators means coding to the stricter contract: atomic targets must be `atomic<u32>` throughout, accessed only via `atomicLoad`/`atomicAdd`, with no aliased non-atomic access. This is a small, permanent authoring tax versus tuning a single-target shader freely.
- **If the MSL fallback ships, a derived third kernel still exists.** The decision designates MSL as derived-from-WGSL rather than a peer, but a transliteration is still hand-written code that must be re-validated whenever the WGSL kernel changes. Drift is reduced, not eliminated, in the fallback case.

### Neutral

- **Bit-exact determinism is explicitly out of scope.** The original GPU binning is already nondeterministic (single-buffered atomic-scatter race). "Faithful" therefore means visually / statistically equivalent cluster formation and drift, never bit-for-bit identical output — this holds across the shared kernel's two backends as well as against the Python reference.
- **Render is not part of the shared compute kernel.** This ADR covers the compute physics only; point rendering (`gl_PointSize=2` equivalent) and the present/composite path are platform-specific and owned by [`../adr/PRIMORDIS-ADR-005-rendering-and-compositing.md`](./PRIMORDIS-ADR-005-rendering-and-compositing.md).
- **House standards apply to the Dart layer, not the shader.** The WGSL kernel, FFI, and JS-interop glue necessarily live outside the standard Riverpod / Freezed feature/data/domain layers; this is expected and is kept behind the `SimBackend` interface (ADR-001) so the UI layer remains standards-compliant.

## Alternatives Considered

### Separate WGSL (web) + MSL (macOS) kernels

Author the web kernel in WGSL and a hand-written MSL kernel for macOS independently, accepting two source-of-truth shaders. This is the documented ADR-004 *fallback* (Approach (b)): the most robust / ship-ready native path, since a hand-written MSL plugin sidesteps the `minigpu` maturity and Tint-vs-Naga parity risks entirely. Rejected as the *primary* approach because it permanently doubles the physics-maintenance surface and reintroduces exactly the drift this ADR exists to prevent (every tuning change applied twice, validated for behavioural equivalence by hand). Retained as the de-risking escape hatch only — see [`../adr/PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md`](./PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md) and [`../tasks/PRIMORDIS-TASK-013-macos-metal-msl-compute-plugin-fallback.md`](../tasks/PRIMORDIS-TASK-013-macos-metal-msl-compute-plugin-fallback.md).

### Keep the original GLSL-430 kernel and run it on desktop

Reuse `Primordis.py`'s GLSL-430 compute shaders directly on macOS to avoid any port. Rejected because macOS OpenGL is frozen at 4.1 and has no compute-shader support — GLSL-430 compute simply cannot execute via OpenGL on a Mac. Metal (directly, or via Dawn/wgpu over Metal) is mandatory on desktop, so the GLSL source has no native runtime regardless. The original GLSL remains useful only as the reference for the WGSL port and parity testing against `Primordis.py`.

## References

- WebGPU / WGSL specification — compute shaders, `var<storage, read_write>`, `atomic<u32>`, `atomicAdd`/`atomicLoad`, `@workgroup_size`, `@builtin(global_invocation_id)`: https://www.w3.org/TR/WGSL/ and https://www.w3.org/TR/webgpu/
- Apple Metal Shading Language — compute kernels, `device` address space, `atomic_uint` / `atomic_fetch_add_explicit`, threadgroups (for the derived MSL fallback): https://developer.apple.com/metal/
- OpenGL 4.3 compute shaders, `std430` layout, SSBOs, `atomicAdd` (reference semantics of the original kernel): Khronos OpenGL 4.3 / GLSL 4.30 specification — https://www.khronos.org/opengl/
- Dawn (Tint WGSL translator) and wgpu/Naga — WebGPU implementations consuming WGSL on native: https://dawn.googlesource.com/dawn and https://github.com/gfx-rs/wgpu
- Flutter `dart:ffi` (native interop for the Dawn/wgpu backend): https://docs.flutter.dev/development/platform-integration/c-interop
- [`./PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md`](./PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md) — SimBackend interface; shared layer placement of the WGSL kernel source.
- [`./PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md`](./PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md) — web WebGPU backend that consumes this kernel.
- [`./PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md`](./PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md) — native Dawn/wgpu primary + MSL fallback; owns the maturity and atomics-parity risks this ADR depends on.
- [`./PRIMORDIS-ADR-005-rendering-and-compositing.md`](./PRIMORDIS-ADR-005-rendering-and-compositing.md) — present/composite paths (out of scope for the compute kernel).
- [`../tasks/PRIMORDIS-TASK-003-port-simulation-to-wgsl-compute-kernel.md`](../tasks/PRIMORDIS-TASK-003-port-simulation-to-wgsl-compute-kernel.md) — port the 3+1 passes to one WGSL kernel; validate at 24k.
- [`../tasks/PRIMORDIS-TASK-004-web-webgpu-backend-js-interop.md`](../tasks/PRIMORDIS-TASK-004-web-webgpu-backend-js-interop.md) — web backend running the shared kernel.
- [`../tasks/PRIMORDIS-TASK-011-macos-target-dawn-wgpu-ffi-backend.md`](../tasks/PRIMORDIS-TASK-011-macos-target-dawn-wgpu-ffi-backend.md) — macOS backend running the same kernel via Dawn/wgpu-over-Metal FFI.
- [`../tasks/PRIMORDIS-TASK-013-macos-metal-msl-compute-plugin-fallback.md`](../tasks/PRIMORDIS-TASK-013-macos-metal-msl-compute-plugin-fallback.md) — MSL fallback kernel (derived from WGSL).
- [`../tasks/PRIMORDIS-TASK-017-atomics-parity-validation-dawn-vs-browser.md`](../tasks/PRIMORDIS-TASK-017-atomics-parity-validation-dawn-vs-browser.md) — Tint-vs-Naga atomic-binning parity validation.
- [`../tasks/PRIMORDIS-TASK-009-parity-test-harness-vs-python-reference.md`](../tasks/PRIMORDIS-TASK-009-parity-test-harness-vs-python-reference.md) — visual/statistical parity vs `Primordis.py`.
- [`../prd/PRIMORDIS-PRD-001-flutter-web-and-macos-port.md`](../prd/PRIMORDIS-PRD-001-flutter-web-and-macos-port.md) — product requirements.
- [`../research/PRIMORDIS-research-summary.md`](../research/PRIMORDIS-research-summary.md) — feasibility synthesis (verified claims, approaches, risks, effort).
