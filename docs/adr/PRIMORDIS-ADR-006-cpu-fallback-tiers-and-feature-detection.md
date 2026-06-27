<!-- Filename convention: <SCOPE>-ADR-NNN-short-title.md -->

# PRIMORDIS-ADR-006: CPU fallback tiers and feature detection

**Status:** Proposed
**Date:** 2026-06-27
**Deciders:** Bruce Abernethy
**Review date:** 2026-09-27
**Supersedes:** N/A
**Superseded by:** N/A
**Compliance/Security:** None (UX/accessibility implications noted under Consequences; reduced-motion handling per org accessibility standard)

## Context

Primordis runs its entire physics in GPU compute (3 passes/frame: clear bin counts; atomic-scatter binning into a 11x7 uniform grid; 3x3 neighbor-bin interaction with Euler integration). At the reference workload of 24,000 particles / 32 types this is ~67M particle-pair tests per frame — trivial on a GPU but only ~1-2.5 fps on a single CPU thread.

Two independent decisions establish that GPU compute is reachable on only *some* targets:

- On the **web**, [PRIMORDIS-ADR-002](./PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md) makes browser **WebGPU** (WGSL compute via `dart:js_interop` + `package:web`) the primary path. But WebGPU is not universal mid-2026: it ships in Chrome/Edge 113+, Safari 26 (macOS Tahoe 26 / iOS 26, GA Sep 2025), and Firefox 141+ (Windows) / 145+ (Apple-Silicon mac). It is **absent** on Firefox Linux/Android, on Intel Macs, and on pre-26 Safari/iOS. A hard `navigator.gpu` feature-detect with a fallback is therefore required.
- On **native macOS**, [PRIMORDIS-ADR-004](./PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md) makes Dawn/wgpu-over-Metal (FFI) the primary GPU path, with a hand-written Metal (MSL) plugin as fallback. GPU device/pipeline creation can still fail (driver, GPU-family gating on old Intel Macs, signing/sandbox issues), so a non-GPU path is needed there too.

Because Flutter exposes no first-party Dart GPU-compute API on any platform (see ADR-002 / ADR-004), the only honest fallback when GPU compute is unavailable is a **CPU** implementation. The CPU ceilings differ sharply by platform:

- **Web has no real multi-core shared memory** for this workload: web "isolates" are web workers that *copy* data, and `SharedArrayBuffer` requires COOP/COEP cross-origin isolation (see [PRIMORDIS-ADR-007](./PRIMORDIS-ADR-007-web-build-and-cross-origin-isolation.md)), which plain static hosts cannot guarantee. So the web CPU fallback is effectively single-threaded Dart→WASM (dart2wasm): only ~3-4k particles @ 60fps (24k ≈ 1-2.5 fps).
- **Native Dart has real isolates** and can share *one* particle buffer across them by allocating it as native memory via `dart:ffi` (`calloc` Pointer; share the integer **address**, not the Pointer object) guarded by a `package:native_synchronization` mutex/barrier. This gives true multi-core shared memory, but Dart AOT SIMD is currently broken (no vectorization), so the realistic ceiling is ~10-14k @ 60fps and ~16-40 fps at 24k — **low-confidence / extrapolated, must be benchmarked before being promised**. It does not reliably hit 24k @ 60fps; GPU remains the real win.

What is missing — and what this ADR decides — is the cross-cutting **capability-detection logic**, the **per-tier particle-count policy**, and the **user-visible "reduced mode" indicator** that ties these backends together behind the single `SimBackend` interface from [PRIMORDIS-ADR-001](./PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md). The implementation is split across [PRIMORDIS-TASK-007](../tasks/PRIMORDIS-TASK-007-webgpu-feature-detection-and-fallback-switch.md) (web feature-detect + switch), [PRIMORDIS-TASK-014](../tasks/PRIMORDIS-TASK-014-native-cpu-isolate-fallback-backend.md) (native isolate CPU backend), and [PRIMORDIS-TASK-015](../tasks/PRIMORDIS-TASK-015-cross-platform-backend-selection-and-reduced-mode-ux.md) (unified selection + reduced-mode UX), which this decision governs.

## Decision

We define a four-tier graceful-degradation model, a deterministic detection-and-selection order, a per-tier particle-count policy, and a visible reduced-mode indicator. All tiers sit behind the single `SimBackend` interface (ADR-001); the Flutter UI never branches on which backend is live.

### 1. Capability tiers and target particle counts

| Tier | Platform / condition | Compute backend | Target particle count | Notes |
|------|----------------------|-----------------|-----------------------|-------|
| **T1 — Native GPU** | macOS, GPU init OK | Dawn/wgpu-over-Metal WGSL (ADR-004), MSL fallback | 24k default; 100k-500k+ on Apple Silicon | Full quality; the payoff of the Mac build |
| **T2 — Web GPU** | Web, `navigator.gpu` present + adapter/device acquired | Browser WebGPU WGSL (ADR-002) | 24k+ @ 60fps | Same WGSL kernel as T1 (ADR-003) |
| **T3 — Native CPU** | macOS, GPU init **fails** | Real isolates over FFI `calloc`'d shared buffer + `native_synchronization` | ~10-14k @ 60fps (estimate — benchmark) | Reduced mode; not a guaranteed 24k |
| **T4 — Web CPU** | Web, no WebGPU | Single-thread Dart→WASM (dart2wasm) | ~3-4k @ 60fps (hard web ceiling) | Reduced mode; lowest ceiling |

These CPU counts are **honest ceilings, not aspirations**: T4 is bounded by single-thread WASM (24k ≈ 1-2.5 fps); T3 is bounded by Dart AOT having no working SIMD and is **low-confidence until benchmarked** (TASK-014). The default particle count for a tier is the count that holds ~60fps for that tier, not the reference 24k.

### 2. Detection logic

Detection is performed **once at startup**, by platform, and produces the highest tier whose preconditions actually succeed (capability is *proven*, never assumed):

**Web (TASK-007):**
1. Hard feature-detect `navigator.gpu` via `dart:js_interop` + `package:web` (truthy check, not a try/catch on first use).
2. If present, attempt `requestAdapter()` then `requestDevice()`. Success → **T2 (Web GPU)**. Any null/throw at adapter or device acquisition → demote.
3. Otherwise (or on demotion) → **T4 (Web CPU, Dart→WASM)**.
   - WebGL2 GPGPU is explicitly **not** a tier (see Alternatives); there is no WebGL middle fallback.

**Native macOS (TASK-014 / TASK-015):**
1. Attempt GPU device + compute-pipeline creation via the primary Dawn/wgpu-over-Metal FFI path (ADR-004), with the MSL plugin as the secondary GPU attempt. Either success → **T1 (Native GPU)**.
2. If GPU init fails (no compatible GPU family, driver/signing/sandbox failure, FFI load failure) → **T3 (Native CPU isolates)**.

The first GPU acquisition is treated as the authoritative probe: device + compute pipeline must both build successfully before T1/T2 is selected. Partial success demotes to the CPU tier rather than risking a runtime stall.

### 3. Particle-count policy

- The selected tier sets the **default and maximum** particle count for that session (T1: 24k, scalable higher on Apple Silicon; T2: 24k; T3: ~10-14k pending benchmark; T4: ~3-4k).
- Particle count is exposed in shared sim params (Freezed model, ADR-001) and marshalled to whichever backend is live; the UI/sliders are identical across tiers.
- A tier may offer a count slider/preset, but it must not exceed that tier's benchmarked ceiling. T3's ceiling is **provisional** and gated on TASK-014 benchmarking before any number is committed in the UI or docs.
- Seeding (32 types, asymmetric 32x32 force/min_distance/radius matrices, random colors) is identical across tiers; only the live count differs.

### 4. "Reduced mode" indicator (UX)

- Whenever a CPU tier (T3 or T4) is selected, the app surfaces a **visible, persistent "reduced mode" indicator** — a small, semantically-labelled status affordance (Material 3, with a `Semantics` label and a tooltip explaining *why*: e.g. "GPU acceleration unavailable; running on CPU at reduced particle count").
- GPU tiers (T1/T2) show **no** reduced-mode chrome (or an unobtrusive "GPU" state).
- The indicator is informational, not motion: it must remain visible and legible while the simulation is **paused** for reduced-motion users (the whole canvas is motion; per org accessibility standard the app already offers pause/static state — see ADR-001 and TASK-018). Reduced-mode messaging must therefore be readable independent of the running animation.

## Consequences

### Positive

- **Every supported target runs *something*.** No browser/driver combination yields a blank screen: WebGPU-less browsers and GPU-init-failed Macs both fall through to a working CPU tier.
- **Honest performance contract.** Per-tier ceilings are documented as benchmarked ceilings, not marketing numbers, so users on T3/T4 see an expected (reduced) experience rather than a stuttering 24k.
- **Clean separation.** All detection and tier selection sits behind `SimBackend` (ADR-001); the Flutter UI/sliders/params layer stays standards-compliant (Riverpod, Freezed, Material 3) and backend-agnostic.
- **Accessibility-positive.** The reduced-mode indicator is a semantic, tooltip-bearing control that coexists with the required pause/reduced-motion state.
- **Native CPU tier is genuinely better than web CPU** (multi-core FFI-shared buffer vs single-thread WASM), so a GPU-less Mac still meaningfully outperforms a GPU-less browser.

### Negative

- **T3 numbers are unverified.** ~10-14k @ 60fps and ~16-40 fps @ 24k are extrapolated (Dart AOT SIMD broken). Committing them in UI before TASK-014 benchmarking would be dishonest; this is a hard gate.
- **Three CPU-related concurrency mechanisms to get right** on native: FFI `calloc` buffer, sharing the integer **address** (not the Pointer), and a `native_synchronization` mutex/barrier — a class of bug (lifetime, alignment, races) the GPU path does not have.
- **More backends to test and maintain.** The CPU tiers need their own parity coverage against the Python reference (cluster formation, drift) and must use deterministic sequential counting-sort binning rather than atomic scatter, diverging from the GPU kernel's binning.
- **Detection edge cases.** `navigator.gpu` present but adapter/device acquisition failing (e.g. blocklisted GPU) must demote cleanly; a missed case strands the user on a dead GPU path.

### Neutral

- The reference 24k / 32-type workload remains the canonical spec; tiers scale the *count*, not the physics, so "faithful" still means visually/statistically equivalent (never bit-exact — the original GPU binning is already nondeterministic).
- The web CPU tier intentionally has **no** WebGL2 middle rung; T4 is the only web fallback.
- This ADR governs *selection and policy*; the concrete backends are decided in ADR-002 (web GPU), ADR-004 (native GPU), and their tasks.

## Alternatives Considered

### WebGL2 GPGPU as a middle web fallback (between WebGPU and Dart-WASM CPU)
Rejected. WebGL2 has no compute shaders, no atomics, and no scatter; the binning pass would become a prefix-sum counting sort over textures — a **rewrite, not a port** — costing more than the WebGPU path itself while still not matching it. It is only justified if "full 24k on non-WebGPU browsers" becomes a hard requirement, which it is not. The web therefore degrades directly from T2 (WebGPU 24k) to T4 (Dart-WASM ~3-4k).

### Assume WebGPU / GPU and skip the CPU tiers entirely
Rejected. WebGPU is absent on Firefox Linux/Android, Intel Macs, and pre-26 Safari/iOS, and native GPU init can fail (GPU-family gating, signing). Without CPU tiers those users get nothing. The whole point of a hard `navigator.gpu` detect plus a GPU-init probe is to guarantee a working fallback.

### Multi-threaded CPU on web via web workers / SharedArrayBuffer
Rejected for the default tier. Web workers copy data (no shared buffer for a 24k SoA), and `SharedArrayBuffer` requires COOP/COEP cross-origin isolation (ADR-007) that plain static hosts cannot provide. So the web CPU tier is single-thread WASM by necessity; multi-core CPU only exists natively (T3) where real isolates + an FFI-shared buffer are available.

### Promote the native CPU tier (T3) to a guaranteed 24k @ 60fps
Rejected. Even with real isolates and shared memory, the realistic ceiling is ~10-14k @ 60fps and Dart AOT SIMD is broken, so 24k lands at an estimated ~16-40 fps. T3 is graceful degradation for GPU-init failure, not a co-equal of T1; its numbers stay provisional until benchmarked (TASK-014).

### Silent degradation (no visible reduced-mode indicator)
Rejected. Dropping particle count and frame budget without telling the user looks like a bug and hides why the experience differs across machines. A visible, semantic, tooltip-bearing reduced-mode indicator (that survives the paused/reduced-motion state) is required by the org accessibility standard and by basic UX honesty.

## References

PRIMORDIS docs (relative path):
- [PRIMORDIS-ADR-001 — Cross-platform architecture & SimBackend](./PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md)
- [PRIMORDIS-ADR-002 — Web GPU compute via WebGPU + JS interop](./PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md)
- [PRIMORDIS-ADR-003 — Shared WGSL compute kernel](./PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md)
- [PRIMORDIS-ADR-004 — Native macOS GPU via Dawn/Metal FFI](./PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md)
- [PRIMORDIS-ADR-005 — Rendering and compositing](./PRIMORDIS-ADR-005-rendering-and-compositing.md)
- [PRIMORDIS-ADR-007 — Web build & cross-origin isolation](./PRIMORDIS-ADR-007-web-build-and-cross-origin-isolation.md)
- [PRIMORDIS-PRD-001 — Flutter Web & macOS port](../prd/PRIMORDIS-PRD-001-flutter-web-and-macos-port.md)
- [PRIMORDIS research summary](../research/PRIMORDIS-research-summary.md)
- [PRIMORDIS-TASK-007 — WebGPU feature detection & fallback switch](../tasks/PRIMORDIS-TASK-007-webgpu-feature-detection-and-fallback-switch.md)
- [PRIMORDIS-TASK-008 — Dart-WASM CPU fallback backend](../tasks/PRIMORDIS-TASK-008-dart-wasm-cpu-fallback-backend.md)
- [PRIMORDIS-TASK-014 — Native CPU isolate fallback backend](../tasks/PRIMORDIS-TASK-014-native-cpu-isolate-fallback-backend.md)
- [PRIMORDIS-TASK-015 — Cross-platform backend selection & reduced-mode UX](../tasks/PRIMORDIS-TASK-015-cross-platform-backend-selection-and-reduced-mode-ux.md)
- [PRIMORDIS-TASK-018 — Test coverage and accessibility](../tasks/PRIMORDIS-TASK-018-test-coverage-and-accessibility.md)

External documentation topics:
- flutter.dev — `dart:ffi` interop and native memory (`Pointer`, `calloc`/`malloc` allocators)
- flutter.dev — Isolates and concurrency in Dart (real isolates on native; web "isolates" as web workers)
- flutter.dev — Compiling to WebAssembly (`flutter build web --wasm`, dart2wasm, Skwasm)
- flutter.dev — Accessibility and `Semantics`; respecting reduced motion / `MediaQuery` disable-animations
- `package:native_synchronization` — cross-isolate mutex/barrier primitives
- WebGPU specification (W3C) — `navigator.gpu`, `GPUAdapter`, `requestAdapter()` / `requestDevice()` feature detection
- MDN — `navigator.gpu` and `SharedArrayBuffer` cross-origin-isolation (COOP/COEP) requirements
- Apple Developer — Metal compute (`MTLDevice`, GPU family / feature-set gating) for native GPU-availability checks
