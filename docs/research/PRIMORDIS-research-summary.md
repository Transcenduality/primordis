# PRIMORDIS Research Summary: Flutter Web (WASM) + Native macOS Port

**Status:** Final (synthesis of feasibility analysis)
**Date:** 2026-06-27
**Author:** Bruce Abernethy (Director of Solution Architecture)
**Scope:** Porting Primordis — a GPU particle-life simulator — from Python (pygame + moderngl + numpy) to Flutter Web (WASM) and native Flutter macOS.

---

## Overview

Primordis is a GPU-accelerated particle-life ("clusters") simulator currently implemented as a single ~350-line Python file (`Primordis.py`) using pygame + moderngl (OpenGL 4.3) + numpy. Its entire physics step runs in **GPU compute shaders** (GLSL `#version 430`, std430 SSBOs, `atomicAdd`).

The team ran a feasibility analysis across two target platforms — Flutter Web (compiled to WASM) and native Flutter macOS — to determine whether the simulation can be reproduced faithfully under the DGROUP Flutter-first standards (Riverpod, Freezed, Retrofit/Dio, GoRouter, `package:lint`, tests, accessibility). Primordis is its **own standalone repo**, not the DGroup monorepo, so template/layout paths are adapted accordingly (Flutter app at repo root or `apps/web`; version in `pubspec.yaml` plus an `AppConfig`/`PrimordisConfig` constant).

The headline finding: the port is **feasible on both platforms**, but it hinges on one hard, verified constraint — **Flutter exposes no first-party Dart GPU-compute API on any platform** as of mid-2026. All GPU compute must therefore live *outside* Flutter: browser WebGPU on web, and Dawn/wgpu-over-Metal (or a native Metal plugin) on macOS. This summary synthesizes the verified findings, the per-platform strategies, particle-count tiers, the recommended architecture, the effort estimate, and the top risks.

---

## What Primordis is (sim spec essentials)

The simulation must be reproduced with the following behavior:

- **Particle-life / "clusters" model.** 24,000 particles, 32 types. World is 1080×720, **toroidal** (wraps on both axes).
- **Per-type-pair parameter matrices.** Three 32×32 `float32` matrices: `forces` (signed attraction/repulsion), `min_distances`, and `radii`. Matrices are **asymmetric** (i→j ≠ j→i). Each type gets a random color.
- **Physics runs entirely in GPU compute shaders** (GLSL `#version 430`, std430 SSBOs, `atomicAdd`). **Three compute passes per frame:**
  1. **Clear bin counts.**
  2. **Bin particles** into a uniform spatial grid via `atomicAdd` scatter. Grid is 11×7 = 77 bins, bin size = `MAX_RADIUS` = 96, `MAX_BIN_PARTICLES` = 512 cap.
  3. **Interaction.** For each particle, scan the 3×3 neighbor bins (toroidal), compute minimum-image distance, apply short-range repulsion (`dist < min_dist`, weighted 5×, `abs(force)`) vs. linear-falloff attraction (`dist < radius`, signed), then Euler-integrate (`v += f*dt; v *= friction; p += v*dt`) and wrap.
- **Render** points with `gl_PointSize = 2`.
- **Three sliders:** Attraction K, Repulsion K, Drift (friction).
- **Cost:** ~67M particle-pair tests/frame at 24k. Trivial on a GPU; ~1–2.5 fps single-thread on CPU.

---

## The core constraint (no first-party Flutter GPU compute)

**Verified SUPPORTED:** Flutter has **no** first-party Dart GPU-compute API on **any** platform as of mid-2026.

Specifics that were verified:

- **`dart:ui` `FragmentProgram` is fragment-shader only.** No compute, no SSBO/UBO, no atomics — and it has historically been fragile on web.
- **`flutter_gpu` is render-only.** There is **no `ComputePass` class** in the Dart API, and it does **not run on web** at all. Even on macOS it exposes no compute. Impeller's Metal backend has internal compute, but it is engine-internal and unexposed.
- **Impeller is still opt-in** (behind a flag) on macOS mid-2026, and is **not on web at all** (web = CanvasKit + Skwasm, both Skia/WebGL).

**Consequence:** All GPU compute must live **outside** Flutter — browser WebGPU via JS interop on web; Dawn/wgpu-over-Metal via `dart:ffi` (or a native Metal plugin) on desktop. **Do not architect around "flutter_gpu compute will ship soon."** This constraint must be re-verified against the pinned Flutter version before sign-off.

---

## Web strategy

### Primary: browser WebGPU (WGSL compute)

The web primary path runs **browser WebGPU** with a WGSL compute kernel, reached from Dart via `dart:js_interop` + `package:web`, on a `<canvas>` **you own** (not Flutter's CanvasKit/Skwasm context). The GLSL-430 → WGSL port is **near 1:1**:

- SSBO → `var<storage, read_write>`
- `atomicAdd(bins[i], 1)` → `atomicAdd(&bins[i], 1u)` (returns the *old* value, which is the scatter offset)
- `local_size_x` → `@workgroup_size`
- `gl_GlobalInvocationID` → `@builtin(global_invocation_id)`
- WGSL atomics must be `atomic<u32>` consistently and read via `atomicLoad`.

This delivers the **full 24k+ particles at 60fps**.

### WebGPU browser support (verified, mid-2026)

| Browser | Support |
|---|---|
| Chrome / Edge | 113+ (since 2023) |
| Safari | 26 (macOS Tahoe 26 / iOS 26, GA Sep 2025) |
| Firefox | 141+ (Windows) / 145+ (Apple-Silicon Mac) |

**Gaps:** Firefox Linux/Android, Intel Macs, and pre-26 Safari/iOS. A **hard `navigator.gpu` feature-detect + fallback is REQUIRED.**

### Fallback: pure Dart → WASM (dart2wasm) CPU sim

When WebGPU is absent, fall back to a **pure Dart → WASM CPU simulation**:

- Single-threaded, viable only at **~3–4k particles @ 60fps** (24k runs ~1–2.5fps — the hard web ceiling).
- The web has **no real isolates** — web "isolates" are web workers that *copy* data. `SharedArrayBuffer` requires COOP/COEP cross-origin-isolation headers, which excludes plain static hosts.
- Render via a single `Canvas.drawRawPoints` / `drawVertices` call from a `Float32List` (never per-point draws).
- Use **sequential counting-sort binning** (deterministic) instead of atomic scatter.

### Why WebGL2 GPGPU is skipped

WebGL2 GPGPU is a *possible* middle fallback, but it is a **rewrite, not a port**: no compute, no atomics, no scatter. Binning would become a prefix-sum counting sort over textures. It costs **more** than the WebGPU path and is **skipped** unless "full 24k on non-WebGPU browsers" becomes a hard requirement.

### Web compositing

Stack the WebGPU `<canvas>` as a **sibling DOM element BEHIND a transparent Flutter glass-pane**, with Flutter UI overlaid on top. **Do not** wrap the canvas in `HtmlElementView` (it forces overlay/canvas-splitting, hits overlay-count limits, and causes jank). Route pointer events explicitly, and sync canvas size to `MediaQuery` `devicePixelRatio` on resize.

---

## Native macOS strategy

The Mac build is the **payoff target**: macOS can run the **full 24k+ (and 100k–500k+)** GPU sim, where web tops out at the CPU fallback unless WebGPU is present.

### The macOS OpenGL 4.1 trap

macOS OpenGL is **frozen at 4.1** — **no compute shaders** (compute needs 4.3). So the original GLSL compute **cannot run via OpenGL on a Mac**. **Metal** (directly, or via Dawn/wgpu over Metal) is **mandatory and sufficient**.

### Primary native path — Approach (a): Dawn/wgpu-over-Metal via FFI

Run the **same single WGSL kernel** via Dawn/wgpu-over-Metal through `dart:ffi` (e.g. the experimental **`minigpu`** package). This realizes **"write the kernel once, two backends"** — browser WebGPU on web + Dawn/wgpu on native. iOS/Windows/Linux then fall out of the same backend for little extra effort.

### Fallback native path — Approach (b): hand-written Metal (MSL) plugin

Hand-written Metal Shading Language compute kernels in a native macOS plugin (Swift/Obj-C++) — a near-1:1 GLSL → MSL port:

- SSBO → `device` buffer
- `atomicAdd` → `atomic_fetch_add_explicit` on a `device atomic_uint`
- `local_size_x` → `threadgroup`

This is the most robust/ship-ready option, but it is a **second kernel to maintain** and is Apple-only. Keep it as the **de-risking escape hatch**.

### macOS compositing — Texture-widget path

Render the sim into an **IOSurface-backed Metal texture** (BGRA8 / CVPixelBuffer) and composite it **UNDER the shared Flutter UI** via the **external-texture `Texture` widget**. The macOS embedder **does support this** (`FlutterPluginRegistrar.textures` / `FlutterTextureRegistry`; engine PR `flutter/engine#24523`). This is **cleaner than the web overlay**. The contract is CVPixelBuffer/IOSurface (`copyPixelBuffer`); watch frame pacing (`textureFrameAvailable` vs. Metal completion) and **confirm no CPU readback** at 24k/60fps.

### Native CPU fallback tier (verified PARTIALLY-TRUE)

Native Dart has **real isolates**, and you can share **one** particle buffer across isolates by allocating it as native memory via `dart:ffi` (`calloc` `Pointer`; share the integer **address**, not the `Pointer` object) plus a `package:native_synchronization` mutex/barrier — true multi-core shared memory, unlike web. The realistic ceiling is **~10–14k @ 60fps** (low-confidence / extrapolated; Dart AOT SIMD is currently broken, so no vectorization). This does **not** reliably hit 24k@60fps — the GPU is the real win. Use this **only** as graceful degradation when GPU init fails.

---

## Particle-count tiers

| Tier | Backend | Realistic ceiling |
|---|---|---|
| Native GPU | Dawn-Metal WGSL or MSL | 24k and **well beyond** (100k–500k+ on Apple Silicon) |
| Web GPU | Browser WebGPU (same WGSL) | Full **24k+** @ 60fps |
| Web CPU (no WebGPU) | Dart → WASM CPU | **~3–4k only** (hard web ceiling) |
| Native CPU fallback | Isolates + FFI shared buffer | **~10–14k** solid; ~16–40fps at 24k (estimate — benchmark before promising) |

---

## Recommended architecture

**One Flutter app, one Dart `SimBackend` interface, swappable compute implementations behind it.** The UI never knows which backend is live.

### Shared across all platforms

- The entire Flutter UI / sliders.
- The 32×32 `forces` / `radii` / `min_distance` matrices + color params (**Freezed** models).
- Particle seeding.
- The frame loop + param marshalling.
- Under Approach (a), the **WGSL kernel source itself**.

### Platform-specific (behind `SimBackend`)

Only the following vary by platform:

- GPU device / pipeline creation.
- Compute dispatch — browser WebGPU vs. Dawn-over-Metal FFI vs. MSL.
- The present/composite path — web stacked-canvas vs. macOS `Texture` widget.

The compute / FFI / shader / JS-interop code necessarily lives **outside** the standard feature/data/domain layers; this is **expected** and is kept behind the `SimBackend` interface so the UI layer stays standards-compliant (Riverpod with plain `Ref`, no `setState` for business logic; Freezed models; Material 3; accessibility).

---

## Effort estimate

| Work | Estimate |
|---|---|
| Web WebGPU path | **~5–8 person-weeks.** The shader port is the *easy* part — time goes to DOM/Flutter compositing, pointer routing, DPR/resize, feature-detect, and the CPU fallback. |
| macOS GPU target, Approach (a) incremental | **~1–2 weeks.** Kernel already written; cost is the Dawn native-asset build, the IOSurface texture bridge + frame pacing, std430 → Metal alignment re-validation, and signing/notarization. |
| macOS Approach (b) add-on | **A few days** for the GLSL → MSL port + parity. |

---

## Top risks

1. **Maturity / bus-factor of the FFI WebGPU layer.** `minigpu` is experimental / solo-maintained. **Spike it first** with the exact 3-pass atomic-binning kernel at 24k / 32-types; keep the Metal plugin warm as fallback.
2. **Atomics parity.** Dawn (Tint) vs. browser/wgpu (Naga) use different WGSL translators with a history of `atomicAdd` bugs. **Validate** that the binning pass produces identical results on both.
3. **Compute → display texture handoff on macOS** is the fiddliest integration risk (frame pacing, no CPU readback).
4. **No first-party compute API.** **Re-verify `flutter_gpu`** against the pinned Flutter version before sign-off.
5. **Three-way kernel drift** if Approach (b) is chosen (GLSL original + WGSL + MSL). Mitigate with **one source-of-truth spec.**
6. **Determinism.** The original GPU binning is already nondeterministic + single-buffered (a race). "Faithful" therefore means **visually / statistically equivalent, never bit-exact.**
7. **Accessibility.** The whole screen is motion — must respect **reduced-motion** (offer a pause / static state) per the org accessibility standard.

---

## Verified-claims appendix

The following claims were **adversarially verified** and found supported (or partially-true) by the feasibility analysis:

- **SUPPORTED** — Flutter has no first-party Dart GPU-compute API on any platform as of mid-2026 (`dart:ui` `FragmentProgram` is fragment-only; `flutter_gpu` is render-only with no `ComputePass` and no web support; Impeller compute is engine-internal/unexposed; Impeller is opt-in on macOS and absent on web).
- **SUPPORTED** — Browser WebGPU (WGSL compute) via `dart:js_interop` + `package:web` on an owned `<canvas>` runs the full 24k+ sim at 60fps on web, and the GLSL-430 → WGSL port is near 1:1.
- **SUPPORTED** — WebGPU browser support mid-2026: Chrome/Edge 113+ (2023); Safari 26 (macOS Tahoe 26 / iOS 26, GA Sep 2025); Firefox 141+ (Windows) / 145+ (Apple-Silicon Mac); with gaps on Firefox Linux/Android, Intel Macs, and pre-26 Safari/iOS — so a hard `navigator.gpu` feature-detect + fallback is required.
- **SUPPORTED** — The web has no real isolates (web workers copy data); `SharedArrayBuffer` requires COOP/COEP cross-origin isolation; the single-thread Dart-WASM CPU fallback tops out at ~3–4k particles @ 60fps.
- **SUPPORTED** — WebGL2 GPGPU is a rewrite (no compute/atomics/scatter; binning → prefix-sum counting sort over textures) that costs more than WebGPU; skip unless full 24k on non-WebGPU browsers is mandatory.
- **SUPPORTED** — macOS OpenGL is frozen at 4.1 (no compute shaders; compute needs 4.3), so the original GLSL compute cannot run via OpenGL on a Mac; Metal (directly or via Dawn/wgpu) is mandatory and sufficient.
- **SUPPORTED** — macOS can run the full 24k+ (and 100k–500k+ on Apple Silicon) GPU sim via Dawn/wgpu-over-Metal (the same WGSL kernel) or a hand-written MSL plugin.
- **SUPPORTED** — The macOS embedder supports external textures (`FlutterPluginRegistrar.textures` / `FlutterTextureRegistry`; engine PR `flutter/engine#24523`); the IOSurface-backed Metal texture composited under the Flutter UI via the `Texture` widget is cleaner than the web overlay.
- **SUPPORTED** — `flutter build web --wasm` (Skwasm renderer) needs WasmGC (Chrome 119+ / FF 120+ / Safari 18.2+) with automatic CanvasKit/dart2js fallback; multi-threaded Skwasm and any `SharedArrayBuffer` CPU path require COOP: same-origin + COEP: require-corp; `--wasm` forbids legacy interop (`dart:html` / `dart:js_util`) anywhere in the dep tree (use `dart:js_interop` + `package:web` only).
- **PARTIALLY-TRUE** — The native CPU fallback can share one particle buffer across real Dart isolates via an FFI `calloc`'d native buffer (sharing the integer address) + `package:native_synchronization`, giving true multi-core shared memory. The ~10–14k @ 60fps ceiling is **low-confidence / extrapolated**: Dart AOT SIMD is currently broken (no vectorization), so this does not reliably hit 24k@60fps and must be benchmarked before being promised.

---

## References

- PRD: [PRIMORDIS-PRD-001 — Flutter Web and macOS Port](../prd/PRIMORDIS-PRD-001-flutter-web-and-macos-port.md)
- ADR-001: [Cross-platform architecture — SimBackend](../adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md)
- ADR-002: [Web GPU compute via WebGPU + JS interop](../adr/PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md)
- ADR-003: [Shared WGSL compute kernel](../adr/PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md)
- ADR-004: [Native macOS GPU — Dawn/Metal via FFI](../adr/PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md)
- ADR-005: [Rendering and compositing](../adr/PRIMORDIS-ADR-005-rendering-and-compositing.md)
- ADR-006: [CPU fallback tiers and feature detection](../adr/PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md)
- ADR-007: [Web build and cross-origin isolation](../adr/PRIMORDIS-ADR-007-web-build-and-cross-origin-isolation.md)
