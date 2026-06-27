# PRIMORDIS-TASK-011: macOS target — Dawn/wgpu-over-Metal SimBackend via dart:ffi

**Status:** Todo
**Priority:** High
**Created:** 2026-06-27
**Updated:** 2026-06-27

## Description

Enable the native Flutter **macOS** target and implement a desktop `SimBackend` that runs the **same single WGSL compute kernel** (from [PRIMORDIS-TASK-003](./PRIMORDIS-TASK-003-port-simulation-to-wgsl-compute-kernel.md)) through **Dawn/wgpu over Metal**, reached from Dart via `dart:ffi`. This is Approach (a) from [PRIMORDIS-ADR-004](../adr/PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md) — "write the kernel once, two backends" — and it is the recommended native GPU path because it eliminates a third kernel and keeps the WGSL source the single source of truth shared with the web WebGPU backend ([PRIMORDIS-TASK-004](./PRIMORDIS-TASK-004-web-webgpu-backend-js-interop.md)).

This task is the **spike that de-risks the experimental FFI WebGPU layer.** Because `minigpu` is experimental and solo-maintained (a top project risk per the research summary), the very first deliverable is a working spike of the exact 3-pass atomic-binning kernel at 24,000 particles / 32 types on Dawn-over-Metal, with the Metal/MSL plugin fallback ([PRIMORDIS-TASK-013](./PRIMORDIS-TASK-013-macos-metal-msl-compute-plugin-fallback.md)) kept warm if the spike fails. It builds only the **device/pipeline creation, buffer allocation, and compute dispatch** half of the macOS backend; the compute→display present path (IOSurface-backed Metal texture) is a separate concern owned by [PRIMORDIS-TASK-012](./PRIMORDIS-TASK-012-macos-metal-texture-present-path.md), and full Dawn-vs-browser atomics parity is owned by [PRIMORDIS-TASK-017](./PRIMORDIS-TASK-017-atomics-parity-validation-dawn-vs-browser.md).

Why Metal is mandatory and not optional: macOS OpenGL is frozen at 4.1, which has **no compute shaders** (compute requires 4.3), so the original GLSL-430 compute kernel cannot run via OpenGL on a Mac at all. Metal — here via Dawn/wgpu over Metal — is the only viable native compute path.

## Scope

**Area:** FFI/Native
**Files/Dirs:**
- `macos/` — Flutter macOS runner (enable target, entitlements, native-asset / Dawn build hooks)
- `lib/sim/backends/macos_dawn_backend.dart` — `SimBackend` implementation over the FFI WebGPU layer
- `lib/sim/ffi/` — `dart:ffi` bindings to the Dawn/wgpu-over-Metal layer (minigpu spike), `ffigen` config, buffer/dispatch wrappers
- `lib/sim/kernel/primordis.wgsl` — **shared, unchanged** WGSL kernel (consumed, not authored, here; authored in TASK-003)
- `pubspec.yaml` — `ffi`/`ffigen` deps, native-asset/build-hook wiring, version bump
- `lib/app_config.dart` (or `PrimordisConfig` constant) — version constant
- `test/sim/backends/macos_dawn_backend_test.dart` — backend contract/marshalling tests
- `tool/spike/` — standalone Dawn-over-Metal kernel spike harness (24k / 32-types)

> **Layering note (house standards):** the FFI bindings, native-asset build hooks, and Dawn/wgpu glue necessarily live **outside** the standard Flutter feature/data/domain layers — this is expected and explicitly sanctioned by [PRIMORDIS-ADR-001](../adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md) and [PRIMORDIS-ADR-004](../adr/PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md). All of it sits **behind the `SimBackend` interface**; the UI layer, Riverpod providers, and Freezed param models ([PRIMORDIS-TASK-002](./PRIMORDIS-TASK-002-simbackend-interface-and-shared-sim-model.md)) stay standards-compliant and never reference `dart:ffi`.

## Acceptance Criteria

- [ ] The Flutter **macOS desktop target is enabled** and the app builds and launches via `flutter run -d macos` (debug) and `flutter build macos` (release).
- [ ] A **standalone spike** (in `tool/spike/`) runs the exact 3-pass kernel (clear bins → `atomicAdd` scatter-bin → interaction + Euler integrate) at **24,000 particles / 32 types** on Dawn-over-Metal and advances the simulation for ≥1000 frames without crash, device-lost, or NaN/Inf positions.
- [ ] `MacosDawnBackend` implements the full `SimBackend` interface from [PRIMORDIS-TASK-002](./PRIMORDIS-TASK-002-simbackend-interface-and-shared-sim-model.md): `init()`, `seed(params)`, `step(dt)`, `updateUniforms(attractionK, repulsionK, drift)`, `dispose()` — with **no UI-visible difference** from the web backend's contract.
- [ ] Backend **creates the WebGPU device/adapter, allocates all SSBO-equivalent storage buffers** (positions, velocities, per-type-pair 32×32 force/min-distance/radius matrices, colors, bin counts, bin contents), and **builds the three compute pipelines + the point-render pipeline** from the shared `primordis.wgsl`.
- [ ] Buffer layouts honor WGSL/`std430` rules: storage buffers declared `var<storage, read_write>`; the bin-count buffer is a consistent `atomic<u32>` array written with `atomicAdd(&bins[i], 1u)` (old value = scatter offset) and read with `atomicLoad`; grid is 11×7 = 77 bins, bin size = `MAX_RADIUS` = 96, `MAX_BIN_PARTICLES` = 512 cap, world 1080×720 toroidal.
- [ ] Compute is dispatched with workgroup counts derived from `@workgroup_size` matching the kernel; param updates (the 3 sliders) are marshalled into the uniform buffer each frame without reallocating storage.
- [ ] The Metal/MSL fallback ([PRIMORDIS-TASK-013](./PRIMORDIS-TASK-013-macos-metal-msl-compute-plugin-fallback.md)) is registered as a **selectable alternate backend** behind `SimBackend` and is reachable if Dawn init fails (selection logic finalized in [PRIMORDIS-TASK-015](./PRIMORDIS-TASK-015-cross-platform-backend-selection-and-reduced-mode-ux.md)).
- [ ] `std430` → Metal buffer alignment/padding is **re-validated** for every shared buffer (a known cost of the Dawn native-asset build); any padding required to satisfy Metal alignment is documented next to the buffer definition.
- [ ] No first-party compute API is assumed: a short note re-confirms `flutter_gpu` exposes no compute against the pinned Flutter version before this backend is relied upon (per [PRIMORDIS-ADR-004](../adr/PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md)).

### Versioning (if Flutter/native code changed)
- [ ] Version bumped in `pubspec.yaml` and the app config constant (`AppConfig`/`PrimordisConfig`); semver (minor bump — new platform target + backend).

### Test Coverage
- [ ] New/modified Dart has unit/widget tests; `flutter test` passes; `flutter analyze` zero warnings (`package:lint`).
- [ ] Backend contract tests cover seed marshalling, uniform updates, and `dispose()` cleanup; FFI buffer-size/offset computations are unit-tested against the documented `std430`/Metal-alignment layout (pure Dart, no GPU device required).

## Implementation Notes

- **One kernel, two backends.** This backend consumes `lib/sim/kernel/primordis.wgsl` verbatim from [PRIMORDIS-TASK-003](./PRIMORDIS-TASK-003-port-simulation-to-wgsl-compute-kernel.md); do **not** fork or edit the WGSL here. The GLSL→WGSL port is near-1:1 (`buffer`/`std430` SSBO → `var<storage, read_write>`; `atomicAdd(bins[i],1)` → `atomicAdd(&bins[i],1u)` returning the old value as the scatter offset; `layout(local_size_x=N)` → `@workgroup_size`; `gl_GlobalInvocationID` → `@builtin(global_invocation_id)`). WGSL atomics must be `atomic<u32>` consistently and read via `atomicLoad`.
- **Spike first (risk-driven order).** Build `tool/spike/` before wiring the backend into the app. The spike must run the *exact* 3-pass atomic-binning kernel at 24k/32-types — not a reduced toy — because the whole point is to prove the experimental FFI layer survives the real workload. If the spike is unstable, escalate to the Metal/MSL plugin ([PRIMORDIS-TASK-013](./PRIMORDIS-TASK-013-macos-metal-msl-compute-plugin-fallback.md)) rather than weakening the kernel.
- **FFI surface.** Reach Dawn/wgpu via `dart:ffi`, spiking the experimental `minigpu` package. Generate bindings with `ffigen`; wrap device/adapter creation, buffer create/map/write, pipeline + bind-group creation, and command-encoder dispatch in a thin Dart wrapper so `MacosDawnBackend` talks to a small, testable surface and never threads raw `Pointer`s into UI code.
- **Native-asset / Dawn build.** The Dawn native-asset build is the main incremental cost of this path. Wire it through Flutter's native-assets/build-hook mechanism so `flutter build macos` produces a self-contained binary; record the exact Dawn/wgpu pin so the kernel translator (Tint) version is known for the parity work in [PRIMORDIS-TASK-017](./PRIMORDIS-TASK-017-atomics-parity-validation-dawn-vs-browser.md).
- **Atomics translator awareness.** Dawn translates WGSL via **Tint**, the browser/wgpu path via **Naga**; these have a history of `atomicAdd` discrepancies. This task only needs the binning pass to run correctly on Tint/Metal; proving *identical* results across Tint and Naga is [PRIMORDIS-TASK-017](./PRIMORDIS-TASK-017-atomics-parity-validation-dawn-vs-browser.md). Keep the binning buffers and atomic ops structured so that parity test can diff them directly.
- **Determinism is not a goal.** The original GPU binning is already nondeterministic (single-buffered scatter race); "faithful" means visually/statistically equivalent, never bit-exact (per [PRIMORDIS-PRD-001](../prd/PRIMORDIS-PRD-001-flutter-web-and-macos-port.md) Non-Goals). Do not add synchronization that the reference lacks in pursuit of repeatability.
- **Scope boundary — presentation.** This task ends at "compute runs and storage buffers hold correct results." Getting those results onto the screen (IOSurface-backed Metal texture, `FlutterTextureRegistry`, `Texture` widget, frame pacing, no CPU readback) is [PRIMORDIS-TASK-012](./PRIMORDIS-TASK-012-macos-metal-texture-present-path.md). For spike-time validation, a CPU readback of buffers is acceptable; it must **not** leak into the shipped frame loop.
- **iOS/Windows/Linux fall-out.** Building the kernel on Dawn/wgpu means iOS/Windows/Linux GPU compute later costs little extra; keep the backend free of macOS-only assumptions except where Metal/IOSurface specifics are unavoidable.

## Testing

- [ ] Run the `tool/spike/` harness at 24k/32-types; confirm ≥1000 frames advance with no device-lost, no NaN/Inf, and bin counts within the `MAX_BIN_PARTICLES`=512 cap.
- [ ] `flutter run -d macos` launches the app with `MacosDawnBackend` selected; the simulation steps (verified via spike/readback path until [PRIMORDIS-TASK-012](./PRIMORDIS-TASK-012-macos-metal-texture-present-path.md) lands the real present path).
- [ ] Unit-test FFI buffer offset/size/alignment math against the documented `std430`/Metal layout (no GPU needed).
- [ ] Force a Dawn-init failure and confirm the backend reports failure cleanly so selection can fall back to the MSL plugin ([PRIMORDIS-TASK-013](./PRIMORDIS-TASK-013-macos-metal-msl-compute-plugin-fallback.md)) / native CPU tier ([PRIMORDIS-TASK-014](./PRIMORDIS-TASK-014-native-cpu-isolate-fallback-backend.md)).
- [ ] `flutter test` and `flutter analyze` pass with zero warnings.
- [ ] Sanity-compare cluster formation/drift against the Python reference and the web WebGPU backend (formalized in [PRIMORDIS-TASK-009](./PRIMORDIS-TASK-009-parity-test-harness-vs-python-reference.md)).

## Related
- PRD: [PRIMORDIS-PRD-001](../prd/PRIMORDIS-PRD-001-flutter-web-and-macos-port.md)
- ADR: [PRIMORDIS-ADR-004 — Native macOS GPU compute (Dawn/Metal via FFI)](../adr/PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md); [PRIMORDIS-ADR-003 — Shared WGSL compute kernel](../adr/PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md); [PRIMORDIS-ADR-001 — Cross-platform architecture (SimBackend)](../adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md)
- Depends on: [PRIMORDIS-TASK-003 — Port simulation to WGSL compute kernel](./PRIMORDIS-TASK-003-port-simulation-to-wgsl-compute-kernel.md); [PRIMORDIS-TASK-004 — Web WebGPU backend (JS interop)](./PRIMORDIS-TASK-004-web-webgpu-backend-js-interop.md)
- Blocks: [PRIMORDIS-TASK-012 — macOS Metal texture present path](./PRIMORDIS-TASK-012-macos-metal-texture-present-path.md); [PRIMORDIS-TASK-013 — macOS Metal/MSL compute plugin fallback](./PRIMORDIS-TASK-013-macos-metal-msl-compute-plugin-fallback.md); [PRIMORDIS-TASK-017 — Atomics parity validation (Dawn vs browser)](./PRIMORDIS-TASK-017-atomics-parity-validation-dawn-vs-browser.md); [PRIMORDIS-TASK-015 — Cross-platform backend selection and reduced-mode UX](./PRIMORDIS-TASK-015-cross-platform-backend-selection-and-reduced-mode-ux.md)
