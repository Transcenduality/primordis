# PRIMORDIS-TASK-002: SimBackend interface and shared sim model

**Status:** Todo
**Priority:** Critical
**Created:** 2026-06-27
**Updated:** 2026-06-27

## Description

Define the load-bearing Dart `SimBackend` interface and the shared, platform-agnostic simulation core that every compute backend (web WebGPU, native Dawn/wgpu-over-Metal, MSL plugin, web CPU-WASM, native CPU isolates) plugs into. This is the contract from [PRIMORDIS-ADR-001](../adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md): one `SimBackend` interface, swappable implementations behind it, and a single shared set of simulation parameters, particle seeding, frame loop, and parameter marshalling that the UI drives without ever knowing which backend is live.

This task owns the **shared** half of the architecture only. The concrete backends are out of scope here: the WGSL kernel is [PRIMORDIS-TASK-003](./PRIMORDIS-TASK-003-port-simulation-to-wgsl-compute-kernel.md), the web WebGPU backend is [PRIMORDIS-TASK-004](./PRIMORDIS-TASK-004-web-webgpu-backend-js-interop.md), and the CPU fallbacks are [PRIMORDIS-TASK-008](./PRIMORDIS-TASK-008-dart-wasm-cpu-fallback-backend.md) / [PRIMORDIS-TASK-014](./PRIMORDIS-TASK-014-native-cpu-isolate-fallback-backend.md). What ships here is: the `SimBackend` interface, the Freezed parameter/color/seed models that faithfully encode the `Primordis.py` simulation constants, the deterministic seeding logic, the frame-loop/tick driver, the param-marshalling surface (the typed bridge that each backend later uploads to a uniform buffer / SSBO / FFI buffer), the Riverpod providers that expose all of this to the UI, and a `FakeSimBackend` so the UI and frame loop are fully unit/widget-testable with no GPU.

The models must capture the simulation exactly as the reference defines it: 24,000 particles, 32 types, a 1080x720 toroidal world, and three **asymmetric** 32x32 float32 matrices — `forces` (signed), `min_distances`, and `radii` — where `i->j != j->i`, plus random per-type colors. These are the inputs that TASK-003's WGSL kernel and the CPU fallbacks all consume; getting the model shape and marshalling layout right here is what lets a single kernel and a single UI serve all platforms.

## Scope

**Area:** Flutter

**Files/Dirs:**
- `lib/src/sim/sim_backend.dart` — the `SimBackend` interface (fills in the seam placeholder from TASK-001)
- `lib/src/sim/models/sim_params.dart` — Freezed `SimParams` (the 3 asymmetric 32x32 matrices, world/grid constants, live slider values)
- `lib/src/sim/models/particle_type.dart` — Freezed per-type model (color, type index)
- `lib/src/sim/models/sim_seed.dart` — Freezed seed descriptor (RNG seed, particle count)
- `lib/src/sim/sim_seeder.dart` — deterministic particle/matrix/color seeding from a `SimSeed`
- `lib/src/sim/sim_marshalling.dart` — typed -> packed (`Float32List`/`Uint32List`) layout for backend upload
- `lib/src/sim/frame_loop.dart` — tick driver (init -> seed -> setParams -> step -> present)
- `lib/src/sim/fake_sim_backend.dart` — test/double backend (no GPU)
- `lib/src/sim/providers/sim_providers.dart` — Riverpod providers (params, seed, backend handle, frame state)
- `test/sim/` — unit tests for models, seeding determinism, marshalling layout, frame loop, fake backend

## Acceptance Criteria

- [ ] `SimBackend` is an abstract Dart interface exposing the lifecycle the PRD names: `init()`, `seed(SimSeed)`, `setParams(SimParams)`, `step(double dt)`, and `present()` (plus `dispose()` and a capability/particle-ceiling query). Method shapes are documented and platform-neutral — they must abstract device/pipeline creation, dispatch, parameter upload, and present across both the owned-`<canvas>` web model and the external-`Texture` macOS model **without leaking platform specifics** ([PRIMORDIS-ADR-001](../adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md), [PRIMORDIS-ADR-005](../adr/PRIMORDIS-ADR-005-rendering-and-compositing.md)).
- [ ] `SimParams` is a Freezed model holding the three **asymmetric** 32x32 float32 matrices — `forces` (signed attraction/repulsion), `minDistances`, `radii` — plus the live slider values (Attraction K, Repulsion K, Drift/friction) and the world/grid constants (world 1080x720, toroidal; grid 11x7 = 77 bins; bin size = `MAX_RADIUS = 96`; `MAX_BIN_PARTICLES = 512`; type count 32; particle count 24000). Asymmetry (`i->j != j->i`) is preserved and asserted in tests.
- [ ] Per-type colors and type indices are Freezed-modeled; color generation is part of deterministic seeding.
- [ ] `SimSeeder` produces particles, the three matrices, and the colors **deterministically** from a `SimSeed` (fixed RNG seed -> identical output), so parity ([PRIMORDIS-TASK-009](./PRIMORDIS-TASK-009-parity-test-harness-vs-python-reference.md)) and tests are reproducible. Note: this is *seed* determinism only — the GPU binning/integration itself remains nondeterministic per the PRD, so "faithful" is statistical, not bit-exact.
- [ ] `sim_marshalling.dart` packs `SimParams` into the exact `Float32List`/`Uint32List` byte layout that backends upload, with the layout documented (field order, offsets, `std140`/`std430`-friendly alignment expectations) so TASK-004 (WebGPU uniform/SSBO) and TASK-003 (WGSL struct) consume an agreed contract. Atomic bin-count buffers are typed as `Uint32List` to match the WGSL `atomic<u32>` requirement noted in [PRIMORDIS-ADR-002](../adr/PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md) / [PRIMORDIS-ADR-003](../adr/PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md).
- [ ] `frame_loop.dart` drives the per-frame sequence (`setParams` when sliders change -> `step(dt)` -> `present()`) decoupled from any concrete backend, and can be ticked by a test without a render surface.
- [ ] Riverpod providers expose: current `SimParams` (with slider mutations), the active `SimBackend` handle, the `SimSeed`, and frame/run state. State is managed with plain `Ref` Riverpod — **no `setState` for business logic** (house standard).
- [ ] `FakeSimBackend` implements `SimBackend` with no GPU (records calls / advances a trivial in-memory state) so the UI, frame loop, and providers are testable in CI without WebGPU/FFI/Metal.
- [ ] All new models are Freezed; `dart run build_runner build` is clean.
- [ ] Backend **selection** is *not* implemented here (that is [PRIMORDIS-TASK-007](./PRIMORDIS-TASK-007-webgpu-feature-detection-and-fallback-switch.md) / [PRIMORDIS-TASK-015](./PRIMORDIS-TASK-015-cross-platform-backend-selection-and-reduced-mode-ux.md)); this task only defines the interface and ships the fake so selection can later inject a concrete backend behind the same provider.

### Versioning (if Flutter/native code changed)
- [ ] Version bumped in `pubspec.yaml` and the app config constant (`PrimordisConfig.version`); semver.

### Test Coverage
- [ ] New/modified Dart has unit/widget tests; `flutter test` passes; `flutter analyze` zero warnings. Specifically: model construction/`copyWith`, matrix asymmetry preserved, deterministic seeding (same seed -> identical particles/matrices/colors), marshalling round-trip/layout assertions, frame-loop ordering against `FakeSimBackend`, and provider state transitions.

## Implementation Notes

- **This is the shared core; the backends are not.** Per [PRIMORDIS-ADR-001](../adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md), the UI, sim params (Freezed), seeding, frame loop, and param marshalling are identical across web and macOS and live in the standard Dart layers. The GPU/FFI/JS-interop/WGSL code is explicitly **out of these layers** and is implemented in the backend tasks behind this interface — keep this file free of any `dart:js_interop`, `dart:ffi`, WGSL, or platform imports so it compiles identically on web and native.
- **Matrix shape drives the kernel.** The three 32x32 float32 matrices are asymmetric (`forces[i][j] != forces[j][i]`), encoding directed per-type-pair behavior exactly as `Primordis.py` does. The interaction physics that consumes them (3x3 toroidal neighbor scan, minimum-image distance, short-range repulsion when `dist < min_dist` weighted 5x using `abs(force)`, linear-falloff signed attraction when `dist < radius`, Euler integrate `v += f*dt; v *= friction; p += v*dt`, then wrap) is implemented in [PRIMORDIS-TASK-003](./PRIMORDIS-TASK-003-port-simulation-to-wgsl-compute-kernel.md) (GPU) and the CPU tasks — but the *marshalled layout* of these matrices must be fixed here so all consumers agree.
- **Marshalling is the contract between this task and every backend.** Define field order and alignment with `std430`/uniform consumption in mind ([PRIMORDIS-ADR-003](../adr/PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md)): matrices flattened row-major into `Float32List`; bin-count buffer as `Uint32List` (the web WebGPU and Dawn backends bind it as `atomic<u32>` and read via `atomicLoad`); particle SoA (`Float32List` positions/velocities/types) sized for 24k. Document any padding so the WGSL struct in TASK-003 and the FFI/WebGPU buffers in TASK-004/TASK-011 map 1:1.
- **The three sliders are params, not a separate channel.** Attraction K, Repulsion K, and Drift/friction live in `SimParams` and flow through `setParams` -> a small uniform block; [PRIMORDIS-TASK-006](./PRIMORDIS-TASK-006-sliders-to-uniforms-and-ui-chrome.md) wires the Flutter sliders to mutate these via Riverpod and marshals them to the live backend's uniforms.
- **Frame loop is backend-agnostic.** It sequences `setParams` (on change) -> `step(dt)` -> `present()`. The *meaning* of `step`/`present` differs per backend (3 compute passes + point render on GPU; counting-sort + `drawRawPoints` on CPU; IOSurface `Texture` present on macOS), but the ordering and the dt-driven Euler tick contract are shared and tested here against `FakeSimBackend`.
- **Determinism scope.** Seeding is deterministic for reproducible parity baselines; the simulation evolution is not bit-exact (original GPU binning is single-buffered with a known scatter race). Tests assert seed determinism, never frame-N bit-equality.
- **Riverpod discipline.** Providers use plain `Ref`; no `ChangeNotifier`/`setState` for sim state. The active backend is exposed behind a provider so [PRIMORDIS-TASK-007](./PRIMORDIS-TASK-007-webgpu-feature-detection-and-fallback-switch.md) / [PRIMORDIS-TASK-015](./PRIMORDIS-TASK-015-cross-platform-backend-selection-and-reduced-mode-ux.md) can swap the concrete implementation (GPU vs CPU, reduced-mode particle count) without UI changes.
- **Accessibility forward-hook.** The frame loop must be pausable (a paused/static state) so the reduced-motion obligation in [PRIMORDIS-ADR-006](../adr/PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md) can be honored by [PRIMORDIS-TASK-015](./PRIMORDIS-TASK-015-cross-platform-backend-selection-and-reduced-mode-ux.md) / [PRIMORDIS-TASK-018](./PRIMORDIS-TASK-018-test-coverage-and-accessibility.md) — expose a pause flag in run state now.

## Testing
- [ ] Unit: `SimParams`/`ParticleType`/`SimSeed` construction, equality, `copyWith`; matrix asymmetry assertion (`forces[i][j] != forces[j][i]` for representative pairs).
- [ ] Unit: `SimSeeder` determinism — identical `SimSeed` yields byte-identical particles, matrices, and colors across two runs.
- [ ] Unit: marshalling layout — packed `Float32List`/`Uint32List` has the documented offsets/length for 32 types and 24k particles; bin-count buffer typed `Uint32List`; round-trip where applicable.
- [ ] Unit: `frame_loop` invokes `setParams` only on param change, then `step(dt)` then `present()` each tick, in order, against `FakeSimBackend`; pause flag suppresses stepping.
- [ ] Unit: Riverpod providers — slider mutation updates `SimParams`; backend provider returns the injected (fake) backend; run-state transitions (running/paused).
- [ ] `dart run build_runner build --delete-conflicting-outputs` is clean; `flutter analyze` zero warnings; `flutter test` passes.

## Related
- PRD: [PRIMORDIS-PRD-001](../prd/PRIMORDIS-PRD-001-flutter-web-and-macos-port.md)
- ADR: [PRIMORDIS-ADR-001](../adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md), [PRIMORDIS-ADR-003](../adr/PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md), [PRIMORDIS-ADR-002](../adr/PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md), [PRIMORDIS-ADR-006](../adr/PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md)
- Depends on: [PRIMORDIS-TASK-001](./PRIMORDIS-TASK-001-project-scaffold-and-build-config.md)
- Blocks: [PRIMORDIS-TASK-003](./PRIMORDIS-TASK-003-port-simulation-to-wgsl-compute-kernel.md), [PRIMORDIS-TASK-004](./PRIMORDIS-TASK-004-web-webgpu-backend-js-interop.md), [PRIMORDIS-TASK-006](./PRIMORDIS-TASK-006-sliders-to-uniforms-and-ui-chrome.md), [PRIMORDIS-TASK-008](./PRIMORDIS-TASK-008-dart-wasm-cpu-fallback-backend.md), [PRIMORDIS-TASK-014](./PRIMORDIS-TASK-014-native-cpu-isolate-fallback-backend.md)
