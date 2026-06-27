# PRIMORDIS-TASK-014: Native multi-core CPU isolate fallback backend

**Status:** Todo
**Priority:** Medium
**Created:** 2026-06-27
**Updated:** 2026-06-27

## Description

Implement the native (macOS-first) **CPU fallback** `SimBackend` for when GPU init fails — tier **T3** in [PRIMORDIS-ADR-006](../adr/PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md). Unlike the web CPU fallback ([PRIMORDIS-TASK-008](./PRIMORDIS-TASK-008-dart-wasm-cpu-fallback-backend.md)), which is single-thread Dart→WASM because the web has no real shared memory, native Dart has **real isolates**. This backend runs the particle physics across multiple isolates that share **one** particle buffer allocated as native memory via `dart:ffi` (`calloc`'d `Pointer`), with the integer **address** — not the `Pointer` object — passed to each worker isolate, and concurrency guarded by a `package:native_synchronization` mutex/barrier. This yields true multi-core shared memory.

The reference workload is 24,000 particles / 32 types over a 1080x720 toroidal world; this backend's realistic ceiling is **~10-14k @ 60fps** (and an estimated ~16-40 fps at 24k). These numbers are **low-confidence / extrapolated** (Dart AOT SIMD is currently broken, so there is no vectorization) and **must be benchmarked in this task before any count is committed to the UI or docs**. T3 is graceful degradation only; the GPU tiers (T1/T2) remain the real win. The backend reuses the exact same shared sim params, seeding, and frame loop from [PRIMORDIS-TASK-002](./PRIMORDIS-TASK-002-simbackend-interface-and-shared-sim-model.md) and the same deterministic **sequential counting-sort binning** physics from the Dart-WASM CPU backend ([PRIMORDIS-TASK-008](./PRIMORDIS-TASK-008-dart-wasm-cpu-fallback-backend.md)) — only the concurrency/sharing layer is new.

## Scope

**Area:** FFI/Native
**Files/Dirs:**
- `lib/sim/backends/native_cpu/native_cpu_sim_backend.dart` — the `SimBackend` implementation (orchestrator isolate)
- `lib/sim/backends/native_cpu/isolate_worker.dart` — worker isolate entrypoint (operates on the shared buffer by address)
- `lib/sim/backends/native_cpu/shared_buffer.dart` — `dart:ffi` `calloc` allocation, SoA layout, address sharing
- `lib/sim/backends/native_cpu/cpu_physics.dart` — counting-sort binning + interaction/integration (shared with TASK-008 where possible)
- `test/sim/native_cpu/native_cpu_sim_backend_test.dart`, `test/sim/native_cpu/shared_buffer_test.dart`, `test/sim/native_cpu/cpu_physics_test.dart`
- `pubspec.yaml` (add `native_synchronization`; `ffi`/`package:ffi` for `calloc`)
- Note: this FFI/isolate code lives **outside** the standard feature/data/domain layers, behind the `SimBackend` interface, by design (see ADR-001).

## Acceptance Criteria

- [ ] A `NativeCpuSimBackend` implements the shared `SimBackend` interface (TASK-002) and is selectable on native targets; the UI/sliders/params layer is unchanged and never branches on it.
- [ ] The particle SoA buffer (positions, velocities, types — `Float32List`/`Int32List`-shaped views over native memory) is allocated once via `calloc` as a `dart:ffi` `Pointer`; its integer **address** (not the `Pointer` object) is what is sent to worker isolates.
- [ ] N worker isolates (N derived from available cores) reconstruct typed-data views over the shared native memory from the passed address and operate in place — no per-frame buffer copy between isolates.
- [ ] A `package:native_synchronization` mutex/barrier coordinates the per-frame phases (bin → interact/integrate → wrap) so isolates do not read a half-written buffer; no data races.
- [ ] Binning uses **deterministic sequential counting-sort** over the 11x7 = 77-bin grid (bin size = MAX_RADIUS = 96), matching the CPU contract — **not** atomic scatter (atomic scatter is the GPU path only).
- [ ] Physics matches the reference: 3x3 toroidal neighbor scan, minimum-image distance, short-range repulsion (`dist < min_dist`, 5x-weighted, `abs(force)`) vs linear-falloff signed attraction (`dist < radius`), Euler integrate (`v += f*dt; v *= friction; p += v*dt`) and toroidal wrap.
- [ ] Seeding (32 types, asymmetric 32x32 force/min_distance/radius matrices, random colors) is identical to the GPU and web-CPU backends; only live particle count differs.
- [ ] Native memory is freed (`calloc.free`) on backend dispose; no leak across backend re-creation; isolates are torn down cleanly.
- [ ] **Benchmark gate:** a documented benchmark records sustained fps at several particle counts on at least one reference Mac; the committed default/max T3 count is the benchmarked ~60fps ceiling, not the extrapolated estimate. The provisional ~10-14k figure is replaced with a measured number before it appears in UI or docs.
- [ ] Reduced-motion: the backend honors pause (frame loop halt holding the last frame) so it composites correctly with the accessibility pause state (see [PRIMORDIS-TASK-018](./PRIMORDIS-TASK-018-test-coverage-and-accessibility.md)).

### Versioning (if Flutter/native code changed)
- [ ] Version bumped in pubspec.yaml and the app config constant (`PrimordisConfig`); semver

### Test Coverage
- [ ] New/modified Dart has unit/widget tests; flutter test passes; flutter analyze zero warnings

## Implementation Notes

- **Backend boundary.** This is one more implementation behind `SimBackend` (ADR-001). All FFI, isolate, and synchronization code is confined here; the Riverpod providers, Freezed param models, and Material 3 UI from TASK-002/006 stay standards-compliant and backend-agnostic. The UI layer must not import `dart:ffi` or `dart:isolate`.
- **Why native differs from web.** Web "isolates" are web workers that **copy** data, and `SharedArrayBuffer` needs COOP/COEP cross-origin isolation (see [PRIMORDIS-ADR-007](../adr/PRIMORDIS-ADR-007-web-build-and-cross-origin-isolation.md)) that plain static hosts cannot guarantee — so the web CPU tier is single-thread (TASK-008). Native Dart has **real** isolates plus `dart:ffi`, which is exactly what makes a shared-memory multi-core path possible here.
- **Sharing the buffer correctly.** Allocate the SoA with `calloc<Float>(...)` / `calloc<Int32>(...)`. `Pointer` objects are **not** sendable across isolates, but their `.address` (an `int`) is. Send the address (plus lengths/strides) in the spawn message; each worker does `Pointer<Float>.fromAddress(addr)` and `.asTypedList(len)` to get a view aliasing the **same** physical memory. Treat the address as a raw, unmanaged handle — lifetime is owned solely by the orchestrator isolate.
- **Synchronization.** Use a `package:native_synchronization` `Mutex`/barrier to fence the frame phases. The counting-sort bin pass must complete (all bins/offsets written) before any worker enters the interaction pass; integration must complete before wrap/next-frame. A barrier per phase boundary is the simplest correct model; partition particles by index range across workers.
- **Determinism / faithfulness.** Use sequential counting-sort binning (deterministic), per the CPU contract in ADR-006. Note that "faithful" means visually/statistically equivalent to `Primordis.py`, never bit-exact — the original GPU binning is itself nondeterministic with a single-buffered race (ADR-001/006). Parity is validated against the Python reference in [PRIMORDIS-TASK-009](./PRIMORDIS-TASK-009-parity-test-harness-vs-python-reference.md).
- **Performance reality.** Dart AOT SIMD is currently broken, so there is no vectorization; the ceiling is bounded by scalar multi-core throughput. The ~10-14k @ 60fps / ~16-40 fps @ 24k figures are **estimates** and a hard benchmark gate (above) exists precisely so the project never promises an unverified number. Do not raise the T3 count ceiling beyond what is measured.
- **Selection.** This backend is only chosen when the native GPU path (Dawn/wgpu-over-Metal FFI, [PRIMORDIS-TASK-011](./PRIMORDIS-TASK-011-macos-target-dawn-wgpu-ffi-backend.md); MSL fallback, [PRIMORDIS-TASK-013](./PRIMORDIS-TASK-013-macos-metal-msl-compute-plugin-fallback.md)) fails device/pipeline creation. The unified detection+selection logic that demotes to T3 lives in [PRIMORDIS-TASK-015](./PRIMORDIS-TASK-015-cross-platform-backend-selection-and-reduced-mode-ux.md); this task provides the backend it switches to.

## Testing

- [ ] Unit test `shared_buffer.dart`: round-trip an address through a worker isolate and confirm writes by the worker are observed by the orchestrator (proves shared, not copied, memory).
- [ ] Unit test `cpu_physics.dart`: counting-sort binning produces correct per-bin counts/offsets and is order-deterministic for a fixed seed; a known small configuration integrates to expected positions.
- [ ] Concurrency test: run the multi-isolate frame loop on a fixed seed and assert no race-induced corruption (compare against a single-threaded reference run of the same physics; results equal within determinism guarantees).
- [ ] Lifetime test: allocate/free across repeated backend create/dispose cycles; assert no native memory growth and clean isolate shutdown.
- [ ] Benchmark (recorded, not a CI gate): sustained fps at multiple counts on a reference Mac; capture the measured ~60fps ceiling and feed it into the T3 default/max count.
- [ ] Manual: force GPU init failure, confirm the app demotes to this backend and runs at the benchmarked count with the reduced-mode indicator (TASK-015) shown.
- [ ] `flutter analyze` zero warnings; `flutter test` passes.

## Related
- PRD: [PRIMORDIS-PRD-001](../prd/PRIMORDIS-PRD-001-flutter-web-and-macos-port.md)
- ADR: [PRIMORDIS-ADR-006](../adr/PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md) (primary — CPU fallback tiers), [PRIMORDIS-ADR-001](../adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md) (SimBackend boundary), [PRIMORDIS-ADR-004](../adr/PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md) (the GPU path this degrades from)
- Depends on: [PRIMORDIS-TASK-002](./PRIMORDIS-TASK-002-simbackend-interface-and-shared-sim-model.md), [PRIMORDIS-TASK-008](./PRIMORDIS-TASK-008-dart-wasm-cpu-fallback-backend.md)
- Blocks: [PRIMORDIS-TASK-015](./PRIMORDIS-TASK-015-cross-platform-backend-selection-and-reduced-mode-ux.md)
