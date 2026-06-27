# PRIMORDIS-TASK-008: Pure Dart→WASM CPU fallback SimBackend

**Status:** Todo
**Priority:** High
**Created:** 2026-06-27
**Updated:** 2026-06-27

## Description

Implement a pure-Dart, single-threaded CPU implementation of the `SimBackend` interface (from [PRIMORDIS-TASK-002](./PRIMORDIS-TASK-002-simbackend-interface-and-shared-sim-model.md)) that runs entirely inside the Dart→WASM (dart2wasm) build, with **no GPU and no FFI/JS-interop**. This is the **web fallback tier (T4)** selected when `navigator.gpu` is absent or adapter/device acquisition fails (per [PRIMORDIS-ADR-006](../adr/PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md)), so that WebGPU-less browsers (Firefox Linux/Android, Intel Macs, pre-26 Safari/iOS) still render a working simulation instead of a blank screen.

This backend reproduces the *behavior* of the reference 3+1-pass GPU pipeline on the CPU, but with two deliberate structural differences forced by the absence of compute/atomics on this tier:

1. **No atomic scatter binning.** The GPU path bins particles with `atomicAdd` scatter; the CPU path uses a **sequential counting-sort** binning pass (count per bin → prefix-sum offsets → scatter into sorted order), which is deterministic and cache-friendly on a single thread.
2. **Single `Canvas` draw call.** All particles are rendered with **one** `Canvas.drawRawPoints` (or `drawVertices`) call from a packed `Float32List`, never per-point draws.

The hard performance ceiling of this tier is **~3-4k particles @ 60fps** (24k ≈ 1-2.5 fps single-thread); the backend must honor the per-tier particle-count policy and not pretend to run 24k. This task also unblocks [PRIMORDIS-TASK-007](./PRIMORDIS-TASK-007-webgpu-feature-detection-and-fallback-switch.md) (the web feature-detect switch) and provides the deterministic counting-sort reference used by the parity harness ([PRIMORDIS-TASK-009](./PRIMORDIS-TASK-009-parity-test-harness-vs-python-reference.md)) and the native isolate CPU backend ([PRIMORDIS-TASK-014](./PRIMORDIS-TASK-014-native-cpu-isolate-fallback-backend.md)), which reuse this counting-sort core.

## Scope

**Area:** Flutter
**Files/Dirs:**
- `lib/sim/backends/cpu_wasm_backend.dart` — the `SimBackend` implementation (CPU/WASM tier)
- `lib/sim/cpu/cpu_sim_step.dart` — the per-frame physics step (clear → counting-sort bin → interaction+integrate)
- `lib/sim/cpu/counting_sort_binning.dart` — deterministic counting-sort binning over the uniform grid
- `lib/sim/cpu/particle_soa.dart` — `Float32List`-backed structure-of-arrays particle store (positions, velocities) + render buffer
- `lib/sim/render/cpu_points_painter.dart` — `CustomPainter` issuing a single `drawRawPoints`/`drawVertices`
- `test/sim/cpu/` — unit tests for binning, step, and SoA
- Shared (consumed, not created here): `lib/sim/sim_backend.dart`, the Freezed sim-param/seed models, and Riverpod providers from TASK-002

> **Layering note (expected, per house standards):** the CPU physics core (`lib/sim/cpu/`) is a numerical/algorithmic layer that lives **outside** the standard feature/data/domain layers, behind the `SimBackend` interface — the same exemption the GPU/FFI/JS-interop backends get. The Flutter UI layer that drives it stays fully standards-compliant (Riverpod plain `Ref`, Freezed params, Material 3). The CPU core itself must remain UI-agnostic and contain no `setState`/widget code.

## Acceptance Criteria

- [ ] A `CpuWasmBackend` class implements the full `SimBackend` interface (init, seed, step/frame, apply-params, particle-count getter, dispose) defined in TASK-002, with **zero** `dart:ffi`, `dart:js_interop`, `package:web`, or GPU references.
- [ ] The backend compiles and runs under `flutter build web --wasm` (dart2wasm); no legacy interop (`dart:html`/`dart:js_util`) anywhere in its dependency tree (per [PRIMORDIS-ADR-007](../adr/PRIMORDIS-ADR-007-web-build-and-cross-origin-isolation.md)).
- [ ] Particles are stored as a **structure-of-arrays** in `Float32List`s (positions, velocities, per-particle type index), not lists of objects; allocation is reused frame-to-frame (no per-frame GC churn).
- [ ] Seeding uses the **shared** seed/params from TASK-002 (32 types; asymmetric 32×32 force / min-distance / radius matrices; per-type random colors) — identical seeding logic to the GPU tiers; only the live particle count differs.
- [ ] Physics reproduces the reference per-frame step: short-range repulsion when `dist < min_dist` (5× weighted, `abs(force)`) vs linear-falloff attraction when `dist < radius` (signed), over a **toroidal** world (1080×720, wrap on both axes) using **minimum-image** distance, with Euler integration (`v += f*dt; v *= friction; p += v*dt`) and position wrap.
- [ ] Neighbor search uses the same uniform spatial grid as the reference: **11×7 = 77 bins**, bin size = `MAX_RADIUS = 96`, scanning the **3×3** neighbor bins with toroidal bin-index wrap.
- [ ] Binning is a **deterministic sequential counting sort** (count per bin → exclusive prefix-sum → stable scatter), **not** atomic scatter; given identical seed + params the binning output is bit-stable across runs on this tier.
- [ ] The 3 live params (Attraction K, Repulsion K, Drift/friction) are read from shared params each frame and affect the step immediately (no rebuild/reseed required to change a slider).
- [ ] Rendering issues exactly **one** `Canvas.drawRawPoints` (or `drawVertices`) call per frame from a packed `Float32List`; point size ≈ 2 px to match the reference `gl_PointSize=2`; per-type colors applied via the single-draw path (vertex colors or per-type batching limited to ≤ 32 draws max, with `drawRawPoints` preferred).
- [ ] The backend honors the **T4 particle-count policy**: default/max ≈ 3-4k (sourced from the tier policy in ADR-006), and never silently runs 24k. The exact default is a named constant, not a magic number.
- [ ] Backend exposes a `pause()`/static state so the **reduced-motion** accessibility requirement can be met (the whole canvas is motion); when paused, it holds the last frame and does no stepping.
- [ ] Reports an honest `actualParticleCount` to the UI so the "reduced mode" indicator (ADR-006 / TASK-015) can display the true count.

### Versioning (if Flutter/native code changed)

- [ ] Version bumped in `pubspec.yaml` and the app config constant (`PrimordisConfig`/`AppConfig`); semver.

### Test Coverage

- [ ] New/modified Dart has unit/widget tests; `flutter test` passes; `flutter analyze` zero warnings.
- [ ] Counting-sort binning has a unit test asserting determinism (same input → same bin order) and correct bin membership over the toroidal grid including wrap-around bins.
- [ ] A small-N step test asserts a known repulsion case (two particles closer than `min_dist`) pushes them apart and a known attraction case pulls them together, with toroidal minimum-image honored across the world seam.
- [ ] A render-buffer test asserts exactly one draw call is produced and the `Float32List` length matches `2 * particleCount`.

## Implementation Notes

- **Why counting sort, not atomics:** this tier has no compute shader and no atomics; replicating the GPU `atomicAdd` scatter on a single thread buys nothing. A sequential counting sort is deterministic (unlike the reference GPU binning, which is intentionally nondeterministic / single-buffered with a known scatter race), which is exactly what we want for the parity harness baseline (TASK-009). Faithful here means **visually/statistically equivalent**, never bit-exact vs the GPU (see ADR-006 and the PRD non-goals).
- **Counting-sort shape:** maintain a `binCount[77]` (`Int32List`), zero it (pass 1 "clear"), increment per particle from its bin index (pass 2 "count"), exclusive prefix-sum into `binStart[77]`, then scatter particle indices into a sorted index array using a moving cursor. `MAX_BIN_PARTICLES=512` from the reference becomes a soft expectation here, not a hard cap — counting sort has no fixed per-bin ceiling, so the 512 cap is **not** ported (note this divergence in code comments and to the parity harness).
- **Bin index:** `bx = floor(x / 96)` clamped/wrapped to `[0,11)`, `by = floor(y / 96)` wrapped to `[0,7)`, `bin = by*11 + bx`. The 3×3 neighbor scan wraps bin indices toroidally (`(bx+dx+11)%11`, `(by+dy+7)%7`).
- **Minimum-image distance:** for each pair, `dx = x_j - x_i`; if `dx > W/2` subtract `W`, if `dx < -W/2` add `W` (same for `dy` with `H`); use the wrapped delta for distance and force direction.
- **Force law (match reference exactly):** if `dist < min_dist[i][j]`: short-range repulsion weighted 5× using `abs(force[i][j])` directed away; else if `dist < radius[i][j]`: linear-falloff attraction using the **signed** `force[i][j]`. Matrices are **asymmetric** (`i→j ≠ j→i`) — index `[i][j]` with `i` = source particle's type acting on neighbor `j`, matching the reference orientation; verify orientation against `Primordis.py` and the parity harness.
- **Integration:** `v += f*dt; v *= friction; p += v*dt; p = wrap(p)`, Euler, same `dt` as the shared frame loop (TASK-002). `friction` is the Drift slider; Attraction K / Repulsion K scale the attraction/repulsion terms respectively.
- **Rendering:** prefer `Canvas.drawRawPoints(PointMode.points, Float32List, Paint()..strokeWidth=2)`. If per-type color is required and `drawRawPoints` can't carry per-vertex color on the target renderer, fall back to `drawVertices` with a per-vertex color list, still **one** draw call. Never iterate particles issuing `drawCircle`/`drawPoints` per particle. Drive the painter from a `CustomPainter` whose `paint` only blits the prebuilt buffer; the physics step runs in the frame loop (TASK-002), not in `paint`.
- **dart2wasm constraints:** keep this backend free of `dart:html`/`dart:js_util` (forbidden under `--wasm`, ADR-007). Use only `dart:typed_data`, `dart:math`, `dart:ui`. No `SharedArrayBuffer` / web-worker path here — this tier is **single-threaded by necessity** (web has no real shared-memory isolates; ADR-006). Multi-core CPU only exists on native (TASK-014).
- **Reuse contract:** factor the counting-sort + step into platform-neutral functions in `lib/sim/cpu/` so TASK-014 (native isolates over an FFI `calloc`'d shared buffer) can call the *same* counting-sort/step logic over a shared buffer instead of reimplementing it. Keep the SoA buffer abstraction injectable (own `Float32List` here; FFI-backed memory there).
- **Particle count:** read default/max from the ADR-006 tier policy surfaced via shared params; expose as `PrimordisConfig` constants. Do not hardcode 24k.

## Testing

- [ ] `flutter test` green; `flutter analyze` zero warnings.
- [ ] Build the WASM target (`flutter build web --wasm`) and confirm the CPU backend compiles and the app boots into the CPU tier when WebGPU is force-disabled (manual `navigator.gpu` override or test seam from TASK-007).
- [ ] On a WebGPU-less browser profile, confirm a visible, animating cluster simulation at the T4 default count holds ~60fps and forms/maintains clusters resembling the reference behavior (qualitative; statistical parity is TASK-009).
- [ ] Toggle each slider (Attraction K, Repulsion K, Drift) and confirm immediate, sensible behavioral change with no reseed.
- [ ] Trigger pause/reduced-motion and confirm the canvas holds a static frame and stepping stops.
- [ ] Profile a frame to confirm exactly one `drawRawPoints`/`drawVertices` call and no per-frame heap growth (reused `Float32List`s).

## Related

- PRD: [PRIMORDIS-PRD-001](../prd/PRIMORDIS-PRD-001-flutter-web-and-macos-port.md)
- ADR: [PRIMORDIS-ADR-006 — CPU fallback tiers and feature detection](../adr/PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md); [PRIMORDIS-ADR-001 — Cross-platform architecture & SimBackend](../adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md); [PRIMORDIS-ADR-007 — Web build & cross-origin isolation](../adr/PRIMORDIS-ADR-007-web-build-and-cross-origin-isolation.md)
- Depends on: [PRIMORDIS-TASK-002 — SimBackend interface & shared sim model](./PRIMORDIS-TASK-002-simbackend-interface-and-shared-sim-model.md)
- Blocks: [PRIMORDIS-TASK-007 — WebGPU feature detection & fallback switch](./PRIMORDIS-TASK-007-webgpu-feature-detection-and-fallback-switch.md); [PRIMORDIS-TASK-009 — Parity test harness vs Python reference](./PRIMORDIS-TASK-009-parity-test-harness-vs-python-reference.md); [PRIMORDIS-TASK-014 — Native CPU isolate fallback backend](./PRIMORDIS-TASK-014-native-cpu-isolate-fallback-backend.md); [PRIMORDIS-TASK-018 — Test coverage and accessibility](./PRIMORDIS-TASK-018-test-coverage-and-accessibility.md)
