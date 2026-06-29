# PRIMORDIS-TASK-003: Port the simulation to a single WGSL compute kernel

**Status:** Complete
**Priority:** Critical
**Created:** 2026-06-27
**Updated:** 2026-06-29

## Description

Port the entire Primordis physics step from the reference GLSL `#version 430` compute shaders (`Primordis.py`, pygame + moderngl + numpy) to **one** WGSL compute kernel source that will be reused unchanged on web (browser WebGPU, [PRIMORDIS-TASK-004](./PRIMORDIS-TASK-004-web-webgpu-backend-js-interop.md)) and native macOS (Dawn/wgpu-over-Metal via FFI, [PRIMORDIS-TASK-011](./PRIMORDIS-TASK-011-macos-target-dawn-wgpu-ffi-backend.md)). This is the foundational physics task: every GPU backend consumes this exact source, so the port must be faithful, self-contained, and validated **standalone** at 24,000 particles / 32 types before any backend integration begins.

The reference simulation runs the full physics in three GPU compute passes per frame, plus a point-render pass:

1. **Clear** the per-bin counters.
2. **Bin** 24,000 particles into a uniform 11x7 = 77-cell toroidal spatial grid (bin size = `MAX_RADIUS` = 96, `MAX_BIN_PARTICLES` = 512 cap) via an `atomicAdd` scatter whose returned old value is each particle's write offset within its bin.
3. **Interaction + integrate**: each particle scans its 3x3 neighbour bins (toroidal, minimum-image distance), applies short-range repulsion (`dist < min_dist`, weighted 5x, `abs(force)`) versus linear-falloff signed attraction (`dist < radius`), then Euler-integrates (`v += f*dt; v *= friction; p += v*dt`) and wraps to the torus.

The world is 1080x720, toroidal on both axes. Per-type-pair parameters are three asymmetric 32x32 `float32` matrices — `forces` (signed), `min_distances`, `radii` — plus a per-type colour. The three live sliders (Attraction K, Repulsion K, Drift/friction) plus `dt`, world dims, grid dims and counts are marshalled into a uniform buffer. At 24k/32-types this is ~67M particle-pair tests per frame — trivial on a GPU.

The WGSL kernel **source** lives in the shared layer per [PRIMORDIS-ADR-001](../adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md) and [PRIMORDIS-ADR-003](../adr/PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md); it is the single source of truth from which the optional MSL fallback ([PRIMORDIS-TASK-013](./PRIMORDIS-TASK-013-macos-metal-msl-compute-plugin-fallback.md)) is derived. Determinism is explicitly **not** a goal: the original binning is already nondeterministic (single-buffered atomic-scatter race), so "faithful" means visually/statistically equivalent, never bit-exact.

## Scope

**Area:** Shader

**Files/Dirs:**

- `lib/sim/kernel/primordis.wgsl` — the canonical WGSL compute kernel (clear / bin / interaction+integrate) plus the point-render shader stages. This is shader source loaded as a string asset; it deliberately lives **outside** the standard Riverpod/Freezed feature/data/domain layers (see Implementation Notes).
- `lib/sim/kernel/kernel_source.dart` — thin Dart accessor that loads/exposes the WGSL source string (e.g. as a `rootBundle` asset or `const` string) and the canonical buffer-binding/dispatch-geometry constants (`WORKGROUP_SIZE`, bind-group indices, buffer sizes) so both backends share them.
- `lib/sim/kernel/buffer_layout.dart` — shared declarations of buffer byte layouts and `std430`/WGSL-`storage` struct sizes (particle SoA/AoS layout, bin-count array, bin-index array, params/uniforms), so web and native marshal identically.
- `pubspec.yaml` — register `lib/sim/kernel/primordis.wgsl` as a Flutter asset.
- `test/sim/kernel/` — standalone validation harness and fixtures (see Testing).
- Reference only (read, do not modify): `Primordis.py` at repo root.

## Acceptance Criteria

- [x] A single WGSL compute kernel source exists at `lib/sim/kernel/primordis.wgsl` and implements all three compute passes plus the point-render stages; there is exactly one copy of the binning and interaction logic in the codebase.
- [x] Pass 1 (clear) zeroes all 77 bin counters before binning each frame.
- [x] Pass 2 (bin) scatters all 24,000 particles into the 11x7 = 77-cell grid using `atomicAdd(&binCounts[cell], 1u)`, with the returned previous value used as the per-bin write offset; bin cell size = `MAX_RADIUS` = 96; per-bin writes are capped at `MAX_BIN_PARTICLES` = 512 and over-cap particles are dropped from the bin index exactly as the reference does (no out-of-bounds writes).
- [x] Pass 3 (interaction+integrate) scans the 3x3 toroidal neighbour bins, computes **minimum-image** distance on the 1080x720 torus, applies short-range repulsion for `dist < min_dist` (weighted 5x, using `abs(force)`) and linear-falloff signed attraction for `dist < radius`, then Euler-integrates (`v += f*dt; v *= friction; p += v*dt`) and wraps both axes.
- [x] The three 32x32 asymmetric matrices (`forces`, `min_distances`, `radii`) are indexed `[typeOf(self)][typeOf(other)]` (i->j, not symmetric) and produce the asymmetric behaviour the reference exhibits.
- [x] The GLSL->WGSL mapping is honoured exactly per [PRIMORDIS-ADR-003](../adr/PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md): SSBO -> `var<storage, read_write>`; `atomicAdd` -> `atomicAdd(&...)`; bin counters declared `atomic<u32>` and read **only** via `atomicLoad` (no aliased non-atomic access to atomic memory); `local_size_x` -> `@workgroup_size`; `gl_GlobalInvocationID` -> `@builtin(global_invocation_id)`.
- [x] All uniforms (Attraction K, Repulsion K, Drift/friction, `dt`, world dims 1080x720, grid dims 11x7, particle count 24000, type count 32) are read from a single uniform buffer whose layout is declared once in `buffer_layout.dart` and reused by both backends.
- [x] The kernel runs standalone (via the test harness) at 24,000 particles / 32 types and produces visually/statistically equivalent cluster formation and drift to `Primordis.py` (handed off to the parity harness in [PRIMORDIS-TASK-009](./PRIMORDIS-TASK-009-parity-test-harness-vs-python-reference.md)).
- [x] WGSL atomics rules are not violated (no mixed atomic/non-atomic access; all atomic targets `atomic<u32>`), so the source compiles cleanly under both the Naga (browser/wgpu) and Tint (Dawn) translators; any translator-specific divergence is flagged for [PRIMORDIS-TASK-017](./PRIMORDIS-TASK-017-atomics-parity-validation-dawn-vs-browser.md) rather than worked around silently.
- [x] The WGSL source contains no backend-specific code (no JS-interop, no FFI, no device/pipeline creation); it is pure kernel source consumable by any WebGPU runtime.

### Versioning (if Flutter/native code changed)

- [x] Version bumped in `pubspec.yaml` and the `PrimordisConfig` app config constant; semver (minor bump — new simulation core).

### Test Coverage

- [x] New/modified Dart (`kernel_source.dart`, `buffer_layout.dart`, the harness glue) has unit tests; `flutter test` passes; `flutter analyze` zero warnings. (The `.wgsl` shader is validated by the standalone harness in Testing, not by Dart unit tests, since it is non-Dart source.)

## Implementation Notes

- **Source-of-truth contract.** This kernel is the normative spec per [PRIMORDIS-ADR-003](../adr/PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md). Both the web backend (TASK-004) and the macOS Dawn/wgpu backend (TASK-011) hand this exact string to their respective WebGPU runtimes; only device/pipeline creation and dispatch differ between them. The MSL fallback (TASK-013) must be a faithful transliteration of **this** file. Keep the source backend-agnostic so it can be diffed against the GLSL original and against MSL.
- **Layer placement is intentionally non-standard.** Per [PRIMORDIS-ADR-001](../adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md), shader source and the buffer-layout constants are GPU artefacts and do **not** belong in the Riverpod/Freezed feature/data/domain structure. Keeping them under `lib/sim/kernel/` (consumed only via the `SimBackend` interface) is expected and explicitly sanctioned; the UI/state layers stay fully standards-compliant. The Freezed sim-param/colour models and seeding live in the shared model layer from [PRIMORDIS-TASK-002](./PRIMORDIS-TASK-002-simbackend-interface-and-shared-sim-model.md) and are marshalled **into** this kernel's buffers, not defined here.
- **GLSL -> WGSL atomic-scatter detail.** In GLSL the binning pass calls `atomicAdd(binCounts[cell], 1)` and uses the **returned old value** as the slot offset into the bin's particle-index list. WGSL's `atomicAdd(&binCounts[cell], 1u)` returns the previous value identically. Declare `binCounts` as `array<atomic<u32>, 77>` and read it back (in pass 3, to know how many particles are in each bin) with `atomicLoad`. Never read `binCounts` through a non-atomic alias — that is undefined in WGSL and is exactly the class of bug Tint/Naga diverge on.
- **Binning geometry.** Grid is 11x7 = 77 cells over the 1080x720 world; cell size = `MAX_RADIUS` = 96 (note 11*96 = 1056 < 1080 and 7*96 = 672 < 720, so cell indexing must clamp/wrap to the 11x7 range exactly as the reference does — preserve the reference's cell-index computation rather than re-deriving it). `MAX_BIN_PARTICLES` = 512 caps per-bin storage; a particle whose bin is already full is simply not added to the index list (it still exists and still moves, it is just invisible to neighbours that frame), matching the reference's single-buffered behaviour.
- **3x3 toroidal neighbour scan + minimum-image.** For each of the 9 neighbour cells, wrap the cell index modulo (11, 7). When computing the displacement to a neighbour particle, apply the minimum-image convention on each axis (1080 / 720): if `|d| > world/2`, fold by `+/- world`. This is what makes attraction/repulsion wrap correctly across the torus seam.
- **Force model (preserve exactly).** For `dist < min_dist`: short-range repulsion, magnitude scaled by `abs(force)` and weighted 5x, always repulsive. For `min_dist <= dist < radius`: signed attraction with linear falloff toward `radius`. Beyond `radius`: no contribution. `force`, `min_dist`, `radius` are read from `forces[selfType][otherType]`, `min_distances[selfType][otherType]`, `radii[selfType][otherType]`. The three sliders scale these (Attraction K and Repulsion K multiply the respective terms; Drift sets friction in `v *= friction`).
- **Integration + wrap.** Euler order matters: `v += f*dt; v *= friction; p += v*dt; p = wrap(p)`. Keep this order to match the reference's trajectories.
- **Render stage.** Point rendering (`gl_PointSize = 2` equivalent) belongs to the present path ([PRIMORDIS-ADR-005](../adr/PRIMORDIS-ADR-005-rendering-and-compositing.md)) but the WGSL vertex/fragment stages that draw particle points-with-per-type-colour can be co-located in this file for convenience; flag clearly that the *compositing* of that output is backend-specific and out of scope here.
- **Workgroup geometry.** Choose `@workgroup_size(N)` once in `kernel_source.dart` (e.g. 64 or 256) and dispatch `ceil(particleCount / N)` workgroups; both backends must read this constant from the shared accessor so dispatch geometry never drifts.
- **Determinism caveat.** Do not attempt to make binning deterministic. The atomic-scatter ordering is intentionally racy (as in the original). Equivalence is judged statistically in TASK-009, and atomic translator parity (Tint vs Naga) is separately validated in TASK-017.

## Testing

- [x] **Standalone kernel harness:** run the WGSL kernel outside Flutter against a known seed (32 types, 24,000 particles, fixed RNG seed shared with the Python reference where possible) using a headless WebGPU runtime (e.g. via the web backend's pipeline or a wgpu/Dawn CLI), stepping N frames and dumping particle positions/velocities.
- [x] **Binning correctness:** assert that, for a seeded frame, the sum of all 77 bin counts (clamped at 512) equals the number of binned particles, and that every binned particle's recorded cell matches a direct CPU re-computation of its cell from its position.
- [x] **Minimum-image / toroidal wrap:** place two particles straddling each world seam (x and y) and assert the computed displacement uses the wrapped (shorter) vector, not the naive one.
- [x] **Force-regime boundaries:** unit-check the three regimes (`dist < min_dist` repulsion 5x/abs; `min_dist <= dist < radius` signed linear falloff; `dist >= radius` zero) with hand-computed expected force vectors for a 2-particle, 2-type setup.
- [x] **Asymmetry:** verify `forces[i][j] != forces[j][i]` produces different forces on i-from-j vs j-from-i.
- [x] **Slider response:** sweep Attraction K, Repulsion K, Drift and assert monotonic effect on the relevant term (more attraction K -> stronger pull; more friction -> faster velocity decay).
- [x] **24k smoke + visual parity hand-off:** run 24,000 particles / 32 types for several hundred frames without crashes, NaNs, or out-of-bounds writes, and capture frames for the parity harness ([PRIMORDIS-TASK-009](./PRIMORDIS-TASK-009-parity-test-harness-vs-python-reference.md)) to compare cluster formation/drift against `Primordis.py`.
- [x] **Translator compile check:** confirm the source compiles under both Naga and Tint (no atomic-aliasing errors); record any divergence for [PRIMORDIS-TASK-017](./PRIMORDIS-TASK-017-atomics-parity-validation-dawn-vs-browser.md).
- [x] `flutter test` (Dart accessor/layout unit tests) passes; `flutter analyze` reports zero warnings.

## Related

- PRD: [PRIMORDIS-PRD-001](../prd/PRIMORDIS-PRD-001-flutter-web-and-macos-port.md)
- ADR: [PRIMORDIS-ADR-003](../adr/PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md) (single shared WGSL kernel — primary), [PRIMORDIS-ADR-001](../adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md) (SimBackend + shared-layer placement), [PRIMORDIS-ADR-002](../adr/PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md) (web consumer of this kernel), [PRIMORDIS-ADR-004](../adr/PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md) (native consumer + MSL fallback)
- Depends on: [PRIMORDIS-TASK-002](./PRIMORDIS-TASK-002-simbackend-interface-and-shared-sim-model.md)
- Blocks: [PRIMORDIS-TASK-004](./PRIMORDIS-TASK-004-web-webgpu-backend-js-interop.md), [PRIMORDIS-TASK-009](./PRIMORDIS-TASK-009-parity-test-harness-vs-python-reference.md), [PRIMORDIS-TASK-011](./PRIMORDIS-TASK-011-macos-target-dawn-wgpu-ffi-backend.md)
