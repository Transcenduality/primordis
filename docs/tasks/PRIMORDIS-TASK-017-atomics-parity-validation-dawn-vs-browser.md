# PRIMORDIS-TASK-017: Atomics parity validation — Dawn (Tint) vs browser (Naga)

**Status:** Todo
**Priority:** High
**Created:** 2026-06-27
**Updated:** 2026-06-27

## Description

Prove that the `atomicAdd` scatter-binning pass of the shared WGSL kernel produces **identical results** when run through the two different WGSL translators the project relies on: **Dawn (Tint)** on native macOS (Dawn/wgpu-over-Metal via `dart:ffi`, [PRIMORDIS-TASK-011](./PRIMORDIS-TASK-011-macos-target-dawn-wgpu-ffi-backend.md) / [PRIMORDIS-ADR-004](../adr/PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md)) and **browser WebGPU / wgpu (Naga)** on web ([PRIMORDIS-ADR-002](../adr/PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md)). This is a named top risk: Dawn (Tint) and browser/wgpu (Naga) are different WGSL translators with a documented history of `atomicAdd` bugs, and the entire spatial-grid binning depends on `atomicAdd(&bins[i], 1u)` returning the correct old value as a per-particle scatter offset.

The kernel is one shared WGSL source ([PRIMORDIS-ADR-003](../adr/PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md)); this task validates that the **same** source behaves equivalently across the two translators on the binning pass specifically, at the reference 24k particles / 32 types over the 11x7 = 77-bin grid (bin size = MAX_RADIUS = 96, MAX_BIN_PARTICLES = 512 cap). Because the original GPU binning is already nondeterministic (single-buffered scatter race, ADR-001/006), "identical" here means **statistically/structurally equivalent** — every particle assigned to its correct bin, per-bin counts and the set of occupants matching — not a bit-exact, order-identical buffer.

## Scope

**Area:** Shader
**Files/Dirs:**
- `test/parity/atomics/binning_parity_test.dart` — harness driving the binning pass on both backends over shared fixtures
- `test/parity/atomics/fixtures/` — fixed seed particle sets (incl. 24k/32-type) + expected per-bin invariants
- `lib/sim/parity/binning_probe.dart` — helper that runs only the clear+bin passes and reads back bin counts/offsets via `atomicLoad`
- `lib/sim/kernels/primordis.wgsl` (read-only here; the shared kernel under test)
- The atomics live in the WGSL kernel and are exercised through the JS-interop (web) and FFI (native) backends — outside the standard layers, behind `SimBackend`.

## Acceptance Criteria

- [ ] A reproducible harness runs **only** the clear-bin-counts pass and the atomic-scatter binning pass (not the full frame) on a fixed seed, on both the browser-WebGPU (Naga) backend and the Dawn/wgpu-over-Metal (Tint) backend.
- [ ] Bin counts are read back via `atomicLoad` (consistent `atomic<u32>` typing), and for every fixture **both translators agree** on: (a) each bin's occupant count, and (b) the **set** of particle indices assigned to each bin.
- [ ] The reference 24k-particle / 32-type fixture passes on both translators; assignment respects the toroidal grid and the MAX_BIN_PARTICLES = 512 cap behaves identically (same overflow handling on both).
- [ ] The `atomicAdd(&bins[i], 1u)` **return value** (old count) is verified to produce correct, collision-free scatter offsets on both translators (no two particles in a bin get the same slot; no out-of-range slot).
- [ ] Equivalence is asserted as **structural/statistical** (per-bin count + occupant set), explicitly **not** bit-exact ordering, consistent with the documented nondeterminism of the single-buffered scatter (ADR-001/006).
- [ ] A discrepancy (e.g. a Tint vs Naga `atomicAdd` bug) fails the harness with a diagnostic identifying the divergent bins and the offending translator, so the issue is caught before sign-off rather than as a runtime visual artifact.
- [ ] Result is recorded as a go/no-go signal for the macOS GPU path: parity passing keeps Approach (a) primary; parity failing escalates to the MSL fallback ([PRIMORDIS-TASK-013](./PRIMORDIS-TASK-013-macos-metal-msl-compute-plugin-fallback.md)) and/or a kernel workaround.

### Versioning (if Flutter/native code changed)
- [ ] Version bumped in pubspec.yaml and the app config constant (`PrimordisConfig`); semver

### Test Coverage
- [ ] New/modified Dart has unit/widget tests; flutter test passes; flutter analyze zero warnings

## Implementation Notes

- **Why this is its own task.** The GLSL→WGSL port is the easy part; the live risk is translator divergence. Dawn uses **Tint**, browser WebGPU and wgpu use **Naga**, and the same WGSL `atomicAdd` can compile to subtly different Metal/SPIR-V/MSL with a known history of atomics bugs (ADR-004 Negative). Binning is the one pass that depends on atomic **return values**, so it is the precise place to validate.
- **WGSL atomics contract.** Bins are `array<atomic<u32>>` in `var<storage, read_write>`; the scatter uses `let offset = atomicAdd(&bins[binIndex], 1u);` (returns the old value = this particle's slot), and counts are read back via `atomicLoad(&bins[i])`. Atomics must be `atomic<u32>` consistently and never read as a plain `u32` (per ADR-003). The probe (`binning_probe.dart`) must read back through `atomicLoad`, not a raw load.
- **Equivalence definition.** Because the scatter is single-buffered and racy by design (faithful to `Primordis.py`, never bit-exact — ADR-001/006), compare **invariants**: each bin's final count, and the **set** (order-independent) of particle indices in each bin. Two runs may order occupants within a bin differently; that is allowed. A difference in counts or membership is a real parity bug.
- **Shared fixtures.** Seed both backends from the **same** particle positions/types and the same 32x32 matrices so the only variable is the translator. Include the full 24k/32-type fixture plus small hand-checkable cases and an overflow case that exercises the 512-per-bin cap on both.
- **Cross-platform execution.** The Naga path runs in a browser-WebGPU test context; the Tint path runs through the Dawn/wgpu FFI backend on macOS. The harness abstracts "run binning, return bin counts + occupant sets" behind one interface so the same assertions apply to both.
- **Decision linkage.** This is the de-risking gate ADR-004 calls for ("validate the binning pass produces identical results on both"). Keeping the MSL plugin warm (TASK-013) is the contingency if parity cannot be achieved on Dawn.

## Testing

- [ ] Run the binning probe on the small hand-checkable fixtures on both translators; assert exact per-bin counts and occupant sets match the hand-computed expectation and each other.
- [ ] Run the 24k/32-type fixture on both; assert per-bin counts and occupant sets are equal between Tint and Naga (order within a bin ignored).
- [ ] Overflow fixture: more than 512 particles targeting one bin; assert identical cap/overflow behavior on both translators.
- [ ] Scatter-offset correctness: assert `atomicAdd` return values yield a collision-free, in-range slot assignment on each translator.
- [ ] Negative/diagnostic: inject a divergent expectation and confirm the harness reports the specific divergent bins and the offending translator.
- [ ] `flutter analyze` zero warnings; `flutter test` passes.

## Related
- PRD: [PRIMORDIS-PRD-001](../prd/PRIMORDIS-PRD-001-flutter-web-and-macos-port.md)
- ADR: [PRIMORDIS-ADR-004](../adr/PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md) (primary — atomics-parity risk, Tint vs Naga), [PRIMORDIS-ADR-003](../adr/PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md) (shared kernel, WGSL atomics contract), [PRIMORDIS-ADR-002](../adr/PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md) (browser/Naga path)
- Depends on: [PRIMORDIS-TASK-011](./PRIMORDIS-TASK-011-macos-target-dawn-wgpu-ffi-backend.md)
- Blocks: (none — validation gate informing TASK-013 escalation)
