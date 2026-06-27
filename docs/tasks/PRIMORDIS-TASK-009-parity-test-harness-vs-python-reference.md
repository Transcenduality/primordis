# PRIMORDIS-TASK-009: Visual/statistical parity test harness vs the Python reference

**Status:** Todo
**Priority:** Medium
**Created:** 2026-06-27
**Updated:** 2026-06-27

## Description

Build a **parity test harness** that validates each `SimBackend` implementation against the original `Primordis.py` reference, and against each other, on **visual and statistical** criteria — **never bit-exact**. The original GPU binning is intentionally nondeterministic (single-buffered, with a known scatter race), so a faithful port means the simulation produces the *same kind of behavior*: clusters of the same character form, drift, and persist, with the same aggregate statistics — not identical particle positions (see [PRIMORDIS-ADR-006](../adr/PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md) and the PRD non-goals).

The harness must cover the backends that exist at this point in the dependency chain — the **WGSL compute kernel** validated standalone in [PRIMORDIS-TASK-003](./PRIMORDIS-TASK-003-port-simulation-to-wgsl-compute-kernel.md) and the **Dart→WASM CPU backend** from [PRIMORDIS-TASK-008](./PRIMORDIS-TASK-008-dart-wasm-cpu-fallback-backend.md) — and be structured so the later GPU backends (web WebGPU, Dawn/Metal FFI, MSL) plug into the **same** parity assertions as they land. It produces the **statistical baselines and golden references** that every backend is measured against, including the cross-backend atomics-parity check ([PRIMORDIS-TASK-017](./PRIMORDIS-TASK-017-atomics-parity-validation-dawn-vs-browser.md)).

Concretely, the harness:
1. Captures a **reference fingerprint** from `Primordis.py` (a fixed seed run → time series of aggregate statistics + a few golden frame snapshots).
2. Defines **statistical metrics** that characterize cluster formation and drift independent of exact positions.
3. Runs each Dart backend from the **identical shared seed** and asserts its metrics fall within tolerance bands of the reference, and that backends agree with one another within their (looser) cross-backend band.

## Scope

**Area:** Infra
**Files/Dirs:**
- `tooling/parity/` — the harness (reference-capture script + metric definitions + comparison runner)
- `tooling/parity/capture_reference.py` — deterministic-seed run of `Primordis.py` exporting per-step aggregate stats (CSV/JSON) and golden frame snapshots (PNG/raw)
- `tooling/parity/metrics.dart` — Dart implementation of the statistical metrics (shared by harness + tests)
- `test/parity/parity_test.dart` — runs the CPU backend (TASK-008) and the standalone WGSL kernel result fixtures (TASK-003) against the reference fingerprint
- `test/parity/fixtures/` — committed reference fingerprint (stats time series + golden snapshots) and the fixed seed/params used to generate them
- `test/parity/golden/` — golden statistical baselines per backend
- Consumes (does not own): the shared seed/params (Freezed models, TASK-002), the WGSL kernel + its standalone validation output (TASK-003), the CPU backend (TASK-008)

> **Layering note (expected, per house standards):** this is test/tooling infrastructure, not app UI; it sits outside the feature/data/domain layers by design. The Dart metric code (`metrics.dart`) is plain, UI-free, fully unit-testable. The `Primordis.py` capture step is reference-only tooling (Python, not shipped in the Flutter app) and lives under `tooling/`.

## Acceptance Criteria

- [ ] A **fixed seed + fixed params** (the 32-type asymmetric 32×32 force/min-distance/radius matrices and per-type colors) are defined **once** and shared by the Python capture and all Dart backends, so every backend starts from the identical initial condition (sourced from the shared seed model in TASK-002 where possible; otherwise a committed fixture both sides load).
- [ ] `capture_reference.py` runs `Primordis.py` headlessly with that seed and exports: (a) a per-step time series of aggregate statistics, and (b) golden frame snapshots at named steps (e.g. early / mid / steady-state).
- [ ] At least the following **statistical metrics** are implemented and compared (all position-invariant / aggregate, never per-particle equality):
  - [ ] **Cluster count / size distribution** over time (via a grid-occupancy or neighbor-density measure on the same 11×7 / `MAX_RADIUS=96` grid the sim uses), capturing that clusters *form* and at roughly the same rate.
  - [ ] **Mean and variance of inter-particle spacing** (nearest-neighbor distance distribution), capturing repulsion/attraction balance.
  - [ ] **Per-type spatial mixing / segregation** measure (do types separate into the same kind of structure given the asymmetric matrices).
  - [ ] **Drift / kinetic energy** proxy: mean velocity magnitude and its decay under the friction term, capturing the Drift slider's effect.
  - [ ] **Population conservation**: particle count is constant; none lost through toroidal wrap.
- [ ] Each metric has an explicit **tolerance band** vs the reference, justified in comments (tighter for conserved quantities like population; looser for nondeterministic cluster geometry). Bands are committed as golden baselines.
- [ ] `parity_test.dart` runs the **CPU backend (TASK-008)** from the shared seed and asserts all metrics fall within the reference tolerance bands over a fixed step budget.
- [ ] The harness ingests the **standalone WGSL kernel** validation output from TASK-003 (the kernel run at 24k / 32-types) and asserts the same metric bands, so the kernel is parity-checked independent of any platform backend.
- [ ] The comparison runner is **backend-pluggable**: adding the web WebGPU, Dawn/Metal FFI, or MSL backend later requires only registering the backend, not new metric code. The cross-backend band (backend-vs-backend) is explicitly **looser** than backend-vs-reference, acknowledging translator/atomics nondeterminism (Tint vs Naga; see TASK-017).
- [ ] The harness asserts **toroidal correctness** explicitly: a wrap-seam scenario (clusters straddling the world edge) is included in a golden snapshot and metrics confirm no artificial edge accumulation.
- [ ] Golden snapshot comparison is **structural/statistical** (e.g. binned occupancy histogram difference under threshold), **not** pixel-exact image diff, since positions are nondeterministic.
- [ ] Harness is runnable in CI (headless) and documents how to regenerate the reference fixtures when the seed/params intentionally change.

### Versioning (if Flutter/native code changed)

- [ ] Version bumped in `pubspec.yaml` and the app config constant (`PrimordisConfig`/`AppConfig`); semver. (If this task only adds test/tooling and no shipped Dart/native code, note "no app version change — tooling/tests only" in the PR.)

### Test Coverage

- [ ] New/modified Dart has unit/widget tests; `flutter test` passes; `flutter analyze` zero warnings.
- [ ] `metrics.dart` has its own unit tests on synthetic inputs with known answers (e.g. a hand-built two-cluster configuration yields the expected cluster count and spacing stats), so the metrics themselves are trusted before they judge backends.
- [ ] The parity test fails loudly (clear diagnostic: which metric, which backend, observed vs band) when a backend regresses out of tolerance.

## Implementation Notes

- **Parity definition (load-bearing):** equality is **statistical**, not bitwise. The reference `Primordis.py` GPU binning is single-buffered and races on scatter, so two runs of the *original itself* differ. The harness therefore must never assert per-particle position equality or pixel-exact frames; it asserts that *distributions and trajectories of aggregate statistics* match within bands. This is the project-wide "faithful = visually/statistically equivalent" contract (PRD + ADR-006).
- **Seed sharing:** the cleanest design is a single committed seed/params fixture (matrices + colors + initial positions/velocities or RNG seed) that both `Primordis.py` (numpy RNG) and the Dart side load identically. Because the two RNGs differ, prefer exporting the *concrete* initial particle array and the *concrete* matrices from one source of truth so initial conditions are byte-identical even though the physics then diverges nondeterministically.
- **Determinism handle:** the CPU backend (TASK-008) uses **deterministic counting-sort binning**, so its run is reproducible and makes the tightest, most stable baseline — use it as the primary Dart anchor. The WGSL/GPU runs are nondeterministic, so they get the looser cross-backend band.
- **Metric grid alignment:** compute cluster/occupancy metrics on the **same** spatial grid the sim uses (11×7 bins, bin size `MAX_RADIUS=96`) so the metric is meaningful relative to the interaction range; account for toroidal wrap when binning the metric, exactly as the sim does (minimum-image).
- **Slider/param coverage:** capture reference fingerprints at a couple of representative (Attraction K, Repulsion K, Drift) settings, not just defaults, so the harness validates that param marshalling produces the same *behavioral response* across backends (e.g. higher Drift → faster velocity decay; the kinetic-energy metric should track this on every backend).
- **Python capture practicality:** `Primordis.py` needs OpenGL 4.3 compute; the capture script should run on a machine/CI runner that has it (or capture once locally and commit the fixtures, with the regeneration procedure documented). Capturing aggregate stats + a few snapshots — not full frame video — keeps fixtures small and committable.
- **Cross-backend role:** this harness is the substrate TASK-017 builds on. TASK-017 specifically compares Dawn(Tint) vs browser(Naga) atomic scatter-binning at 24k/32-types; it should reuse these metrics and the binning-occupancy check rather than inventing its own. Keep the binning-occupancy metric exported as a reusable function.
- **No bit-exact temptation:** if a future contributor wants a bit-exact mode, the only place it could exist is the deterministic CPU counting-sort path comparing two CPU runs — never CPU-vs-GPU or GPU-vs-GPU. Document this explicitly so nobody files "parity flaky" bugs against inherent GPU nondeterminism.

## Testing

- [ ] `flutter test` green (including `test/parity/parity_test.dart` and `metrics.dart` unit tests); `flutter analyze` zero warnings.
- [ ] Regenerate the reference fixtures from `capture_reference.py` on a machine with OpenGL 4.3 and confirm committed fixtures match (or update them via the documented procedure).
- [ ] Run the parity suite against the CPU backend (TASK-008) and confirm all metrics are within band over the fixed step budget.
- [ ] Run the parity suite against the standalone WGSL kernel output (TASK-003) and confirm within (looser) cross-backend band.
- [ ] Deliberately introduce a known regression (e.g. swap `i,j` matrix indexing to break asymmetry, or drop the 5× repulsion weight) and confirm the harness flags the correct metric — proving the harness has teeth.
- [ ] Confirm the wrap-seam scenario shows no artificial edge accumulation in either snapshot or metric.

## Related

- PRD: [PRIMORDIS-PRD-001](../prd/PRIMORDIS-PRD-001-flutter-web-and-macos-port.md)
- ADR: [PRIMORDIS-ADR-003 — Shared WGSL compute kernel](../adr/PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md); [PRIMORDIS-ADR-006 — CPU fallback tiers and feature detection](../adr/PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md); [PRIMORDIS-ADR-001 — Cross-platform architecture & SimBackend](../adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md)
- Depends on: [PRIMORDIS-TASK-003 — Port simulation to WGSL compute kernel](./PRIMORDIS-TASK-003-port-simulation-to-wgsl-compute-kernel.md); [PRIMORDIS-TASK-008 — Dart→WASM CPU fallback backend](./PRIMORDIS-TASK-008-dart-wasm-cpu-fallback-backend.md)
- Blocks: [PRIMORDIS-TASK-017 — Atomics parity validation (Dawn vs browser)](./PRIMORDIS-TASK-017-atomics-parity-validation-dawn-vs-browser.md)
