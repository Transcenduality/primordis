# PRIMORDIS-TASK-018: Test coverage and accessibility compliance

**Status:** Todo
**Priority:** High
**Created:** 2026-06-27
**Updated:** 2026-06-27

## Description

Bring the project to the org's quality bar: **widget tests** for the UI (the three sliders — Attraction K, Repulsion K, Drift/friction — plus reset/seed chrome), **unit tests** for the shared sim params and the CPU simulation (including the deterministic sequential counting-sort binning), **reduced-motion / pause accessibility** compliance for a full-screen-motion app, and **`flutter analyze` zero warnings**. This is the cross-cutting standards task that makes the standards-compliant layers (Riverpod, Freezed, Material 3, `package:lint`) demonstrably correct and accessible, independent of which GPU/CPU backend is live.

Accessibility is a top-level goal (org standard): because the **entire screen is motion**, the app must respect reduced-motion by offering a **pause / static state**, and all controls must be semantic and tooltip-bearing. This task verifies those obligations — which ADR-005 and ADR-006 assign to the UI layer — are actually met and tested.

## Scope

**Area:** Flutter
**Files/Dirs:**
- `test/features/simulation/sliders_test.dart` — widget tests for the three sliders + reset/seed controls
- `test/features/simulation/accessibility_test.dart` — reduced-motion/pause + semantics/tooltip tests
- `test/sim/sim_params_test.dart` — unit tests for the Freezed sim params (32x32 matrices, colors, count clamping)
- `test/sim/cpu/counting_sort_binning_test.dart`, `test/sim/cpu/cpu_sim_test.dart` — CPU sim + deterministic binning
- `lib/features/simulation/...` (add `Semantics`/`Tooltip`/pause control where missing; no business logic in `setState`)
- `analysis_options.yaml` (ensure `package:lint` is wired and zero-warning enforced in CI)
- UI/test code stays inside the standard layers; backend internals (JS-interop/FFI/shader) are exercised through `SimBackend` seams, not imported into UI tests.

## Acceptance Criteria

- [ ] **Widget tests** cover the three sliders (Attraction K, Repulsion K, Drift/friction): each renders, has a `Semantics` label and a `Tooltip`, and a drag updates the corresponding Riverpod-held param (no `setState` for business logic; plain `Ref`).
- [ ] Widget tests cover reset/seed controls: reset returns params to defaults; seed regenerates the 32-type asymmetric matrices/colors; both are semantic, tooltip-bearing controls.
- [ ] **Unit tests** for the shared Freezed sim params: the three asymmetric 32x32 float32 matrices (forces, min_distances, radii; i→j ≠ j→i) and per-type colors round-trip via Freezed; particle count clamps to the active tier ceiling (from [PRIMORDIS-TASK-015](./PRIMORDIS-TASK-015-cross-platform-backend-selection-and-reduced-mode-ux.md)).
- [ ] **Unit tests** for the CPU sim ([PRIMORDIS-TASK-008](./PRIMORDIS-TASK-008-dart-wasm-cpu-fallback-backend.md)): the deterministic **sequential counting-sort** binning over the 11x7 = 77-bin grid (bin size 96) produces correct, order-deterministic per-bin counts/offsets for a fixed seed; the interaction/integration step (3x3 toroidal scan, min-distance repulsion vs linear-falloff attraction, Euler integrate, wrap) advances a known fixture to expected positions.
- [ ] **Reduced-motion accessibility:** with `MediaQuery` disable-animations / reduced-motion enabled, the app presents a **paused / static** state (frame loop halted, last frame held) and offers an explicit, semantic pause/play control; the simulation does not auto-animate against the user's preference.
- [ ] All interactive controls expose `Semantics` labels and `Tooltip`s; the reduced-mode indicator (TASK-015) remains legible while paused.
- [ ] `flutter analyze` reports **zero** warnings under `package:lint`; `flutter test` passes; CI enforces both.
- [ ] Tests assert UI behavior **independent of backend**: the same widget/param tests pass regardless of which `SimBackend` is selected (UI never branches on the concrete backend).

### Versioning (if Flutter/native code changed)
- [ ] Version bumped in pubspec.yaml and the app config constant (`PrimordisConfig`); semver

### Test Coverage
- [ ] New/modified Dart has unit/widget tests; flutter test passes; flutter analyze zero warnings

## Implementation Notes

- **Scope of testability.** Per ADR-001, the UI, params, seeding, and frame loop are shared and standards-compliant; the non-standard JS-interop/FFI/shader code lives behind `SimBackend`. This task tests the **standards-compliant** surface directly and treats the backend as an injected seam (mock/fake `SimBackend`) so widget/unit tests never need a GPU. That separation is what makes the UI fully unit-testable.
- **Sliders → params via Riverpod.** The sliders drive params through Riverpod providers (TASK-006), not `setState`. Widget tests pump the widget with a `ProviderScope` (overriding the backend with a fake) and assert the provider state changes on drag. The marshalling of params to a live backend is the backend's concern; here we assert the **provider** is updated.
- **CPU binning is the deterministic one.** Only the CPU path uses sequential counting-sort binning (the GPU path uses atomic scatter, validated separately in [PRIMORDIS-TASK-017](./PRIMORDIS-TASK-017-atomics-parity-validation-dawn-vs-browser.md)). Counting-sort is deterministic, so its unit tests can assert exact per-bin counts/offsets for a fixed seed. Faithfulness to `Primordis.py` overall is statistical, not bit-exact (ADR-001/006); the determinism here is of the **CPU algorithm**, not of the reference GPU sim.
- **Reduced-motion is mandatory, not optional.** The whole canvas is motion; the org accessibility standard requires honoring reduced-motion. ADR-005 and ADR-006 assign the pause/static-state obligation to the UI/frame loop (pause halts the loop and holds the last composited frame). This task makes that behavior real and tested via `MediaQuery` disable-animations.
- **Lint gate.** `package:lint` with zero warnings is a house standard; wire it in `analysis_options.yaml` and enforce in CI so regressions fail the build. Material 3 + `GoogleFonts` for any new UI.
- **Dependency timing.** Depends on the sliders/UI chrome (TASK-006) and the CPU fallback backend (TASK-008) existing so there is real code to cover; it also references the tier/reduced-mode UX (TASK-015) for the reduced-mode-while-paused assertion.

## Testing

- [ ] `flutter test` runs the full widget + unit suite green.
- [ ] Widget: each slider renders, exposes `Semantics` + `Tooltip`, and a simulated drag updates its Riverpod provider (verified with a `ProviderScope` + fake `SimBackend`).
- [ ] Widget: reset restores defaults; seed regenerates matrices/colors; both controls are semantic and tooltipped.
- [ ] Unit: Freezed sim-params equality/copyWith/round-trip for the 32x32 matrices and colors; count clamps to tier ceiling.
- [ ] Unit: counting-sort binning gives exact, order-deterministic per-bin counts/offsets for a fixed seed; integration advances a known fixture to expected positions with toroidal wrap.
- [ ] Accessibility: with reduced-motion on, the app is in paused/static state and exposes a semantic pause/play control; reduced-mode indicator stays legible while paused.
- [ ] CI: `flutter analyze` returns zero warnings under `package:lint`; pipeline fails on any warning or test failure.

## Related
- PRD: [PRIMORDIS-PRD-001](../prd/PRIMORDIS-PRD-001-flutter-web-and-macos-port.md)
- ADR: [PRIMORDIS-ADR-006](../adr/PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md) (reduced-motion / reduced-mode, CPU binning), [PRIMORDIS-ADR-001](../adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md) (standards-compliant UI behind SimBackend), [PRIMORDIS-ADR-005](../adr/PRIMORDIS-ADR-005-rendering-and-compositing.md) (pause/static-state obligation)
- Depends on: [PRIMORDIS-TASK-006](./PRIMORDIS-TASK-006-sliders-to-uniforms-and-ui-chrome.md), [PRIMORDIS-TASK-008](./PRIMORDIS-TASK-008-dart-wasm-cpu-fallback-backend.md)
- Blocks: (none — quality/accessibility gate)
