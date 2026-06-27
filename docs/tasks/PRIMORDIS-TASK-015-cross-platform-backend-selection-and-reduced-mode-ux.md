# PRIMORDIS-TASK-015: Cross-platform backend selection and reduced-mode UX

**Status:** Todo
**Priority:** High
**Created:** 2026-06-27
**Updated:** 2026-06-27

## Description

Unify capability detection and `SimBackend` selection across **all** platforms behind one decision point, and surface the user-visible **"reduced mode"** indicator and per-tier **particle-count policy** decided in [PRIMORDIS-ADR-006](../adr/PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md). This is the convergence task: it composes the web feature-detect+switch ([PRIMORDIS-TASK-007](./PRIMORDIS-TASK-007-webgpu-feature-detection-and-fallback-switch.md)), the native GPU path ([PRIMORDIS-TASK-011](./PRIMORDIS-TASK-011-macos-target-dawn-wgpu-ffi-backend.md)), and the native CPU isolate fallback ([PRIMORDIS-TASK-014](./PRIMORDIS-TASK-014-native-cpu-isolate-fallback-backend.md)) into a single, deterministic, platform-aware selector that picks the **highest tier whose preconditions actually succeed** and exposes which tier is live to the UI without the UI branching on a specific backend type.

The four tiers (ADR-006): **T1 Native GPU** (Dawn/wgpu-over-Metal WGSL, MSL fallback; 24k default, 100k-500k+ on Apple Silicon) → **T3 Native CPU** (isolates; benchmarked ceiling ~10-14k, gated by TASK-014) on native; **T2 Web GPU** (browser WebGPU WGSL; 24k+) → **T4 Web CPU** (single-thread Dart→WASM; ~3-4k) on web. The reduced-mode indicator appears for CPU tiers (T3/T4) and is a semantic, tooltip-bearing Material 3 affordance that must remain legible while the simulation is **paused** for reduced-motion users.

## Scope

**Area:** Flutter
**Files/Dirs:**
- `lib/sim/selection/backend_selector.dart` — platform-aware detection + tier selection returning a `SimBackend` + `CapabilityTier`
- `lib/sim/selection/capability_tier.dart` — Freezed model: tier id, default/max particle count, isReducedMode, human-readable reason
- `lib/sim/selection/backend_provider.dart` — Riverpod provider exposing the selected backend + tier to the UI (plain `Ref`, no setState for business logic)
- `lib/features/simulation/widgets/reduced_mode_indicator.dart` — Material 3 status chip with `Semantics` + tooltip
- `lib/sim/sim_params.dart` (extend shared Freezed params so particle count is tier-clamped)
- `test/sim/selection/backend_selector_test.dart`, `test/features/simulation/reduced_mode_indicator_test.dart`
- Detection internals that touch `navigator.gpu` (web) or GPU device creation (native) remain behind `SimBackend`/the per-platform backends, outside the standard layers.

## Acceptance Criteria

- [ ] A single `BackendSelector` runs detection **once at startup**, by platform, and returns the highest tier whose preconditions are **proven** (not assumed): capability must be demonstrated by a successful adapter/device/pipeline acquisition, never inferred.
- [ ] Web order: hard `navigator.gpu` feature-detect (truthy via `dart:js_interop` + `package:web`, not try/catch on first use) → `requestAdapter()` → `requestDevice()`; full success ⇒ **T2**; any null/throw ⇒ **T4**. WebGL2 is explicitly **not** a tier.
- [ ] Native order: attempt GPU device + compute-pipeline creation via the primary Dawn/wgpu-over-Metal FFI path (MSL plugin as secondary GPU attempt); either success ⇒ **T1**; GPU init failure (no compatible GPU family, driver/signing/sandbox/FFI-load failure) ⇒ **T3**.
- [ ] Partial GPU success (e.g. device but pipeline build fails; adapter but no device) **demotes** cleanly to the CPU tier rather than stranding the user on a dead GPU path or risking a runtime stall.
- [ ] The selected tier sets the **default and maximum** particle count for the session (T1 24k+, T2 24k, T3 benchmarked ceiling from TASK-014, T4 ~3-4k); the shared params model clamps any count slider/preset to the tier ceiling. T3's number is the **benchmarked** value, not the provisional estimate.
- [ ] Seeding (32 types, asymmetric 32x32 matrices, random colors) is identical across tiers; only the live count differs.
- [ ] A `reducedModeIndicator` is shown **only** for T3/T4, with a `Semantics` label and a tooltip explaining *why* (e.g. "GPU acceleration unavailable; running on CPU at reduced particle count"). T1/T2 show no reduced-mode chrome (or an unobtrusive "GPU" state).
- [ ] The indicator remains visible and legible while the simulation is **paused** (reduced-motion state, ADR-006 / [PRIMORDIS-TASK-018](./PRIMORDIS-TASK-018-test-coverage-and-accessibility.md)); its messaging does not depend on the animation running.
- [ ] The selected backend and tier are exposed through a Riverpod provider; the UI/sliders consume tier metadata but never switch on the concrete backend class.

### Versioning (if Flutter/native code changed)
- [ ] Version bumped in pubspec.yaml and the app config constant (`PrimordisConfig`); semver

### Test Coverage
- [ ] New/modified Dart has unit/widget tests; flutter test passes; flutter analyze zero warnings

## Implementation Notes

- **One selector, two probe strategies.** Branch on platform at the top (web vs native), then run the platform-specific probe sequence from ADR-006 §2. Each probe **proves** capability by actually acquiring the resource: web acquires adapter+device; native builds device+compute pipeline. The first GPU acquisition is the authoritative probe — both device and pipeline must build before T1/T2 is committed.
- **Composes existing tasks.** This task does not re-implement detection internals: web `navigator.gpu` detect+switch is TASK-007, the native GPU backend is TASK-011 (with TASK-013 MSL secondary), the native CPU backend is TASK-014. TASK-015 wires them into one selector + tier model + UX and is the place the demotion policy is centralized.
- **Tier model (Freezed).** `CapabilityTier` carries the tier id (T1-T4), default count, max count, `isReducedMode`, and a localized reason string for the tooltip. Keep it a pure Freezed value object so it is trivially testable and so the indicator widget is a pure function of it.
- **Reduced-mode indicator (accessibility).** Material 3 chip/badge with `GoogleFonts` text, a `Semantics(label: ...)`, and a `Tooltip`. Per the org accessibility standard the whole canvas is motion, so the app offers a pause/static state; the indicator must read correctly there too — render it as static UI chrome independent of the frame loop. This is informational, never motion.
- **Particle-count policy.** Count lives in the shared Freezed sim params (ADR-001) and is clamped to the live tier's ceiling at the params layer, so neither the slider nor a preset can exceed it. T3's ceiling is **provisional until TASK-014 benchmarks it** — do not hardcode 24k for T3.
- **Riverpod, no setState.** Selection result and tier are exposed via a provider using a plain `Ref`; UI reads `tier`/`isReducedMode`. No business logic in `setState`.
- **Standards boundary.** The selector itself is ordinary Dart; only the probes reach into JS-interop (web) or FFI/native (macOS), which already live behind the per-platform backends. The selector, tier model, provider, and indicator are all standards-compliant and fully covered by tests.

## Testing

- [ ] Unit test `BackendSelector` with injected/mocked probe outcomes for each branch: web {gpu absent / adapter null / device null / full success}; native {pipeline build fail / device fail / full success} → assert the expected tier (T1/T2/T3/T4) and that partial success demotes.
- [ ] Unit test that the chosen tier clamps the params particle count to its ceiling (slider above ceiling is capped).
- [ ] Widget test `ReducedModeIndicator`: present for T3/T4, absent for T1/T2; assert `Semantics` label and tooltip text; assert it renders while paused.
- [ ] Widget test: with `MediaQuery` disable-animations / reduced motion on, the indicator stays visible and the simulation is in paused/static state.
- [ ] Manual: on a non-WebGPU browser confirm T4 + reduced mode; on a Mac with GPU-init forced to fail confirm T3 + reduced mode; on healthy GPU confirm T1/T2 + no reduced-mode chrome.
- [ ] `flutter analyze` zero warnings; `flutter test` passes.

## Related
- PRD: [PRIMORDIS-PRD-001](../prd/PRIMORDIS-PRD-001-flutter-web-and-macos-port.md)
- ADR: [PRIMORDIS-ADR-006](../adr/PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md) (primary — tiers, detection, particle-count policy, reduced-mode UX), [PRIMORDIS-ADR-001](../adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md) (SimBackend boundary)
- Depends on: [PRIMORDIS-TASK-007](./PRIMORDIS-TASK-007-webgpu-feature-detection-and-fallback-switch.md), [PRIMORDIS-TASK-011](./PRIMORDIS-TASK-011-macos-target-dawn-wgpu-ffi-backend.md), [PRIMORDIS-TASK-014](./PRIMORDIS-TASK-014-native-cpu-isolate-fallback-backend.md)
- Blocks: (none — convergence/UX task)
