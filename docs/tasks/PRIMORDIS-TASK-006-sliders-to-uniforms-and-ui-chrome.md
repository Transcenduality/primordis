# PRIMORDIS-TASK-006: Sliders to uniforms and UI chrome

**Status:** Todo
**Priority:** High
**Created:** 2026-06-27
**Updated:** 2026-06-27

## Description

Build the shared Flutter UI chrome and wire its live controls into the simulation. Primordis exposes three sliders — **Attraction K**, **Repulsion K**, and **Drift** (friction) — plus reset/seed controls. This task implements those controls as standards-compliant Material 3 widgets driven by Riverpod, and marshals their values into the running backend so they take effect each frame: on the web GPU backend they become WebGPU **uniforms** consumed by the WGSL kernel ([PRIMORDIS-TASK-003](./PRIMORDIS-TASK-003-port-simulation-to-wgsl-compute-kernel.md), [PRIMORDIS-TASK-004](./PRIMORDIS-TASK-004-web-webgpu-backend-js-interop.md)); on any other backend the same values flow through the identical `SimBackend` parameter path.

The control values feed the integration step of the interaction pass: Attraction K scales the linear-falloff attraction term, Repulsion K scales the short-range repulsion term (the `dist < min_dist`, 5×-weighted, `abs(force)` branch), and Drift scales the per-step friction multiplier in the Euler integrate (`v += f*dt; v *= friction; p += v*dt`). The control widgets and their providers are **shared across all platforms**; only the marshalling target (a WebGPU uniform buffer vs. a CPU param struct vs. an FFI-shared param block) differs behind `SimBackend`, and the UI never knows which.

This task also owns the chrome: reset/seed controls (re-seed particles, regenerate the random per-type colors and the asymmetric 32×32 force/min-distance/radius matrices via the shared sim model), the app shell, and a play/pause control that doubles as the reduced-motion affordance.

## Scope

**Area:** Flutter
**Files/Dirs:**
- `lib/features/simulation/widgets/control_panel.dart` — the three sliders + reset/seed + play/pause chrome (Material 3, GoogleFonts).
- `lib/features/simulation/widgets/labeled_slider.dart` — reusable semantic slider (label, value, tooltip, `Semantics`).
- `lib/features/simulation/providers/sim_params_provider.dart` — Riverpod providers holding live control values (Attraction K, Repulsion K, Drift) and seed/reset intents, plain `Ref`.
- `lib/sim/sim_params.dart` — shared Freezed param/seed models reused by the marshalling path (defined in [PRIMORDIS-TASK-002](./PRIMORDIS-TASK-002-simbackend-interface-and-shared-sim-model.md); referenced/extended here for the live-tunable fields).
- `lib/sim/sim_backend.dart` — `SimBackend` interface (from [PRIMORDIS-TASK-002](./PRIMORDIS-TASK-002-simbackend-interface-and-shared-sim-model.md)); the `updateParams(...)` / `reseed(...)` entry points used here.
- `lib/sim/web/web_sim_backend.dart` — web impl marshalling the params into the WebGPU uniform buffer.
- `test/features/simulation/control_panel_test.dart`, `test/features/simulation/providers/sim_params_provider_test.dart` — widget/unit tests.

Note (house standards): the slider/chrome widgets and their providers are **standard feature-layer Flutter code** and must stay fully Riverpod/Freezed/Material 3 compliant. The only non-standard surface this task touches is the **uniform-buffer write inside the web backend**, which is JS-interop/WebGPU code that lives outside the feature/data/domain layers **by design**, quarantined under `lib/sim/web/` behind `SimBackend`. The UI must reach the backend only through the interface — never through `dart:js_interop` directly.

## Acceptance Criteria

- [ ] Three Material 3 sliders — **Attraction K**, **Repulsion K**, **Drift** — render with clear labels, current value display, and **tooltips** describing each effect.
- [ ] Each slider has accessible semantics (`Semantics` label/value, screen-reader announceable, keyboard-operable); ranges and default values match the Primordis reference behavior.
- [ ] Slider changes propagate to the live backend within one frame via `SimBackend.updateParams(...)`; dragging Attraction K / Repulsion K / Drift visibly changes clustering, repulsion strength, and drift respectively, with no restart.
- [ ] On the web GPU backend the three values are written into a **WebGPU uniform buffer** consumed by the WGSL kernel; the marshalling respects WGSL/`std140`-style uniform alignment so values land in the correct fields.
- [ ] **Reset/seed** control re-seeds the 24,000 particles and regenerates the random per-type colors and the **asymmetric** 32×32 force / min-distance / radius matrices (via the shared Freezed sim model, [PRIMORDIS-TASK-002](./PRIMORDIS-TASK-002-simbackend-interface-and-shared-sim-model.md)); the running backend picks up the new seed/params without a full reinitialization where avoidable.
- [ ] A **play/pause** control stops and resumes the frame loop; when paused the last rendered frame is held and controls stay interactive.
- [ ] **Reduced-motion compliance:** the app honors `prefers-reduced-motion` (web) / `MediaQuery.disableAnimations` by starting in (or offering) a paused/static state, satisfying the top-level accessibility goal for a full-screen-motion app (see [PRIMORDIS-ADR-006](../adr/PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md)).
- [ ] All business logic (current param values, seed/reset/pause intents) lives in **Riverpod providers with plain `Ref`** — **no `setState`** for business state. UI uses Material 3 + GoogleFonts.
- [ ] The control panel is **backend-agnostic**: identical widget code drives web GPU, web CPU fallback, native GPU, and native CPU backends through the `SimBackend` interface.

### Versioning (if Flutter/native code changed)

- [ ] Version bumped in `pubspec.yaml` and the app config constant (`PrimordisConfig`/`AppConfig`); semver.

### Test Coverage

- [ ] New/modified Dart has unit/widget tests; `flutter test` passes; `flutter analyze` zero warnings.

## Implementation Notes

- **Parameter semantics (from the brief / [PRIMORDIS-ADR-003](../adr/PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md)):** Attraction K scales the signed linear-falloff attraction (`dist < radius`); Repulsion K scales the short-range repulsion branch (`dist < min_dist`, 5× weighted, `abs(force)`); Drift is the friction multiplier in `v *= friction`. Keep these as named fields in the shared Freezed param model so the meaning is identical on every backend.
- **Marshalling boundary:** `updateParams` takes the shared param value object and the backend translates it. On web that means a single `device.queue.writeBuffer(...)`-style write into the uniform buffer each time a slider moves (or once per frame, coalesced); do **not** rebuild pipelines on a slider change. The static, expensive data — the 32×32 matrices and per-type colors — belongs in storage buffers and only changes on reseed, not on slider drags.
- **Uniform layout:** WGSL uniform buffers have alignment rules (16-byte `vec4`/struct alignment). Define the uniform struct layout once and assert the Dart-side byte offsets match the WGSL declaration so a future field addition cannot silently shift the live K values. This is a small, high-leverage parity check.
- **Coalescing:** debounce/coalesce rapid slider deltas to at most one uniform write per frame to avoid queue spam; the frame loop ([PRIMORDIS-TASK-002](./PRIMORDIS-TASK-002-simbackend-interface-and-shared-sim-model.md)) reads the latest provider value before dispatching the compute passes.
- **Reset vs. reseed:** distinguish "reset params to defaults" (slider values only → uniform write) from "reseed" (new particle positions, new colors, new asymmetric matrices → storage-buffer rewrite). The shared sim model exposes both; the chrome wires both intents through providers.
- **Determinism caveat:** reseed may take an optional fixed RNG seed for reproducibility in tests/parity ([PRIMORDIS-TASK-009](./PRIMORDIS-TASK-009-parity-test-harness-vs-python-reference.md)), but "faithful" is visual/statistical, never bit-exact (the original GPU binning is nondeterministic).
- **Accessibility:** every control needs a `Semantics` label and a tooltip; the play/pause control is the primary reduced-motion affordance and must be reachable by keyboard and screen reader. Reduced-motion detection and the broader policy are owned by [PRIMORDIS-ADR-006](../adr/PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md); this task implements the UI side.
- **No direct interop in the UI:** the control panel imports the `SimBackend` interface and providers only. The WebGPU uniform write lives in `lib/sim/web/web_sim_backend.dart`, not in `features/`.

## Testing

- [ ] Widget test: `ControlPanel` renders three labeled sliders, reset/seed, and play/pause; each slider exposes the expected `Semantics` label/value and a tooltip.
- [ ] Widget test: dragging a slider updates its Riverpod provider value; a fake/mock `SimBackend` records the corresponding `updateParams` call with the new value.
- [ ] Unit test: the param→uniform-bytes marshalling produces the correct byte offsets/values for the WGSL uniform struct layout (offset assertions).
- [ ] Widget test: tapping reset/seed triggers `reseed(...)` on the mock backend and regenerates the shared sim model (new matrices/colors), without tearing down the widget tree.
- [ ] Widget test: play/pause toggles the frame-loop state in the provider; paused holds and resume restarts.
- [ ] Accessibility test: with `MediaQuery.disableAnimations`/`prefers-reduced-motion` set, the app starts paused (or surfaces the static-state affordance); controls remain operable by keyboard.
- [ ] Manual: on the live web GPU backend, confirm each slider visibly changes the simulation in real time with no restart.
- [ ] `flutter analyze` zero warnings; `flutter test` passes.

## Related

- PRD: [PRIMORDIS-PRD-001](../prd/PRIMORDIS-PRD-001-flutter-web-and-macos-port.md)
- ADR: [PRIMORDIS-ADR-001 — Cross-platform architecture (SimBackend)](../adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md) (primary)
- ADR: [PRIMORDIS-ADR-003 — Shared WGSL compute kernel](../adr/PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md), [PRIMORDIS-ADR-002 — Web GPU compute via WebGPU + JS interop](../adr/PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md), [PRIMORDIS-ADR-006 — CPU fallback tiers and feature detection](../adr/PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md)
- Depends on: [PRIMORDIS-TASK-002 — SimBackend interface and shared sim model](./PRIMORDIS-TASK-002-simbackend-interface-and-shared-sim-model.md), [PRIMORDIS-TASK-004 — Web WebGPU backend (JS interop)](./PRIMORDIS-TASK-004-web-webgpu-backend-js-interop.md)
- Blocks: [PRIMORDIS-TASK-018 — Test coverage and accessibility](./PRIMORDIS-TASK-018-test-coverage-and-accessibility.md)
