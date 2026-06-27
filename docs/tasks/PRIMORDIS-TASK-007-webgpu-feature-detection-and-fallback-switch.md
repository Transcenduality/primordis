# PRIMORDIS-TASK-007: WebGPU feature detection and fallback switch

**Status:** Todo
**Priority:** High
**Created:** 2026-06-27
**Updated:** 2026-06-27

## Description

Implement a **hard `navigator.gpu` feature-detect** and the backend-selection logic that, on the web, chooses the WebGPU GPU backend when it is available and otherwise **switches to the Dart→WASM CPU fallback backend**. WebGPU is not universally available mid-2026 — present on Chrome/Edge 113+, Safari 26 (macOS Tahoe 26 / iOS 26), and Firefox 141+ (Windows) / 145+ (Apple-Silicon Mac), but **absent** on Firefox Linux/Android, Intel Macs, and pre-26 Safari/iOS — so a feature-detect with a graceful fallback is mandatory, not optional ([PRIMORDIS-ADR-002](../adr/PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md), [PRIMORDIS-ADR-006](../adr/PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md)).

Detection must be **hard**: a present `navigator.gpu` is necessary but not sufficient — the adapter/device request can still fail. The selector requests the adapter and device and only commits to the GPU backend once a device is in hand; any failure at any step (no `navigator.gpu`, no adapter, device-lost) routes to the CPU-WASM backend ([PRIMORDIS-TASK-008](./PRIMORDIS-TASK-008-dart-wasm-cpu-fallback-backend.md)). The chosen backend is constructed behind the `SimBackend` interface so the UI never learns which tier is live, and the active tier (and its particle-count policy) is surfaced for the "reduced mode" indicator.

This task owns the **web** selection seam. Cross-platform selection across native GPU / native CPU isolate tiers and the unified reduced-mode UX are composed on top of this in [PRIMORDIS-TASK-015](./PRIMORDIS-TASK-015-cross-platform-backend-selection-and-reduced-mode-ux.md).

## Scope

**Area:** Flutter
**Files/Dirs:**
- `lib/sim/web/web_capability_detector.dart` — `navigator.gpu` presence + adapter/device acquisition probe (JS-interop, behind `SimBackend`).
- `lib/sim/backend_selector.dart` — async selection returning the chosen `SimBackend` (web GPU vs. web CPU-WASM) and the resolved capability tier.
- `lib/sim/capability_tier.dart` — Freezed model describing the resolved tier (backend kind, particle-count ceiling, reduced-mode flag, reason).
- `lib/sim/providers/active_backend_provider.dart` — Riverpod provider exposing the selected backend + tier to the UI, plain `Ref`.
- `lib/sim/web/web_sim_backend.dart`, `lib/sim/web/web_cpu_sim_backend.dart` — the two web candidate backends (from [PRIMORDIS-TASK-004](./PRIMORDIS-TASK-004-web-webgpu-backend-js-interop.md) and [PRIMORDIS-TASK-008](./PRIMORDIS-TASK-008-dart-wasm-cpu-fallback-backend.md)).
- `test/sim/backend_selector_test.dart`, `test/sim/web/web_capability_detector_test.dart` — unit tests with injected/mocked capability probes.

Note (house standards): the `navigator.gpu`/adapter/device probe is **`dart:js_interop` + `package:web` code that lives outside the standard feature/data/domain layers by design**, quarantined under `lib/sim/web/` behind `SimBackend`. The selector and the tier model are plain Dart/Freezed and Riverpod-exposed, so the UI layer that reacts to the tier stays standards-compliant. Use `dart:js_interop`/`package:web` only — **no `dart:html` / `dart:js_util`** (required by `--wasm`, see [PRIMORDIS-ADR-007](../adr/PRIMORDIS-ADR-007-web-build-and-cross-origin-isolation.md)).

## Acceptance Criteria

- [ ] A **hard `navigator.gpu` feature-detect** runs at startup: when `navigator.gpu` is absent, the selector commits to the web CPU-WASM backend without attempting any WebGPU calls.
- [ ] When `navigator.gpu` is present, the selector **requests the adapter and device** and only commits to the WebGPU backend once a device is acquired; failure to get an adapter or device (including async `null`/throw) routes to the CPU-WASM fallback.
- [ ] A **device-lost** event after successful init is handled: the selector/active-backend provider degrades to the CPU-WASM tier (or surfaces a recoverable error per policy) rather than crashing.
- [ ] The chosen backend is delivered **behind the `SimBackend` interface**; the UI consumes only `activeBackendProvider` and never branches on WebGPU vs. CPU directly.
- [ ] The resolved **capability tier** is exposed (Freezed model): backend kind, particle-count ceiling (web GPU full 24k+ vs. web CPU ~3–4k), reduced-mode flag, and a human-readable reason — consumed by the reduced-mode indicator ([PRIMORDIS-TASK-015](./PRIMORDIS-TASK-015-cross-platform-backend-selection-and-reduced-mode-ux.md)).
- [ ] **Particle-count policy** is applied by tier per [PRIMORDIS-ADR-006](../adr/PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md): the CPU-WASM tier seeds the reduced count (~3–4k, the hard web ceiling) rather than 24k.
- [ ] All probe/interop uses **`dart:js_interop` + `package:web` only**; no `dart:html` / `dart:js_util` anywhere reachable.
- [ ] Selection is async and non-blocking: the UI shows a determinate "detecting / initializing" state and then the resolved simulation; no white-screen hang if WebGPU init stalls.
- [ ] Selection state and tier live in **Riverpod providers with plain `Ref`** — no `setState` for selection logic.

### Versioning (if Flutter/native code changed)

- [ ] Version bumped in `pubspec.yaml` and the app config constant (`PrimordisConfig`/`AppConfig`); semver.

### Test Coverage

- [ ] New/modified Dart has unit/widget tests; `flutter test` passes; `flutter analyze` zero warnings.

## Implementation Notes

- **Hard detect, not feature-flag (from [PRIMORDIS-ADR-002](../adr/PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md)):** `navigator.gpu != null` is the gate, but the real commitment point is a successful `requestAdapter()` → `requestDevice()`. Treat a missing `navigator.gpu`, a `null` adapter, a rejected device request, and an immediate device-loss all as "no GPU tier." This covers the verified gaps (Firefox Linux/Android, Intel Macs, pre-26 Safari/iOS) without hard-coding browser/version strings — detect capability, do not sniff user agents.
- **Selector contract:** `selectBackend()` is `Future<SimBackendSelection>` returning the constructed backend + the `CapabilityTier`. Inject the capability probe so tests can simulate "GPU available", "no `navigator.gpu`", "adapter null", and "device request throws" without a real browser.
- **Tier model:** keep `CapabilityTier` Freezed and platform-neutral (web GPU, web CPU; native tiers added in [PRIMORDIS-TASK-015](./PRIMORDIS-TASK-015-cross-platform-backend-selection-and-reduced-mode-ux.md)) so the same model carries the reduced-mode flag and particle ceiling everywhere.
- **Particle-count ceiling:** the CPU-WASM fallback is single-threaded Dart→WASM and tops out at ~3–4k particles @ 60fps (24k runs ~1–2.5 fps), and the web has no real isolates / `SharedArrayBuffer` needs COOP/COEP. So the selector must hand the CPU tier the reduced seed count, not the full 24k. The GPU tier keeps the full 24k+.
- **Compositor coupling:** when the CPU tier is selected there is **no sibling WebGPU canvas to stack** — the CPU backend draws inside Flutter via a single `Canvas.drawRawPoints` ([PRIMORDIS-TASK-008](./PRIMORDIS-TASK-008-dart-wasm-cpu-fallback-backend.md)). The selector's result must let the compositor ([PRIMORDIS-TASK-005](./PRIMORDIS-TASK-005-web-canvas-compositing-and-pointer-routing.md)) skip creating/attaching the sibling canvas. Keep that branch in the present path, driven by the tier, not in the UI.
- **`flutter_gpu` premise:** this selection assumes no first-party web compute API exists; re-verify `flutter_gpu` against the pinned Flutter version at sign-off but do not architect around it ([PRIMORDIS-ADR-001](../adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md)). The selector should not need changing if that premise holds.
- **Reduced-motion is orthogonal:** `prefers-reduced-motion` pause is an accessibility concern owned by [PRIMORDIS-TASK-006](./PRIMORDIS-TASK-006-sliders-to-uniforms-and-ui-chrome.md) / [PRIMORDIS-ADR-006](../adr/PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md); it is not the same as the GPU/CPU capability tier and must not be conflated in the selector.
- **Logging:** record the resolved tier and reason (no `navigator.gpu` / no adapter / device-lost / GPU OK) so support can diagnose why a given browser landed on the fallback.

## Testing

- [ ] Unit test: probe returns "no `navigator.gpu`" → selector yields the CPU-WASM backend and a reduced tier with the ~3–4k ceiling and a reason string.
- [ ] Unit test: probe returns "GPU available, device acquired" → selector yields the WebGPU backend and the full-24k tier.
- [ ] Unit test: probe has `navigator.gpu` but adapter is `null` → CPU-WASM fallback.
- [ ] Unit test: probe has `navigator.gpu` but `requestDevice()` throws/rejects → CPU-WASM fallback.
- [ ] Unit test: device-lost after init → active-backend provider transitions to the CPU-WASM tier without throwing.
- [ ] Widget test: while selection is pending the UI shows the detecting/initializing state; after resolution it shows the simulation; no indefinite hang.
- [ ] Manual on a **WebGPU browser** (Chrome 113+ / Safari 26 / Firefox 145+ Apple-Silicon): GPU tier selected, 24k particles.
- [ ] Manual on a **non-WebGPU context** (e.g. Firefox Linux, or `navigator.gpu` shimmed out): CPU tier selected, reduced particle count, single `drawRawPoints` render, no attempt to create a WebGPU canvas.
- [ ] `--wasm` sanity: no `dart:html` / `dart:js_util` in the dependency tree introduced by the probe.
- [ ] `flutter analyze` zero warnings; `flutter test` passes.

## Related

- PRD: [PRIMORDIS-PRD-001](../prd/PRIMORDIS-PRD-001-flutter-web-and-macos-port.md)
- ADR: [PRIMORDIS-ADR-006 — CPU fallback tiers and feature detection](../adr/PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md) (primary)
- ADR: [PRIMORDIS-ADR-002 — Web GPU compute via WebGPU + JS interop](../adr/PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md), [PRIMORDIS-ADR-001 — Cross-platform architecture (SimBackend)](../adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md), [PRIMORDIS-ADR-007 — Web build and cross-origin isolation](../adr/PRIMORDIS-ADR-007-web-build-and-cross-origin-isolation.md)
- Depends on: [PRIMORDIS-TASK-004 — Web WebGPU backend (JS interop)](./PRIMORDIS-TASK-004-web-webgpu-backend-js-interop.md), [PRIMORDIS-TASK-008 — Dart-WASM CPU fallback backend](./PRIMORDIS-TASK-008-dart-wasm-cpu-fallback-backend.md)
- Blocks: [PRIMORDIS-TASK-015 — Cross-platform backend selection and reduced-mode UX](./PRIMORDIS-TASK-015-cross-platform-backend-selection-and-reduced-mode-ux.md)
