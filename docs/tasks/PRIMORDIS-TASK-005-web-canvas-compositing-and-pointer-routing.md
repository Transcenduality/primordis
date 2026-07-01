# PRIMORDIS-TASK-005: Web canvas compositing and pointer routing

**Status:** Complete
**Priority:** High
**Created:** 2026-06-27
**Updated:** 2026-07-01

> **Delivered via [PR #6](https://github.com/babernethy/primordis/pull/6)** — merged 2026-07-01 (merge commit `83a2b46`). Retained here as a historical file alongside TASK-001..004. Pending work (TASK-006 onward) now lives in the `dgroup-standards` MCP server under the `PRIMORDIS_WEB` scope.

## Description

Composite the web WebGPU simulation with the Flutter UI so the point field renders *behind* the Flutter controls, pointer input reaches the correct layer, and the image stays sharp across display densities and through resizes.

The web backend (see [PRIMORDIS-TASK-004](./PRIMORDIS-TASK-004-web-webgpu-backend-js-interop.md)) already owns a WebGPU `<canvas>` and runs the shared WGSL kernel ([PRIMORDIS-TASK-003](./PRIMORDIS-TASK-003-port-simulation-to-wgsl-compute-kernel.md)) plus a point-render pass on it. That GPU output does **not** flow through Flutter's web rendering context (CanvasKit/Skwasm over Skia/WebGL), so it cannot be drawn into a Flutter `Canvas`. This task implements the present/composite mechanism decided in [PRIMORDIS-ADR-005](../adr/PRIMORDIS-ADR-005-rendering-and-compositing.md): stack the owned WebGPU `<canvas>` in the DOM as a **sibling element positioned behind** a **transparent Flutter glass-pane**, overlay the Flutter widget tree (sliders, chrome) on top, route pointer events explicitly, and sync the canvas backing store to `devicePixelRatio` on every resize.

This is the effort-heavy part of the web work: the shader port is the easy part; the DOM/Flutter compositing, pointer routing, and DPR/resize plumbing are where the time goes.

## Scope

**Area:** Flutter
**Files/Dirs:**
- `lib/sim/web/web_canvas_compositor.dart` — owns the sibling WebGPU `<canvas>` lifecycle, DOM stacking, DPR/resize sync, and pointer-event wiring (JS-interop layer, behind `SimBackend`).
- `lib/sim/web/web_pointer_router.dart` — explicit pointer-event routing between the sibling canvas and the Flutter glass-pane.
- `lib/sim/web/web_sim_backend.dart` — web `SimBackend` impl (from [PRIMORDIS-TASK-004](./PRIMORDIS-TASK-004-web-webgpu-backend-js-interop.md)); extended to surface the canvas handle and resize/DPR hooks to the compositor.
- `lib/features/simulation/widgets/simulation_view.dart` — the Flutter glass-pane host widget (transparent background) that overlays the controls over the sibling canvas region.
- `web/index.html`, `web/styles.css` (or equivalent) — DOM container/stacking context and z-order for the sibling canvas vs. the Flutter host.
- `test/sim/web/web_pointer_router_test.dart`, `test/features/simulation/simulation_view_test.dart` — unit/widget tests.

Note (house standards): the `<canvas>` ownership, DOM stacking, `dart:js_interop`/`package:web` pointer wiring, and DPR sync are **JS-interop code that necessarily lives outside the standard feature/data/domain layers**. This is expected and sanctioned provided it stays quarantined under `lib/sim/web/` behind the `SimBackend` interface, so the `features/simulation` UI layer remains plain Riverpod/Freezed/Material 3.

## Acceptance Criteria

- [x] The WebGPU `<canvas>` is created/attached as a **sibling DOM element positioned behind** the Flutter view; it is **NOT** wrapped in `HtmlElementView` (no platform-view/overlay/canvas-splitting path).
- [x] The Flutter view is configured as a **transparent glass-pane**: the simulation shows through all background/empty regions while the sliders and chrome render normally on top.
- [x] All DOM/canvas/pointer interop uses **`dart:js_interop` + `package:web` only**; no `dart:html` and no `dart:js_util` are introduced anywhere reachable from the dependency tree (required by the `--wasm` build, see [PRIMORDIS-ADR-007](../adr/PRIMORDIS-ADR-007-web-build-and-cross-origin-isolation.md)).
- [x] **Pointer routing is explicit:** gestures over the slider/chrome region reach Flutter widgets; pointer interaction over the open field reaches the simulation/backend; no events are lost or double-handled at the seam.
- [x] The canvas backing store is sized to `logicalSize * MediaQuery.devicePixelRatio`, and **re-applied on every resize** (including DPR change, e.g. dragging a window between displays); the point field stays sharp and aligned with the Flutter overlay.
- [x] On resize, the canvas backing store, the WebGPU surface configuration, and the Flutter glass-pane bounds stay in lockstep with no visible tearing, gap, or misalignment between the simulation and the overlay.
- [x] The 1080×720 toroidal world maps correctly into the displayed canvas region (aspect/letterboxing behavior is defined and consistent across densities).
- [x] Reduced-motion: when the frame loop is paused (accessibility pause / `prefers-reduced-motion`), the last composited frame is held and the overlay remains fully interactive (pause owned by [PRIMORDIS-TASK-006](./PRIMORDIS-TASK-006-sliders-to-uniforms-and-ui-chrome.md) / [PRIMORDIS-ADR-006](../adr/PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md); this task must not break holding the last frame).
- [x] No `setState`-driven business logic: compositor/canvas state that the UI observes is exposed via Riverpod providers with a plain `Ref`.

### Versioning (if Flutter/native code changed)

- [x] Version bumped in `pubspec.yaml` and the app config constant (`PrimordisConfig`/`AppConfig`); semver.

### Test Coverage

- [x] New/modified Dart has unit/widget tests; `flutter test` passes; `flutter analyze` zero warnings.

## Implementation Notes

- **Compositing model (from [PRIMORDIS-ADR-005](../adr/PRIMORDIS-ADR-005-rendering-and-compositing.md)):** web composites Flutter *over* a sibling DOM canvas (the inverse of macOS, which composites the GPU texture *under* Flutter via the `Texture` widget). Do not try to make the two symmetric.
- **Why not `HtmlElementView`:** embedding the canvas as a platform view forces Flutter's web compositor into overlay/canvas-splitting mode, is subject to overlay-count limits, and causes jank. The sibling-canvas + transparent-glass-pane approach is chosen precisely to avoid that; the accepted cost is the manual pointer routing and DPR sync implemented here.
- **DOM stacking:** the sibling `<canvas>` and the Flutter host element share a positioned container; the canvas sits at a lower z-index, the Flutter host above it with a transparent background. Establish the stacking context in `web/index.html` / CSS so Flutter's own canvas does not paint an opaque background over the simulation.
- **Transparent glass-pane:** ensure the Flutter web view background is transparent so only the widget tree paints; verify under both Skwasm (`--wasm`) and the CanvasKit/dart2js fallback that the simulation remains visible behind it (build/renderer matrix owned by [PRIMORDIS-ADR-007](../adr/PRIMORDIS-ADR-007-web-build-and-cross-origin-isolation.md)).
- **Pointer routing:** because the WebGPU canvas lives outside Flutter's hit-test tree, wire pointer/touch listeners explicitly via `dart:js_interop` + `package:web`. Decide and document the hit-test policy at the seam: control region → Flutter (let events fall through to the glass-pane / be consumed by widgets); field region → forward to the backend. Use `pointer-events` CSS and/or explicit listener targeting; account for capture/bubble so a drag started over the field is not stolen by the overlay and vice versa.
- **DPR/resize:** observe size via `MediaQuery` (logical) and `devicePixelRatio`; on change, resize the canvas backing store to `logical * dpr`, re-`configure()` the WebGPU `GPUCanvasContext` if required, and update the Flutter glass-pane bounds in the same frame to avoid a one-frame mismatch. Debounce rapid resizes but always converge to the final size. Mistakes here surface as blurriness or simulation/overlay misalignment — the recurring bug class flagged in the ADR.
- **Backend boundary:** the compositor talks to the web `SimBackend` only through the interface ([PRIMORDIS-ADR-001](../adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md)); the UI never learns which backend is live. Keep canvas-handle, present, and resize hooks on the backend, not in the widget layer.
- **No GPU output to composite when the CPU fallback is active:** if WebGPU is absent and the app falls back to the Dart→WASM CPU backend ([PRIMORDIS-TASK-008](./PRIMORDIS-TASK-008-dart-wasm-cpu-fallback-backend.md)), the sim is drawn inside Flutter via a single `Canvas.drawRawPoints` and there is no sibling canvas to stack. The compositor must cleanly detach/not-create the sibling canvas in that tier (selection logic owned by [PRIMORDIS-TASK-007](./PRIMORDIS-TASK-007-webgpu-feature-detection-and-fallback-switch.md)).

## Testing

- [x] Widget test: `SimulationView` renders the transparent glass-pane and overlays slider/chrome widgets; background is transparent (no opaque paint over the sibling-canvas region).
- [x] Unit test (`web_pointer_router`): given pointer positions in the control region vs. the field region, the router dispatches to the Flutter target vs. the backend target respectively, with no double-dispatch.
- [x] Manual/integration on **Chrome/Edge 113+** and **Safari 26 / Firefox 145+ (Apple-Silicon)**: simulation renders behind the controls; sliders are draggable; field interaction reaches the backend.
- [x] DPR test: load at `devicePixelRatio` 1, 2, and 3 (or drag the window between a non-Retina and Retina display); confirm the point field is crisp and aligned with the overlay at each density, and re-aligns after the move.
- [x] Resize test: continuously resize the window; confirm no tearing/gaps and that the canvas + glass-pane converge to the final size without leftover misalignment.
- [x] `--wasm` build sanity: confirm no `dart:html` / `dart:js_util` appears in the dependency tree (build does not fall back to legacy interop); verify the same behavior under the CanvasKit/dart2js fallback renderer.
- [x] Reduced-motion: pause the loop and confirm the last frame holds while the overlay stays interactive.
- [x] `flutter analyze` reports zero warnings; `flutter test` passes.

## Related

- PRD: [PRIMORDIS-PRD-001](../prd/PRIMORDIS-PRD-001-flutter-web-and-macos-port.md)
- ADR: [PRIMORDIS-ADR-005 — Rendering and compositing](../adr/PRIMORDIS-ADR-005-rendering-and-compositing.md) (primary)
- ADR: [PRIMORDIS-ADR-002 — Web GPU compute via WebGPU + JS interop](../adr/PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md), [PRIMORDIS-ADR-001 — Cross-platform architecture (SimBackend)](../adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md), [PRIMORDIS-ADR-007 — Web build and cross-origin isolation](../adr/PRIMORDIS-ADR-007-web-build-and-cross-origin-isolation.md)
- Depends on: [PRIMORDIS-TASK-004 — Web WebGPU backend (JS interop)](./PRIMORDIS-TASK-004-web-webgpu-backend-js-interop.md)
- Blocks: [PRIMORDIS-TASK-010 — Web build, hosting, and cross-origin isolation](./PRIMORDIS-TASK-010-web-build-hosting-and-cross-origin-isolation.md)
