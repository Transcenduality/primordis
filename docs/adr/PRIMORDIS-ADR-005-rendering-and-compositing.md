<!-- Filename convention: <SCOPE>-ADR-NNN-short-title.md -->

# PRIMORDIS-ADR-005: Rendering and Compositing of GPU Output with Flutter UI

**Status:** Proposed
**Date:** 2026-06-27
**Deciders:** Bruce Abernethy
**Review date:** 2026-09-27
**Supersedes:** N/A
**Superseded by:** N/A
**Compliance/Security:** None

## Context

Primordis runs its entire physics in GPU compute shaders and presents the result as a field of 24,000 points (`gl_PointSize=2` equivalent) over a 1080x720 toroidal world. The compute path lives **outside** Flutter on every platform — browser WebGPU on web (see [PRIMORDIS-ADR-002](./PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md)) and Dawn/wgpu-over-Metal via `dart:ffi` on macOS (see [PRIMORDIS-ADR-004](./PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md)), driving one shared WGSL kernel ([PRIMORDIS-ADR-003](./PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md)). That GPU output does not flow through Flutter's own rendering context (web = CanvasKit/Skwasm over Skia/WebGL; macOS = Impeller/Skia), so it cannot simply be drawn into a Flutter `Canvas`. We must decide **how the GPU-rendered point field is composited together with the Flutter UI** (the three sliders — Attraction K, Repulsion K, Drift/friction — plus reset/seed chrome) so that the simulation appears behind the controls, pointer input reaches the right layer, and the image stays sharp across display densities.

This is a per-platform integration decision because the two platforms expose fundamentally different compositing surfaces:

- **Web.** Flutter renders into its own `<canvas>`/DOM subtree. There is no first-party way to hand an external GPU-rendered canvas to the Flutter compositor as a true layer. The available embedding primitive, `HtmlElementView`, forces Flutter's web compositor into platform-view/overlay mode, which splits the Flutter canvas around the embedded element and is subject to overlay-count limits and jank.
- **macOS.** The Flutter desktop embedder exposes an external-texture path: a GPU texture can be registered with `FlutterTextureRegistry` and displayed by the `Texture` widget as a real layer inside the Flutter scene. This is a cleaner contract than the web overlay, but it carries its own integration risk in the compute-to-display handoff (frame pacing, avoiding CPU readback).

The render cost itself is trivial (one point-list draw of 24k vertices per frame); the engineering cost and risk are entirely in the **compositing, pointer routing, and density/resize plumbing**, which the effort estimate in the research summary explicitly calls out as where web time is spent.

Note on the CPU-fallback render path: when no GPU is available the simulation is drawn entirely inside Flutter via a single `Canvas.drawRawPoints` call from a `Float32List`. That is a different render path with different compositing properties and is owned by [PRIMORDIS-ADR-006](./PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md); it is listed under Alternatives below only to make the boundary explicit.

## Decision

We will composite GPU output with Flutter UI using a **platform-specific present path behind the `SimBackend` interface** ([PRIMORDIS-ADR-001](./PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md)). The UI layer never knows which present path is live.

**Web — stacked sibling canvas behind a transparent Flutter glass-pane:**

1. Render the simulation into a **WebGPU `<canvas>` that we own**, configured on the WebGPU device created by the web backend ([PRIMORDIS-ADR-002](./PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md)). The point field is drawn with a WebGPU render pipeline (point-list, the WGSL equivalent of `gl_PointSize=2`).
2. Place that `<canvas>` in the DOM as a **sibling element positioned BEHIND** Flutter's view, and make the Flutter view a **transparent glass-pane** so the Flutter widget tree (sliders, chrome) overlays the simulation. Flutter draws only the UI; the empty/background regions are transparent so the WebGPU canvas shows through.
3. **Do NOT wrap the WebGPU canvas in `HtmlElementView`.** Embedding it as a platform view forces Flutter's web compositor into overlay/canvas-splitting mode, is subject to overlay-count limits, and introduces jank — exactly the failure modes we are avoiding.
4. **Route pointer events explicitly.** Because the WebGPU canvas is a sibling DOM element outside Flutter's hit-test tree, pointer interaction (e.g. dragging in the field, and ensuring slider/chrome gestures still reach Flutter) is wired by hand via `dart:js_interop` + `package:web` rather than relying on Flutter's automatic hit testing.
5. **Sync to `devicePixelRatio`.** Size the WebGPU canvas backing store to the logical size multiplied by `MediaQuery.devicePixelRatio`, and re-apply on every resize so the point field stays sharp and aligned with the Flutter overlay at all display densities.

**macOS — IOSurface-backed Metal texture via `FlutterTextureRegistry` + `Texture` widget:**

1. Render the simulation into an **IOSurface-backed Metal texture** (BGRA8 / `CVPixelBuffer`).
2. **Register that texture with `FlutterTextureRegistry`** (obtained via the plugin registrar) and display it **under** the shared Flutter UI using the Flutter **`Texture` widget**, which is supported on the macOS desktop embedder. The simulation becomes a real layer in the Flutter scene with the sliders/chrome composited on top.
3. The texture **contract is `CVPixelBuffer`/IOSurface** (the `copyPixelBuffer`-style handoff). **Watch frame pacing** — coordinate `textureFrameAvailable` notifications against Metal completion so the displayed frame is the one that finished — and **avoid CPU readback**: the texture must stay GPU-resident from compute through present at 24k/60fps. No readback to CPU is permitted on this path.

In both cases the GPU sits **behind** the Flutter UI and the Flutter UI is rendered normally (Material 3, the standards-compliant widget tree). Only the present/composite step is platform-specific; particle seeding, the frame loop, parameter marshalling, and the WGSL kernel source are shared.

## Consequences

### Positive

- **No first-party compute API is required for presentation either.** GPU output is shown by mechanisms that already exist and are verified-supported (sibling DOM canvas on web; `FlutterTextureRegistry`/`Texture` on macOS), consistent with the "all GPU work lives outside Flutter" posture of ADRs 002–004.
- **Avoids the known web embedding failure modes.** Rejecting `HtmlElementView` for the WebGPU canvas sidesteps overlay/canvas-splitting, overlay-count limits, and the associated jank, keeping the simulation at full frame rate while Flutter overlays the controls.
- **macOS gets the cleaner contract.** The external-texture `Texture` widget composites the simulation as a true Flutter layer, which is cleaner than the web overlay and lets the same Flutter UI sit on top unchanged.
- **UI stays standards-compliant.** Because the GPU surface is composited behind a normal Flutter widget tree, the sliders/chrome remain ordinary Material 3 + Riverpod widgets with semantics/tooltips; the non-standard JS-interop/FFI/texture-bridge code is confined to the present path behind `SimBackend`.
- **Sharp output across densities.** Explicit `devicePixelRatio` synchronization on web keeps the point field crisp and aligned with the overlay on high-DPI displays and through resizes.

### Negative

- **Manual pointer routing on web is bespoke and fragile.** Because the WebGPU canvas lives outside Flutter's hit-test tree, every interaction has to be wired explicitly via `dart:js_interop`/`package:web`; getting the layering of pointer events between the sibling canvas and the Flutter glass-pane correct is real work and a maintenance surface.
- **DPR/resize plumbing is a recurring source of bugs.** Backing-store sizing must track `MediaQuery.devicePixelRatio` on every resize; mistakes show up as blurriness or misalignment between the simulation and the overlay.
- **macOS compute-to-display handoff is the fiddliest integration risk in the project.** Frame pacing (`textureFrameAvailable` vs Metal completion) and the hard "no CPU readback at 24k/60fps" requirement make this the most delicate part of the native build; a stray readback silently tanks performance.
- **Two present paths to build and maintain.** The web stacked-canvas path and the macOS texture path share no presentation code, doubling the surface that must be kept working as Flutter and the GPU layers evolve.

### Neutral

- **Two different compositing models by design.** Web composites Flutter *over* a sibling DOM canvas; macOS composites the GPU texture *under* Flutter as a scene layer. Both satisfy "GPU behind UI," but they are not symmetric and are documented as such.
- **`std430` → Metal alignment and atomics parity** are validated on the compute side ([PRIMORDIS-ADR-003](./PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md), [PRIMORDIS-ADR-004](./PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md), and [PRIMORDIS-TASK-017](../tasks/PRIMORDIS-TASK-017-atomics-parity-validation-dawn-vs-browser.md)); this ADR governs only the present/composite step downstream of those.
- **Reduced-motion accessibility** (offering a pause/static state for a full-screen-motion app) is a cross-cutting requirement satisfied by pausing the frame loop and holding the last composited frame; it is owned by [PRIMORDIS-ADR-006](./PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md) and the UI tasks rather than by the compositing mechanism itself.

## Alternatives Considered

### Web: embed the WebGPU canvas with `HtmlElementView` (platform view)

Wrap the owned WebGPU `<canvas>` in a Flutter `HtmlElementView` so Flutter "owns" it as a platform view in the widget tree. **Rejected for web.** This forces Flutter's web compositor into overlay/canvas-splitting mode, is subject to overlay-count limits, and introduces jank — the exact problems the stacked sibling-canvas + transparent glass-pane approach exists to avoid. The trade is that the chosen approach requires explicit pointer routing and DPR sync (see Negative), which is the accepted cost of avoiding the compositor pathology.

### macOS: present via a sibling/native overlay window instead of an external texture

Composite on macOS the way we do on web (a separate native surface stacked behind a transparent Flutter window) rather than through `FlutterTextureRegistry`. **Not chosen.** The macOS embedder genuinely supports the external-texture `Texture` widget path (`FlutterPluginRegistrar.textures` / `FlutterTextureRegistry`), which composites the simulation as a real Flutter scene layer and is cleaner than an overlay-window scheme. The texture path's only cost — frame pacing and the no-CPU-readback constraint — is accepted and tracked in [PRIMORDIS-TASK-012](../tasks/PRIMORDIS-TASK-012-macos-metal-texture-present-path.md).

### CPU `Canvas.drawRawPoints` rendering inside Flutter

Draw the point field directly in Flutter from a `Float32List` using a single `Canvas.drawRawPoints` (or `drawVertices`) call, with no external GPU surface to composite at all. **Out of scope for this ADR — it is the CPU-fallback render path**, used only when GPU initialization fails (web ~3–4k particles; native isolate fallback ~10–14k). It is owned by [PRIMORDIS-ADR-006](./PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md) and is mentioned here solely to fix the boundary: when this path is active there is no GPU output to composite, so the compositing decisions above do not apply.

## References

PRIMORDIS docs:

- [PRIMORDIS-ADR-001 — Cross-platform architecture (SimBackend)](./PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md)
- [PRIMORDIS-ADR-002 — Web GPU compute via WebGPU + JS interop](./PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md)
- [PRIMORDIS-ADR-003 — Shared WGSL compute kernel](./PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md)
- [PRIMORDIS-ADR-004 — Native macOS GPU compute (Dawn/Metal via FFI)](./PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md)
- [PRIMORDIS-ADR-006 — CPU fallback tiers and feature detection](./PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md)
- [PRIMORDIS-ADR-007 — Web build and cross-origin isolation](./PRIMORDIS-ADR-007-web-build-and-cross-origin-isolation.md)
- [PRIMORDIS-PRD-001 — Flutter Web and macOS port](../prd/PRIMORDIS-PRD-001-flutter-web-and-macos-port.md)
- [PRIMORDIS research summary](../research/PRIMORDIS-research-summary.md)

Related tasks:

- [PRIMORDIS-TASK-005 — Web canvas compositing and pointer routing](../tasks/PRIMORDIS-TASK-005-web-canvas-compositing-and-pointer-routing.md)
- [PRIMORDIS-TASK-004 — Web WebGPU backend (JS interop)](../tasks/PRIMORDIS-TASK-004-web-webgpu-backend-js-interop.md)
- [PRIMORDIS-TASK-012 — macOS Metal texture present path](../tasks/PRIMORDIS-TASK-012-macos-metal-texture-present-path.md)
- [PRIMORDIS-TASK-011 — macOS target: Dawn/wgpu FFI backend](../tasks/PRIMORDIS-TASK-011-macos-target-dawn-wgpu-ffi-backend.md)
- [PRIMORDIS-TASK-008 — Dart-WASM CPU fallback backend](../tasks/PRIMORDIS-TASK-008-dart-wasm-cpu-fallback-backend.md)
- [PRIMORDIS-TASK-018 — Test coverage and accessibility](../tasks/PRIMORDIS-TASK-018-test-coverage-and-accessibility.md)

External documentation topics:

- flutter.dev — `Texture` widget and external textures (`TextureRegistry` / `FlutterTextureRegistry`; macOS desktop embedder support; engine PR flutter/engine#24523).
- flutter.dev — `HtmlElementView` and web platform views (overlay / canvas-splitting behavior and overlay-count limits).
- flutter.dev — `MediaQuery.devicePixelRatio` and high-DPI handling.
- flutter.dev — `dart:js_interop` and `package:web` for owning a DOM canvas and routing pointer events on web.
- WebGPU (W3C) — `GPUCanvasContext` / `configure()`, render pipelines, and point-list (`primitive.topology = "point-list"`) rendering.
- Apple Developer (Metal) — `IOSurface`-backed `MTLTexture`, `CVPixelBuffer`/`CVMetalTextureCache`, and BGRA8 texture formats; frame completion handlers for pacing.
