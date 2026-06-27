# PRIMORDIS-TASK-012: macOS present path — IOSurface-backed Metal texture via FlutterTextureRegistry

**Status:** Todo
**Priority:** High
**Created:** 2026-06-27
**Updated:** 2026-06-27

## Description

Implement the macOS **compute→display present path**: render the simulation's point field into an **IOSurface-backed Metal texture** (BGRA8 / `CVPixelBuffer`), register that texture with **`FlutterTextureRegistry`** (obtained via the macOS plugin registrar), and display it **under** the shared Flutter UI with the Flutter **`Texture` widget**. The simulation becomes a real layer in the Flutter scene, with the sliders/chrome composited on top — the cleaner contract that the macOS embedder supports, decided in [PRIMORDIS-ADR-005](../adr/PRIMORDIS-ADR-005-rendering-and-compositing.md).

This is the downstream half of the macOS backend: [PRIMORDIS-TASK-011](./PRIMORDIS-TASK-011-macos-target-dawn-wgpu-ffi-backend.md) makes compute *run* and fills the storage buffers; this task gets those results onto the screen without a CPU round-trip. The research summary flags this handoff as **the fiddliest integration risk in the project** — frame pacing (`textureFrameAvailable` notifications vs Metal completion) and the hard requirement of **no CPU readback at 24k/60fps**. A stray readback silently tanks performance, so this task's success is defined as much by what must *not* happen (no readback) as by what appears on screen.

## Scope

**Area:** FFI/Native
**Files/Dirs:**
- `macos/Runner/` or `macos/` plugin sources — Swift / Obj-C++ texture bridge implementing the `FlutterTexture` / `copyPixelBuffer` contract and registering with `FlutterTextureRegistry`
- `lib/sim/present/macos_texture_present.dart` — Dart side: receives the registered texture id, drives the `Texture` widget, signals new frames
- `lib/sim/backends/macos_dawn_backend.dart` — present hook wired into the shared frame loop (render-into-texture step)
- `lib/ui/` — `Texture` widget placement beneath the Flutter glass/overlay (sliders + chrome on top)
- `pubspec.yaml` / app config constant — version bump
- `test/sim/present/macos_texture_present_test.dart` — Dart-side present-controller tests

> **Layering note (house standards):** the native texture bridge (Swift/Obj-C++, `IOSurface`, `CVPixelBuffer`, `MTLTexture`, `FlutterTextureRegistry`) lives **outside** the standard Flutter feature/data/domain layers, as expected per [PRIMORDIS-ADR-001](../adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md) and [PRIMORDIS-ADR-005](../adr/PRIMORDIS-ADR-005-rendering-and-compositing.md). It sits behind the present path of `SimBackend`. The `Texture` widget itself is an ordinary Material 3 / Riverpod-driven widget; the UI layer stays standards-compliant and never touches Metal or `IOSurface` directly.

## Acceptance Criteria

- [ ] The simulation is rendered into an **IOSurface-backed Metal texture** (BGRA8 / `CVPixelBuffer`) produced by the macOS GPU backend's render pipeline (the WGSL point-render pass — the `gl_PointSize=2` equivalent, 24k point-list vertices).
- [ ] A native texture object implementing the `copyPixelBuffer`-style **`CVPixelBuffer`/IOSurface contract** is **registered with `FlutterTextureRegistry`** via the plugin registrar, returning a texture id consumed on the Dart side.
- [ ] A Flutter **`Texture` widget** displays that texture id **beneath** the Flutter UI; sliders and chrome ([PRIMORDIS-TASK-006](./PRIMORDIS-TASK-006-sliders-to-uniforms-and-ui-chrome.md)) composite on top as normal Material 3 widgets.
- [ ] **No CPU readback** occurs in the steady-state frame loop: the texture stays GPU-resident from compute through present. Verified by profiling at **24,000 particles / 60fps** (no `getBytes`/`blit`-to-CPU, no `CVPixelBuffer` lock-for-read on the hot path).
- [ ] **Frame pacing is correct:** the frame signalled to Flutter (`textureFrameAvailable`) is the one Metal has actually *completed* (gated on the Metal completion handler / fence), so no torn or stale frames are displayed and the displayed frame matches the latest finished compute step.
- [ ] The `Texture` layer tracks **`devicePixelRatio`** and window resize so the point field stays sharp and correctly sized; the toroidal 1080×720 world maps to the view as it does on web/native parity.
- [ ] The present path is **swappable behind `SimBackend`** and works for both macOS GPU backends — Dawn/wgpu-over-Metal (Approach a, [PRIMORDIS-TASK-011](./PRIMORDIS-TASK-011-macos-target-dawn-wgpu-ffi-backend.md)) and the MSL plugin (Approach b, [PRIMORDIS-TASK-013](./PRIMORDIS-TASK-013-macos-metal-msl-compute-plugin-fallback.md)) — since both render into a Metal texture.
- [ ] **Reduced-motion / pause** holds the **last composited frame** (static texture) without re-dispatching compute, satisfying the accessibility requirement for a full-screen-motion app (per [PRIMORDIS-ADR-005](../adr/PRIMORDIS-ADR-005-rendering-and-compositing.md) and the org accessibility standard); resuming continues the loop.

### Versioning (if Flutter/native code changed)
- [ ] Version bumped in `pubspec.yaml` and the app config constant; semver (minor bump — new present path).

### Test Coverage
- [ ] New/modified Dart has unit/widget tests; `flutter test` passes; `flutter analyze` zero warnings (`package:lint`).
- [ ] Widget test confirms the `Texture` widget renders for a given texture id and sits beneath the slider/chrome overlay; present-controller unit tests cover frame-available signalling, resize/DPR updates, and the pause/last-frame-hold path.

## Implementation Notes

- **Contract is `CVPixelBuffer`/IOSurface.** Implement the macOS `FlutterTexture` `copyPixelBuffer` contract with an `IOSurface`-backed `CVPixelBuffer` (BGRA8). The GPU backend renders into a `MTLTexture` that wraps the same `IOSurface` (e.g. via `CVMetalTextureCache`), so there is **one shared surface** from Metal render to Flutter present — that shared surface is what makes "no CPU readback" achievable.
- **Frame pacing is the core risk.** Do not call `textureFrameAvailable` when the encoder is *submitted*; call it when Metal *completes* (in the command-buffer completion handler, or gated on a fence/event). Mismatching these produces tearing or stale frames. Use double/triple buffering of the IOSurface if needed so compute for frame N+1 does not stomp the surface Flutter is still presenting for frame N.
- **No CPU readback — make it observable.** Profile with the Metal frame capture / GPU timeline to confirm zero CPU→GPU or GPU→CPU copies of the particle texture per frame. A CPU readback at 24k/60fps silently destroys performance, so treat any `getBytes`/buffer-lock-for-read on the hot path as a failing condition, not a warning.
- **Engine support is real.** The macOS desktop embedder supports external textures via `FlutterPluginRegistrar.textures` / `FlutterTextureRegistry` (engine PR flutter/engine#24523). This is *not* the web overlay scheme — on macOS the GPU texture is a true Flutter scene layer composited *under* the UI, whereas web composites Flutter *over* a sibling DOM canvas ([PRIMORDIS-TASK-005](./PRIMORDIS-TASK-005-web-canvas-compositing-and-pointer-routing.md)). The two present paths are asymmetric by design and share no presentation code.
- **Backend-agnostic.** Keep the texture bridge independent of *which* compute backend produced the render: both the Dawn/wgpu path and the MSL plugin path end in a Metal texture, so the bridge consumes a `MTLTexture`/`IOSurface` handle and does not care how it was filled.
- **Pointer/UI on top.** Unlike web, no bespoke pointer routing is needed here — the `Texture` widget is inside Flutter's hit-test tree, so sliders/chrome and any field interaction use Flutter's normal gesture system.
- **Determinism is not a goal** (per [PRIMORDIS-PRD-001](../prd/PRIMORDIS-PRD-001-flutter-web-and-macos-port.md)); the present path neither adds nor removes nondeterminism — it only displays whatever compute produced.

## Testing

- [ ] Launch `flutter run -d macos`; confirm the point field appears **behind** the sliders/chrome and animates at 24k particles.
- [ ] Profile a 24k/60fps run with Metal frame capture / Instruments; confirm **no CPU readback** of the particle texture and stable 60fps frame pacing (no tearing, no stale frames).
- [ ] Resize the window and change display (DPI) and confirm the texture rescales sharply and stays aligned with the overlay.
- [ ] Trigger reduced-motion / pause and confirm the last frame is held statically with compute halted; resume and confirm motion continues.
- [ ] Run with both the Dawn backend ([PRIMORDIS-TASK-011](./PRIMORDIS-TASK-011-macos-target-dawn-wgpu-ffi-backend.md)) and, once available, the MSL plugin ([PRIMORDIS-TASK-013](./PRIMORDIS-TASK-013-macos-metal-msl-compute-plugin-fallback.md)) selected; the present path works unchanged for both.
- [ ] `flutter test` and `flutter analyze` pass with zero warnings.

## Related
- PRD: [PRIMORDIS-PRD-001](../prd/PRIMORDIS-PRD-001-flutter-web-and-macos-port.md)
- ADR: [PRIMORDIS-ADR-005 — Rendering and compositing](../adr/PRIMORDIS-ADR-005-rendering-and-compositing.md); [PRIMORDIS-ADR-004 — Native macOS GPU compute (Dawn/Metal via FFI)](../adr/PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md); [PRIMORDIS-ADR-001 — Cross-platform architecture (SimBackend)](../adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md)
- Depends on: [PRIMORDIS-TASK-011 — macOS target: Dawn/wgpu FFI backend](./PRIMORDIS-TASK-011-macos-target-dawn-wgpu-ffi-backend.md)
- Blocks: [PRIMORDIS-TASK-016 — macOS packaging, signing, and GPU gating](./PRIMORDIS-TASK-016-macos-packaging-signing-and-gpu-gating.md)
