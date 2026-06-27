# PRIMORDIS-TASK-004: Web WebGPU SimBackend via `dart:js_interop` + `package:web`

**Status:** Todo
**Priority:** Critical
**Created:** 2026-06-27
**Updated:** 2026-06-27

## Description

Implement the **web** `SimBackend` ([PRIMORDIS-TASK-002](./PRIMORDIS-TASK-002-simbackend-interface-and-shared-sim-model.md)) that runs the shared WGSL compute kernel ([PRIMORDIS-TASK-003](./PRIMORDIS-TASK-003-port-simulation-to-wgsl-compute-kernel.md)) on **browser WebGPU**, reached from Dart through **`dart:js_interop` + `package:web`** on a `<canvas>` the app owns. The backend acquires `navigator.gpu` -> adapter -> device, creates the storage/uniform buffers, the three compute pipelines (clear / bin / interaction+integrate) and the point-render pipeline, and dispatches all four passes per frame, presenting particle points (`gl_PointSize = 2` equivalent) onto its own canvas.

This is the GPU path that delivers the **full 24,000+ particles at 60fps** where WebGPU is present (Chrome/Edge 113+, Safari 26, Firefox 141+ Windows / 145+ Apple-Silicon Mac). It is gated by a hard `navigator.gpu` feature-detect; when WebGPU is absent the app degrades to the Dart->WASM CPU fallback ([PRIMORDIS-TASK-008](./PRIMORDIS-TASK-008-dart-wasm-cpu-fallback-backend.md)) via the selection logic in [PRIMORDIS-TASK-007](./PRIMORDIS-TASK-007-webgpu-feature-detection-and-fallback-switch.md). This task delivers the backend and its render; the DOM stacking/compositing and pointer routing are owned by [PRIMORDIS-TASK-005](./PRIMORDIS-TASK-005-web-canvas-compositing-and-pointer-routing.md), and slider-to-uniform wiring by [PRIMORDIS-TASK-006](./PRIMORDIS-TASK-006-sliders-to-uniforms-and-ui-chrome.md). The interop/WebGPU code lives **outside** the standard feature/data/domain layers, quarantined behind `SimBackend` per [PRIMORDIS-ADR-002](../adr/PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md).

The build target is `flutter build web --wasm` (Skwasm), which **forbids legacy interop** (`dart:html` / `dart:js_util`) anywhere in the dependency tree — so this backend must use `dart:js_interop` + `package:web` **exclusively** ([PRIMORDIS-ADR-007](../adr/PRIMORDIS-ADR-007-web-build-and-cross-origin-isolation.md)).

## Scope

**Area:** FFI/Native (JS-interop layer; lives outside the standard Dart layers behind `SimBackend`)

**Files/Dirs:**

- `lib/sim/backends/web/web_webgpu_backend.dart` — the `SimBackend` implementation: device acquisition, buffer/pipeline creation, per-frame dispatch + present, dispose.
- `lib/sim/backends/web/webgpu_interop.dart` — `dart:js_interop` `extension type` bindings for the WebGPU JS API surface used (`navigator.gpu`, `GPUAdapter`, `GPUDevice`, `GPUQueue`, `GPUBuffer`, `GPUShaderModule`, `GPUComputePipeline`, `GPURenderPipeline`, `GPUBindGroup`/`GPUBindGroupLayout`, `GPUCommandEncoder`, `GPUComputePassEncoder`, `GPURenderPassEncoder`, `GPUCanvasContext`) and `package:web` `HTMLCanvasElement`.
- `lib/sim/backends/web/web_canvas_handle.dart` — owns the `HTMLCanvasElement` + its `GPUCanvasContext` configuration (format, alpha mode for transparency under the Flutter glass-pane); shared with TASK-005.
- `lib/sim/backends/web/buffer_marshalling.dart` — packs the Freezed sim params (32x32 `forces`/`min_distances`/`radii`, per-type colours), the particle seed, and the uniform block into `ByteData`/typed-list views matching `buffer_layout.dart` from TASK-003; uploads via `GPUQueue.writeBuffer`.
- `lib/sim/providers/` — Riverpod provider(s) exposing the web backend instance behind the `SimBackend` interface (conditional-import wiring so non-web builds do not pull web-only code).
- `pubspec.yaml` — add `web` (`package:web`) dependency; confirm no `dart:html`/`dart:js_util` anywhere in the dep tree.
- `test/sim/backends/web/` — unit tests for marshalling and feature-detect; integration smoke harness for the dispatch loop.

## Acceptance Criteria

- [ ] The web `SimBackend` acquires `navigator.gpu` -> `requestAdapter()` -> `requestDevice()` and fails gracefully (returns an unavailable/error state for the selector in [PRIMORDIS-TASK-007](./PRIMORDIS-TASK-007-webgpu-feature-detection-and-fallback-switch.md)) when any step is null/throws, rather than crashing the app.
- [ ] All WebGPU access uses `dart:js_interop` `extension type`s + `package:web`; there is **no** `dart:html` and **no** `dart:js_util` anywhere in the dependency tree (verified for the `--wasm` build, [PRIMORDIS-ADR-007](../adr/PRIMORDIS-ADR-007-web-build-and-cross-origin-isolation.md)).
- [ ] The backend creates storage buffers for particle state (SoA/AoS per `buffer_layout.dart`), the 32x32 parameter matrices, the bin-count array (`atomic<u32>` x 77), the bin-index array (capped at `MAX_BIN_PARTICLES` = 512 per bin), and a uniform buffer — all with byte layouts identical to TASK-003 so the **same** WGSL source binds correctly.
- [ ] Three compute pipelines (clear / bin / interaction+integrate) and one point-render pipeline are created from the **shared** WGSL source string loaded via `kernel_source.dart` (TASK-003); no web-specific copy of the kernel exists.
- [ ] Per frame the backend encodes: clear pass -> bin pass -> interaction pass (each a `GPUComputePassEncoder` with `dispatchWorkgroups(ceil(count / WORKGROUP_SIZE))`) -> point-render pass, submits one command buffer, and presents to the owned canvas; runs 24,000 particles / 32 types at 60fps on a WebGPU-capable browser.
- [ ] The three live sliders (Attraction K, Repulsion K, Drift/friction) plus `dt` and counts are written into the uniform buffer each frame via `GPUQueue.writeBuffer` (full wiring from Riverpod state is TASK-006; this task exposes the marshalling entry point).
- [ ] Particles render as points with per-type colour (`gl_PointSize = 2` equivalent) onto the owned `<canvas>`, configured with the alpha mode required to sit transparently under the Flutter glass-pane (compositing itself is TASK-005).
- [ ] The backend implements the full `SimBackend` lifecycle (init / seed / step / resize / dispose), releasing all GPU resources on dispose with no leaks across hot-restart.
- [ ] The backend is exposed only through the `SimBackend` Riverpod provider using a plain `Ref`; no `setState` is used for sim/business logic, and the UI never references WebGPU types directly (standards quarantine per [PRIMORDIS-ADR-001](../adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md)).
- [ ] Web-only code is conditionally imported so native/macOS builds compile without pulling `package:web`/WebGPU interop.

### Versioning (if Flutter/native code changed)

- [ ] Version bumped in `pubspec.yaml` and the `PrimordisConfig` app config constant; semver (minor bump — web GPU backend).

### Test Coverage

- [ ] New/modified Dart has unit/widget tests (buffer marshalling round-trips against `buffer_layout.dart`; `navigator.gpu` feature-detect branch; graceful-failure path when adapter/device is null); `flutter test` passes; `flutter analyze` zero warnings.

## Implementation Notes

- **Why this path (not Flutter compute).** Per [PRIMORDIS-ADR-002](../adr/PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md): `dart:ui` `FragmentProgram` is fragment-only (no SSBO/atomics) and `flutter_gpu` is render-only and absent on web. The only compute-capable web surface is **browser WebGPU**, so all compute lives outside Flutter's CanvasKit/Skwasm stack on a canvas the app owns.
- **Owned canvas, not `HtmlElementView`.** The WebGPU `<canvas>` is created and managed as a sibling DOM element (via `package:web`), **not** wrapped in `HtmlElementView` — that forces overlay/canvas-splitting and jank ([PRIMORDIS-ADR-005](../adr/PRIMORDIS-ADR-005-rendering-and-compositing.md)). This task creates and configures the canvas + `GPUCanvasContext`; TASK-005 handles stacking it behind the transparent Flutter glass-pane, pointer routing, and DPR/resize sync. Configure the context's alpha mode (e.g. premultiplied) so the canvas can sit transparently under the UI.
- **`extension type` interop pattern.** Model the WebGPU API as `dart:js_interop` `extension type`s over `JSObject`. Async calls (`requestAdapter`, `requestDevice`, `createRenderPipelineAsync` if used) return JS promises -> use `.toDart` `Future`s. Pass shader source as a `JSString`; pass buffer data as typed-list JS views (`toJS` on `Float32List`/`Uint32List`). Do **not** reach for `dart:js_util` helpers — everything must be expressible with `js_interop`.
- **Shared kernel, shared layout.** The WGSL string and the canonical constants (`WORKGROUP_SIZE`, bind-group indices, buffer sizes) come from `kernel_source.dart`/`buffer_layout.dart` (TASK-003). Buffer byte offsets and the uniform block layout must match the kernel's `var<storage>`/uniform declarations exactly; a mismatch silently corrupts the sim. This identical source is also what the native Dawn/wgpu backend ([PRIMORDIS-TASK-011](./PRIMORDIS-TASK-011-macos-target-dawn-wgpu-ffi-backend.md)) consumes — keep all binding indices in the shared accessor so the two backends cannot drift.
- **Atomic bin buffer.** The bin-count buffer is `atomic<u32>` in WGSL (TASK-003); on the Dart side it is just a `u32` storage buffer of length 77 — correctness depends on the kernel using `atomicAdd`/`atomicLoad`. Naga (the browser WGSL translator) has a history of `atomicAdd` bugs; any anomaly in scatter-binning here feeds [PRIMORDIS-TASK-017](./PRIMORDIS-TASK-017-atomics-parity-validation-dawn-vs-browser.md) (Tint-vs-Naga parity) rather than being patched per-backend.
- **Frame loop + param marshalling.** Drive `step()` from the shared frame loop (TASK-002). Each frame: write the uniform buffer (sliders/`dt`/counts), encode the four passes, submit. Avoid per-frame buffer re-creation; reuse buffers and only `writeBuffer` the small uniform block. Avoid any CPU readback of particle state in the hot path (it would kill 60fps at 24k).
- **`--wasm` constraint.** The build forbids `dart:html`/`dart:js_util` in the entire dep tree ([PRIMORDIS-ADR-007](../adr/PRIMORDIS-ADR-007-web-build-and-cross-origin-isolation.md)). Use `package:web` for DOM (`HTMLCanvasElement`) and `dart:js_interop` for the WebGPU bindings. Add a CI/dep-tree check that fails if either legacy library appears.
- **Standards quarantine.** This is interop/GPU glue and necessarily lives outside the Riverpod/Freezed feature/data/domain layers — expected and sanctioned by [PRIMORDIS-ADR-001](../adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md)/[PRIMORDIS-ADR-002](../adr/PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md). Keep it strictly behind `SimBackend`; expose it to the app only via a Riverpod provider (plain `Ref`, no `setState`). The Freezed sim-param/colour models (TASK-002) are inputs marshalled into buffers here, not redefined.
- **Accessibility note.** This backend must support a paused/static state so the reduced-motion path (TASK-018 / [PRIMORDIS-ADR-006](../adr/PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md)) can stop the animation; expose a `pause()`/single-step capability on the backend rather than only a free-running loop.

## Testing

- [ ] **Feature-detect unit test:** mock the absence and presence of `navigator.gpu` and assert the backend reports availability correctly and never throws on absence (feeds TASK-007).
- [ ] **Graceful failure:** simulate `requestAdapter()`/`requestDevice()` returning null/throwing and assert the backend surfaces an error state instead of crashing.
- [ ] **Marshalling round-trip:** pack the 32x32 matrices, colours, seed, and uniform block, then read the bytes back and assert offsets/values match `buffer_layout.dart` (TASK-003) exactly.
- [ ] **Browser integration smoke (Chrome/Edge 113+ headless or Safari 26 / Firefox 141+/145+):** initialize the device, run several hundred frames at 24,000 particles / 32 types, and assert no exceptions, no device-lost, and a sustained ~60fps; capture frames for the parity harness ([PRIMORDIS-TASK-009](./PRIMORDIS-TASK-009-parity-test-harness-vs-python-reference.md)).
- [ ] **Dispose/leak check:** init -> dispose -> re-init repeatedly (and across hot-restart) and assert GPU resources are released (no growing buffer/pipeline count).
- [ ] **`--wasm` dep-tree guard:** assert the build contains no `dart:html` / `dart:js_util` import anywhere (CI check).
- [ ] **Pause/static:** assert `pause()`/single-step halts the dispatch loop (reduced-motion support hand-off).
- [ ] `flutter test` passes; `flutter analyze` reports zero warnings; `flutter build web --wasm` succeeds.

## Related

- PRD: [PRIMORDIS-PRD-001](../prd/PRIMORDIS-PRD-001-flutter-web-and-macos-port.md)
- ADR: [PRIMORDIS-ADR-002](../adr/PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md) (web WebGPU via js_interop — primary), [PRIMORDIS-ADR-001](../adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md) (SimBackend quarantine), [PRIMORDIS-ADR-003](../adr/PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md) (shared kernel consumed here), [PRIMORDIS-ADR-005](../adr/PRIMORDIS-ADR-005-rendering-and-compositing.md) (owned-canvas present path), [PRIMORDIS-ADR-007](../adr/PRIMORDIS-ADR-007-web-build-and-cross-origin-isolation.md) (`--wasm` / legacy-interop ban)
- Depends on: [PRIMORDIS-TASK-003](./PRIMORDIS-TASK-003-port-simulation-to-wgsl-compute-kernel.md)
- Blocks: [PRIMORDIS-TASK-005](./PRIMORDIS-TASK-005-web-canvas-compositing-and-pointer-routing.md), [PRIMORDIS-TASK-006](./PRIMORDIS-TASK-006-sliders-to-uniforms-and-ui-chrome.md), [PRIMORDIS-TASK-007](./PRIMORDIS-TASK-007-webgpu-feature-detection-and-fallback-switch.md), [PRIMORDIS-TASK-011](./PRIMORDIS-TASK-011-macos-target-dawn-wgpu-ffi-backend.md)
