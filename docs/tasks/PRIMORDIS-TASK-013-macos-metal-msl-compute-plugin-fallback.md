# PRIMORDIS-TASK-013: macOS Metal/MSL compute plugin — de-risking fallback backend

**Status:** Todo
**Priority:** Medium
**Created:** 2026-06-27
**Updated:** 2026-06-27

## Description

Implement Approach (b) from [PRIMORDIS-ADR-004](../adr/PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md): a hand-written **Metal Shading Language (MSL)** compute plugin (Swift / Obj-C++) for macOS, exposing a `SimBackend` implementation that runs a **near-1:1 GLSL→MSL port** of the three compute passes. This is the **de-risking escape hatch** for the experimental, solo-maintained FFI WebGPU layer (`minigpu`) used by the primary Dawn path ([PRIMORDIS-TASK-011](./PRIMORDIS-TASK-011-macos-target-dawn-wgpu-ffi-backend.md)). It is the most robust, ship-ready Metal path — but it is a **second kernel to maintain** and is Apple-only, so it is built and kept *warm* rather than promoted to primary.

Per [PRIMORDIS-ADR-004](../adr/PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md), this backend is selected only when the Dawn/wgpu-over-Metal backend fails to initialize or proves unstable in the spike; selection logic is finalized in [PRIMORDIS-TASK-015](./PRIMORDIS-TASK-015-cross-platform-backend-selection-and-reduced-mode-ux.md). It renders into a Metal texture and reuses the **same present path** as the Dawn backend ([PRIMORDIS-TASK-012](./PRIMORDIS-TASK-012-macos-metal-texture-present-path.md)), so the UI and compositing are unchanged. Because choosing this approach introduces **three-way kernel drift** (GLSL original + WGSL + MSL), the MSL kernel must be ported against the **single source-of-truth spec** that governs the WGSL kernel ([PRIMORDIS-ADR-003](../adr/PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md)), not against an independent reading of `Primordis.py`.

## Scope

**Area:** Shader
**Files/Dirs:**
- `macos/` plugin sources (Swift / Obj-C++) — Metal device/queue setup, pipeline-state creation, dispatch
- `macos/.../Shaders/primordis.metal` — hand-written MSL compute kernels (clear / bin-scatter / interaction+integrate) + point-render
- `lib/sim/backends/macos_metal_backend.dart` — `SimBackend` implementation over the MSL plugin (method-channel / FFI bridge)
- `docs/` — reference to the single source-of-truth kernel spec the MSL port must track
- `pubspec.yaml` / app config constant — version bump
- `test/sim/backends/macos_metal_backend_test.dart` — backend contract/marshalling tests

> **Layering note (house standards):** the MSL kernels and the native Metal plugin (Swift/Obj-C++) live **outside** the standard Flutter feature/data/domain layers, as expected per [PRIMORDIS-ADR-001](../adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md) and [PRIMORDIS-ADR-004](../adr/PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md), and sit behind the `SimBackend` interface. The UI, Riverpod providers, and Freezed param models stay standards-compliant and never reference Metal.

## Acceptance Criteria

- [ ] A native macOS Metal plugin exposes a `MacosMetalBackend` that implements the full `SimBackend` interface ([PRIMORDIS-TASK-002](./PRIMORDIS-TASK-002-simbackend-interface-and-shared-sim-model.md)) with **no UI-visible difference** from the Dawn backend.
- [ ] The MSL kernel reproduces the **three compute passes**: (1) clear bin counts; (2) bin particles into the uniform spatial grid via atomic scatter; (3) interaction over a 3×3 toroidal neighbor scan (minimum-image distance, short-range repulsion `dist<min_dist` 5× weighted using `abs(force)`, linear-falloff signed attraction `dist<radius`) + Euler integrate (`v += f*dt; v *= friction; p += v*dt`; wrap) — plus the point-render pass.
- [ ] The port honors the canonical sim constants: **24,000 particles, 32 types**, world **1080×720 toroidal**, grid **11×7 = 77 bins**, bin size = `MAX_RADIUS` = **96**, `MAX_BIN_PARTICLES` = **512** cap, asymmetric 32×32 force/min-distance/radius matrices (i→j ≠ j→i), per-type random colors, `gl_PointSize=2` equivalent.
- [ ] GLSL→MSL mappings are applied per [PRIMORDIS-ADR-004](../adr/PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md): SSBO → `device` buffer pointer (`device float* positions [[buffer(0)]]`); `atomicAdd(bins[i],1)` → `atomic_fetch_add_explicit(&bins[i], 1, memory_order_relaxed)` on a `device atomic_uint` (returns old value = scatter offset); `layout(local_size_x=N)` → threadgroup size; `gl_GlobalInvocationID.x` → `[[thread_position_in_grid]]`; `[[thread_position_in_threadgroup]]` / `[[threadgroup_position_in_grid]]` as needed.
- [ ] Atomic bin counters are a consistent `device atomic_uint` type, read back via the atomic load API (mirroring the WGSL `atomic<u32>` + `atomicLoad` constraint).
- [ ] The backend renders into a **Metal texture** and is consumed by the shared present path ([PRIMORDIS-TASK-012](./PRIMORDIS-TASK-012-macos-metal-texture-present-path.md)) — no new compositing code.
- [ ] The 3 sliders (Attraction K, Repulsion K, Drift/friction) are marshalled into a Metal uniform/constant buffer per frame, matching the Dawn backend's parameter semantics.
- [ ] The backend is **registered as a selectable fallback** and is reachable when the Dawn backend init fails (selection per [PRIMORDIS-TASK-015](./PRIMORDIS-TASK-015-cross-platform-backend-selection-and-reduced-mode-ux.md)).
- [ ] The MSL kernel is ported against the **single source-of-truth kernel spec** (not an independent re-read of `Primordis.py`), and that linkage is documented to mitigate three-way kernel drift.

### Versioning (if Flutter/native code changed)
- [ ] Version bumped in `pubspec.yaml` and the app config constant; semver (minor bump — new fallback backend).

### Test Coverage
- [ ] New/modified Dart has unit/widget tests; `flutter test` passes; `flutter analyze` zero warnings (`package:lint`).
- [ ] Backend contract tests cover seed marshalling, uniform updates, and `dispose()`; parameter-buffer layout math is unit-tested against the documented Metal alignment.

## Implementation Notes

- **Single source of truth, not a re-port.** Port the MSL kernel from the same spec that defines the WGSL kernel ([PRIMORDIS-ADR-003](../adr/PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md)), so the three kernels (GLSL original, WGSL, MSL) cannot silently diverge. Any constant or formula change must update the spec first, then both kernels — this is the explicit mitigation for the three-way-drift risk called out in [PRIMORDIS-ADR-004](../adr/PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md).
- **Near-1:1 port, but validate atomics.** The GLSL→MSL mapping is mechanical (table above / [PRIMORDIS-ADR-004](../adr/PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md)), but the scatter-bin pass is where parity bites: `atomic_fetch_add_explicit` must return the *old* value to use as the per-bin write offset, exactly as the WGSL `atomicAdd(&bins[i],1u)` does. Validate the binning pass produces visually/statistically equivalent clustering to the Dawn path; broader Dawn-vs-browser atomics parity is [PRIMORDIS-TASK-017](./PRIMORDIS-TASK-017-atomics-parity-validation-dawn-vs-browser.md), but this MSL path should be diffed against the Dawn binning output as part of its acceptance.
- **Reuse the present path.** Do not invent compositing here. Render into a Metal texture and hand it to the IOSurface/`FlutterTextureRegistry`/`Texture` path from [PRIMORDIS-TASK-012](./PRIMORDIS-TASK-012-macos-metal-texture-present-path.md); the texture bridge is already backend-agnostic.
- **Warm, not primary.** This is the escape hatch: it must be buildable and runnable on demand, but the project ships on the Dawn path if the spike succeeds. Keep it green in CI so it is genuinely "warm." It is Apple-only and a second kernel to maintain — accepted costs of the de-risking value.
- **Bridge choice.** Either a method-channel/plugin bridge or a thin `dart:ffi` surface to the plugin is acceptable; keep `MacosMetalBackend` talking to a small, testable Dart wrapper so no Metal/Obj-C types leak into UI code.
- **Determinism is not a goal.** The original scatter is already a single-buffered race; MSL parity means visually/statistically equivalent, never bit-exact (per [PRIMORDIS-PRD-001](../prd/PRIMORDIS-PRD-001-flutter-web-and-macos-port.md)). Do not add ordering the reference lacks.
- **No first-party compute API.** This task exists precisely because Flutter exposes no Dart compute API and macOS OpenGL is frozen at 4.1 (no compute); hand-written Metal is the most robust native route. It does not depend on `flutter_gpu`.

## Testing

- [ ] Build the macOS plugin and run `MacosMetalBackend` at 24k/32-types; confirm ≥1000 frames advance with no crash, device-lost, or NaN/Inf.
- [ ] Diff cluster formation/drift against the Dawn backend ([PRIMORDIS-TASK-011](./PRIMORDIS-TASK-011-macos-target-dawn-wgpu-ffi-backend.md)) and the Python reference via the parity harness ([PRIMORDIS-TASK-009](./PRIMORDIS-TASK-009-parity-test-harness-vs-python-reference.md)); confirm visual/statistical equivalence.
- [ ] Verify the MSL backend renders correctly through the shared present path ([PRIMORDIS-TASK-012](./PRIMORDIS-TASK-012-macos-metal-texture-present-path.md)) with the same sliders/chrome on top.
- [ ] Force a Dawn-init failure and confirm backend selection falls through to `MacosMetalBackend` (per [PRIMORDIS-TASK-015](./PRIMORDIS-TASK-015-cross-platform-backend-selection-and-reduced-mode-ux.md)).
- [ ] Unit-test parameter/uniform buffer layout math against the documented Metal alignment (no GPU needed).
- [ ] `flutter test` and `flutter analyze` pass with zero warnings.

## Related
- PRD: [PRIMORDIS-PRD-001](../prd/PRIMORDIS-PRD-001-flutter-web-and-macos-port.md)
- ADR: [PRIMORDIS-ADR-004 — Native macOS GPU compute (Dawn/Metal via FFI), MSL fallback](../adr/PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md); [PRIMORDIS-ADR-003 — Shared WGSL compute kernel (single source of truth)](../adr/PRIMORDIS-ADR-003-shared-wgsl-compute-kernel.md); [PRIMORDIS-ADR-001 — Cross-platform architecture (SimBackend)](../adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md)
- Depends on: [PRIMORDIS-TASK-011 — macOS target: Dawn/wgpu FFI backend](./PRIMORDIS-TASK-011-macos-target-dawn-wgpu-ffi-backend.md)
- Blocks: [PRIMORDIS-TASK-015 — Cross-platform backend selection and reduced-mode UX](./PRIMORDIS-TASK-015-cross-platform-backend-selection-and-reduced-mode-ux.md)
