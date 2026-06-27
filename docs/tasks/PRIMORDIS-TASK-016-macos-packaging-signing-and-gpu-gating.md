# PRIMORDIS-TASK-016: macOS packaging, signing/notarization, and GPU gating

**Status:** Todo
**Priority:** Medium
**Created:** 2026-06-27
**Updated:** 2026-06-27

## Description

Make the native macOS build **shippable**: code signing and notarization, GPU-family gating so old/unsupported Macs (notably **Intel Macs**) cleanly fall back instead of crashing or stalling, and CI to build/sign/notarize the app. This task is downstream of the macOS present path ([PRIMORDIS-TASK-012](./PRIMORDIS-TASK-012-macos-metal-texture-present-path.md)) because a signed, notarized app must bundle the working Dawn/wgpu-over-Metal native asset (and the IOSurface/`Texture` present bridge) and run inside the macOS sandbox. The signing/notarization cost is explicitly part of the "~1-2 person-weeks incremental" macOS GPU estimate in [PRIMORDIS-ADR-004](../adr/PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md).

GPU gating ties into the tier model from [PRIMORDIS-ADR-006](../adr/PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md): on a Mac whose GPU family cannot run the compute path, GPU device/pipeline creation must fail **gracefully** so the selector ([PRIMORDIS-TASK-015](./PRIMORDIS-TASK-015-cross-platform-backend-selection-and-reduced-mode-ux.md)) demotes to the native CPU isolate tier (T3) rather than producing a blank screen or a hang. This task ensures the failure is detected at init and surfaced as a clean demotion, and that the packaged binary's entitlements/signing do not themselves break GPU init.

## Scope

**Area:** Infra
**Files/Dirs:**
- `macos/Runner.entitlements` / `macos/Runner/*.entitlements` (sandbox, GPU/Metal, hardened-runtime entitlements)
- `macos/Runner.xcodeproj` / `macos/Podfile` (bundle the Dawn/wgpu native asset + any MSL plugin)
- `lib/sim/backends/native_gpu/gpu_gating.dart` — GPU-family / compute-capability check that returns a clean unavailable result (consumed by the selector)
- `.github/workflows/macos-release.yml` (or equivalent CI) — build, sign, notarize, staple, archive
- `scripts/macos/sign_and_notarize.sh` — `codesign` + `notarytool` + `stapler` automation
- `pubspec.yaml` / `PrimordisConfig` (version bump for a release build)
- Signing/entitlements and native-asset packaging sit outside the standard Dart layers, as expected for the native build.

## Acceptance Criteria

- [ ] The macOS app builds via `flutter build macos` (release) with the Dawn/wgpu-over-Metal native asset (TASK-011) and the IOSurface/`Texture` present bridge (TASK-012) bundled and loadable inside the packaged `.app`.
- [ ] The app is **code-signed** with a Developer ID, **notarized** (`notarytool`), and the notarization ticket is **stapled** (`stapler`); Gatekeeper accepts the app on a clean machine (`spctl`/launch passes).
- [ ] Hardened runtime is enabled and entitlements are the minimal set required for Metal/GPU compute and FFI/native-asset loading; entitlements/sandbox do **not** break GPU device/pipeline creation.
- [ ] **GPU gating:** on a Mac whose GPU family cannot run the compute path (e.g. older Intel Macs), GPU init fails **at startup probe time** with a clean unavailable signal — no crash, no hang, no blank canvas.
- [ ] On gated/failed GPU init, the backend selector (TASK-015) demotes to the **native CPU isolate** tier (T3, TASK-014) and the reduced-mode indicator is shown; the app remains usable.
- [ ] CI produces a signed, notarized, stapled artifact on a release build and fails loudly if signing/notarization fails (no silently unsigned artifact).
- [ ] No CPU readback regression is introduced by packaging: the signed build still runs the GPU present path GPU-resident at 24k/60fps where the GPU is supported (re-confirm against TASK-012's no-readback contract).

### Versioning (if Flutter/native code changed)
- [ ] Version bumped in pubspec.yaml and the app config constant (`PrimordisConfig`); semver

### Test Coverage
- [ ] New/modified Dart has unit/widget tests; flutter test passes; flutter analyze zero warnings

## Implementation Notes

- **Why signing matters here.** Approach (a) loads a native Dawn/wgpu asset via `dart:ffi` (ADR-004). Bundling and loading an FFI native asset inside a hardened-runtime, notarized `.app` is exactly where signing/entitlement mistakes surface (dyld load failures, sandbox denials). The notarization step is called out in the macOS effort estimate (ADR-004) and is non-optional for distribution.
- **GPU gating = clean demotion, not detection theater.** The gate is the same authoritative probe used by the selector (ADR-006 §2): attempt GPU **device + compute-pipeline** creation; if the GPU family/feature set can't support the compute path, return unavailable. Intel Macs are the canonical gated case (and also lack browser WebGPU on the web side — ADR-006). The point is that the probe **fails fast and cleanly** so T3 is selected; this task makes sure packaging/entitlements don't turn that clean failure into a crash.
- **Entitlements minimalism.** Enable hardened runtime; include only the entitlements GPU compute + FFI loading actually need. Over-broad entitlements complicate notarization; missing ones break Metal/FFI at runtime. Validate on a clean, non-developer machine, not just the build host.
- **CI.** The release workflow runs `flutter build macos --release`, then `scripts/macos/sign_and_notarize.sh` (`codesign` → `notarytool submit --wait` → `stapler staple`), then verifies with `spctl`. Secrets (Developer ID cert, App Store Connect API key) come from CI secrets, never committed. The web build/hosting pipeline is separate ([PRIMORDIS-TASK-010](./PRIMORDIS-TASK-010-web-build-hosting-and-cross-origin-isolation.md)); this is macOS-only.
- **Boundary.** All of this is Infra/native packaging and lives outside the standard feature/data/domain layers; the only Dart touched is the gating helper (`gpu_gating.dart`) that returns a clean unavailable result, which is unit-testable with the probe outcome mocked.

## Testing

- [ ] Unit test `gpu_gating.dart`: a mocked unsupported-GPU-family probe returns "unavailable" and feeds the selector a demotion (asserts T3 is chosen via TASK-015's selector with this probe result).
- [ ] CI dry run: confirm the workflow builds, signs, notarizes, staples, and that `spctl --assess` / a Gatekeeper check passes on the produced `.app`.
- [ ] Manual on supported GPU (Apple Silicon): signed build runs at 24k/60fps on the GPU present path with no CPU readback (re-verify TASK-012 contract).
- [ ] Manual on an unsupported/Intel Mac (or simulated gated probe): app launches, GPU init fails cleanly, T3 CPU tier + reduced-mode indicator appears, no crash/hang/blank screen.
- [ ] Negative CI test: a broken/missing signing identity causes the pipeline to fail rather than emit an unsigned artifact.
- [ ] `flutter analyze` zero warnings; `flutter test` passes.

## Related
- PRD: [PRIMORDIS-PRD-001](../prd/PRIMORDIS-PRD-001-flutter-web-and-macos-port.md)
- ADR: [PRIMORDIS-ADR-004](../adr/PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md) (primary — native macOS GPU, signing/notarization called out), [PRIMORDIS-ADR-006](../adr/PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md) (GPU-init failure → CPU tier demotion), [PRIMORDIS-ADR-005](../adr/PRIMORDIS-ADR-005-rendering-and-compositing.md) (present path being packaged)
- Depends on: [PRIMORDIS-TASK-012](./PRIMORDIS-TASK-012-macos-metal-texture-present-path.md)
- Blocks: (none — release/packaging task)
