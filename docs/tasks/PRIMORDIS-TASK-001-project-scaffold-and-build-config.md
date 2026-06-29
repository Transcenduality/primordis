# PRIMORDIS-TASK-001: Project scaffold and build config

**Status:** Complete
**Priority:** Critical
**Created:** 2026-06-27
**Updated:** 2026-06-29

## Description

Stand up the Flutter application that will host Primordis on web (WASM) and native macOS, with the DGROUP-approved stack wired in from the first commit and the `--wasm` web build posture established. This is the foundational task that every other Primordis task depends on: it fixes the repo layout, the dependency set, the lint posture, the version/config-constant convention, and the build-config seam that the WebGPU canvas and (later) the macOS `Texture` widget will plug into.

Primordis is its **own standalone repo** — not the DGroup monorepo — so the Flutter app lives at the repo root (with `apps/web` reserved as an alternative layout if a multi-package split is later needed), and the version lives in `pubspec.yaml` plus a single `PrimordisConfig`/`AppConfig` constant. No simulation logic, no GPU code, and no compute backends are implemented here; this task delivers a runnable, lint-clean, standards-compliant Flutter shell that builds for web with `--wasm` and runs on macOS desktop, plus the `SimBackend` seam location (the interface itself is defined in [PRIMORDIS-TASK-002](./PRIMORDIS-TASK-002-simbackend-interface-and-shared-sim-model.md)).

The single governing constraint from [PRIMORDIS-ADR-001](../adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md) and [PRIMORDIS-ADR-002](../adr/PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md) is baked in here at the dependency level: because the web build will be compiled with `flutter build web --wasm` (Skwasm), the project may **not** depend — anywhere in its dependency tree — on legacy interop (`dart:html` / `dart:js_util`). The scaffold must therefore pin `dart:js_interop` + `package:web` as the only interop path from day one, so no later task accidentally pulls in a `--wasm`-incompatible package. See [PRIMORDIS-ADR-007](../adr/PRIMORDIS-ADR-007-web-build-and-cross-origin-isolation.md).

## Scope

**Area:** Infra

**Files/Dirs:**
- `pubspec.yaml` (repo root) — dependencies, dev_dependencies, environment SDK constraints, `version:`
- `analysis_options.yaml` — `package:lint` ruleset, zero-warning posture
- `lib/main.dart` — app entrypoint, `ProviderScope` root, `MaterialApp.router`
- `lib/src/app/` — `PrimordisApp` widget, Material 3 theme (GoogleFonts), GoRouter config
- `lib/src/config/primordis_config.dart` — `PrimordisConfig` constant (version mirror + world/grid/particle constants placeholder)
- `lib/src/sim/sim_backend.dart` — placeholder file marking the `SimBackend` seam (interface filled in TASK-002)
- `web/index.html`, `web/manifest.json` — web bootstrap; reserved DOM container for the owned WebGPU canvas (wired in TASK-005)
- `macos/` — macOS runner enabled (`flutter create --platforms macos .` output)
- `test/` — smoke/widget test for app boot
- `README.md` — repo layout, build commands, `--wasm` note
- `.github/workflows/` — CI skeleton (analyze + test) — full web/macOS pipelines land in TASK-010/TASK-016

## Acceptance Criteria

- [x] `flutter create` scaffold exists at repo root with **web** and **macos** platforms enabled (no android/ios/windows/linux runners committed unless trivially free; they are out of initial scope per [PRIMORDIS-ADR-001](../adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md)).
- [x] `pubspec.yaml` declares the DGROUP-approved stack: `flutter_riverpod` (Riverpod), `freezed` + `freezed_annotation` + `json_serializable` + `build_runner`, `go_router`, `google_fonts`, and `lint` (`package:lint`) as a dev dependency. Retrofit/Dio are listed as available house standards but are **not** added unless a networking need exists (Primordis has none at scaffold time) — note this explicitly in `pubspec.yaml` comments.
- [x] `analysis_options.yaml` includes `package:lint` and the project passes `flutter analyze` with **zero warnings**.
- [x] No part of the dependency tree pulls in `dart:html` or `dart:js_util`; `package:web` + `dart:js_interop` are the only declared interop path (verifiable via `flutter pub deps` inspection / a documented audit step). This is required for the `--wasm` build to succeed later ([PRIMORDIS-ADR-007](../adr/PRIMORDIS-ADR-007-web-build-and-cross-origin-isolation.md)).
- [x] App boots to a Material 3 home scaffold using `GoogleFonts`, with the root wrapped in a Riverpod `ProviderScope`; routing goes through `GoRouter` (`MaterialApp.router`), even if there is only one route initially. No `setState` is used for business logic (none exists yet, but the pattern is established).
- [x] `flutter build web --wasm` completes successfully producing a Skwasm build (with automatic CanvasKit/dart2js fallback enabled by default). The build need not yet render the simulation — only prove the `--wasm` toolchain and dependency tree are clean. (Full Skwasm/WasmGC target browsers and COOP/COEP headers are TASK-010 scope.)
- [x] `flutter build macos` (debug) completes and the app launches as a native macOS window.
- [x] `PrimordisConfig` constant exists exposing at minimum the app `version` (kept in sync with `pubspec.yaml`) and placeholders for the simulation constants that TASK-002 will own: world size `1080x720`, particle count `24000`, type count `32`, grid `11x7 = 77` bins, `MAX_RADIUS = 96` (bin size), `MAX_BIN_PARTICLES = 512`. Values are declared as named constants here; their authoritative typed model lives in TASK-002.
- [x] A `lib/src/sim/sim_backend.dart` placeholder marks the `SimBackend` seam and documents (in a header comment) that all GPU/FFI/JS-interop/shader code lives **behind this seam, outside the standard feature/data/domain layers**, as sanctioned by [PRIMORDIS-ADR-001](../adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md).
- [x] `README.md` documents the repo layout, the `flutter build web --wasm` and `flutter build macos` commands, the `dart:js_interop`-only interop constraint, and the standalone-repo (root, not monorepo) posture.

### Versioning (if Flutter/native code changed)
- [x] Version bumped in `pubspec.yaml` and the app config constant (`PrimordisConfig.version`); semver. Initial scaffold sets the baseline (e.g. `1.4.0` succeeding the Python `1.3` line, or `0.1.0` for the Flutter rewrite — record the chosen scheme in the README).

### Test Coverage
- [x] New/modified Dart has unit/widget tests; `flutter test` passes; `flutter analyze` zero warnings. At minimum a widget smoke test asserts the app boots, mounts a `ProviderScope`, and renders the home route through `GoRouter`.

## Implementation Notes

- **Repo layout.** Keep the Flutter app at the repo root (the existing `Primordis.py` reference stays in place as the parity reference for [PRIMORDIS-TASK-009](./PRIMORDIS-TASK-009-parity-test-harness-vs-python-reference.md); do not delete it). `apps/web` is reserved only if a later multi-package split (e.g. separating an FFI native-asset package) forces it — do not pre-split now.
- **Interop discipline is a build constraint, not a style choice.** `flutter build web --wasm` forbids `dart:html`/`dart:js_util` anywhere in the dependency tree ([PRIMORDIS-ADR-007](../adr/PRIMORDIS-ADR-007-web-build-and-cross-origin-isolation.md)). Add `web` (`package:web`) now and forbid legacy interop in code review. This is what makes the WebGPU `js_interop` bindings in [PRIMORDIS-TASK-004](./PRIMORDIS-TASK-004-web-webgpu-backend-js-interop.md) and the canvas compositing in [PRIMORDIS-TASK-005](./PRIMORDIS-TASK-005-web-canvas-compositing-and-pointer-routing.md) viable.
- **`web/index.html` prepares for the stacked canvas.** Per [PRIMORDIS-ADR-005](../adr/PRIMORDIS-ADR-005-rendering-and-compositing.md), the WebGPU `<canvas>` will be a **sibling DOM element behind a transparent Flutter glass-pane**, *not* an `HtmlElementView`. Reserve a documented container/insertion point in `index.html` (or document where TASK-005 will inject the sibling canvas) so the compositing task has a known anchor; do not implement the canvas or pointer routing here.
- **Standards live above the seam, non-standard code lives below it.** The UI/state/router layers established here are fully Riverpod/Freezed/GoRouter/Material 3 compliant. The `SimBackend` placeholder explicitly demarcates where standards-noncompliant GPU/FFI/JS-interop/WGSL code will live — this quarantine is the whole point of the architecture ([PRIMORDIS-ADR-001](../adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md)). No shader/FFI/interop code is written in this task; it only reserves the boundary.
- **Riverpod, not `setState`.** Establish the convention (plain `Ref`, providers in their own files) even though there is no business state yet, so TASK-002's `SimBackend`/param providers and TASK-006's slider providers drop in cleanly.
- **Material 3 + GoogleFonts** for the theme; keep the home scaffold minimal (title + placeholder area where the simulation canvas/`Texture` will eventually composite). Accessibility hooks (semantics, reduced-motion handling) are not implemented here but the home scaffold must not introduce continuous motion that would later violate the reduced-motion goal ([PRIMORDIS-ADR-006](../adr/PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md)).
- **No GPU/native code yet.** macOS is enabled at the runner level only; the Dawn/wgpu FFI backend and `Texture` present path are [PRIMORDIS-TASK-011](./PRIMORDIS-TASK-011-macos-target-dawn-wgpu-ffi-backend.md) / [PRIMORDIS-TASK-012](./PRIMORDIS-TASK-012-macos-metal-texture-present-path.md). The `--wasm` web build here only needs to compile and serve the shell.
- **CI skeleton** runs `flutter analyze` + `flutter test` on push. Web hosting (COOP/COEP) and macOS signing/notarization pipelines are deferred to [PRIMORDIS-TASK-010](./PRIMORDIS-TASK-010-web-build-hosting-and-cross-origin-isolation.md) and [PRIMORDIS-TASK-016](./PRIMORDIS-TASK-016-macos-packaging-signing-and-gpu-gating.md).

## Testing
- [x] `flutter pub get` resolves with no version conflicts and no legacy-interop packages in the tree (`flutter pub deps` audited; document the audit in the PR).
- [x] `flutter analyze` reports zero issues.
- [x] `flutter test` passes, including the app-boot widget smoke test (ProviderScope mounted, home route rendered via GoRouter).
- [x] `flutter build web --wasm` succeeds end-to-end; the resulting build serves and loads the shell in a WasmGC-capable browser.
- [x] `flutter build macos` (debug) succeeds and the app launches as a native window.
- [x] `dart run build_runner build --delete-conflicting-outputs` runs clean (Freezed/json_serializable codegen wired even before any models exist, so TASK-002 codegen is turnkey).
- [x] Manual check: `PrimordisConfig.version` matches `pubspec.yaml` `version:`.

## Related
- PRD: [PRIMORDIS-PRD-001](../prd/PRIMORDIS-PRD-001-flutter-web-and-macos-port.md)
- ADR: [PRIMORDIS-ADR-001](../adr/PRIMORDIS-ADR-001-cross-platform-architecture-simbackend.md), [PRIMORDIS-ADR-007](../adr/PRIMORDIS-ADR-007-web-build-and-cross-origin-isolation.md), [PRIMORDIS-ADR-002](../adr/PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md), [PRIMORDIS-ADR-005](../adr/PRIMORDIS-ADR-005-rendering-and-compositing.md)
- Depends on: none
- Blocks: [PRIMORDIS-TASK-002](./PRIMORDIS-TASK-002-simbackend-interface-and-shared-sim-model.md) (and transitively every downstream Primordis task)
