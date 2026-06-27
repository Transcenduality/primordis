# PRIMORDIS-TASK-010: Web build, hosting, and cross-origin isolation

**Status:** Todo
**Priority:** High
**Created:** 2026-06-27
**Updated:** 2026-06-27

## Description

Establish the production **web build and hosting** configuration for Primordis: compile with `flutter build web --wasm` (Skwasm renderer) with automatic CanvasKit/dart2js fallback, and serve the app with **cross-origin isolation** headers (`COOP: same-origin` + `COEP: require-corp`) so that `SharedArrayBuffer`-dependent paths and multi-threaded Skwasm can run where supported (per [PRIMORDIS-ADR-007](../adr/PRIMORDIS-ADR-007-web-build-and-cross-origin-isolation.md)).

This task wires together the compositing produced in [PRIMORDIS-TASK-005](./PRIMORDIS-TASK-005-web-canvas-compositing-and-pointer-routing.md) — the WebGPU `<canvas>` stacked **behind** a transparent Flutter glass-pane — into a deployable artifact and hosting setup. Two web-specific constraints from the brief shape it:

1. **`--wasm` forbids legacy interop** (`dart:html` / `dart:js_util`) anywhere in the dependency tree. The whole app (and especially the web WebGPU backend from [PRIMORDIS-TASK-004](./PRIMORDIS-TASK-004-web-webgpu-backend-js-interop.md)) must use `dart:js_interop` + `package:web` only. This task adds the CI gate that enforces it.
2. **Cross-origin isolation requires header control**, which plain static hosts without header configuration cannot provide. The hosting target must be one where `COOP`/`COEP` (and the WASM MIME types / WasmGC-capable serving) can be set. This task picks and configures that host.

A distinct posture from the DGroup app is recorded here: `DGROUP_WEB-ADR-020` (Flutter Web Rendering and Compilation, in the DGroup monorepo — external to this repo) chose CanvasKit + dart2js for the DGroup web app and noted a *future* `--wasm` switch; Primordis needs `--wasm` **now** plus its **own** WebGPU canvas — a deliberately different web posture captured in [PRIMORDIS-ADR-007](../adr/PRIMORDIS-ADR-007-web-build-and-cross-origin-isolation.md).

## Scope

**Area:** Infra
**Files/Dirs:**
- `web/` — `index.html` (renderer config / bootstrap), `manifest.json`, headers metadata as the host requires
- Hosting/deploy config (e.g. `firebase.json` headers block, `_headers` file, nginx/Caddy snippet, or equivalent for the chosen host) carrying `COOP`/`COEP` + WASM MIME + caching
- CI workflow under `.github/workflows/` (or repo CI equivalent): build `--wasm`, run analyze/tests, enforce the no-legacy-interop gate, deploy
- `analysis_options.yaml` / a lint or grep gate forbidding `dart:html` and `dart:js_util` in the dependency tree
- `docs/` deployment runbook section (how to deploy, how to verify cross-origin isolation)
- Consumes (does not own): the compositing/canvas wiring from TASK-005 and the WebGPU backend from TASK-004

> **Layering note (expected, per house standards):** this is build/infra/hosting configuration, outside the feature/data/domain layers by design. It must **not** introduce business logic; Riverpod/Freezed/GoRouter conventions are unaffected. Its job is to package and serve the standards-compliant UI plus the JS-interop WebGPU backend (which itself legitimately lives outside the standard layers behind `SimBackend`).

## Acceptance Criteria

- [ ] `flutter build web --wasm` produces a working artifact using the **Skwasm** renderer, with **automatic CanvasKit/dart2js fallback** for browsers lacking WasmGC (WasmGC requires Chrome 119+ / Firefox 120+ / Safari 18.2+); the app loads and runs the simulation on both a WasmGC browser and a fallback browser.
- [ ] **No legacy interop** (`dart:html`, `dart:js_util`) appears anywhere in the app's dependency tree; a CI gate fails the build if either is imported. Web interop is exclusively `dart:js_interop` + `package:web`.
- [ ] The deployed site serves **`Cross-Origin-Opener-Policy: same-origin`** and **`Cross-Origin-Embedder-Policy: require-corp`** on the document, so `self.crossOriginIsolated === true` in the running app.
- [ ] WASM assets are served with correct MIME (`application/wasm`) and any required `Cross-Origin-Resource-Policy` / caching headers so Skwasm and the app's own WebGPU/wasm assets load under COEP `require-corp`.
- [ ] Multi-threaded Skwasm and any `SharedArrayBuffer`-dependent path **activate only when** `crossOriginIsolated` is true, and **degrade gracefully** (single-threaded) when it is not, without crashing — consistent with the ADR-006 web tier policy (single-thread CPU fallback is the web ceiling regardless).
- [ ] The chosen hosting target supports header configuration (not a plain static host); the header config is committed as code (e.g. `firebase.json` / `_headers` / server snippet), not set manually in a console.
- [ ] The WebGPU `<canvas>` from TASK-005 composites correctly in the deployed build: it sits behind the transparent Flutter glass-pane, pointer routing works, and DPR/resize sync behaves on the live host (no `HtmlElementView` wrapping — per [PRIMORDIS-ADR-005](../adr/PRIMORDIS-ADR-005-rendering-and-compositing.md)).
- [ ] The WebGPU feature-detect + CPU fallback ([PRIMORDIS-TASK-007](./PRIMORDIS-TASK-007-webgpu-feature-detection-and-fallback-switch.md)) works in the deployed artifact: a WebGPU browser gets the GPU tier; a non-WebGPU browser gets the Dart→WASM CPU tier ([PRIMORDIS-TASK-008](./PRIMORDIS-TASK-008-dart-wasm-cpu-fallback-backend.md)) — neither shows a blank screen.
- [ ] CI builds `--wasm`, runs `flutter analyze` (zero warnings) and `flutter test`, enforces the interop gate, and deploys (or produces a deployable artifact) on the main branch.
- [ ] A deployment runbook documents: how to build, the exact headers and why, how to verify cross-origin isolation (`crossOriginIsolated`), and the WasmGC/fallback matrix.

### Versioning (if Flutter/native code changed)

- [ ] Version bumped in `pubspec.yaml` and the app config constant (`PrimordisConfig`/`AppConfig`); semver. The deployed build surfaces this version (e.g. in an about/diagnostics affordance) so the live site is traceable to a commit.

### Test Coverage

- [ ] New/modified Dart has unit/widget tests; `flutter test` passes; `flutter analyze` zero warnings.
- [ ] CI step asserts the `--wasm` build succeeds and the no-legacy-interop gate is enforced (a deliberate `dart:html` import in a throwaway test branch must fail CI).
- [ ] A post-deploy smoke check (script or manual checklist) verifies `crossOriginIsolated === true` and that both a WebGPU and a non-WebGPU browser profile boot into their correct tier.

## Implementation Notes

- **Renderer choice:** `--wasm` selects Skwasm where WasmGC is available and falls back to CanvasKit/dart2js otherwise — this is Flutter's built-in behavior; configure/verify it in `web/index.html` bootstrap rather than fighting it. Note that Primordis does **not** render the simulation through Skwasm/CanvasKit — the sim lives on the **owned WebGPU canvas** behind the glass-pane (ADR-005). Skwasm/CanvasKit only paints the Flutter UI chrome (sliders, indicators). This is exactly why the project diverges from DGROUP_WEB-ADR-020's CanvasKit posture.
- **Why COOP/COEP:** cross-origin isolation (`crossOriginIsolated`) is the precondition for `SharedArrayBuffer` and multi-threaded Skwasm. The brief is explicit that plain static hosts without header control **cannot** do this. So the hosting decision is constrained: pick a host that lets you set response headers as committed config. Set `COOP: same-origin` and `COEP: require-corp` on the HTML document and ensure all sub-resources (including third-party fonts via GoogleFonts, the WebGPU/wasm assets) are COEP-compatible (`require-corp` means cross-origin sub-resources need `Cross-Origin-Resource-Policy`/CORS, or they must be self-hosted). If GoogleFonts CDN fonts break under COEP, self-host the font assets.
- **Header carrier examples (pick per host):** Firebase Hosting `firebase.json` `headers` block; Netlify/Cloudflare Pages `_headers` file; a reverse proxy (nginx/Caddy) snippet. Whatever the host, the headers live in committed config so the isolation guarantee is reproducible.
- **Interop gate:** add a CI step that greps/analyzes the resolved dependency tree for `dart:html` and `dart:js_util` and fails on any hit. The `--wasm` compiler will also reject them, but a fast explicit gate gives a clear error and catches transitive deps before a full build. Keep the web WebGPU bindings (TASK-004) strictly on `dart:js_interop` + `package:web`.
- **MIME/WasmGC serving:** ensure `.wasm` is served as `application/wasm` and that the host doesn't strip/rename the dart2wasm output; WasmGC browsers need the WasmGC module served intact, and the fallback path must be reachable for older browsers.
- **Graceful isolation degradation:** do not hard-require `crossOriginIsolated`. Per ADR-006 the web CPU tier is single-threaded regardless (web has no real shared-memory isolates), and the WebGPU GPU path does not need `SharedArrayBuffer`. So if a deployment lands somewhere isolation can't be set, the app must still run (GPU tier if WebGPU present; single-thread CPU tier otherwise) — isolation is an enablement, not a hard dependency, for the *current* tiers. Document this so a future threaded-CPU experiment knows isolation is its precondition.
- **Compositing in production:** verify on the live host that the stacked-canvas approach (not `HtmlElementView`) survives the deployed bootstrap — overlay/canvas-splitting and DPR mismatches sometimes only appear on a real host with real DPR/zoom. Sync canvas size to `MediaQuery` devicePixelRatio on resize (the wiring is TASK-005; this task confirms it post-deploy).
- **CI/CD:** main-branch pipeline = analyze → test → `flutter build web --wasm` → interop gate → deploy. Surface the `PrimordisConfig` version in a diagnostics/about affordance so the deployed site is traceable to a commit, satisfying the versioning criterion.

## Testing

- [ ] `flutter build web --wasm` succeeds locally and in CI; artifact boots in a WasmGC browser (Skwasm) and a non-WasmGC browser (CanvasKit/dart2js fallback).
- [ ] `flutter analyze` zero warnings; `flutter test` green.
- [ ] Deploy to the chosen host; open the site and confirm in console `self.crossOriginIsolated === true` and that `COOP`/`COEP` headers are present on the document response.
- [ ] In a WebGPU browser: confirm the GPU tier runs the 24k simulation on the owned canvas behind the glass-pane, sliders work, pointer routing works, resize/DPR is correct.
- [ ] In a non-WebGPU browser profile: confirm graceful fallback to the Dart→WASM CPU tier (TASK-008) with the reduced-mode indicator, no blank screen, no crash.
- [ ] Confirm the interop gate fails CI when a `dart:html`/`dart:js_util` import is deliberately introduced, then revert.
- [ ] Confirm GoogleFonts/font assets load under COEP `require-corp` (self-hosted if necessary) with no console CORP/CORS errors.

## Related

- PRD: [PRIMORDIS-PRD-001](../prd/PRIMORDIS-PRD-001-flutter-web-and-macos-port.md)
- ADR: [PRIMORDIS-ADR-007 — Web build & cross-origin isolation](../adr/PRIMORDIS-ADR-007-web-build-and-cross-origin-isolation.md); [PRIMORDIS-ADR-005 — Rendering and compositing](../adr/PRIMORDIS-ADR-005-rendering-and-compositing.md); [PRIMORDIS-ADR-002 — Web GPU compute via WebGPU + JS interop](../adr/PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md); [PRIMORDIS-ADR-006 — CPU fallback tiers and feature detection](../adr/PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md)
- Depends on: [PRIMORDIS-TASK-005 — Web canvas compositing & pointer routing](./PRIMORDIS-TASK-005-web-canvas-compositing-and-pointer-routing.md)
- Blocks: (web release readiness; no downstream task hard-depends on this)
