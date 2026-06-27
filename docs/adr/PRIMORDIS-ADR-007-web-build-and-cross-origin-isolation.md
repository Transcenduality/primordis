<!-- Filename convention: <SCOPE>-ADR-NNN-short-title.md -->

# PRIMORDIS-ADR-007: Web build and cross-origin isolation

**Status:** Proposed
**Date:** 2026-06-27
**Deciders:** Bruce Abernethy
**Review date:** 2026-09-27
**Supersedes:** N/A
**Superseded by:** N/A
**Compliance/Security:** COOP `same-origin` + COEP `require-corp` cross-origin-isolation headers are a security-relevant serving requirement (they gate `SharedArrayBuffer` and restrict which subresources may be embedded). See Context and Consequences.

## Context

Primordis is being ported from a single ~350-line Python file (pygame + moderngl, OpenGL 4.3 compute) to one Flutter app that runs the same particle-life simulation on Flutter Web (WASM) and native macOS. The web compute path is browser WebGPU reached from Dart via `dart:js_interop` + `package:web` on an owned `<canvas>` (see [PRIMORDIS-ADR-002](./PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md)), with a pure-Dart CPU fallback when `navigator.gpu` is absent (see [PRIMORDIS-ADR-006](./PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md)).

This ADR records how the Flutter side of the web target is **compiled and served**: the Flutter web renderer/compiler toolchain, the HTTP headers required for cross-origin isolation, and the interop constraints that toolchain choice imposes on the entire dependency tree. It does not re-decide the WebGPU compute approach (ADR-002), the shared kernel (ADR-003), or the compositing strategy (ADR-005); it covers the build and hosting envelope those decisions run inside.

Relevant facts from the feasibility analysis:

- `flutter build web --wasm` compiles Dart to WebAssembly (dart2wasm) and uses the **Skwasm** renderer. Skwasm depends on **WasmGC**, which requires **Chrome 119+ / Firefox 120+ / Safari 18.2+**. The `--wasm` build ships with **automatic fallback to CanvasKit + dart2js** on engines that lack WasmGC, so older browsers still load the app.
- **Multi-threaded Skwasm** and **any `SharedArrayBuffer`-based path** require the page to be **cross-origin isolated**, which means serving **COOP: `same-origin`** and **COEP: `require-corp`** headers. Plain static hosts that do not allow setting response headers cannot achieve this.
- The pure-Dart CPU fallback (ADR-006) is single-threaded by default. Web has no real isolates — web "isolates" are web workers that copy data — and true shared-memory threading on web depends on `SharedArrayBuffer`, which in turn depends on the same COOP/COEP isolation. So cross-origin isolation is the precondition for *any future* worker/`SharedArrayBuffer`-backed CPU fallback, even though the initial CPU tier is single-threaded.
- The `--wasm` build **forbids legacy interop anywhere in the dependency tree**: `dart:html` and `dart:js_util` are not permitted. Only `dart:js_interop` + `package:web` may be used. The Primordis WebGPU bindings (ADR-002) are already written to that constraint; this ADR makes the constraint binding for the *whole* dep tree.
- DGroup's web standard, **DGROUP_WEB-ADR-020 (Flutter Web Rendering and Compilation)**, chose **CanvasKit + dart2js** for the DGroup app and noted a *future* `--wasm` switch. Primordis takes a different posture — `--wasm` now, plus an **owned WebGPU canvas** outside Flutter's renderer — which is worth recording as a deliberate divergence rather than an accident.

Primordis is its own standalone repo (Flutter app at the repo root or `apps/web`), with the version in `pubspec.yaml` and a `PrimordisConfig`/`AppConfig` constant; it is not the DGroup monorepo, so DGROUP_WEB-ADR-020 is a *reference* standard, not an inherited configuration.

## Decision

1. **Compile the Flutter web target with `flutter build web --wasm`** (dart2wasm + Skwasm renderer), relying on the toolchain's **automatic CanvasKit + dart2js fallback** for engines without WasmGC. WasmGC support — and therefore the Skwasm path — is expected on **Chrome 119+ / Firefox 120+ / Safari 18.2+**; older engines transparently receive the CanvasKit/dart2js build.

2. **Serve the app cross-origin isolated**: set **`Cross-Origin-Opener-Policy: same-origin`** and **`Cross-Origin-Embedder-Policy: require-corp`** on the served HTML/asset responses. This enables `SharedArrayBuffer`, which is the precondition for multi-threaded Skwasm and for any worker/`SharedArrayBuffer`-backed CPU fallback tier. Hosting therefore **requires header control**; plain static hosts that cannot set these headers are excluded.

3. **Enforce a `dart:js_interop` + `package:web`-only interop rule across the entire dependency tree.** `dart:html` and `dart:js_util` are prohibited in app code and in any dependency reachable from the web build, because `--wasm` rejects legacy interop. The Primordis WebGPU bindings (ADR-002) already comply; this decision extends the rule to all transitive dependencies and is enforced in CI.

4. **Record this as a deliberate divergence from DGROUP_WEB-ADR-020.** DGroup's app standardized on CanvasKit + dart2js with a future `--wasm` move; Primordis adopts `--wasm` immediately and owns a separate WebGPU canvas outside Flutter's Skia/WebGL renderer. The divergence is justified by Primordis's GPU-compute requirement and is documented here so the two postures stay legible to anyone moving between repos.

The build/hosting and interop configuration is implemented and validated in [PRIMORDIS-TASK-010](../tasks/PRIMORDIS-TASK-010-web-build-hosting-and-cross-origin-isolation.md), within the scaffold/build-config baseline established by [PRIMORDIS-TASK-001](../tasks/PRIMORDIS-TASK-001-project-scaffold-and-build-config.md).

## Consequences

### Positive

- **Faster Flutter UI baseline on modern engines.** The Skwasm/dart2wasm path is the modern Flutter web target; the dart2js + CanvasKit fallback is the slower baseline, reserved for engines without WasmGC.
- **Cross-origin isolation unlocks shared-memory threading.** Once COOP/COEP are in place, `SharedArrayBuffer` becomes available, enabling multi-threaded Skwasm and a future worker-backed CPU fallback — the only route to real shared-memory threading on web.
- **`--wasm`-clean interop is future-proof.** Mandating `dart:js_interop` + `package:web` only means the build cannot regress into legacy-interop breakage, and it matches the discipline the WebGPU bindings (ADR-002) already follow.
- **Automatic fallback preserves reach.** Older browsers still load the app via CanvasKit/dart2js without a separate build pipeline or manual gating.
- **Explicit divergence record.** Capturing the difference from DGROUP_WEB-ADR-020 prevents confusion and gives future DGroup `--wasm` migrations a reference data point.

### Negative

- **Hosting is constrained.** Requiring COOP/COEP excludes plain static hosts that cannot set response headers; the deploy target must allow header configuration (or sit behind a layer that does). This is a hard hosting constraint, not a nice-to-have.
- **COEP `require-corp` constrains embedded subresources.** Every cross-origin subresource the page loads must be CORP/CORS-eligible or it will be blocked. The owned WebGPU canvas and assets must be served compatibly with cross-origin isolation.
- **Dependency-tree interop audit cost.** A single transitive dependency pulling in `dart:html`/`dart:js_util` breaks the `--wasm` build, so dependency selection and upgrades require ongoing vigilance.
- **Two render paths to reason about.** The Skwasm vs CanvasKit/dart2js split means behavior and performance must be considered on both, even though fallback is automatic.

### Neutral

- The **initial CPU fallback (ADR-006) is single-threaded**, so cross-origin isolation is not strictly required for it to *function*; COOP/COEP are provisioned now so the threaded/`SharedArrayBuffer` upgrade path is available without re-architecting hosting.
- WasmGC browser thresholds (Chrome 119+ / Firefox 120+ / Safari 18.2+) are *engine* requirements distinct from the WebGPU support matrix in ADR-002/ADR-006; a browser can run Skwasm yet lack `navigator.gpu` (and vice versa). The two capability checks are orthogonal.
- The Flutter renderer here (Skwasm/CanvasKit, Skia/WebGL) is intentionally separate from the WebGPU compute canvas (ADR-002/ADR-005); this ADR governs only the Flutter-side compilation and serving.

## Alternatives Considered

### dart2js + CanvasKit only (no `--wasm`)

Build the Flutter web target the way DGROUP_WEB-ADR-020 specifies for the DGroup app: dart2js + CanvasKit, no Skwasm/dart2wasm. **Rejected** as the default because it is the slower fallback baseline rather than the modern target, and it forgoes the WasmGC/Skwasm path on capable engines. The `--wasm` build already *includes* this configuration as its automatic fallback, so choosing it exclusively would mean accepting the slower baseline for all users with no upside. It remains the path older browsers transparently receive.

### Static host without COOP/COEP

Deploy to a plain static host that cannot set response headers, skipping cross-origin isolation. **Rejected** because without COOP `same-origin` + COEP `require-corp` the page is not cross-origin isolated, `SharedArrayBuffer` is unavailable, and therefore multi-threaded Skwasm and any future worker/`SharedArrayBuffer`-backed CPU fallback are impossible — it permanently caps the CPU tier at single-thread and forecloses threading. The marginal convenience of a header-less static host does not justify giving up shared-memory threading.

## References

- [PRIMORDIS-ADR-002: Web GPU compute via WebGPU and JS interop](./PRIMORDIS-ADR-002-web-gpu-compute-webgpu-js-interop.md) — owned WebGPU canvas; `dart:js_interop` + `package:web` bindings.
- [PRIMORDIS-ADR-006: CPU fallback tiers and feature detection](./PRIMORDIS-ADR-006-cpu-fallback-tiers-and-feature-detection.md) — `navigator.gpu` detection and the CPU-WASM fallback this build serves.
- [PRIMORDIS-ADR-005: Rendering and compositing](./PRIMORDIS-ADR-005-rendering-and-compositing.md) — stacked WebGPU canvas behind the transparent Flutter glass-pane.
- DGROUP_WEB-ADR-020: Flutter Web Rendering and Compilation — DGroup standard (CanvasKit + dart2js, future `--wasm`); Primordis diverges to `--wasm` now.
- [PRIMORDIS-TASK-010: Web build, hosting, and cross-origin isolation](../tasks/PRIMORDIS-TASK-010-web-build-hosting-and-cross-origin-isolation.md) — implements `--wasm`/Skwasm build, CanvasKit/dart2js fallback, COOP/COEP headers, deploy.
- [PRIMORDIS-TASK-001: Project scaffold and build config](../tasks/PRIMORDIS-TASK-001-project-scaffold-and-build-config.md) — `--wasm` build config baseline and interop lint rule.
- [PRIMORDIS-PRD-001: Flutter Web and macOS port](../prd/PRIMORDIS-PRD-001-flutter-web-and-macos-port.md)
- [PRIMORDIS research summary](../research/PRIMORDIS-research-summary.md)
- flutter.dev — Web renderers and the `flutter build web --wasm` (WebAssembly/dart2wasm + Skwasm) build mode, including CanvasKit/dart2js fallback.
- flutter.dev — JavaScript interoperability with `dart:js_interop` and `package:web` (legacy `dart:html`/`dart:js_util` incompatibility with `--wasm`).
- WebAssembly — WasmGC proposal and engine support (Chrome 119+ / Firefox 120+ / Safari 18.2+) underlying Skwasm.
- MDN / web platform — `Cross-Origin-Opener-Policy` (`same-origin`), `Cross-Origin-Embedder-Policy` (`require-corp`), cross-origin isolation, and `SharedArrayBuffer` requirements.
- WebGPU specification — `navigator.gpu` feature detection (see ADR-002/ADR-006), referenced for the orthogonal compute-capability check.
- Apple Developer (Metal) — referenced via the native compute path in [PRIMORDIS-ADR-004](./PRIMORDIS-ADR-004-native-macos-gpu-dawn-metal-ffi.md); out of scope for this web-build ADR.
