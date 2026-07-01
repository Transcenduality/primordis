/// Conditional-import facade for the platform compositor.
///
/// The single entry point the provider ([compositor_provider.dart]) uses to
/// obtain a [SimCompositor] WITHOUT importing web-only code unconditionally. The
/// import below resolves to:
///
/// - [sim_compositor_stub.dart] off-web (Dart VM / `flutter test`, macOS) — no
///   `dart:js_interop` / `package:web`; always [NoopSimCompositor].
/// - [sim_compositor_web.dart] on web (`dart.library.js_interop`) — the DOM
///   canvas compositor when a WebGPU backend is live, else no-op.
///
/// This keeps the DOM/JS-interop compositing strictly below the `SimBackend`
/// seam and out of every non-web build ([PRIMORDIS-ADR-001] / [PRIMORDIS-ADR-007]).
library;

import 'package:primordis/sim/backends/web/sim_compositor.dart';
import 'package:primordis/sim/backends/web/sim_compositor_stub.dart'
    if (dart.library.js_interop) 'package:primordis/sim/backends/web/sim_compositor_web.dart'
    as impl;
import 'package:primordis/sim/sim_backend.dart';

export 'package:primordis/sim/backends/web/sim_compositor.dart'
    show CompositorLayout, NoopSimCompositor, SimCompositor;

/// Builds the compositor for [backend]: the DOM canvas compositor on web when a
/// WebGPU backend is live, otherwise a [NoopSimCompositor].
SimCompositor createSimCompositor(SimBackend backend) =>
    impl.createSimCompositor(backend);
