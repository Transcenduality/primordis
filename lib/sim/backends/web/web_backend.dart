/// Conditional-import facade for the web WebGPU backend.
///
/// The single entry point the rest of the app (the provider, and later the
/// selector in [PRIMORDIS-TASK-007]) uses to reach the web backend WITHOUT
/// importing web-only code unconditionally. The import below resolves to:
///
/// - [web_backend_stub.dart] off-web (Dart VM / `flutter test`, macOS, native) —
///   no `dart:js_interop` / `package:web` pulled in; reports unsupported.
/// - [web_backend_web.dart] on web (`dart.library.js_interop`) — the real
///   WebGPU backend.
///
/// This keeps the WebGPU/JS-interop strictly below the `SimBackend` seam and out
/// of every non-web build ([PRIMORDIS-ADR-001] / [PRIMORDIS-ADR-007]).
library;

import 'package:primordis/sim/backends/web/web_backend_stub.dart'
    if (dart.library.js_interop) 'package:primordis/sim/backends/web/web_backend_web.dart'
    as impl;
import 'package:primordis/sim/backends/web/webgpu_support.dart';
import 'package:primordis/sim/sim_backend.dart';

export 'package:primordis/sim/backends/web/webgpu_support.dart'
    show WebGpuSupport;

/// Reports whether browser WebGPU is usable on this platform (never throws).
/// Off-web this is always [WebGpuSupport.unsupportedNoApi].
Future<WebGpuSupport> probeWebGpu() => impl.probeWebGpu();

/// Constructs the web WebGPU [SimBackend], or null off-web. The returned backend
/// is un-initialized — the caller awaits `init()` then `seed()` (the lifecycle
/// on [SimBackend]) before stepping.
SimBackend? createWebSimBackend() => impl.createWebSimBackend();
