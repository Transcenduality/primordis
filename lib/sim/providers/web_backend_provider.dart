import 'dart:async';

import 'package:primordis/sim/backends/web/web_backend.dart';
import 'package:primordis/sim/sim_backend.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'web_backend_provider.g.dart';

/// Riverpod providers exposing the web WebGPU backend behind the [SimBackend]
/// interface, via the conditional-import facade ([web_backend.dart]).
///
/// These are the seam the backend selector ([PRIMORDIS-TASK-007]) consumes: it
/// reads [webGpuSupportProvider] and, when supported, overrides the app's
/// `simBackendProvider` with [webSimBackendProvider]. Off-web both resolve to
/// the stub (unsupported / null), so reading them on the Dart VM or native is
/// safe and pulls in no WebGPU/JS-interop code ([PRIMORDIS-ADR-001]).

/// Whether browser WebGPU is available on this platform. A [Future] because the
/// probe awaits `requestAdapter()`. Off-web: [WebGpuSupport.unsupportedNoApi].
@riverpod
Future<WebGpuSupport> webGpuSupport(Ref ref) => probeWebGpu();

/// The web WebGPU backend instance (un-initialized), or null off-web.
///
/// Kept alive so the backend persists for the app's lifetime; disposed with the
/// container. Lifecycle bring-up (`init`/`seed`) is the driver's job
/// ([PRIMORDIS-TASK-005] / [PRIMORDIS-TASK-006]), mirroring `simBackendProvider`.
@Riverpod(keepAlive: true)
SimBackend? webSimBackend(Ref ref) {
  final backend = createWebSimBackend();
  if (backend != null) {
    ref.onDispose(() => unawaited(backend.dispose()));
  }
  return backend;
}
