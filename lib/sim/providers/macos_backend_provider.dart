import 'dart:async';

import 'package:primordis/sim/backends/macos/macos_backend.dart';
import 'package:primordis/sim/sim_backend.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'macos_backend_provider.g.dart';

/// Riverpod provider exposing the macOS Dawn backend behind the [SimBackend]
/// interface via the conditional-import facade ([macos_backend.dart]) —
/// mirroring `web_backend_provider.dart`.
///
/// This is the seam the cross-platform backend selector
/// ([PRIMORDIS-TASK-805]) consumes: on macOS it resolves to an un-initialized
/// [MacosDawnBackend]; anywhere else it is null and pulls in no FFI/minigpu
/// code ([PRIMORDIS-ADR-001] / [PRIMORDIS-ADR-004]). If `init()` later throws
/// (Dawn/device failure) the selector falls back to the Metal/MSL plugin tier
/// ([PRIMORDIS-TASK-803], currently deferred) or the native CPU tier
/// ([PRIMORDIS-TASK-804]).
@Riverpod(keepAlive: true)
SimBackend? macosDawnSimBackend(Ref ref) {
  final backend = createMacosDawnBackend();
  if (backend != null) {
    ref.onDispose(() => unawaited(backend.dispose()));
  }
  return backend;
}
