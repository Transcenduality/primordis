import 'dart:io';

import 'package:flutter/foundation.dart';

import 'package:primordis/sim/backends/macos/macos_dawn_backend.dart';
import 'package:primordis/sim/ffi/minigpu_dawn_gpu.dart';
import 'package:primordis/sim/sim_backend.dart';

/// `dart:io` side of the macOS-backend facade.
///
/// Returns an un-initialized [MacosDawnBackend] on macOS, null elsewhere.
/// Lifecycle bring-up (`init`/`seed`) — and falling back to the CPU tier when
/// `init` throws (Dawn/device failure) — is the selector's job
/// (PRIMORDIS-TASK-805).
SimBackend? createMacosDawnBackend() {
  if (!Platform.isMacOS) return null;
  return MacosDawnBackend(
    MinigpuDawnGpu(
      onLog: (level, message) => debugPrint('[mgpu:$level] $message'),
    ),
  );
}
