// Web implementation for the conditional facade ([web_backend.dart]).
//
// Selected only when `dart.library.js_interop` is available (the
// `flutter build web` / `--wasm` target). Delegates to the real WebGPU backend;
// importing this file is what pulls in `dart:js_interop` + `package:web`, so it
// is reached ONLY through the conditional import.
import 'package:primordis/sim/backends/web/web_webgpu_backend.dart';
import 'package:primordis/sim/backends/web/webgpu_support.dart';
import 'package:primordis/sim/sim_backend.dart';

/// Probes browser WebGPU availability (never throws).
Future<WebGpuSupport> probeWebGpu() => WebWebGpuBackend.probe();

/// Constructs the web WebGPU backend (un-initialized; the caller `init()`s it).
SimBackend? createWebSimBackend() => WebWebGpuBackend();
