// Non-web default for the conditional facade ([web_backend.dart]).
//
// Selected on every target WITHOUT `dart.library.js_interop` — the Dart VM
// (`flutter test`), macOS, and any native build. It pulls in NO web-only code
// (`dart:js_interop` / `package:web`), so non-web builds never compile the
// WebGPU backend ([PRIMORDIS-TASK-004] conditional-import requirement).
import 'package:primordis/sim/backends/web/webgpu_support.dart';
import 'package:primordis/sim/sim_backend.dart';

/// Always reports no WebGPU API off-web.
Future<WebGpuSupport> probeWebGpu() async => WebGpuSupport.unsupportedNoApi;

/// No web backend exists off-web.
SimBackend? createWebSimBackend() => null;
