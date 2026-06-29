// Web implementation for the compositor facade ([sim_compositor_factory.dart]).
//
// Selected only when `dart.library.js_interop` is available (the
// `flutter build web` / `--wasm` target). Importing this file is what pulls in
// the DOM compositor (and through it `dart:js_interop` + `package:web`), so it is
// reached ONLY through the conditional import.
//
// The real DOM compositor only exists when the active backend is the WebGPU
// backend (it needs that backend's owned canvas). On the web CPU fallback tier
// ([PRIMORDIS-TASK-008]) the active backend is not a [WebWebGpuBackend], so we
// return the no-op compositor — cleanly NOT creating a sibling canvas, as
// [PRIMORDIS-TASK-005] requires.
import 'package:primordis/sim/backends/web/sim_compositor.dart';
import 'package:primordis/sim/backends/web/web_canvas_compositor.dart';
import 'package:primordis/sim/backends/web/web_webgpu_backend.dart';
import 'package:primordis/sim/sim_backend.dart';

/// The DOM canvas compositor when a WebGPU backend is live; otherwise no-op.
SimCompositor createSimCompositor(SimBackend backend) => switch (backend) {
      final WebWebGpuBackend b => WebCanvasCompositor(b),
      _ => const NoopSimCompositor(),
    };
