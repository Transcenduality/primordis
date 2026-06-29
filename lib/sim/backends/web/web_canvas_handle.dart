// Owns the WebGPU `<canvas>` and its configured `GPUCanvasContext`.
//
// Per [PRIMORDIS-ADR-005], the WebGPU surface is a sibling DOM `<canvas>` the
// app creates and manages itself — NOT an `HtmlElementView` (which forces
// overlay/canvas-splitting and jank). This task ([PRIMORDIS-TASK-004]) creates
// and configures the canvas + context; the DOM stacking behind the transparent
// Flutter glass-pane, pointer routing, and DPR/resize syncing are owned by
// [PRIMORDIS-TASK-005], which reuses this handle.
//
// Web-only: imports `dart:js_interop` + `package:web` and is reachable only
// through the conditional facade ([web_backend.dart]).
@JS()
library;

import 'dart:js_interop';

import 'package:primordis/sim/backends/web/webgpu_interop.dart';
import 'package:web/web.dart' as web;

/// Creates, configures, and owns the WebGPU canvas + `GPUCanvasContext`.
///
/// The context is configured with the device's preferred format and a
/// premultiplied alpha mode so the canvas can sit transparently *under* the
/// Flutter glass-pane ([PRIMORDIS-ADR-005]); the actual compositing/stacking is
/// TASK-005. Sizing here is the raw backing-store pixel size — DPR handling is
/// also TASK-005.
class WebCanvasHandle {
  WebCanvasHandle._(this.canvas, this.context, this.format);

  /// Creates an off-document `<canvas>` of [width]×[height] device pixels and
  /// configures its `webgpu` context against [device].
  ///
  /// Returns null when the `webgpu` context cannot be obtained (a browser that
  /// exposes `navigator.gpu` yet fails to hand back a context — rare, but
  /// handled rather than thrown so the backend can degrade gracefully).
  static WebCanvasHandle? create({
    required GPUDevice device,
    required GPU gpu,
    required int width,
    required int height,
  }) {
    final canvas =
        web.document.createElement('canvas') as web.HTMLCanvasElement
          ..width = width
          ..height = height;

    final context = canvas.getContext('webgpu') as GPUCanvasContext?;
    if (context == null) return null;

    final format = gpu.getPreferredCanvasFormat();
    context.configure(
      GPUCanvasConfiguration(
        device: device,
        format: format,
        // Premultiplied so the transparent regions of the render let the
        // Flutter UI show through when stacked (TASK-005 / ADR-005).
        alphaMode: 'premultiplied',
      ),
    );
    return WebCanvasHandle._(canvas, context, format);
  }

  /// The owned canvas element (stacked + sized by TASK-005).
  final web.HTMLCanvasElement canvas;

  /// The configured WebGPU context. [present] targets its current texture.
  final GPUCanvasContext context;

  /// The swap-chain texture format the context was configured with — the render
  /// pipeline's colour-target format must match this.
  final String format;

  /// A fresh view of the context's current swap-chain texture, the render
  /// pass's colour attachment for this frame.
  GPUTextureView currentView() => context.getCurrentTexture().createView();

  /// Resizes the canvas backing store, reconfiguring the context against
  /// [device] (WebGPU requires reconfigure after a size change). Reuses the
  /// format chosen at creation.
  void resize({
    required GPUDevice device,
    required int width,
    required int height,
  }) {
    canvas
      ..width = width
      ..height = height;
    context.configure(
      GPUCanvasConfiguration(
        device: device,
        format: format,
        alphaMode: 'premultiplied',
      ),
    );
  }

  /// Releases the context's swap-chain resources and detaches the canvas from
  /// the DOM if it was attached (TASK-005 owns attachment, but dispose is safe
  /// either way).
  void dispose() {
    context.unconfigure();
    canvas.remove();
  }
}
