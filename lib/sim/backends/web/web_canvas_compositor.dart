// The DOM side of the web present path: stacks the backend's owned WebGPU
// `<canvas>` BEHIND the transparent Flutter glass-pane and keeps its backing
// store synced to `devicePixelRatio` ([PRIMORDIS-ADR-005] / [PRIMORDIS-TASK-005]).
//
// Web-only: `dart:js_interop` + `package:web`, reachable only via the conditional
// facade ([sim_compositor_factory.dart]). No `dart:html`, no `dart:js_util`
// ([PRIMORDIS-ADR-007]).
//
// Input is NOT routed here. The canvas is set `pointer-events: none` so it never
// participates in hit-testing; the Flutter glass-pane on top owns every pointer
// and the field/control split is decided by [PointerRouter] in the widget layer.
// This file only does placement, stacking, and DPR/backing-store sizing — the
// two recurring web bug classes the ADR flags.
@JS()
library;

import 'dart:js_interop';

import 'package:primordis/sim/backends/web/sim_compositor.dart';
import 'package:primordis/sim/backends/web/web_webgpu_backend.dart';
import 'package:web/web.dart' as web;

/// The DOM id given to the sibling canvas (matches the reserved anchor comment
/// in `web/index.html`).
const String kGpuCanvasId = 'primordis-gpu-canvas';

/// Stacks and sizes the WebGPU backend's canvas under the Flutter view.
///
/// Created by [createSimCompositor] only when the live backend is a
/// [WebWebGpuBackend]; the CPU tier gets a [NoopSimCompositor] instead. The
/// canvas itself is created at `seed()` time inside the backend, so this
/// compositor can be constructed before the canvas exists — [syncLayout]
/// lazily attaches it once it appears, and no-ops until then.
class WebCanvasCompositor implements SimCompositor {
  WebCanvasCompositor(this._backend);

  final WebWebGpuBackend _backend;

  /// The last layout applied, so an unchanged [syncLayout] skips all DOM work —
  /// avoiding a redundant WebGPU `configure()` (which would discard the
  /// swap-chain and could blank a paused last frame, [PRIMORDIS-ADR-006]).
  CompositorLayout? _last;

  @override
  void syncLayout(CompositorLayout layout) {
    final canvas = _backend.canvasElement;
    // No canvas yet (pre-seed): nothing to place.
    if (canvas == null) return;
    if (layout == _last) return;

    _ensureStacked(canvas);

    // CSS placement (cheap; always applied on a real change). The field rect is
    // in page CSS pixels and the canvas is `position: fixed`, so these line the
    // simulation up exactly under the Flutter overlay region.
    final style = canvas.style;
    style
      ..setProperty('left', '${layout.cssLeft}px')
      ..setProperty('top', '${layout.cssTop}px')
      ..setProperty('width', '${layout.cssWidth}px')
      ..setProperty('height', '${layout.cssHeight}px');

    // Backing store (device pixels). Reconfiguring the WebGPU context is the
    // expensive part, so only resize when the device-pixel size actually
    // changed — DPR shifts and region resizes converge here.
    if (canvas.width != layout.backingWidth ||
        canvas.height != layout.backingHeight) {
      _backend.resizeCanvas(
        width: layout.backingWidth,
        height: layout.backingHeight,
      );
    }

    _last = layout;
  }

  @override
  void detach() {
    // The backend owns the canvas lifecycle (its `dispose()` removes the
    // element). Here we only drop our cached layout so a later re-attach
    // re-applies placement from scratch.
    _last = null;
  }

  /// Inserts the canvas as a sibling positioned behind the Flutter view and
  /// applies the one-time base styling. Idempotent: a no-op once connected.
  void _ensureStacked(web.HTMLCanvasElement canvas) {
    if (canvas.isConnected) return;
    canvas.id = kGpuCanvasId;
    canvas.style
      // Fixed + low z-index puts it behind the Flutter glass-pane; the page CSS
      // (`web/index.html`) gives the Flutter view a higher z-index and a
      // transparent background so the simulation shows through.
      ..setProperty('position', 'fixed')
      ..setProperty('z-index', '0')
      // The canvas must never steal pointers — Flutter owns input.
      ..setProperty('pointer-events', 'none')
      ..setProperty('display', 'block');
    // Prepend so it sits before (visually behind, with z-index) the bootstrap-
    // injected Flutter host.
    final body = web.document.body;
    if (body != null) body.insertBefore(canvas, body.firstChild);
  }
}
