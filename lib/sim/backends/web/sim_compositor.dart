// The neutral compositor seam: the present surface placement contract the UI
// drives, with no platform specifics leaking above it.
//
// [PRIMORDIS-ADR-005] composites the web simulation as a sibling DOM `<canvas>`
// stacked BEHIND a transparent Flutter glass-pane. The DOM/JS-interop that
// actually does the stacking is web-only ([web_canvas_compositor.dart]); this
// file declares the interface + the geometry value object so the widget layer
// ([features/simulation]) and the provider can talk to "the compositor" without
// importing `dart:js_interop` / `package:web`. Off-web and on the CPU fallback
// tier the implementation is [NoopSimCompositor] — there is no sibling canvas to
// place ([PRIMORDIS-TASK-008]).
//
// Pure Dart: no JS interop, no `dart:ui`. Constructed through the conditional
// facade [createSimCompositor] in [sim_compositor_factory.dart].

/// Where to place the sibling canvas, in the DOM's coordinate space.
///
/// CSS fields are page CSS pixels (the canvas is `position: fixed`); the backing
/// fields are device pixels (`css * devicePixelRatio`). Produced by
/// [WorldViewport.toCompositorLayout]. Value-equal so the compositor can skip a
/// redundant style write / WebGPU reconfigure when the layout has not changed —
/// which both saves work and lets a paused last frame stay on screen
/// ([PRIMORDIS-ADR-006]).
class CompositorLayout {
  const CompositorLayout({
    required this.cssLeft,
    required this.cssTop,
    required this.cssWidth,
    required this.cssHeight,
    required this.backingWidth,
    required this.backingHeight,
  });

  /// Field left edge in page CSS pixels.
  final double cssLeft;

  /// Field top edge in page CSS pixels.
  final double cssTop;

  /// Field width in CSS pixels (the canvas element's logical size).
  final double cssWidth;

  /// Field height in CSS pixels (the canvas element's logical size).
  final double cssHeight;

  /// Backing-store width in device pixels.
  final int backingWidth;

  /// Backing-store height in device pixels.
  final int backingHeight;

  @override
  bool operator ==(Object other) =>
      other is CompositorLayout &&
      other.cssLeft == cssLeft &&
      other.cssTop == cssTop &&
      other.cssWidth == cssWidth &&
      other.cssHeight == cssHeight &&
      other.backingWidth == backingWidth &&
      other.backingHeight == backingHeight;

  @override
  int get hashCode => Object.hash(
        cssLeft,
        cssTop,
        cssWidth,
        cssHeight,
        backingWidth,
        backingHeight,
      );

  @override
  String toString() =>
      'CompositorLayout(css: ${cssWidth}x$cssHeight @ ($cssLeft,$cssTop), '
      'backing: ${backingWidth}x$backingHeight)';
}

/// Places and sizes the GPU present surface behind the Flutter glass-pane.
///
/// The widget layer holds one of these (via `simCompositorProvider`) and calls
/// [syncLayout] whenever the simulation region's size or `devicePixelRatio`
/// changes. Implementations must be idempotent and tolerate being called before
/// the backend has a canvas (the web canvas is created at `seed()` time, after
/// the provider/compositor exist).
abstract interface class SimCompositor {
  /// Stacks (if needed) and sizes the sibling canvas to [layout]. Idempotent;
  /// a no-op when [layout] is unchanged or there is no sibling canvas yet.
  void syncLayout(CompositorLayout layout);

  /// Releases compositor-held DOM state. Safe to call more than once. The
  /// canvas element itself is owned by the backend, which disposes it.
  void detach();
}

/// The compositor used off-web and on the CPU fallback tier, where the
/// simulation is drawn inside Flutter and there is no external canvas to stack
/// ([PRIMORDIS-ADR-005] / [PRIMORDIS-TASK-008]). Every method is a no-op, so the
/// same widget code runs unchanged on macOS, in `flutter test`, and in the
/// reduced web tier.
class NoopSimCompositor implements SimCompositor {
  const NoopSimCompositor();

  @override
  void syncLayout(CompositorLayout layout) {}

  @override
  void detach() {}
}
