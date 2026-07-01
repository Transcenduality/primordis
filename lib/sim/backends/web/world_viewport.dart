// Pure geometry for fitting the toroidal world into the displayed canvas region.
//
// This is the maths behind the two recurring bug classes [PRIMORDIS-ADR-005]
// flags for the web present path — DPR/backing-store sizing and
// world/letterbox alignment — pulled OUT of the JS-interop layer so it is plain,
// deterministic Dart that runs under `flutter test`. No `dart:js_interop`, no
// `package:web`: only `dart:ui` value types ([Offset]/[Rect]). The web
// compositor ([web_canvas_compositor.dart]) consumes the [CompositorLayout] this
// produces; the pointer router ([web_pointer_router.dart]) consumes the
// world-coordinate mapping.
import 'dart:math' as math;
import 'dart:ui';

import 'package:primordis/sim/backends/web/sim_compositor.dart';

/// How the fixed-aspect [worldWidth]×[worldHeight] world is placed inside a
/// CSS-logical canvas region of [regionWidth]×[regionHeight] at a given
/// [devicePixelRatio].
///
/// The fit is **contain** (aspect-preserving, centred): the world is scaled by
/// the larger axis constraint so it never overflows, and the unused axis becomes
/// symmetric letterbox margin. The sibling `<canvas>` is sized to the fitted
/// **field** rect (not the whole region), so the existing full-clip-space render
/// pipeline ([PRIMORDIS-TASK-003], which draws to the full NDC cube and knows
/// nothing about letterboxing) fills the canvas with the correct aspect, and the
/// margins are simply the transparent page showing through behind the glass-pane.
///
/// The backing store is `field * devicePixelRatio` device pixels — the
/// [PRIMORDIS-ADR-005] sharpness requirement — recomputed whenever the region or
/// DPR changes (e.g. dragging the window between a Retina and non-Retina
/// display).
///
/// All fields are CSS-logical pixels except [backingWidth]/[backingHeight].
/// Value-equal so the compositor can skip redundant DOM writes / context
/// reconfigures while idle (which keeps a paused last frame held —
/// [PRIMORDIS-ADR-006]).
class WorldViewport {
  const WorldViewport({
    required this.regionWidth,
    required this.regionHeight,
    required this.devicePixelRatio,
    required this.worldWidth,
    required this.worldHeight,
  });

  /// Available region width in CSS-logical pixels.
  final double regionWidth;

  /// Available region height in CSS-logical pixels.
  final double regionHeight;

  /// `MediaQuery.devicePixelRatio` — device pixels per logical pixel.
  final double devicePixelRatio;

  /// Toroidal world width in world units (1080 — [PrimordisConfig]).
  final double worldWidth;

  /// Toroidal world height in world units (720 — [PrimordisConfig]).
  final double worldHeight;

  /// Logical pixels per world unit under the contain fit. Zero when the region
  /// or world has collapsed to a non-positive extent (first layout / minimised
  /// window), which makes every derived quantity degrade safely.
  double get scale {
    if (regionWidth <= 0 ||
        regionHeight <= 0 ||
        worldWidth <= 0 ||
        worldHeight <= 0) {
      return 0;
    }
    return math.min(regionWidth / worldWidth, regionHeight / worldHeight);
  }

  /// Fitted field width in CSS-logical pixels.
  double get fieldWidth => worldWidth * scale;

  /// Fitted field height in CSS-logical pixels.
  double get fieldHeight => worldHeight * scale;

  /// Left letterbox margin (centres the field horizontally).
  double get offsetX => (regionWidth - fieldWidth) / 2;

  /// Top letterbox margin (centres the field vertically).
  double get offsetY => (regionHeight - fieldHeight) / 2;

  /// The fitted field rectangle within the region, in region-local CSS pixels.
  Rect get fieldRect => Rect.fromLTWH(offsetX, offsetY, fieldWidth, fieldHeight);

  /// Canvas backing-store width in device pixels (`field * dpr`, min 1).
  int get backingWidth => math.max(1, (fieldWidth * devicePixelRatio).round());

  /// Canvas backing-store height in device pixels (`field * dpr`, min 1).
  int get backingHeight =>
      math.max(1, (fieldHeight * devicePixelRatio).round());

  /// Maps a region-local pointer position to world coordinates
  /// (`0 <= x < worldWidth`, `0 <= y < worldHeight` — half-open at the max
  /// edges because the fitted field uses [Rect.contains]), or null when the
  /// point lies in a
  /// letterbox margin (outside the simulated field). The router uses null to
  /// keep margin taps with the Flutter overlay rather than the backend.
  Offset? worldFromRegionLocal(Offset regionLocal) {
    final field = fieldRect;
    if (scale <= 0 || !field.contains(regionLocal)) return null;
    return Offset(
      (regionLocal.dx - field.left) / scale,
      (regionLocal.dy - field.top) / scale,
    );
  }

  /// Builds the DOM-facing [CompositorLayout] for the sibling canvas, translating
  /// the region-local field rect into page CSS pixels by adding the region's
  /// global [regionOrigin] (the Flutter view sits at the page origin, so a
  /// Flutter-global logical offset is a page CSS offset).
  CompositorLayout toCompositorLayout(Offset regionOrigin) => CompositorLayout(
        cssLeft: regionOrigin.dx + offsetX,
        cssTop: regionOrigin.dy + offsetY,
        cssWidth: fieldWidth,
        cssHeight: fieldHeight,
        backingWidth: backingWidth,
        backingHeight: backingHeight,
      );

  @override
  bool operator ==(Object other) =>
      other is WorldViewport &&
      other.regionWidth == regionWidth &&
      other.regionHeight == regionHeight &&
      other.devicePixelRatio == devicePixelRatio &&
      other.worldWidth == worldWidth &&
      other.worldHeight == worldHeight;

  @override
  int get hashCode => Object.hash(
        regionWidth,
        regionHeight,
        devicePixelRatio,
        worldWidth,
        worldHeight,
      );

  @override
  String toString() =>
      'WorldViewport(region: ${regionWidth}x$regionHeight, dpr: '
      '$devicePixelRatio, field: ${fieldWidth}x$fieldHeight @ '
      '($offsetX,$offsetY), backing: ${backingWidth}x$backingHeight)';
}
