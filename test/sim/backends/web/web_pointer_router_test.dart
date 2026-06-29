import 'dart:ui' show Offset, Rect;

import 'package:flutter_test/flutter_test.dart';
import 'package:primordis/sim/backends/web/web_pointer_router.dart';
import 'package:primordis/sim/backends/web/world_viewport.dart';

const PointerRouter _router = PointerRouter();

/// A 1080×720 field exactly filling the region (scale 1, no letterbox).
const WorldViewport _fullField = WorldViewport(
  regionWidth: 1080,
  regionHeight: 720,
  devicePixelRatio: 1,
  worldWidth: 1080,
  worldHeight: 720,
);

void main() {
  group('field vs control routing', () {
    test('open field forwards to the backend with the world coordinate', () {
      final r = _router.routeAt(
        const Offset(500, 400),
        controlRects: const <Rect>[],
        viewport: _fullField,
      );
      expect(r.route, PointerRoute.field);
      expect(r.world, const Offset(500, 400));
    });

    test('a point inside a control rect stays with Flutter (no world)', () {
      final r = _router.routeAt(
        const Offset(50, 40),
        controlRects: <Rect>[const Rect.fromLTWH(0, 0, 200, 100)],
        viewport: _fullField,
      );
      expect(r.route, PointerRoute.flutter);
      expect(r.world, isNull);
    });

    test('control wins even when the point is also over the field', () {
      // (50,40) is inside both the field and the control: control takes it, so
      // the seam dispatches exactly once (no double-handling).
      final r = _router.routeAt(
        const Offset(50, 40),
        controlRects: <Rect>[const Rect.fromLTWH(0, 0, 200, 100)],
        viewport: _fullField,
      );
      expect(r.route, PointerRoute.flutter);
    });

    test('honours any of several control rects', () {
      const rects = <Rect>[
        Rect.fromLTWH(0, 0, 100, 100),
        Rect.fromLTWH(980, 620, 100, 100),
      ];
      expect(
        _router.routeAt(const Offset(1000, 650), controlRects: rects, viewport: _fullField).route,
        PointerRoute.flutter,
      );
      expect(
        _router.routeAt(const Offset(540, 360), controlRects: rects, viewport: _fullField).route,
        PointerRoute.field,
      );
    });
  });

  group('letterbox margins', () {
    const letterboxed = WorldViewport(
      regionWidth: 1080,
      regionHeight: 540, // height-bound → vertical bars at x<135, x>=945
      devicePixelRatio: 1,
      worldWidth: 1080,
      worldHeight: 720,
    );

    test('a margin tap (no control) is inert — stays with Flutter', () {
      final r = _router.routeAt(
        const Offset(50, 270),
        controlRects: const <Rect>[],
        viewport: letterboxed,
      );
      expect(r.route, PointerRoute.flutter);
      expect(r.world, isNull);
    });

    test('inside the fitted field still routes to the backend', () {
      final r = _router.routeAt(
        const Offset(540, 270),
        controlRects: const <Rect>[],
        viewport: letterboxed,
      );
      expect(r.route, PointerRoute.field);
      expect(r.world!.dx, closeTo(540, 1e-9));
      expect(r.world!.dy, closeTo(360, 1e-9));
    });
  });
}
