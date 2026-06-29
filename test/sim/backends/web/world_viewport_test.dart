import 'package:flutter_test/flutter_test.dart';
import 'package:primordis/sim/backends/web/world_viewport.dart';

/// Builds a viewport over the 1080×720 world (the fixed sim world).
WorldViewport vp(double w, double h, double dpr) => WorldViewport(
      regionWidth: w,
      regionHeight: h,
      devicePixelRatio: dpr,
      worldWidth: 1080,
      worldHeight: 720,
    );

void main() {
  group('contain fit', () {
    test('exact world aspect fills the region with no letterbox', () {
      final v = vp(1080, 720, 1);
      expect(v.scale, 1.0);
      expect(v.fieldWidth, 1080);
      expect(v.fieldHeight, 720);
      expect(v.offsetX, 0);
      expect(v.offsetY, 0);
    });

    test('wider-than-world region letterboxes horizontally (height-bound)', () {
      // 1080×540 is wider (2.0) than the world (1.5): height constrains.
      final v = vp(1080, 540, 1);
      expect(v.scale, closeTo(0.75, 1e-9));
      expect(v.fieldWidth, closeTo(810, 1e-9));
      expect(v.fieldHeight, closeTo(540, 1e-9));
      expect(v.offsetX, closeTo(135, 1e-9)); // (1080-810)/2
      expect(v.offsetY, 0);
    });

    test('taller-than-world region letterboxes vertically (width-bound)', () {
      // 1080×900 is taller (1.2) than the world (1.5): width constrains.
      final v = vp(1080, 900, 1);
      expect(v.scale, 1.0);
      expect(v.fieldWidth, 1080);
      expect(v.fieldHeight, 720);
      expect(v.offsetX, 0);
      expect(v.offsetY, closeTo(90, 1e-9)); // (900-720)/2
    });
  });

  group('backing store = field × dpr', () {
    test('scales with devicePixelRatio, css size unchanged', () {
      expect(vp(1080, 720, 1).backingWidth, 1080);
      expect(vp(1080, 720, 1).backingHeight, 720);

      final hi = vp(1080, 720, 2);
      expect(hi.backingWidth, 2160);
      expect(hi.backingHeight, 1440);
      expect(hi.fieldWidth, 1080, reason: 'css/logical size is dpr-independent');

      expect(vp(1080, 720, 3).backingWidth, 3240);
    });

    test('fractional dpr rounds to whole device pixels', () {
      final v = vp(1080, 720, 1.5);
      expect(v.backingWidth, 1620); // round(1080 * 1.5)
      expect(v.backingHeight, 1080); // round(720 * 1.5)
    });

    test('collapsed region degrades safely (scale 0, backing min 1)', () {
      final v = vp(0, 0, 2);
      expect(v.scale, 0);
      expect(v.backingWidth, 1);
      expect(v.backingHeight, 1);
      expect(v.worldFromRegionLocal(Offset.zero), isNull);
    });
  });

  group('worldFromRegionLocal', () {
    test('maps the field interior to world coordinates', () {
      final v = vp(1080, 720, 1);
      expect(v.worldFromRegionLocal(const Offset(540, 360)), const Offset(540, 360));
      expect(v.worldFromRegionLocal(const Offset(108, 72)), const Offset(108, 72));
    });

    test('accounts for letterbox offset and scale', () {
      final v = vp(1080, 540, 1); // scale 0.75, offsetX 135
      // Field centre maps to the world centre.
      final c = v.worldFromRegionLocal(const Offset(540, 270)); // 135 + 810/2
      expect(c!.dx, closeTo(540, 1e-9));
      expect(c.dy, closeTo(360, 1e-9));
    });

    test('returns null in a letterbox margin (outside the field)', () {
      final v = vp(1080, 540, 1); // horizontal bars at x<135 and x>=945
      expect(v.worldFromRegionLocal(const Offset(50, 270)), isNull);
      expect(v.worldFromRegionLocal(const Offset(1000, 270)), isNull);
    });
  });

  group('toCompositorLayout', () {
    test('translates the field rect into page CSS pixels at the origin', () {
      final layout = vp(1080, 720, 2).toCompositorLayout(const Offset(200, 100));
      expect(layout.cssLeft, 200);
      expect(layout.cssTop, 100);
      expect(layout.cssWidth, 1080);
      expect(layout.cssHeight, 720);
      expect(layout.backingWidth, 2160);
      expect(layout.backingHeight, 1440);
    });

    test('adds the letterbox offset to the region origin', () {
      // 1080×540 → offsetX 135; region origin (10, 20).
      final layout = vp(1080, 540, 1).toCompositorLayout(const Offset(10, 20));
      expect(layout.cssLeft, closeTo(145, 1e-9)); // 10 + 135
      expect(layout.cssTop, 20);
    });
  });

  group('value equality', () {
    test('equal inputs are ==; differing dpr is not', () {
      expect(vp(800, 600, 2), vp(800, 600, 2));
      expect(vp(800, 600, 2).hashCode, vp(800, 600, 2).hashCode);
      expect(vp(800, 600, 2), isNot(vp(800, 600, 1)));
    });
  });
}
