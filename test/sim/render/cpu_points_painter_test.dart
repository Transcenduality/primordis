import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/rendering.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:primordis/shared/constants/primordis_config.dart';
import 'package:primordis/sim/models/particle_type.dart';
import 'package:primordis/sim/render/cpu_points_painter.dart';

/// Records `drawRawPoints` calls so the painter's single-draw contract can be
/// asserted. Every other `Canvas` method is a no-op via [noSuchMethod].
class _RecordingCanvas implements Canvas {
  final List<({ui.PointMode mode, Float32List points, ui.Color color})> draws =
      <({ui.PointMode mode, Float32List points, ui.Color color})>[];

  @override
  void drawRawPoints(ui.PointMode pointMode, Float32List points, Paint paint) {
    draws.add((mode: pointMode, points: points, color: paint.color));
  }

  @override
  void noSuchMethod(Invocation invocation) {}
}

void main() {
  group('CpuPointsPainter', () {
    test('empty frame draws nothing', () {
      final canvas = _RecordingCanvas();
      CpuPointsPainter(CpuFrame.empty).paint(canvas, const Size(1080, 720));
      expect(canvas.draws, isEmpty);
    });

    test('single type -> exactly one drawRawPoints of length 2*count', () {
      const count = 5;
      final points = Float32List(count * 2);
      for (var k = 0; k < points.length; k++) {
        points[k] = k.toDouble();
      }
      final frame = CpuFrame(
        pointsByType: <Float32List>[points],
        colors: const <ui.Color>[ui.Color(0xFFFF0000)],
        pointSize: PrimordisConfig.pointSize,
      );
      final canvas = _RecordingCanvas();

      CpuPointsPainter(frame).paint(canvas, const Size(1080, 720));

      expect(canvas.draws.length, 1, reason: 'exactly one draw call');
      expect(canvas.draws.single.points.length, 2 * count);
      expect(canvas.draws.single.mode, ui.PointMode.points);
    });

    test('per-type batching issues one draw per non-empty type', () {
      final frame = CpuFrame(
        pointsByType: <Float32List>[
          Float32List.fromList(<double>[1, 2]),
          Float32List(0), // empty type -> skipped
          Float32List.fromList(<double>[3, 4, 5, 6]),
        ],
        colors: const <ui.Color>[
          ui.Color(0xFF0000FF),
          ui.Color(0xFF00FF00),
          ui.Color(0xFFFF0000),
        ],
        pointSize: PrimordisConfig.pointSize,
      );
      final canvas = _RecordingCanvas();

      CpuPointsPainter(frame).paint(canvas, const Size(1080, 720));

      // Two non-empty types -> two draws; the empty one is skipped.
      expect(canvas.draws.length, 2);
      expect(canvas.draws.first.color, const ui.Color(0xFF0000FF));
      expect(canvas.draws.last.color, const ui.Color(0xFFFF0000));
    });

    test('draw count never exceeds the type count (<=32)', () {
      final frame = CpuFrame(
        pointsByType: <Float32List>[
          for (var t = 0; t < PrimordisConfig.typeCount; t++)
            Float32List.fromList(<double>[t.toDouble(), t.toDouble()]),
        ],
        colors: <ui.Color>[
          for (var t = 0; t < PrimordisConfig.typeCount; t++)
            const ui.Color(0xFFFFFFFF),
        ],
        pointSize: PrimordisConfig.pointSize,
      );
      final canvas = _RecordingCanvas();

      CpuPointsPainter(frame).paint(canvas, const Size(1080, 720));

      expect(canvas.draws.length, lessThanOrEqualTo(32));
      expect(canvas.draws.length, PrimordisConfig.typeCount);
    });

    test('shouldRepaint only when the frame identity changes', () {
      final frame = CpuFrame(
        pointsByType: <Float32List>[Float32List.fromList(<double>[0, 0])],
        colors: const <ui.Color>[ui.Color(0xFFFFFFFF)],
        pointSize: PrimordisConfig.pointSize,
      );
      final painter = CpuPointsPainter(frame);
      expect(painter.shouldRepaint(CpuPointsPainter(frame)), isFalse);
      expect(painter.shouldRepaint(CpuPointsPainter(CpuFrame.empty)), isTrue);
    });

    test('particleTypeColors maps per-type RGB to opaque colours', () {
      final colors = particleTypeColors(const <ParticleType>[
        ParticleType(index: 0, r: 1, g: 0, b: 0),
        ParticleType(index: 1, r: 0, g: 0.5, b: 1),
      ]);
      expect(colors.length, 2);
      expect(colors[0].a, 1.0);
      expect(colors[0].r, 1.0);
      expect(colors[1].b, 1.0);
    });
  });
}
