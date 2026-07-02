import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter/rendering.dart';
import 'package:primordis/shared/constants/primordis_config.dart';
import 'package:primordis/sim/models/particle_type.dart';

/// An immutable, ready-to-blit snapshot of one simulated frame.
///
/// Produced by the CPU backend's `present()` and consumed by
/// [CpuPointsPainter]. It carries the packed point coordinates grouped by type
/// plus the per-type colours, so the painter can blit them with **one
/// `drawRawPoints` call per type** (<= 32 draws, the cap allowed by
/// [PRIMORDIS-TASK-008]) rather than iterating particles with `drawCircle`.
///
/// `drawRawPoints` takes a single [Paint] colour and cannot carry per-vertex
/// colour, and `drawVertices` only draws triangles (not points) — so genuine
/// per-particle colour in a *single* Flutter draw call is not achievable for
/// points. Grouping by type and issuing one `drawRawPoints` per type is the
/// faithful compromise the task calls out, and keeps the reference's per-type
/// colouring while capping draws at the type count.
@immutable
class CpuFrame {
  /// Wraps per-type point buffers and their colours.
  ///
  /// [pointsByType] has one entry per particle type; entry `t` is a packed
  /// `x, y` [Float32List] of that type's live particles (length `2 * count_t`).
  /// [colors] is parallel: `colors[t]` is the [ui.Color] for type `t`.
  const CpuFrame({
    required this.pointsByType,
    required this.colors,
    required this.pointSize,
  });

  /// One packed `x, y` buffer per particle type (may be empty for a type with
  /// no live particles this frame).
  final List<Float32List> pointsByType;

  /// Per-type point colour, parallel to [pointsByType].
  final List<ui.Color> colors;

  /// Point diameter in logical pixels (matches the reference `gl_PointSize`).
  final double pointSize;

  /// An empty frame (nothing to draw) — the initial painter state.
  static const CpuFrame empty = CpuFrame(
    pointsByType: <Float32List>[],
    colors: <ui.Color>[],
    pointSize: PrimordisConfig.pointSize,
  );
}

/// Builds the per-type packed colour buffers a [CpuFrame] needs from the
/// per-type colours, so the backend can cache the immutable colour list and the
/// per-type index groupings once at seed time.
List<ui.Color> particleTypeColors(List<ParticleType> types) => <ui.Color>[
      for (final t in types)
        ui.Color.from(alpha: 1, red: t.r, green: t.g, blue: t.b),
    ];

/// A [CustomPainter] that blits a prebuilt [CpuFrame] with one `drawRawPoints`
/// call per particle type.
///
/// It performs **no physics** — the frame loop ([PRIMORDIS-TASK-005]) runs the
/// step and hands the painter a finished [CpuFrame]; `paint` only draws. This
/// keeps the render path allocation-light (the [Paint] is the only per-paint
/// object) and satisfies the "exactly one draw per type, <=32 draws" contract.
class CpuPointsPainter extends CustomPainter {
  /// Paints [frame]; repaints when [frame] identity changes.
  CpuPointsPainter(this.frame) : super(repaint: null);

  /// The finished frame to blit.
  final CpuFrame frame;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..strokeWidth = frame.pointSize
      ..strokeCap = StrokeCap.round;
    for (var t = 0; t < frame.pointsByType.length; t++) {
      final points = frame.pointsByType[t];
      if (points.isEmpty) continue;
      paint.color = frame.colors[t];
      canvas.drawRawPoints(ui.PointMode.points, points, paint);
    }
  }

  @override
  bool shouldRepaint(covariant CpuPointsPainter oldDelegate) =>
      !identical(oldDelegate.frame, frame);
}
