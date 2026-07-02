import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:primordis/shared/constants/primordis_config.dart';
import 'package:primordis/sim/cpu/counting_sort_binning.dart';
import 'package:primordis/sim/cpu/cpu_sim_step.dart';
import 'package:primordis/sim/cpu/particle_soa.dart';
import 'package:primordis/sim/models/sim_params.dart';
import 'package:primordis/sim/models/type_matrix.dart';

GridGeometry _grid() => GridGeometry(
      worldWidth: PrimordisConfig.worldWidth.toDouble(),
      worldHeight: PrimordisConfig.worldHeight.toDouble(),
      binSize: PrimordisConfig.binSize.toDouble(),
    );

/// Builds a two-type params block with explicit, hand-chosen matrices so the
/// force law is exercised in isolation. `friction = 1` so drift does not scale
/// away the signal; K's = 1 so the raw force law is asserted.
SimParams _params({
  required double forceAB,
  required double forceBA,
  required double minDist,
  required double radius,
  double attractionK = 1,
  double repulsionK = 1,
  double friction = 1,
}) {
  // 2x2 matrices, index [i][j] = i*2 + j.
  final forces = TypeMatrix.fromRows(<List<double>>[
    <double>[0, forceAB],
    <double>[forceBA, 0],
  ]);
  final minDistances = TypeMatrix.fromRows(<List<double>>[
    <double>[minDist, minDist],
    <double>[minDist, minDist],
  ]);
  final radii = TypeMatrix.fromRows(<List<double>>[
    <double>[radius, radius],
    <double>[radius, radius],
  ]);
  return SimParams(
    forces: forces,
    minDistances: minDistances,
    radii: radii,
    attractionK: attractionK,
    repulsionK: repulsionK,
    friction: friction,
    particleCount: 2,
    typeCount: 2,
  );
}

/// A 2-particle SoA at the given positions with types [t0], [t1] and zero
/// initial velocity.
ParticleSoa _twoParticles(
  double x0,
  double y0,
  double x1,
  double y1, {
  int t0 = 0,
  int t1 = 1,
}) {
  final grid = _grid();
  final soa = ParticleSoa(
    particleCount: 2,
    typeCount: 2,
    binCount: grid.binCount,
  );
  soa.positions.setAll(0, <double>[x0, y0, x1, y1]);
  soa.types[0] = t0;
  soa.types[1] = t1;
  return soa;
}

void main() {
  const w = PrimordisConfig.worldWidth; // 1080
  const h = PrimordisConfig.worldHeight; // 720

  group('cpuSimStep force law', () {
    test('repulsion: particles closer than min_dist are pushed apart', () {
      // Two particles 5px apart on x; min_dist = 10 so they repel.
      final grid = _grid();
      final soa = _twoParticles(500, 400, 505, 400);
      // Positive force -> repulsion uses abs(force); direction is away.
      final params = _params(
        forceAB: 0.5,
        forceBA: 0.5,
        minDist: 10,
        radius: 96,
      );

      cpuSimStep(soa, params, grid, 1);

      final x0 = soa.positions[0];
      final x1 = soa.positions[2];
      // p0 (left) should move further left; p1 (right) further right.
      expect(x0, lessThan(500), reason: 'left particle repelled left');
      expect(x1, greaterThan(505), reason: 'right particle repelled right');
      // Separation increased.
      expect(x1 - x0, greaterThan(5));
    });

    test('attraction: particles between min_dist and radius are pulled in', () {
      final grid = _grid();
      // 40px apart; min_dist small (4), radius large (96) -> attraction band.
      final soa = _twoParticles(500, 400, 540, 400);
      final params = _params(
        forceAB: 0.6,
        forceBA: 0.6,
        minDist: 4,
        radius: 96,
      );

      cpuSimStep(soa, params, grid, 1);

      final x0 = soa.positions[0];
      final x1 = soa.positions[2];
      expect(x0, greaterThan(500), reason: 'left particle pulled right');
      expect(x1, lessThan(540), reason: 'right particle pulled left');
      expect(x1 - x0, lessThan(40), reason: 'separation decreased');
    });

    test('negative signed force flips attraction into repulsion', () {
      final grid = _grid();
      final soa = _twoParticles(500, 400, 540, 400);
      // Same band as attraction test but force is negative -> pushes apart.
      final params = _params(
        forceAB: -0.6,
        forceBA: -0.6,
        minDist: 4,
        radius: 96,
      );

      cpuSimStep(soa, params, grid, 1);

      expect(soa.positions[2] - soa.positions[0], greaterThan(40));
    });

    test('beyond radius there is no interaction', () {
      final grid = _grid();
      // 90px apart but radius only 50 -> outside band, no force. Also outside
      // min_dist. Velocity stays zero, position unchanged (friction=1,dt=1).
      final soa = _twoParticles(400, 400, 490, 400);
      final params = _params(
        forceAB: 0.6,
        forceBA: 0.6,
        minDist: 4,
        radius: 50,
      );

      cpuSimStep(soa, params, grid, 1);

      expect(soa.positions[0], 400);
      expect(soa.positions[2], 490);
      expect(soa.velocities[0], 0);
      expect(soa.velocities[2], 0);
    });
  });

  group('asymmetric matrix orientation', () {
    test('[i][j] uses the SOURCE particle type acting on the neighbour', () {
      final grid = _grid();
      // Type-0 attracts (positive) toward type-1; type-1 repels (negative)
      // type-0. In the attraction band both particles feel their OWN row's
      // force toward the other. p0 is type0 (row0 -> forceAB), p1 is type1
      // (row1 -> forceBA). Set forceAB>0 (0 attracts 1) and forceBA<0
      // (1 repels 0). Net: p0 pulled right, p1 pushed right -> both drift right,
      // and because forces differ the motion is asymmetric.
      final soa = _twoParticles(500, 400, 540, 400);
      final params = _params(
        forceAB: 0.6, // type0 -> type1 : attract
        forceBA: -0.6, // type1 -> type0 : repel
        minDist: 4,
        radius: 96,
      );

      cpuSimStep(soa, params, grid, 1);

      // p0 feels attraction toward p1 (to the right): moves right.
      expect(soa.positions[0], greaterThan(500));
      // p1 feels repulsion from p0 (away from p0, i.e. to the right): moves
      // right too.
      expect(soa.positions[2], greaterThan(540));
      // Orientation proof: the two particles' velocity magnitudes are equal
      // here (|forceAB| == |forceBA|) but had we swapped only one sign the
      // directions would differ — covered by the negative-force test above.
      expect(soa.velocities[0], isNot(0));
      expect(soa.velocities[2], isNot(0));
    });
  });

  group('toroidal minimum-image across the world seam', () {
    test('repulsion acts across the x seam (wrap), not through the middle', () {
      final grid = _grid();
      // p0 near right edge, p1 near left edge: 6px apart across the seam,
      // but 1074px apart the "long" way. Min-image must see 6px -> repel.
      final soa = _twoParticles(w - 3.0, 400, 3, 400);
      final params = _params(
        forceAB: 0.5,
        forceBA: 0.5,
        minDist: 10,
        radius: 96,
      );

      cpuSimStep(soa, params, grid, 1);

      // p0 is at x=1077; repelled AWAY from p1-across-the-seam means to the
      // right, which wraps. p1 at x=3 repelled left, which wraps to high x.
      // Assert the min-image separation grew: recompute wrapped dx.
      double wrappedDx(double a, double b) {
        var d = b - a;
        if (d > w / 2) d -= w;
        if (d < -w / 2) d += w;
        return d;
      }

      final sepBefore = wrappedDx(w - 3.0, 3).abs(); // 6
      final sepAfter = wrappedDx(soa.positions[0], soa.positions[2]).abs();
      expect(sepBefore, closeTo(6, 1e-3));
      expect(sepAfter, greaterThan(sepBefore),
          reason: 'repelled across the seam -> larger min-image separation');
    });

    test('attraction acts across the y seam (wrap)', () {
      final grid = _grid();
      // 30px apart across the top/bottom seam.
      final soa = _twoParticles(500, h - 15.0, 500, 15);
      final params = _params(
        forceAB: 0.6,
        forceBA: 0.6,
        minDist: 4,
        radius: 96,
      );

      cpuSimStep(soa, params, grid, 1);

      double wrappedDy(double a, double b) {
        var d = b - a;
        if (d > h / 2) d -= h;
        if (d < -h / 2) d += h;
        return d;
      }

      final sepAfter = wrappedDy(soa.positions[1], soa.positions[3]).abs();
      expect(sepAfter, lessThan(30),
          reason: 'attracted across the seam -> smaller min-image separation');
    });

    test('positions stay wrapped into [0, world) after stepping', () {
      final grid = _grid();
      final soa = _twoParticles(w - 1.0, h - 1.0, 1, 1);
      final params = _params(
        forceAB: 0.8,
        forceBA: 0.8,
        minDist: 200, // force a big repulsion so a particle crosses the edge
        radius: 96,
        repulsionK: 200,
      );

      cpuSimStep(soa, params, grid, 1);

      for (var i = 0; i < 2; i++) {
        expect(soa.positions[i * 2], inInclusiveRange(0, w.toDouble()));
        expect(soa.positions[i * 2 + 1], inInclusiveRange(0, h.toDouble()));
        expect(soa.positions[i * 2], lessThan(w));
        expect(soa.positions[i * 2 + 1], lessThan(h));
      }
    });
  });

  group('integration', () {
    test('friction scales velocity each tick (v *= friction)', () {
      final grid = _grid();
      // Two particles far apart (no force) but with initial velocity; friction
      // 0.5 halves velocity each tick.
      final soa = ParticleSoa(
        particleCount: 2,
        typeCount: 2,
        binCount: grid.binCount,
      );
      soa.positions.setAll(0, <double>[100, 100, 900, 600]);
      soa.velocities.setAll(0, <double>[10, 0, 0, 0]);
      soa.types[0] = 0;
      soa.types[1] = 1;
      final params = _params(
        forceAB: 0,
        forceBA: 0,
        minDist: 4,
        radius: 10, // tiny radius so the two never interact
        friction: 0.5,
      );

      cpuSimStep(soa, params, grid, 1);

      // No force: v0 = (10 + 0) * 0.5 = 5.
      expect(soa.velocities[0], closeTo(5, 1e-4));
      // p0 moved by v*dt = 5.
      expect(soa.positions[0], closeTo(105, 1e-4));
    });

    test('is deterministic: identical setup -> identical result', () {
      final grid = _grid();
      final params = _params(
        forceAB: 0.4,
        forceBA: -0.3,
        minDist: 8,
        radius: 80,
        friction: 0.6,
      );
      Float32List runOnce() {
        final soa = _twoParticles(500, 400, 520, 405);
        for (var f = 0; f < 25; f++) {
          cpuSimStep(soa, params, grid, 1 / 60);
        }
        return Float32List.fromList(soa.positions);
      }

      expect(runOnce(), orderedEquals(runOnce()));
    });
  });

  group('fillRenderBuffer', () {
    test('copies positions into a 2*N packed buffer', () {
      final soa = _twoParticles(11, 22, 33, 44);
      fillRenderBuffer(soa);
      expect(soa.renderXY.length, 2 * soa.particleCount);
      expect(soa.renderXY, orderedEquals(<double>[11, 22, 33, 44]));
    });
  });
}
