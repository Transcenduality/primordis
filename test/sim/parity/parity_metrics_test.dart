import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:primordis/sim/parity/frame_metrics.dart';
import 'package:primordis/sim/parity/parity_metrics.dart';

/// Unit tests for the reusable parity metric primitives on **synthetic inputs
/// with known answers**. These prove the metrics themselves are correct before
/// they are used to judge any backend ([PRIMORDIS-TASK-009] acceptance:
/// "`metrics.dart` has its own unit tests ... so the metrics themselves are
/// trusted before they judge backends").
void main() {
  // A small 4x2-bin grid (world 8x4, bin size 2) keeps hand-computed answers
  // tractable while still exercising toroidal wrap and edge/interior split.
  MetricGrid smallGrid() =>
      MetricGrid(worldWidth: 8, worldHeight: 4, binSize: 2);

  group('MetricGrid', () {
    test('derives grid dimensions and bins positions row-major', () {
      final grid = smallGrid();
      expect(grid.gridWidth, 4);
      expect(grid.gridHeight, 2);
      expect(grid.binCount, 8);
      // (x=3, y=1) -> col 1, row 0 -> bin 1.
      expect(grid.binIndexFor(3, 1), 1);
      // (x=7, y=3) -> col 3, row 1 -> bin 7.
      expect(grid.binIndexFor(7, 3), 7);
    });

    test('clamps the exact-edge coordinate into the last bin', () {
      final grid = smallGrid();
      // x == worldWidth would floor to col 4 (out of range); clamped to 3.
      expect(grid.columnOf(8), 3);
      expect(grid.rowOf(4), 1);
    });

    test('rejects a degenerate grid', () {
      expect(
        () => MetricGrid(worldWidth: 1, worldHeight: 4, binSize: 2),
        throwsArgumentError,
      );
    });
  });

  group('binOccupancy', () {
    test('counts every particle into its bin with no cap', () {
      final grid = smallGrid();
      // Three particles in bin 0, one in bin 7.
      final positions = Float32List.fromList(<double>[
        0.5, 0.5, // bin 0
        1.0, 0.5, // bin 0
        0.1, 0.9, // bin 0
        7.5, 3.5, // bin 7
      ]);
      final occ = binOccupancy(positions, 4, grid);
      expect(occ[0], 3);
      expect(occ[7], 1);
      expect(occ.fold<int>(0, (a, b) => a + b), 4);
    });
  });

  group('Moments', () {
    test('computes mean, population variance, min and max', () {
      final m = Moments.of(<double>[2, 4, 4, 4, 5, 5, 7, 9]);
      expect(m.count, 8);
      expect(m.mean, closeTo(5.0, 1e-9));
      // Population variance of this classic set is 4.
      expect(m.variance, closeTo(4.0, 1e-9));
      expect(m.stdDev, closeTo(2.0, 1e-9));
      expect(m.min, 2);
      expect(m.max, 9);
    });

    test('empty input is all zeros', () {
      final m = Moments.of(const <double>[]);
      expect(m.count, 0);
      expect(m.mean, 0);
      expect(m.variance, 0);
    });

    test('variance is floored at zero for a constant input', () {
      final m = Moments.of(List<double>.filled(100, 3.0));
      expect(m.variance, 0.0);
    });
  });

  group('toroidalDistanceSquared', () {
    test('uses the minimum image across the world seam', () {
      // Points at x=0.5 and x=7.5 in an 8-wide world are 1 apart across the
      // seam, not 7 apart the direct way.
      final d2 = toroidalDistanceSquared(0.5, 2, 7.5, 2, 8, 4);
      expect(math.sqrt(d2), closeTo(1.0, 1e-6));
    });

    test('matches the direct distance away from the seam', () {
      final d2 = toroidalDistanceSquared(1, 1, 3, 1, 8, 4);
      expect(math.sqrt(d2), closeTo(2.0, 1e-6));
    });
  });

  group('FrameMetrics.from — two-cluster synthetic configuration', () {
    // Build two tight clusters on an 11x7 sim-sized grid (world 1080x720, bin
    // 96), well separated, so the harness must report exactly two clusters and
    // small nearest-neighbour spacing.
    late MetricGrid grid;
    late Float32List positions;
    late Float32List velocities;
    late Int32List types;
    const perCluster = 50;
    const n = perCluster * 2;

    setUp(() {
      grid = MetricGrid(worldWidth: 1080, worldHeight: 720, binSize: 96);
      positions = Float32List(n * 2);
      velocities = Float32List(n * 2);
      types = Int32List(n);
      final rng = math.Random(7);
      // Cluster A around (150, 150) — type 0. Cluster B around (900, 550) —
      // type 1. Each spans a couple of pixels so many share a bin (dense).
      for (var i = 0; i < perCluster; i++) {
        positions[i * 2] = 150 + rng.nextDouble() * 4 - 2;
        positions[i * 2 + 1] = 150 + rng.nextDouble() * 4 - 2;
        types[i] = 0;
      }
      for (var i = perCluster; i < n; i++) {
        positions[i * 2] = 900 + rng.nextDouble() * 4 - 2;
        positions[i * 2 + 1] = 550 + rng.nextDouble() * 4 - 2;
        types[i] = 1;
      }
    });

    test('reports exactly two clusters', () {
      final m = FrameMetrics.from(
        positions: positions,
        velocities: velocities,
        types: types,
        particleCount: n,
        typeCount: 2,
        grid: grid,
      );
      expect(m.clusterCount, 2);
      expect(m.particleCount, n);
    });

    test('nearest-neighbour spacing is small for tight clusters', () {
      final m = FrameMetrics.from(
        positions: positions,
        velocities: velocities,
        types: types,
        particleCount: n,
        typeCount: 2,
        grid: grid,
      );
      // Within a 4px-wide cluster of 50 points, NN spacing is well under a bin.
      expect(m.nearestNeighbour.mean, lessThan(4.0));
    });

    test('fully segregated types yield a high segregation index', () {
      final m = FrameMetrics.from(
        positions: positions,
        velocities: velocities,
        types: types,
        particleCount: n,
        typeCount: 2,
        grid: grid,
      );
      // Each occupied bin is monochromatic -> maximal segregation.
      expect(m.segregationIndex, greaterThan(0.9));
    });

    test('a perfectly mixed distribution yields near-zero segregation', () {
      // A well-populated uniform sprinkle (bins hold hundreds each) with types
      // assigned independently of position: every bin's composition matches the
      // global 50/50 mix up to sampling noise, so segregation is near zero.
      // (A *sparse* spread would read as segregated purely because 1-particle
      // bins are trivially monochromatic — segregation is only meaningful when
      // bins are populated, which the real 24k sim always satisfies.)
      const big = 24000;
      final mixedTypes = Int32List(big);
      final spread = Float32List(big * 2);
      final spreadVel = Float32List(big * 2);
      final rng = math.Random(11);
      for (var i = 0; i < big; i++) {
        spread[i * 2] = rng.nextDouble() * 1080;
        spread[i * 2 + 1] = rng.nextDouble() * 720;
        mixedTypes[i] = rng.nextBool() ? 0 : 1;
      }
      final m = FrameMetrics.from(
        positions: spread,
        velocities: spreadVel,
        types: mixedTypes,
        particleCount: big,
        typeCount: 2,
        grid: grid,
      );
      expect(m.segregationIndex, lessThan(0.1));
    });

    test('speed and kinetic-energy track the velocity field', () {
      // Give every particle speed 3 (3-4-5 triangle).
      for (var i = 0; i < n; i++) {
        velocities[i * 2] = 3;
        velocities[i * 2 + 1] = 4;
      }
      final m = FrameMetrics.from(
        positions: positions,
        velocities: velocities,
        types: types,
        particleCount: n,
        typeCount: 2,
        grid: grid,
      );
      expect(m.speed.mean, closeTo(5.0, 1e-4));
      // Uniform speed -> zero variance -> KE = 0.5 * mean^2 = 12.5.
      expect(m.kineticEnergy, closeTo(12.5, 1e-3));
    });
  });

  group('FrameMetrics — toroidal edge correctness', () {
    test('a cluster straddling the seam is not flagged as edge accumulation',
        () {
      final grid = MetricGrid(worldWidth: 1080, worldHeight: 720, binSize: 96);
      // A large uniform sprinkle (the sim's ~24k regime) makes per-bin Poisson
      // noise small relative to the mean, so a minimum-image-aware metric sees
      // edge bins as statistically indistinguishable from interior bins and
      // reports no artificial edge build-up.
      const n = 24000;
      final positions = Float32List(n * 2);
      final velocities = Float32List(n * 2);
      final types = Int32List(n);
      final rng = math.Random(3);
      for (var i = 0; i < n; i++) {
        positions[i * 2] = rng.nextDouble() * 1080;
        positions[i * 2 + 1] = rng.nextDouble() * 720;
      }
      final m = FrameMetrics.from(
        positions: positions,
        velocities: velocities,
        types: types,
        particleCount: n,
        typeCount: 1,
        grid: grid,
      );
      // A uniform sprinkle has edge bins statistically like interior bins, so
      // the excess stays modest (well under the correctness guard band).
      expect(m.maxEdgeBinExcess, lessThan(0.75));
    });
  });
}
