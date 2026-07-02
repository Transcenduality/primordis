import 'package:flutter_test/flutter_test.dart';
import 'package:primordis/shared/constants/primordis_config.dart';
import 'package:primordis/sim/cpu/counting_sort_binning.dart';
import 'package:primordis/sim/cpu/particle_soa.dart';

/// Builds a [ParticleSoa] whose positions come from `(x, y)` pairs on the
/// reference 1080x720 / bin-96 grid, so bin membership can be asserted directly.
ParticleSoa _soaFromPositions(List<double> xy, GridGeometry grid) {
  final count = xy.length ~/ 2;
  final soa = ParticleSoa(
    particleCount: count,
    typeCount: PrimordisConfig.typeCount,
    binCount: grid.binCount,
  );
  soa.positions.setAll(0, xy);
  return soa;
}

/// The reference grid: 1080x720, bin 96 -> 11x7 = 77 bins.
GridGeometry _referenceGrid() => GridGeometry(
      worldWidth: PrimordisConfig.worldWidth.toDouble(),
      worldHeight: PrimordisConfig.worldHeight.toDouble(),
      binSize: PrimordisConfig.binSize.toDouble(),
    );

/// Reads back the particle indices the sort placed into bin [bin].
List<int> _binMembers(ParticleSoa soa, int bin) {
  final start = soa.binStarts[bin];
  final end = start + soa.binCounts[bin];
  return <int>[for (var s = start; s < end; s++) soa.sortedIndices[s]];
}

void main() {
  group('GridGeometry', () {
    test('reproduces the reference 11x7 = 77-bin grid', () {
      final grid = _referenceGrid();
      expect(grid.gridWidth, 11);
      expect(grid.gridHeight, 7);
      expect(grid.binCount, 77);
    });

    test('bin index is row-major by*gridWidth + bx (matches reference)', () {
      final grid = _referenceGrid();
      // x in [96,192) -> column 1; y in [192,288) -> row 2.
      expect(grid.columnOf(100), 1);
      expect(grid.rowOf(200), 2);
      expect(grid.binIndexFor(100, 200), 2 * 11 + 1);
    });

    test('clamps the exact far edge (x==worldWidth) into the last bin', () {
      final grid = _referenceGrid();
      expect(grid.columnOf(1080), 10);
      expect(grid.rowOf(720), 6);
    });

    test('wraps neighbour bin indices toroidally on both axes', () {
      final grid = _referenceGrid();
      // Left of column 0 wraps to 10; above row 0 wraps to 6.
      expect(grid.wrapColumn(-1), 10);
      expect(grid.wrapColumn(11), 0);
      expect(grid.wrapRow(-1), 6);
      expect(grid.wrapRow(7), 0);
    });
  });

  group('countingSortBinning', () {
    test('places every particle (no per-bin cap / no drops)', () {
      final grid = _referenceGrid();
      // 50 particles all in bin 0 — far exceeds nothing, but proves the 512 cap
      // is NOT ported: all 50 must be present in the one bin.
      final xy = <double>[for (var i = 0; i < 50; i++) ...<double>[10, 10]];
      final soa = _soaFromPositions(xy, grid);

      countingSortBinning(soa, grid);

      expect(soa.binCounts[0], 50);
      // Total placed equals particleCount (nothing dropped).
      final total = soa.binCounts.fold<int>(0, (a, b) => a + b);
      expect(total, 50);
      expect(_binMembers(soa, 0), List<int>.generate(50, (i) => i));
    });

    test('assigns particles to the correct bins across the grid', () {
      final grid = _referenceGrid();
      // p0 -> bin 0 (col 0,row 0); p1 -> col 1,row 0 = bin 1;
      // p2 -> col 0,row 1 = bin 11; p3 -> col 10,row 6 = bin 76.
      final soa = _soaFromPositions(<double>[
        10, 10, // 0 -> bin 0
        100, 10, // 1 -> bin 1
        10, 100, // 2 -> bin 11
        1070, 710, // 3 -> bin 76
      ], grid);

      countingSortBinning(soa, grid);

      expect(_binMembers(soa, 0), <int>[0]);
      expect(_binMembers(soa, 1), <int>[1]);
      expect(_binMembers(soa, 11), <int>[2]);
      expect(_binMembers(soa, 76), <int>[3]);
    });

    test('output is stable and within-bin ordering is ascending by index', () {
      final grid = _referenceGrid();
      // Three particles in bin 0 seeded out of index order in space, plus one
      // elsewhere; scatter must be stable (ascending original index per bin).
      final soa = _soaFromPositions(<double>[
        20, 20, // 0 -> bin 0
        500, 500, // 1 -> some middle bin
        30, 30, // 2 -> bin 0
        40, 40, // 3 -> bin 0
      ], grid);

      countingSortBinning(soa, grid);

      expect(_binMembers(soa, 0), <int>[0, 2, 3]);
    });

    test('is deterministic: identical input -> identical sortedIndices', () {
      final grid = _referenceGrid();
      final xy = <double>[
        for (var i = 0; i < 200; i++)
          ...<double>[(i * 37 % 1080).toDouble(), (i * 53 % 720).toDouble()],
      ];
      final a = _soaFromPositions(xy, grid)..let((s) => countingSortBinning(s, grid));
      final b = _soaFromPositions(xy, grid)..let((s) => countingSortBinning(s, grid));

      expect(a.sortedIndices, orderedEquals(b.sortedIndices));
      expect(a.binStarts, orderedEquals(b.binStarts));
      expect(a.binCounts, orderedEquals(b.binCounts));
    });

    test('binStarts is an exclusive prefix-sum of binCounts', () {
      final grid = _referenceGrid();
      final xy = <double>[
        for (var i = 0; i < 77; i++)
          // one particle centred in each bin, in bin order
          ...<double>[
            (i % 11) * 96 + 48,
            (i ~/ 11) * 96 + 48,
          ],
      ];
      final soa = _soaFromPositions(xy, grid);

      countingSortBinning(soa, grid);

      var running = 0;
      for (var b = 0; b < grid.binCount; b++) {
        expect(soa.binStarts[b], running, reason: 'prefix-sum mismatch at $b');
        running += soa.binCounts[b];
      }
      expect(running, 77);
    });
  });
}

/// Small cascade helper so a freshly-built SoA can be sorted inline.
extension _Let<T> on T {
  R let<R>(R Function(T) fn) => fn(this);
}
