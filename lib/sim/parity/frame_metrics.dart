import 'dart:math' as math;
import 'dart:typed_data';

import 'package:primordis/sim/parity/parity_metrics.dart';

/// The full set of position-invariant statistics for a single simulation frame.
///
/// This is the fingerprint the parity harness compares across backends and
/// against the committed reference. Every field is an **aggregate** — a count, a
/// moment, a distribution summary — so two runs that produce the same *kind* of
/// behaviour (clusters of the same character forming and drifting) compare equal
/// within tolerance even though their per-particle positions differ. Nothing
/// here is per-particle or pixel-exact, by design ([PRIMORDIS-ADR-001]).
///
/// Build one with [FrameMetrics.from]; compare two with [FrameMetrics.compare].
class FrameMetrics {
  const FrameMetrics({
    required this.particleCount,
    required this.clusterCount,
    required this.occupiedBinFraction,
    required this.occupancy,
    required this.nearestNeighbour,
    required this.segregationIndex,
    required this.speed,
    required this.kineticEnergy,
    required this.maxEdgeBinExcess,
  });

  /// Total live particles this frame. A **conserved** quantity: the toroidal
  /// wrap must never lose or duplicate a particle, so this is asserted with the
  /// tightest (exact) tolerance in [FrameMetrics.compare].
  final int particleCount;

  /// Number of clusters this frame, via connected-component labelling of
  /// *dense* grid bins (a bin whose occupancy exceeds the mean by a factor),
  /// with 8-neighbour toroidal connectivity. Captures that clusters *form* and
  /// roughly how many — not where.
  final int clusterCount;

  /// Fraction of grid bins that hold at least one particle (`[0, 1]`). A coarse
  /// spread measure: as particles condense into clusters this drops from ~1
  /// (uniform fill) toward the clustered steady state.
  final double occupiedBinFraction;

  /// Moments of the per-bin occupancy histogram. High variance relative to the
  /// mean signals clustering (a few very full bins, many empty); a uniform
  /// distribution has near-zero variance. This is the reusable atomics-parity
  /// vector's summary ([PRIMORDIS-TASK-017]).
  final Moments occupancy;

  /// Moments of the per-particle nearest-neighbour distance (minimum-image on
  /// the torus, searched within the 3x3 bin neighbourhood the sim uses).
  /// Captures the repulsion/attraction balance: tighter spacing under strong
  /// attraction, wider under strong repulsion.
  final Moments nearestNeighbour;

  /// Per-type spatial segregation in `[0, 1]`: 0 when every type is perfectly
  /// mixed (each bin's type composition matches the global mix), approaching 1
  /// as types separate into distinct bins. Computed as a chi-square-style
  /// deviation of per-bin type composition from the global composition,
  /// normalised by the maximum possible deviation.
  final double segregationIndex;

  /// Moments of per-particle speed (velocity magnitude). The kinetic proxy that
  /// tracks the Drift/friction slider: higher friction (lower retention) decays
  /// speed faster, so this mean falls over time on every backend.
  final Moments speed;

  /// Mean kinetic energy proxy: `mean(0.5 * |v|^2)` (unit mass). A single
  /// scalar summary of drift, redundant with [speed] but convenient for the
  /// Drift-response assertion.
  final double kineticEnergy;

  /// Toroidal-correctness probe: the largest *excess* occupancy in any grid bin
  /// touching the world edge, measured as a fraction above the interior-bin
  /// mean. A wrap bug that piles particles against a seam shows up here as a
  /// large positive value; a correct torus keeps edge bins statistically
  /// indistinguishable from interior bins, so this stays small.
  final double maxEdgeBinExcess;

  /// Computes every metric for one frame from the SoA buffers.
  ///
  /// [positions]/[velocities] are interleaved `x, y`; [types] is per-particle.
  /// Only the first [particleCount] particles are read. [grid] should mirror the
  /// sim's own grid so bins equal interaction cells.
  ///
  /// [clusterDensityFactor] sets the "dense bin" threshold for cluster
  /// labelling (a bin is part of a cluster when its occupancy is at least this
  /// factor times the mean occupancy); the default 1.5 is justified in the
  /// harness's golden baselines.
  factory FrameMetrics.from({
    required Float32List positions,
    required Float32List velocities,
    required Int32List types,
    required int particleCount,
    required int typeCount,
    required MetricGrid grid,
    double clusterDensityFactor = 1.5,
  }) {
    final occ = binOccupancy(positions, particleCount, grid);
    final occMoments = Moments.of(occ.map((c) => c.toDouble()));

    var occupiedBins = 0;
    for (final c in occ) {
      if (c > 0) occupiedBins++;
    }
    final occupiedFraction =
        grid.binCount == 0 ? 0.0 : occupiedBins / grid.binCount;

    final clusters = _countClusters(
      occ,
      grid,
      occMoments.mean * clusterDensityFactor,
    );

    final nn = _nearestNeighbourMoments(
      positions,
      particleCount,
      grid,
    );

    final segregation = _segregationIndex(
      positions,
      types,
      particleCount,
      typeCount,
      grid,
    );

    final speedMoments = _speedMoments(velocities, particleCount);
    final ke = 0.5 * (speedMoments.mean * speedMoments.mean +
        speedMoments.variance);

    final edgeExcess = _maxEdgeBinExcess(occ, grid);

    return FrameMetrics(
      particleCount: particleCount,
      clusterCount: clusters,
      occupiedBinFraction: occupiedFraction,
      occupancy: occMoments,
      nearestNeighbour: nn,
      segregationIndex: segregation,
      speed: speedMoments,
      kineticEnergy: ke,
      maxEdgeBinExcess: edgeExcess,
    );
  }

  /// Connected-component count of dense bins, 8-neighbour toroidal.
  static int _countClusters(
    Int32List occupancy,
    MetricGrid grid,
    double denseThreshold,
  ) {
    final w = grid.gridWidth;
    final h = grid.gridHeight;
    final dense = List<bool>.generate(
      occupancy.length,
      (i) => occupancy[i] >= denseThreshold && occupancy[i] > 0,
    );
    final seen = List<bool>.filled(occupancy.length, false);
    var clusters = 0;
    final stack = <int>[];
    for (var start = 0; start < occupancy.length; start++) {
      if (!dense[start] || seen[start]) continue;
      clusters++;
      stack
        ..clear()
        ..add(start);
      seen[start] = true;
      while (stack.isNotEmpty) {
        final cell = stack.removeLast();
        final cx = cell % w;
        final cy = cell ~/ w;
        for (var dx = -1; dx <= 1; dx++) {
          for (var dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy == 0) continue;
            final nx = (cx + dx + w) % w;
            final ny = (cy + dy + h) % h;
            final n = ny * w + nx;
            if (dense[n] && !seen[n]) {
              seen[n] = true;
              stack.add(n);
            }
          }
        }
      }
    }
    return clusters;
  }

  /// Per-particle nearest-neighbour distance moments (minimum-image), searching
  /// the 3x3 bin neighbourhood the sim's force law scans.
  static Moments _nearestNeighbourMoments(
    Float32List positions,
    int particleCount,
    MetricGrid grid,
  ) {
    if (particleCount < 2) return Moments.of(const <double>[]);
    // Bucket particle indices by bin for a local neighbour search.
    final byBin = List<List<int>>.generate(grid.binCount, (_) => <int>[]);
    for (var i = 0; i < particleCount; i++) {
      byBin[grid.binIndexFor(positions[i * 2], positions[i * 2 + 1])].add(i);
    }
    final distances = Float64List(particleCount);
    final w = grid.gridWidth;
    final h = grid.gridHeight;
    for (var i = 0; i < particleCount; i++) {
      final px = positions[i * 2];
      final py = positions[i * 2 + 1];
      final cx = grid.columnOf(px);
      final cy = grid.rowOf(py);
      var best = double.infinity;
      for (var dx = -1; dx <= 1; dx++) {
        final nx = (cx + dx + w) % w;
        for (var dy = -1; dy <= 1; dy++) {
          final ny = (cy + dy + h) % h;
          for (final j in byBin[ny * w + nx]) {
            if (j == i) continue;
            final d2 = toroidalDistanceSquared(
              px,
              py,
              positions[j * 2],
              positions[j * 2 + 1],
              grid.worldWidth,
              grid.worldHeight,
            );
            if (d2 < best) best = d2;
          }
        }
      }
      // No neighbour within the 3x3 window: cap at the bin size so an isolated
      // particle contributes a finite, bounded distance rather than infinity.
      distances[i] =
          best.isFinite ? math.sqrt(best) : grid.binSize;
    }
    return Moments.of(distances);
  }

  /// Chi-square-style per-type segregation in `[0, 1]`.
  static double _segregationIndex(
    Float32List positions,
    Int32List types,
    int particleCount,
    int typeCount,
    MetricGrid grid,
  ) {
    if (particleCount == 0 || typeCount < 2) return 0.0;
    final byType = binOccupancyByType(
      positions,
      types,
      particleCount,
      typeCount,
      grid,
    );
    // Global type fractions.
    final typeTotals = List<int>.filled(typeCount, 0);
    for (var i = 0; i < particleCount; i++) {
      typeTotals[types[i]]++;
    }
    // Per-bin total occupancy.
    final binTotals = Int32List(grid.binCount);
    for (var t = 0; t < typeCount; t++) {
      final vec = byType[t];
      for (var b = 0; b < grid.binCount; b++) {
        binTotals[b] += vec[b];
      }
    }
    // Sum over occupied bins of the L1 deviation between each bin's type
    // composition and the global composition, weighted by bin population.
    //
    // Normalisation: when every occupied bin is monochromatic (maximal
    // segregation), a bin of `binTotal` particles all of type `t` contributes
    // `binTotal * (1 - p_t) + sum_{u != t} binTotal * p_u = 2 * binTotal *
    // (1 - p_t)` where `p_t` is the global fraction of the resident type.
    // Summed over all particles this is `2 * particleCount * (1 - sum_t p_t^2)`
    // — the maximum achievable L1 deviation given the *global* type mix. We
    // divide by that so a fully-segregated population scores 1.0 regardless of
    // how balanced the type populations are, and a perfectly mixed one scores 0.
    var deviation = 0.0;
    for (var b = 0; b < grid.binCount; b++) {
      final binTotal = binTotals[b];
      if (binTotal == 0) continue;
      for (var t = 0; t < typeCount; t++) {
        final expected = binTotal * (typeTotals[t] / particleCount);
        final actual = byType[t][b].toDouble();
        deviation += (actual - expected).abs();
      }
    }
    var sumP2 = 0.0;
    for (var t = 0; t < typeCount; t++) {
      final p = typeTotals[t] / particleCount;
      sumP2 += p * p;
    }
    final maxDeviation = 2.0 * particleCount * (1.0 - sumP2);
    return maxDeviation <= 0
        ? 0.0
        : (deviation / maxDeviation).clamp(0.0, 1.0);
  }

  /// Per-particle speed (`|v|`) moments.
  static Moments _speedMoments(Float32List velocities, int particleCount) {
    if (particleCount == 0) return Moments.of(const <double>[]);
    final speeds = Float64List(particleCount);
    for (var i = 0; i < particleCount; i++) {
      final vx = velocities[i * 2];
      final vy = velocities[i * 2 + 1];
      speeds[i] = math.sqrt(vx * vx + vy * vy);
    }
    return Moments.of(speeds);
  }

  /// Largest edge-bin occupancy-*density* excess over the interior-bin mean
  /// density, as a fraction. A correct torus keeps this near 0; an
  /// edge-accumulation bug drives it up.
  ///
  /// Compares **density** (particles per px²), not raw counts: because the
  /// world is not an integer number of bins wide/tall, the right/bottom edge
  /// bins physically cover more area and would otherwise show a spurious excess
  /// from geometry alone ([MetricGrid.binArea]). Normalising by area removes
  /// that artifact so only a genuine seam pile-up registers.
  static double _maxEdgeBinExcess(Int32List occupancy, MetricGrid grid) {
    final w = grid.gridWidth;
    final h = grid.gridHeight;
    if (w < 3 || h < 3) {
      // Too small to distinguish edge from interior; not meaningful.
      return 0.0;
    }
    var interiorDensitySum = 0.0;
    var interiorCount = 0;
    var maxEdgeDensity = 0.0;
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        final isEdge = x == 0 || y == 0 || x == w - 1 || y == h - 1;
        final density = occupancy[y * w + x] / grid.binArea(x, y);
        if (isEdge) {
          if (density > maxEdgeDensity) maxEdgeDensity = density;
        } else {
          interiorDensitySum += density;
          interiorCount++;
        }
      }
    }
    if (interiorCount == 0) return 0.0;
    final interiorMean = interiorDensitySum / interiorCount;
    if (interiorMean <= 0) return maxEdgeDensity > 0 ? double.infinity : 0.0;
    return (maxEdgeDensity - interiorMean) / interiorMean;
  }

  /// A per-field diff of two frames' metrics, for a diagnostic failure message.
  ///
  /// Returns a list of human-readable `field: a vs b (|delta|)` lines, one per
  /// scalar metric, so a parity failure can point at exactly which statistic
  /// drifted and by how much rather than just "not equal".
  static List<String> compare(FrameMetrics a, FrameMetrics b) {
    String line(String name, double x, double y) {
      final delta = (x - y).abs();
      return '$name: ${x.toStringAsFixed(4)} vs ${y.toStringAsFixed(4)} '
          '(|Δ|=${delta.toStringAsFixed(4)})';
    }

    return <String>[
      line('particleCount', a.particleCount.toDouble(),
          b.particleCount.toDouble()),
      line('clusterCount', a.clusterCount.toDouble(),
          b.clusterCount.toDouble()),
      line('occupiedBinFraction', a.occupiedBinFraction,
          b.occupiedBinFraction),
      line('occupancy.mean', a.occupancy.mean, b.occupancy.mean),
      line('occupancy.variance', a.occupancy.variance, b.occupancy.variance),
      line('nearestNeighbour.mean', a.nearestNeighbour.mean,
          b.nearestNeighbour.mean),
      line('nearestNeighbour.variance', a.nearestNeighbour.variance,
          b.nearestNeighbour.variance),
      line('segregationIndex', a.segregationIndex, b.segregationIndex),
      line('speed.mean', a.speed.mean, b.speed.mean),
      line('speed.variance', a.speed.variance, b.speed.variance),
      line('kineticEnergy', a.kineticEnergy, b.kineticEnergy),
      line('maxEdgeBinExcess', a.maxEdgeBinExcess, b.maxEdgeBinExcess),
    ];
  }

  @override
  String toString() => 'FrameMetrics(n=$particleCount, '
      'clusters=$clusterCount, '
      'occupiedFrac=${occupiedBinFraction.toStringAsFixed(3)}, '
      'nnMean=${nearestNeighbour.mean.toStringAsFixed(2)}, '
      'segregation=${segregationIndex.toStringAsFixed(3)}, '
      'speedMean=${speed.mean.toStringAsFixed(2)}, '
      'ke=${kineticEnergy.toStringAsFixed(2)}, '
      'edgeExcess=${maxEdgeBinExcess.toStringAsFixed(3)})';
}
