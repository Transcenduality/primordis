/// Position-invariant, aggregate statistics that characterise a Primordis
/// simulation frame **without** reference to exact particle positions.
///
/// Parity across the Primordis backends (`Primordis.py`, the deterministic CPU
/// tier, the web WebGPU tier, the macOS Dawn/Metal tier) is **statistical, never
/// bit-exact** ([PRIMORDIS-ADR-001], [PRIMORDIS-ADR-006], PRD non-goals). The
/// reference GPU binning is a single-buffered atomic scatter with a known race,
/// so two runs of the reference *itself* diverge per-particle. A faithful port
/// therefore means the *distributions and trajectories of aggregate statistics*
/// match within tolerance bands — clusters of the same character form, drift,
/// and persist — not identical positions or pixels.
///
/// This library is the **reusable substrate** those bands are computed over. It
/// is intentionally pure Dart (`dart:math` + `dart:typed_data` only — no Flutter,
/// no `dart:ui`, no I/O), fully unit-tested on synthetic inputs with known
/// answers, so the metrics are trusted *before* they judge any backend. The
/// cross-backend atomics-parity check ([PRIMORDIS-TASK-017], the Dawn/Tint vs
/// browser/Naga bin-count comparison at 24k/32-types) consumes the same
/// functions here rather than inventing its own — in particular
/// [binOccupancy] and [FrameMetrics.from] are exported as the reusable
/// building blocks.
///
/// All inputs are the SoA buffers the simulation already maintains:
/// - `positions`: interleaved `x, y`, length `2 * particleCount`.
/// - `velocities`: interleaved `x, y`, length `2 * particleCount`.
/// - `types`: per-particle type index, length `particleCount`.
///
/// Metrics that involve neighbourhood/distance always account for the toroidal
/// wrap via **minimum-image**, exactly as the sim does, so a cluster straddling
/// the world seam is not double-counted or torn apart.
library;

import 'dart:math' as math;
import 'dart:typed_data';

/// The uniform spatial grid the metrics bin into.
///
/// Deliberately mirrors the simulation's own grid (11x7 bins of size
/// `MAX_RADIUS = 96` over the 1080x720 toroidal world) so occupancy and cluster
/// metrics are meaningful *relative to the interaction range* — a bin is exactly
/// one interaction cell. Constructed from world + bin size the same way
/// `GridGeometry` in the CPU tier is, but kept independent here so the metrics
/// library has zero dependency on `lib/sim/cpu/**` and can be reused by any
/// backend or the standalone kernel harness.
class MetricGrid {
  /// Builds a grid from toroidal world size and bin edge length.
  MetricGrid({
    required this.worldWidth,
    required this.worldHeight,
    required this.binSize,
  })  : gridWidth = (worldWidth / binSize).floor(),
        gridHeight = (worldHeight / binSize).floor() {
    if (gridWidth < 1 || gridHeight < 1) {
      throw ArgumentError('grid must have at least one bin per axis');
    }
  }

  /// Toroidal world width in pixels.
  final double worldWidth;

  /// Toroidal world height in pixels.
  final double worldHeight;

  /// Bin edge length (== the sim's `MAX_RADIUS`).
  final double binSize;

  /// Grid columns (`worldWidth ~/ binSize`).
  final int gridWidth;

  /// Grid rows (`worldHeight ~/ binSize`).
  final int gridHeight;

  /// Total bins (`gridWidth * gridHeight`).
  int get binCount => gridWidth * gridHeight;

  /// Column index of a world x-coordinate, clamped to `[0, gridWidth)`.
  ///
  /// The clamp only guards the exact-edge `x == worldWidth` float-rounding case,
  /// matching the reference binning shader's `clamp`.
  int columnOf(double x) {
    final bx = (x / binSize).floor();
    if (bx < 0) return 0;
    if (bx >= gridWidth) return gridWidth - 1;
    return bx;
  }

  /// Row index of a world y-coordinate, clamped to `[0, gridHeight)`.
  int rowOf(double y) {
    final by = (y / binSize).floor();
    if (by < 0) return 0;
    if (by >= gridHeight) return gridHeight - 1;
    return by;
  }

  /// Flat bin index of world position `(x, y)`, row-major: `row * gridWidth + col`.
  int binIndexFor(double x, double y) => rowOf(y) * gridWidth + columnOf(x);

  /// The world-space **area** a bin covers, in px².
  ///
  /// The world is not always an integer number of bins wide/tall (1080/96 =
  /// 11.25, 720/96 = 7.5), so the sim clamps the leftover strip into the last
  /// row/column — making the right and bottom edge bins physically larger. Any
  /// density-based metric (e.g. edge-accumulation checks) must divide by this
  /// per-bin area or it will mistake that geometry for a real pile-up.
  double binArea(int col, int row) {
    final binW = (col == gridWidth - 1) ? worldWidth - col * binSize : binSize;
    final binH = (row == gridHeight - 1) ? worldHeight - row * binSize : binSize;
    return binW * binH;
  }
}

/// Per-bin particle counts over [grid] for the given [positions].
///
/// This is the **atomics-parity primitive** ([PRIMORDIS-TASK-017] reuses it):
/// the GPU tiers build this same histogram via `atomicAdd`, and comparing two
/// backends' occupancy vectors (Dawn/Tint vs browser/Naga) is exactly the
/// bin-count agreement check. Returned length is `grid.binCount`, row-major.
///
/// [positions] is interleaved `x, y`; only the first `2 * particleCount`
/// entries are read. Every particle is counted (no per-bin cap): the metric
/// characterises the *true* spatial distribution, independent of any backend's
/// `MAX_BIN_PARTICLES` membership cap.
Int32List binOccupancy(
  Float32List positions,
  int particleCount,
  MetricGrid grid,
) {
  final counts = Int32List(grid.binCount);
  for (var i = 0; i < particleCount; i++) {
    final bin = grid.binIndexFor(positions[i * 2], positions[i * 2 + 1]);
    counts[bin]++;
  }
  return counts;
}

/// Per-`(type, bin)` occupancy: `result[t]` is the occupancy vector for type `t`.
///
/// Used by the segregation metric: if types self-organise into distinct
/// structures (as the asymmetric matrices drive them to), each type's occupancy
/// vector concentrates in different bins than a uniform mixture would.
List<Int32List> binOccupancyByType(
  Float32List positions,
  Int32List types,
  int particleCount,
  int typeCount,
  MetricGrid grid,
) {
  final result = <Int32List>[
    for (var t = 0; t < typeCount; t++) Int32List(grid.binCount),
  ];
  for (var i = 0; i < particleCount; i++) {
    final bin = grid.binIndexFor(positions[i * 2], positions[i * 2 + 1]);
    result[types[i]][bin]++;
  }
  return result;
}

/// A one-dimensional summary of a distribution: count, mean, variance, min, max.
///
/// Sample variance is the population form (`E[x^2] - E[x]^2`) — the metrics
/// describe an entire particle population, not a sample of it. Immutable and
/// comparable so golden baselines can be asserted field-by-field.
class Moments {
  const Moments({
    required this.count,
    required this.mean,
    required this.variance,
    required this.min,
    required this.max,
  });

  /// Computes the moments of [values]. Empty input yields all-zero moments.
  factory Moments.of(Iterable<double> values) {
    var count = 0;
    var sum = 0.0;
    var sumSq = 0.0;
    var min = double.infinity;
    var max = double.negativeInfinity;
    for (final v in values) {
      count++;
      sum += v;
      sumSq += v * v;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (count == 0) {
      return const Moments(
        count: 0,
        mean: 0,
        variance: 0,
        min: 0,
        max: 0,
      );
    }
    final mean = sum / count;
    final variance = math.max(0.0, sumSq / count - mean * mean);
    return Moments(
      count: count,
      mean: mean,
      variance: variance,
      min: min,
      max: max,
    );
  }

  /// Number of samples.
  final int count;

  /// Arithmetic mean.
  final double mean;

  /// Population variance (`E[x^2] - E[x]^2`, floored at 0 for FP safety).
  final double variance;

  /// Smallest sample (0 when [count] is 0).
  final double min;

  /// Largest sample (0 when [count] is 0).
  final double max;

  /// Standard deviation (`sqrt(variance)`).
  double get stdDev => math.sqrt(variance);

  @override
  String toString() => 'Moments(n=$count, mean=${mean.toStringAsFixed(3)}, '
      'var=${variance.toStringAsFixed(3)}, min=${min.toStringAsFixed(3)}, '
      'max=${max.toStringAsFixed(3)})';
}

/// Minimum-image separation between two world points on the torus.
///
/// Returns the squared distance to avoid a `sqrt` in the hot path; callers that
/// need the true distance take the root. Wraps each axis by `±world` when the
/// naive separation exceeds half the world, exactly as the sim's force law does.
double toroidalDistanceSquared(
  double ax,
  double ay,
  double bx,
  double by,
  double worldWidth,
  double worldHeight,
) {
  var dx = bx - ax;
  var dy = by - ay;
  final halfW = worldWidth * 0.5;
  final halfH = worldHeight * 0.5;
  if (dx > halfW) {
    dx -= worldWidth;
  } else if (dx < -halfW) {
    dx += worldWidth;
  }
  if (dy > halfH) {
    dy -= worldHeight;
  } else if (dy < -halfH) {
    dy += worldHeight;
  }
  return dx * dx + dy * dy;
}
