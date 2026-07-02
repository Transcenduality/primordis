import 'package:primordis/sim/cpu/particle_soa.dart';

/// The geometry of the uniform spatial grid the CPU physics bins into.
///
/// Mirrors the reference grid exactly ([PRIMORDIS-ADR-006], `Primordis.py`):
/// 1080/96 = 11 columns by 720/96 = 7 rows -> 77 bins, bin size 96.
/// Positions are always wrapped into `[0, world)`, so a particle's column/row
/// falls in `[0, gridWidth)` / `[0, gridHeight)` naturally; the `min` clamp in
/// [binIndexFor] only guards the exact-edge case `x == worldWidth` produced by
/// float rounding, matching the reference binning shader's `clamp`.
class GridGeometry {
  /// Builds a grid from world size and bin size.
  GridGeometry({
    required this.worldWidth,
    required this.worldHeight,
    required this.binSize,
  })  : gridWidth = (worldWidth / binSize).floor(),
        gridHeight = (worldHeight / binSize).floor() {
    assert(gridWidth > 0 && gridHeight > 0, 'grid must have >=1 bin per axis');
  }

  /// Toroidal world width in pixels.
  final double worldWidth;

  /// Toroidal world height in pixels.
  final double worldHeight;

  /// Bin edge length (== `MAX_RADIUS`).
  final double binSize;

  /// Grid columns (`worldWidth ~/ binSize`).
  final int gridWidth;

  /// Grid rows (`worldHeight ~/ binSize`).
  final int gridHeight;

  /// Total bins (`gridWidth * gridHeight`).
  int get binCount => gridWidth * gridHeight;

  /// Column index of a world x-coordinate, clamped to `[0, gridWidth)`.
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

  /// Flat bin index of world position `(x, y)`, row-major: `by * gridWidth + bx`.
  ///
  /// Matches the reference `bin_idx = y * grid_width + x` orientation.
  int binIndexFor(double x, double y) => rowOf(y) * gridWidth + columnOf(x);

  /// Toroidally-wrapped column: `(bx + gridWidth) % gridWidth`.
  int wrapColumn(int bx) => (bx % gridWidth + gridWidth) % gridWidth;

  /// Toroidally-wrapped row: `(by + gridHeight) % gridHeight`.
  int wrapRow(int by) => (by % gridHeight + gridHeight) % gridHeight;
}

/// Deterministic sequential counting-sort binning over the uniform grid.
///
/// This is the CPU tier's replacement for the GPU `atomicAdd` scatter binning
/// ([PRIMORDIS-ADR-006]). It is a textbook counting sort in three passes, all in
/// place over the reused [SimBuffers] scratch arrays (no per-frame allocation):
///
/// 1. **count** — zero `binCounts`, then increment `binCounts[bin]` per particle.
/// 2. **exclusive prefix-sum** — `binStarts[b] = sum(binCounts[0..b))`.
/// 3. **stable scatter** — walk particles in index order, placing each into
///    `sortedIndices` at a moving per-bin cursor (`binStarts` advanced through a
///    scratch copy), so the output is grouped by bin and, within a bin, in
///    ascending particle index. Stable + deterministic: identical seed+params
///    produce a **bit-stable** `sortedIndices` across runs on this tier.
///
/// ### Deliberate divergence from the reference / GPU tier
///
/// The reference caps each bin at `MAX_BIN_PARTICLES = 512` and silently drops
/// overflow (`if (offset < MAX_BIN_PARTICLES)`). A counting sort has **no fixed
/// per-bin ceiling** — every particle is always placed — so **the 512 cap is NOT
/// ported here**. This makes the CPU tier *more* faithful (no dropped particles)
/// while diverging from the GPU binning; the parity harness ([PRIMORDIS-TASK-009])
/// treats this counting-sort output as the deterministic baseline and must not
/// expect the GPU's capped/nondeterministic membership.
///
/// After this returns, for bin `b`: its particle indices occupy
/// `sortedIndices[binStarts[b] .. binStarts[b] + binCounts[b])`.
void countingSortBinning(SimBuffers buffers, GridGeometry grid) {
  final n = buffers.particleCount;
  final binCount = buffers.binCount;
  final positions = buffers.positions;
  final binCounts = buffers.binCounts;
  final binStarts = buffers.binStarts;
  final sortedIndices = buffers.sortedIndices;

  assert(
    binCount == grid.binCount,
    'buffer binCount $binCount != grid ${grid.binCount}',
  );

  // Pass 1 — count (also serves as the "clear bin counts" pass).
  for (var b = 0; b < binCount; b++) {
    binCounts[b] = 0;
  }
  for (var i = 0; i < n; i++) {
    final bin = grid.binIndexFor(positions[i * 2], positions[i * 2 + 1]);
    binCounts[bin]++;
  }

  // Pass 2 — exclusive prefix-sum into binStarts, and seed the moving cursor
  // (reuse sortedIndices space is not safe; use a running accumulator instead).
  var running = 0;
  for (var b = 0; b < binCount; b++) {
    binStarts[b] = running;
    running += binCounts[b];
  }

  // Pass 3 — stable scatter using a per-bin moving cursor. We advance a local
  // copy of the start offsets so binStarts stays as the neighbour-scan base.
  // `cursor` reuses no extra frame allocation beyond a single Int32List the
  // length of binCount; to honour the "no per-frame allocation" contract we
  // instead walk with a running offset recomputed from binStarts + a temporary
  // count already consumed. We keep a compact cursor by writing into
  // sortedIndices at binStarts[bin] + (placed so far in that bin), tracked via
  // binCounts being decremented back down is destructive; instead use the
  // prefix offsets directly with an auxiliary consumed-count folded into the
  // scatter loop below.
  //
  // To keep it strictly allocation-free AND non-destructive to binStarts, we
  // temporarily repurpose binCounts as the moving cursor: after this loop
  // binCounts is rebuilt to its true per-bin counts so callers (the neighbour
  // scan) can rely on both binStarts and binCounts.
  for (var b = 0; b < binCount; b++) {
    binCounts[b] = binStarts[b];
  }
  for (var i = 0; i < n; i++) {
    final bin = grid.binIndexFor(positions[i * 2], positions[i * 2 + 1]);
    sortedIndices[binCounts[bin]] = i;
    binCounts[bin]++;
  }
  // Restore binCounts to true per-bin counts: count[b] = start[b+1] - start[b]
  // (and for the last bin, n - start[last]).
  for (var b = 0; b < binCount; b++) {
    final end = (b + 1 < binCount) ? binStarts[b + 1] : n;
    binCounts[b] = end - binStarts[b];
  }
}
