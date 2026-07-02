import 'dart:math' as math;

import 'package:primordis/sim/cpu/counting_sort_binning.dart';
import 'package:primordis/sim/cpu/particle_soa.dart';
import 'package:primordis/sim/models/sim_params.dart';
import 'package:primordis/sim/models/type_matrix.dart';

/// The per-particle-pair short-range repulsion weight from the reference
/// (`f -= dn * abs(force) * 5.0 * ...`, `Primordis.py`).
const double _repulsionWeight = 5.0;

/// The pair-skip radius floor from the reference (`dist < 0.1 -> continue`),
/// which also avoids a divide-by-zero when normalising the separation.
const double _minPairDistance = 0.1;

/// Advances the CPU simulation by [dt] seconds: bin, then 3x3-neighbour
/// interaction with Euler integration and toroidal wrap.
///
/// This is the platform-neutral hot path shared by the web CPU tier (T4,
/// [PRIMORDIS-TASK-008]) and, later, the native isolate tier (T3,
/// [PRIMORDIS-TASK-014]) — it operates only on the injectable [SimBuffers]
/// abstraction and the plain-data [params], so the native tier reuses it
/// verbatim over FFI-backed buffers. It contains no `dart:ui`, no widget code,
/// and no per-frame allocation.
///
/// Faithful to `Primordis.py`'s binning + interaction compute shaders (see
/// [PRIMORDIS-ADR-006]); "faithful" is statistical, not bit-exact vs the GPU
/// ([PRIMORDIS-ADR-001]). The exact force law reproduced here:
///
/// - separation `d = pos[j] - pos[i]` under **minimum-image** on the toroidal
///   world (wrap each axis by `±world` when `|d| > world/2`);
/// - `dist = length(d)`; skip the pair when `dist > maxRadius` or
///   `dist < 0.1`; direction `dn = d / dist`;
/// - matrix lookup is **asymmetric**, indexed `[my_type][other_type]` i.e.
///   `my_type * typeCount + other_type` (source particle's type acting on the
///   neighbour), matching the reference `idx = my_type * NUM_TYPES + other_type`;
/// - if `dist < minDistances[idx]`: repulsion
///   `f -= dn * abs(force) * 5 * (1 - dist/mind) * K_repulsion`;
/// - else if `dist < radii[idx]`: attraction
///   `f += dn * force * (1 - dist/rad) * K_attraction` (signed force);
/// - integration `v += f*dt; v *= friction; p += v*dt; p = wrap(p)`.
///
/// [grid] must match `params`' world/bin geometry and the buffer sizes.
void cpuSimStep(
  SimBuffers buffers,
  SimParams params,
  GridGeometry grid,
  double dt,
) {
  // Fail fast if the buffers, params, and grid disagree on dimensions — a drift
  // (e.g. a reduced-tier count change that didn't propagate to params) would
  // otherwise read past matrix bounds or mis-wrap distances with no clear error.
  assert(
    buffers.typeCount == params.typeCount,
    'buffer typeCount ${buffers.typeCount} != params ${params.typeCount}',
  );
  assert(
    buffers.binCount == grid.binCount,
    'buffer binCount ${buffers.binCount} != grid ${grid.binCount}',
  );
  assert(
    params.forces.dimension == params.typeCount,
    'forces matrix side ${params.forces.dimension} != typeCount '
    '${params.typeCount}',
  );
  assert(
    grid.worldWidth == params.worldWidth.toDouble() &&
        grid.worldHeight == params.worldHeight.toDouble(),
    'grid world size does not match params world size',
  );

  countingSortBinning(buffers, grid);

  final n = buffers.particleCount;
  if (n == 0) return;

  final positions = buffers.positions;
  final velocities = buffers.velocities;
  final types = buffers.types;
  final binCounts = buffers.binCounts;
  final binStarts = buffers.binStarts;
  final sortedIndices = buffers.sortedIndices;

  final forces = params.forces;
  final minDistances = params.minDistances;
  final radii = params.radii;
  final typeCount = params.typeCount;

  final worldWidth = params.worldWidth.toDouble();
  final worldHeight = params.worldHeight.toDouble();
  final halfWorldWidth = worldWidth * 0.5;
  final halfWorldHeight = worldHeight * 0.5;
  final maxRadius = params.maxRadius;
  final kAttraction = params.attractionK;
  final kRepulsion = params.repulsionK;
  final friction = params.friction;

  final gridWidth = grid.gridWidth;

  for (var i = 0; i < n; i++) {
    final px = positions[i * 2];
    final py = positions[i * 2 + 1];
    final myType = types[i];
    final rowBase = myType * typeCount;

    final cx = grid.columnOf(px);
    final cy = grid.rowOf(py);

    var fx = 0.0;
    var fy = 0.0;

    for (var dx = -1; dx <= 1; dx++) {
      final nx = grid.wrapColumn(cx + dx);
      for (var dy = -1; dy <= 1; dy++) {
        final ny = grid.wrapRow(cy + dy);
        final binIdx = ny * gridWidth + nx;
        final start = binStarts[binIdx];
        final end = start + binCounts[binIdx];
        for (var s = start; s < end; s++) {
          final j = sortedIndices[s];
          if (j == i) continue;

          var ddx = positions[j * 2] - px;
          var ddy = positions[j * 2 + 1] - py;
          // Minimum-image on the torus.
          if (ddx > halfWorldWidth) {
            ddx -= worldWidth;
          } else if (ddx < -halfWorldWidth) {
            ddx += worldWidth;
          }
          if (ddy > halfWorldHeight) {
            ddy -= worldHeight;
          } else if (ddy < -halfWorldHeight) {
            ddy += worldHeight;
          }

          final dist = math.sqrt(ddx * ddx + ddy * ddy);
          if (dist > maxRadius || dist < _minPairDistance) continue;

          final invDist = 1.0 / dist;
          final dnx = ddx * invDist;
          final dny = ddy * invDist;

          final idx = rowBase + types[j];
          final mind = minDistances.valueAtFlat(idx);
          final rad = radii.valueAtFlat(idx);
          final forceStrength = forces.valueAtFlat(idx);

          if (dist < mind) {
            // Short-range repulsion: 5x-weighted, magnitude of the (signed)
            // force, directed away from the neighbour.
            final mag = forceStrength.abs() *
                _repulsionWeight *
                (1.0 - dist / mind) *
                kRepulsion;
            fx -= dnx * mag;
            fy -= dny * mag;
          } else if (dist < rad) {
            // Linear-falloff attraction using the SIGNED force.
            final mag = forceStrength * (1.0 - dist / rad) * kAttraction;
            fx += dnx * mag;
            fy += dny * mag;
          }
        }
      }
    }

    // Euler integration + drift, then toroidal position wrap.
    var vx = velocities[i * 2] + fx * dt;
    var vy = velocities[i * 2 + 1] + fy * dt;
    vx *= friction;
    vy *= friction;
    var newX = px + vx * dt;
    var newY = py + vy * dt;
    if (newX < 0.0) {
      newX += worldWidth;
    } else if (newX >= worldWidth) {
      newX -= worldWidth;
    }
    if (newY < 0.0) {
      newY += worldHeight;
    } else if (newY >= worldHeight) {
      newY -= worldHeight;
    }

    positions[i * 2] = newX;
    positions[i * 2 + 1] = newY;
    velocities[i * 2] = vx;
    velocities[i * 2 + 1] = vy;
  }
}

/// Copies live positions into the reused packed render buffer for a single
/// `Canvas.drawRawPoints` call.
///
/// Kept separate from [cpuSimStep] so `present()` (which fills the render
/// buffer) never mutates physics state and a paused frame can re-blit the last
/// buffer without stepping. Writes exactly `2 * particleCount` floats into the
/// pre-allocated [SimBuffers.renderXY]; no allocation.
void fillRenderBuffer(SimBuffers buffers) {
  final n = buffers.particleCount;
  final positions = buffers.positions;
  final renderXY = buffers.renderXY;
  for (var k = 0; k < n * 2; k++) {
    renderXY[k] = positions[k];
  }
}

/// Flat-index accessor on [TypeMatrix] used by the hot loop.
///
/// [TypeMatrix.at] takes `(row, col)` and asserts bounds twice per call; the
/// physics core already computes the row-major flat index (`i*n + j`) once, so
/// this reads the backing store directly to keep the inner loop tight.
extension _FlatMatrixAccess on TypeMatrix {
  double valueAtFlat(int flatIndex) => values[flatIndex];
}
