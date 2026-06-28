import 'dart:math';
import 'dart:typed_data';

import 'package:primordis/shared/constants/primordis_config.dart';
import 'package:primordis/sim/models/particle_type.dart';
import 'package:primordis/sim/models/seeded_sim.dart';
import 'package:primordis/sim/models/sim_seed.dart';
import 'package:primordis/sim/models/type_matrix.dart';

/// Deterministically materializes particles, the three matrices, and the
/// per-type colours from a [SimSeed].
///
/// Faithful to `Primordis.py`'s `set_parameters` / `random_type_colors` /
/// particle initialization (same distributions and ranges), but uses Dart's
/// [Random] rather than NumPy's PRNG — so output is **reproducible across Dart
/// runs for a given seed**, not bit-identical to the Python reference (parity is
/// statistical; [PRIMORDIS-TASK-009]).
///
/// ### Draw order (fixed, so the output is stable)
///
/// All randomness comes from a single `Random(seed.seed)` consumed in this exact
/// order; changing the order changes the output for a given seed:
///
/// 1. positions — per particle: `x = u()*worldWidth`, `y = u()*worldHeight`.
/// 2. velocities — per particle: `vx = u()*16 - 8`, `vy = u()*16 - 8` (-8..8).
/// 3. types — per particle: `nextInt(typeCount)`.
/// 4. colours — per type: `r, g, b = u(), u(), u()` (each in `[0, 1)`).
/// 5. forces — value block then sign block: `0.1 + u()*0.7` (0.1..0.8) for every
///    cell, then negate each cell where `u() < 0.5` ⇒ signed, asymmetric.
/// 6. minDistances — per cell: `4 + u()*8` (4..12).
/// 7. radii — per cell: `20 + u()*76` (20..[PrimordisConfig.maxRadius]).
///
/// where `u()` is `Random.nextDouble()`. Matrices are row-major over
/// `(i = my_type, j = other_type)`.
SeededSim seedSimulation(SimSeed seed) {
  assert(seed.typeCount > 0, 'typeCount must be positive, got ${seed.typeCount}');
  assert(
    seed.particleCount >= 0,
    'particleCount must be non-negative, got ${seed.particleCount}',
  );
  final rng = Random(seed.seed);
  final n = seed.typeCount;
  final count = seed.particleCount;
  const worldWidth = PrimordisConfig.worldWidth;
  const worldHeight = PrimordisConfig.worldHeight;

  // 1. Positions (interleaved x, y).
  final positions = Float32List(count * 2);
  for (var i = 0; i < count; i++) {
    positions[i * 2] = rng.nextDouble() * worldWidth;
    positions[i * 2 + 1] = rng.nextDouble() * worldHeight;
  }

  // 2. Velocities (interleaved x, y; uniform -8..8).
  final velocities = Float32List(count * 2);
  for (var i = 0; i < count; i++) {
    velocities[i * 2] = rng.nextDouble() * 16.0 - 8.0;
    velocities[i * 2 + 1] = rng.nextDouble() * 16.0 - 8.0;
  }

  // 3. Per-particle type indices.
  final types = Uint32List(count);
  for (var i = 0; i < count; i++) {
    types[i] = rng.nextInt(n);
  }

  // 4. Per-type colours.
  final particleTypes = <ParticleType>[
    for (var t = 0; t < n; t++)
      ParticleType(
        index: t,
        r: rng.nextDouble(),
        g: rng.nextDouble(),
        b: rng.nextDouble(),
      ),
  ];

  // 5. Forces: a value block (0.1..0.8) followed by an independent sign block.
  final forcesData = Float32List(n * n);
  for (var k = 0; k < forcesData.length; k++) {
    forcesData[k] = 0.1 + rng.nextDouble() * 0.7;
  }
  for (var k = 0; k < forcesData.length; k++) {
    if (rng.nextDouble() < 0.5) forcesData[k] = -forcesData[k];
  }
  final forces = TypeMatrix(n, forcesData);

  // 6. Minimum distances (4..12).
  final minDistances = TypeMatrix.generate(
    n,
    (_, _) => 4.0 + rng.nextDouble() * 8.0,
  );

  // 7. Radii (20..maxRadius).
  const radiusSpan = PrimordisConfig.maxRadius - 20.0;
  final radii = TypeMatrix.generate(
    n,
    (_, _) => 20.0 + rng.nextDouble() * radiusSpan,
  );

  return SeededSim(
    seed: seed,
    particleCount: count,
    typeCount: n,
    positions: positions,
    velocities: velocities,
    types: types,
    particleTypes: particleTypes,
    forces: forces,
    minDistances: minDistances,
    radii: radii,
  );
}
