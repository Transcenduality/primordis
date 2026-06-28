import 'dart:typed_data';

import 'package:primordis/sim/models/particle_type.dart';
import 'package:primordis/sim/models/sim_seed.dart';
import 'package:primordis/sim/models/type_matrix.dart';

/// The complete deterministic output of seeding: initial particle buffers, the
/// three matrices, and the per-type colours.
///
/// This is the bundle the `SimSeeder` produces from a [SimSeed] and that a
/// backend uploads at `seed()` time. It is a plain immutable carrier rather than
/// a Freezed model on purpose: its payload is typed-data buffers
/// ([Float32List]/[Uint32List]), for which Freezed's generated `==` would fall
/// back to identity — value comparison is done element-wise in tests instead.
///
/// ## Structure-of-arrays layout (the marshalling contract)
///
/// Particle data is stored SoA, sized for [particleCount] particles, in the
/// exact `std430`-friendly layout backends bind ([PRIMORDIS-ADR-003]):
///
/// - [positions] / [velocities]: `Float32List` of length `2 * particleCount`,
///   interleaved `x, y` per particle (an array of tightly-packed `vec2<f32>`).
/// - [types]: `Uint32List` of length `particleCount`, the per-particle type
///   index (`array<u32>` in WGSL).
///
/// See `sim_marshalling.dart` for the matrix/colour/uniform packing built on top
/// of this.
class SeededSim {
  const SeededSim({
    required this.seed,
    required this.particleCount,
    required this.typeCount,
    required this.positions,
    required this.velocities,
    required this.types,
    required this.particleTypes,
    required this.forces,
    required this.minDistances,
    required this.radii,
  });

  /// The seed this output was produced from.
  final SimSeed seed;

  /// Number of particles in the SoA buffers.
  final int particleCount;

  /// Number of particle types (matrix side length, [particleTypes] length).
  final int typeCount;

  /// Interleaved `x, y` positions; length `2 * particleCount`.
  final Float32List positions;

  /// Interleaved `x, y` velocities; length `2 * particleCount`.
  final Float32List velocities;

  /// Per-particle type index; length `particleCount`.
  final Uint32List types;

  /// The [typeCount] particle types with their seeded colours.
  final List<ParticleType> particleTypes;

  /// Signed attraction/repulsion matrix.
  final TypeMatrix forces;

  /// Repulsion-onset distance matrix.
  final TypeMatrix minDistances;

  /// Attraction cutoff radius matrix.
  final TypeMatrix radii;
}
