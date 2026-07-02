import 'dart:typed_data';

import 'package:primordis/sim/models/seeded_sim.dart';

/// The mutable, structure-of-arrays working set the CPU physics core operates
/// on, expressed as an **injectable abstraction** rather than a concrete store.
///
/// The web CPU tier (T4, [PRIMORDIS-TASK-008]) backs this with plain
/// [Float32List]/[Int32List]s ([ParticleSoa]); the native multi-core isolate
/// tier (T3, [PRIMORDIS-TASK-014]) will back the *same* interface with typed-
/// list **views over an FFI `calloc`'d shared buffer** so isolates can share one
/// particle set by address ([PRIMORDIS-ADR-006]). Because the counting-sort and
/// step live in `lib/sim/cpu/` against this interface (never against a concrete
/// buffer), that native tier reuses the identical physics without a rewrite.
///
/// ## Layout contract (SoA, reused frame-to-frame)
///
/// All arrays are allocated once at [particleCount]/[binCount]/[typeCount] and
/// reused every frame — there is **no per-frame allocation** (the hot path must
/// not churn GC; [PRIMORDIS-TASK-008] acceptance criteria).
///
/// - [positions] / [velocities]: length `2 * particleCount`, interleaved `x, y`.
/// - [types]: length `particleCount`, per-particle type index.
/// - [binCounts]: length `binCount`, particles per bin (counting-sort pass 2).
/// - [binStarts]: length `binCount`, exclusive prefix-sum offsets (pass 3).
/// - [sortedIndices]: length `particleCount`, particle indices grouped by bin in
///   ascending bin order (the counting-sort output the neighbour scan reads).
/// - [renderXY]: length `2 * particleCount`, the packed point buffer handed to a
///   single `Canvas.drawRawPoints`; kept separate from [positions] so present
///   never mutates physics state.
///
/// Implementations expose typed-data **views**; the physics functions index them
/// directly. Whether those views wrap a Dart-owned buffer or FFI memory is the
/// implementation's concern, not the physics'.
abstract interface class SimBuffers {
  /// Number of live particles the buffers are sized for.
  int get particleCount;

  /// Number of particle types (matrix side length).
  int get typeCount;

  /// Number of spatial-grid bins.
  int get binCount;

  /// Interleaved `x, y` positions; length `2 * particleCount`.
  Float32List get positions;

  /// Interleaved `x, y` velocities; length `2 * particleCount`.
  Float32List get velocities;

  /// Per-particle type index; length `particleCount`.
  Int32List get types;

  /// Per-bin particle counts (counting-sort count pass); length `binCount`.
  Int32List get binCounts;

  /// Per-bin exclusive prefix-sum start offsets; length `binCount`.
  Int32List get binStarts;

  /// Particle indices grouped by bin in ascending bin order; length
  /// `particleCount`.
  Int32List get sortedIndices;

  /// Packed `x, y` render buffer for a single draw call; length
  /// `2 * particleCount`.
  Float32List get renderXY;
}

/// The Dart-owned [SimBuffers] used by the web CPU tier (T4).
///
/// Every array is a plain [Float32List]/[Int32List], allocated once in the
/// constructor and reused for the life of the store. Call [loadFrom] to copy a
/// freshly [SeededSim] into the working set without reallocating (a reseed at
/// the same count reuses the buffers).
class ParticleSoa implements SimBuffers {
  /// Allocates SoA storage for [particleCount] particles over a [binCount]-bin
  /// grid with [typeCount] types. All buffers are zero-initialized.
  ParticleSoa({
    required this.particleCount,
    required this.typeCount,
    required this.binCount,
  })  : assert(particleCount >= 0, 'particleCount must be non-negative'),
        assert(typeCount > 0, 'typeCount must be positive'),
        assert(binCount > 0, 'binCount must be positive'),
        positions = Float32List(particleCount * 2),
        velocities = Float32List(particleCount * 2),
        types = Int32List(particleCount),
        binCounts = Int32List(binCount),
        binStarts = Int32List(binCount),
        sortedIndices = Int32List(particleCount),
        renderXY = Float32List(particleCount * 2);

  @override
  final int particleCount;

  @override
  final int typeCount;

  @override
  final int binCount;

  @override
  final Float32List positions;

  @override
  final Float32List velocities;

  @override
  final Int32List types;

  @override
  final Int32List binCounts;

  @override
  final Int32List binStarts;

  @override
  final Int32List sortedIndices;

  @override
  final Float32List renderXY;

  /// Copies the deterministic [seeded] state into this working set in place.
  ///
  /// Requires the seed to match this store's dimensions ([particleCount],
  /// [typeCount]); a reseed at a different count needs a fresh [ParticleSoa]
  /// (backend selection sizes the store to the tier's count before seeding).
  /// The seeder emits [Uint32List] type indices; they are copied element-wise
  /// into the signed [Int32List] the physics core indexes with (values are in
  /// `[0, typeCount)`, so the reinterpretation is lossless).
  void loadFrom(SeededSim seeded) {
    assert(
      seeded.particleCount == particleCount,
      'seed particleCount ${seeded.particleCount} != store $particleCount',
    );
    assert(
      seeded.typeCount == typeCount,
      'seed typeCount ${seeded.typeCount} != store $typeCount',
    );
    positions.setAll(0, seeded.positions);
    velocities.setAll(0, seeded.velocities);
    for (var i = 0; i < particleCount; i++) {
      types[i] = seeded.types[i];
    }
  }
}
