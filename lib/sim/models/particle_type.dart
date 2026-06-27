import 'package:freezed_annotation/freezed_annotation.dart';

part 'particle_type.freezed.dart';

/// One of the simulation's particle types: a stable [index] and its colour.
///
/// The reference (`Primordis.py`) assigns each of the 32 types a random RGB
/// colour (`np.random.rand(NUM_TYPES, 3)`, channels in `[0, 1)`) and renders
/// every particle in its type's colour. Colour generation is part of
/// deterministic seeding, so these are produced by the `SimSeeder` from a
/// `SimSeed` rather than constructed ad hoc.
///
/// Channels are kept as raw `float` components (not a `dart:ui` `Color`) so this
/// model stays platform-neutral and marshals 1:1 into the float colour buffer
/// the render pipeline uploads ([PRIMORDIS-ADR-001], [PRIMORDIS-ADR-005]).
@freezed
abstract class ParticleType with _$ParticleType {
  const factory ParticleType({
    /// Type index in `[0, typeCount)`. Also the row/column used to look up this
    /// type's behaviour in the force/min-distance/radius matrices.
    required int index,

    /// Red channel, `[0, 1)`.
    required double r,

    /// Green channel, `[0, 1)`.
    required double g,

    /// Blue channel, `[0, 1)`.
    required double b,
  }) = _ParticleType;
}
