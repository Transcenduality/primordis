import 'package:freezed_annotation/freezed_annotation.dart';
import 'package:primordis/shared/constants/primordis_config.dart';

part 'sim_seed.freezed.dart';

/// Describes a reproducible starting state for the simulation.
///
/// Everything random in Primordis — particle positions/velocities/types, the
/// three asymmetric per-type-pair matrices, and the per-type colours — is
/// derived deterministically from a [SimSeed] by the `SimSeeder`. The same
/// [SimSeed] yields byte-identical output on every run, which is what makes the
/// parity harness ([PRIMORDIS-TASK-009]) and unit tests reproducible.
///
/// Note: this is *seed* determinism only. The simulation's per-frame evolution
/// remains nondeterministic (the GPU binning is a single-buffered atomic
/// scatter with a known race), so "faithful" is statistical, not bit-exact
/// ([PRIMORDIS-ADR-001]).
@freezed
abstract class SimSeed with _$SimSeed {
  const factory SimSeed({
    /// RNG seed fed to the deterministic seeder.
    @Default(1) int seed,

    /// Number of particles to seed. Defaults to the reference 24,000; reduced
    /// tiers ([PRIMORDIS-ADR-006]) lower this without changing the physics.
    @Default(PrimordisConfig.particleCount) int particleCount,

    /// Number of particle types (and the side length of the 3 matrices).
    @Default(PrimordisConfig.typeCount) int typeCount,
  }) = _SimSeed;
}
