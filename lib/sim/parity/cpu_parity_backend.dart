import 'dart:typed_data';

import 'package:primordis/sim/cpu/counting_sort_binning.dart';
import 'package:primordis/sim/cpu/cpu_sim_step.dart';
import 'package:primordis/sim/cpu/particle_soa.dart';
import 'package:primordis/sim/models/sim_params.dart';
import 'package:primordis/sim/models/sim_seed.dart';
import 'package:primordis/sim/parity/parity_runner.dart';
import 'package:primordis/sim/sim_seeder.dart';

/// [ParityBackend] adapter over the deterministic CPU tier
/// ([PRIMORDIS-TASK-008]).
///
/// This is the harness's **primary anchor**: the counting-sort binning is
/// sequential and bit-stable per seed+params, so its fingerprint is the most
/// reproducible reference the other (nondeterministic GPU) backends are measured
/// against. It reuses the shipping `cpuSimStep` / `countingSortBinning` /
/// `ParticleSoa` verbatim — the parity harness must exercise the *real* physics,
/// not a reimplementation, or it proves nothing.
///
/// The one deliberate divergence from the reference/GPU tiers is documented at
/// `countingSortBinning`: the CPU tier does **not** port the `MAX_BIN_PARTICLES
/// = 512` per-bin cap, so it never drops over-cap particles from bin membership.
/// The parity metrics are all position-invariant aggregates that tolerate this
/// (they never assume capped membership), which is exactly why parity is
/// statistical.
class CpuParityBackend implements ParityBackend {
  /// Builds a backend for [seed] with the given live-slider [params].
  ///
  /// [params] carries the matrices *shape* (world/grid/type/particle counts) and
  /// the three sliders; the actual matrix *contents* come from deterministic
  /// seeding so both this backend and the Python reference start from the
  /// identical initial condition. The seed's `particleCount`/`typeCount` must
  /// match [params].
  CpuParityBackend({
    required this.seedSpec,
    required this.params,
  })  : _grid = GridGeometry(
          worldWidth: params.worldWidth.toDouble(),
          worldHeight: params.worldHeight.toDouble(),
          binSize: params.binSize,
        ),
        _buffers = ParticleSoa(
          particleCount: seedSpec.particleCount,
          typeCount: seedSpec.typeCount,
          binCount: params.binCount,
        );

  /// The reproducible seed spec (RNG seed + counts).
  final SimSeed seedSpec;

  /// The live-slider + geometry params the step uses. The seeded matrices are
  /// injected into a copy of this at [seed] time.
  final SimParams params;

  final GridGeometry _grid;
  final ParticleSoa _buffers;
  late SimParams _liveParams;

  @override
  String get label => 'cpu';

  @override
  int get particleCount => seedSpec.particleCount;

  @override
  int get typeCount => seedSpec.typeCount;

  @override
  void seed() {
    final seeded = seedSimulation(seedSpec);
    _buffers.loadFrom(seeded);
    // Inject the deterministically-seeded matrices into the live params so the
    // step uses the same asymmetric force/min-distance/radius contents the
    // Python reference would from the same conceptual seed.
    _liveParams = params.copyWith(
      forces: seeded.forces,
      minDistances: seeded.minDistances,
      radii: seeded.radii,
    );
  }

  @override
  void step(double dt) => cpuSimStep(_buffers, _liveParams, _grid, dt);

  @override
  Float32List get positions => _buffers.positions;

  @override
  Float32List get velocities => _buffers.velocities;

  @override
  Int32List get types => _buffers.types;
}
