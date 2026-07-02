import 'dart:typed_data';

import 'package:primordis/sim/models/sim_params.dart';
import 'package:primordis/sim/models/sim_seed.dart';
import 'package:primordis/sim/models/type_matrix.dart';
import 'package:primordis/sim/parity/cpu_parity_backend.dart';
import 'package:primordis/sim/parity/parity_metrics.dart';

/// Shared fixture parameters for the parity suite.
///
/// The **seed + params are defined once here** and reused by the reference
/// fixture generator, the CPU backend under test, and (later) every GPU
/// backend — so every backend starts from the byte-identical initial condition
/// the parity contract requires. The three matrices are injected by the backend
/// at `seed()` time from deterministic seeding, so only their *shape* (counts)
/// lives in the params; a small placeholder matrix satisfies the constructor.
class ParityHarnessConfig {
  ParityHarnessConfig({
    required this.seed,
    required this.particleCount,
    required this.typeCount,
    required this.attractionK,
    required this.repulsionK,
    required this.friction,
    required this.totalSteps,
    required this.checkpoints,
    this.dt = _fixedDt,
    this.worldWidth = 1080,
    this.worldHeight = 720,
    this.binSize = 96.0,
  });

  /// A fixed 60 fps timestep so the reference and every backend integrate the
  /// same dt (the live sim uses a variable clock dt; the harness pins it for
  /// reproducibility).
  static const double _fixedDt = 1.0 / 60.0;

  final int seed;
  final int particleCount;
  final int typeCount;
  final double attractionK;
  final double repulsionK;
  final double friction;
  final int totalSteps;
  final Map<String, int> checkpoints;
  final double dt;
  final int worldWidth;
  final int worldHeight;
  final double binSize;

  /// The default parity scenario: a modest population the CPU tier can step
  /// many times quickly in CI, at the reference default sliders, over a step
  /// budget long enough to reach a clustered steady state.
  ///
  /// Population is deliberately below the 24k reference (this is the CPU tier's
  /// regime, [PRIMORDIS-ADR-006]); parity is over *statistics*, and the metrics
  /// are density/fraction based so they compare across populations. The
  /// standalone WGSL kernel ([PRIMORDIS-TASK-003]) is validated at 24k against
  /// the looser cross-backend band.
  factory ParityHarnessConfig.defaults() => ParityHarnessConfig(
        seed: 42,
        particleCount: 3000,
        typeCount: 32,
        attractionK: 32,
        repulsionK: 32,
        friction: 0.25,
        totalSteps: 240,
        checkpoints: const <String, int>{
          'early': 0,
          'mid': 60,
          'steady': 239,
        },
      );

  /// A high-Drift variant (low friction/retention) used to assert every backend
  /// reproduces the same *behavioural response* to the Drift slider: kinetic
  /// energy must decay faster here than in [defaults]. Same seed/geometry.
  factory ParityHarnessConfig.highDrift() => ParityHarnessConfig(
        seed: 42,
        particleCount: 3000,
        typeCount: 32,
        attractionK: 32,
        repulsionK: 32,
        friction: 0.10,
        totalSteps: 240,
        checkpoints: const <String, int>{
          'early': 0,
          'mid': 60,
          'steady': 239,
        },
      );

  /// The metric grid mirroring the sim's own grid.
  MetricGrid get grid => MetricGrid(
        worldWidth: worldWidth.toDouble(),
        worldHeight: worldHeight.toDouble(),
        binSize: binSize,
      );

  /// The reproducible seed spec.
  SimSeed get simSeed => SimSeed(
        seed: seed,
        particleCount: particleCount,
        typeCount: typeCount,
      );

  /// The params block (matrices are placeholders — the backend injects the
  /// deterministically-seeded matrices at `seed()` time).
  SimParams get params {
    final placeholder = TypeMatrix(
      typeCount,
      Float32List(typeCount * typeCount),
    );
    return SimParams(
      forces: placeholder,
      minDistances: placeholder,
      radii: placeholder,
      attractionK: attractionK,
      repulsionK: repulsionK,
      friction: friction,
      particleCount: particleCount,
      typeCount: typeCount,
      worldWidth: worldWidth,
      worldHeight: worldHeight,
      binSize: binSize,
      binCount: grid.binCount,
      gridWidth: grid.gridWidth,
      gridHeight: grid.gridHeight,
    );
  }

  /// A ready-to-run CPU backend for this config.
  CpuParityBackend buildCpuBackend() =>
      CpuParityBackend(seedSpec: simSeed, params: params);
}
