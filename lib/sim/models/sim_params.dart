import 'package:freezed_annotation/freezed_annotation.dart';
import 'package:primordis/shared/constants/primordis_config.dart';
import 'package:primordis/sim/models/type_matrix.dart';

part 'sim_params.freezed.dart';

/// Live-slider ranges and defaults, mirroring the three `Primordis.py` sliders.
///
/// The UI ([PRIMORDIS-TASK-006]) clamps user input to these bounds before
/// mutating [SimParams]; the seeder/marshalling use the defaults as the starting
/// values. Kept here next to [SimParams] so the value and its bounds live
/// together.
abstract final class SimSliders {
  /// Attraction strength multiplier (`K_attraction`).
  static const double attractionMin = 0.1;
  static const double attractionMax = 128.0;
  static const double attractionDefault = 32.0;

  /// Repulsion strength multiplier (`K_repulsion`).
  static const double repulsionMin = 0.1;
  static const double repulsionMax = 128.0;
  static const double repulsionDefault = 32.0;

  /// Drift/friction: per-tick velocity retention (`v *= friction`). Labelled
  /// "Particle Drift Strength" in the reference.
  static const double frictionMin = 0.05;
  static const double frictionMax = 0.99;
  static const double frictionDefault = 0.25;
}

/// The complete, platform-agnostic parameter block the UI drives and every
/// backend uploads.
///
/// It bundles three things the simulation needs, all in one Freezed value:
///
/// 1. **The three asymmetric 32x32 `float32` matrices** — [forces] (signed
///    attraction/repulsion), [minDistances], and [radii]. `m.at(i, j)` is
///    independent of `m.at(j, i)`; this directed per-type-pair encoding is the
///    heart of the simulation (see [TypeMatrix]).
/// 2. **The three live sliders** — [attractionK], [repulsionK], [friction] —
///    which the user mutates and which flow into the per-frame uniform block.
/// 3. **The world/grid constants** — world size, toroidal grid geometry, bin
///    sizing, type and particle counts — carried here so marshalling is
///    self-contained and a reduced-mode tier can lower [particleCount] without
///    touching the physics ([PRIMORDIS-ADR-006]).
///
/// Defaults mirror [PrimordisConfig] (asserted in tests to prevent drift). The
/// matrices have no default — they come from deterministic seeding.
///
/// Value equality matters: the frame loop calls `setParams` only when
/// [SimParams] changes, which relies on Freezed's generated `==` over these
/// fields (the matrices supply their own value equality via [TypeMatrix]).
@freezed
abstract class SimParams with _$SimParams {
  const factory SimParams({
    /// Signed attraction/repulsion strength per ordered type pair.
    required TypeMatrix forces,

    /// Repulsion onset distance per ordered type pair (reference range 4..12).
    required TypeMatrix minDistances,

    /// Attraction cutoff radius per ordered type pair (reference range 20..96).
    required TypeMatrix radii,

    /// Live attraction multiplier. See [SimSliders].
    @Default(SimSliders.attractionDefault) double attractionK,

    /// Live repulsion multiplier. See [SimSliders].
    @Default(SimSliders.repulsionDefault) double repulsionK,

    /// Live drift/friction. See [SimSliders].
    @Default(SimSliders.frictionDefault) double friction,

    /// Active particle count (default 24,000; reduced tiers lower it).
    @Default(PrimordisConfig.particleCount) int particleCount,

    /// Number of particle types and the matrix side length.
    @Default(PrimordisConfig.typeCount) int typeCount,

    /// Toroidal world width in pixels.
    @Default(PrimordisConfig.worldWidth) int worldWidth,

    /// Toroidal world height in pixels.
    @Default(PrimordisConfig.worldHeight) int worldHeight,

    /// Max interaction radius; also the spatial-grid bin size. Literal `96.0`
    /// because const defaults can't call `.toDouble()`; mirrors
    /// [PrimordisConfig.maxRadius] (asserted in tests).
    @Default(96.0) double maxRadius,

    /// Uniform-grid bin size (== [maxRadius]); mirrors [PrimordisConfig.binSize].
    @Default(96.0) double binSize,

    /// Grid columns (`worldWidth ~/ binSize`).
    @Default(PrimordisConfig.gridWidth) int gridWidth,

    /// Grid rows (`worldHeight ~/ binSize`).
    @Default(PrimordisConfig.gridHeight) int gridHeight,

    /// Total bins (`gridWidth * gridHeight`).
    @Default(PrimordisConfig.binCount) int binCount,

    /// Per-bin particle-index capacity; overflow is dropped (as in reference).
    @Default(PrimordisConfig.maxBinParticles) int maxBinParticles,
  }) = _SimParams;
}
