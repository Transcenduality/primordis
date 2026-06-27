/// App-wide configuration constants for Primordis.
///
/// The simulation constants below mirror the `Primordis.py` reference. Their
/// authoritative, typed model (the Freezed `SimParams` with the 32x32 force /
/// min-distance / radius matrices) is introduced in PRIMORDIS-TASK-002; the
/// named constants here let the scaffold and config compile before that lands.
abstract final class PrimordisConfig {
  /// App version. Keep in sync with `pubspec.yaml` `version:`.
  static const String version = '0.1.0';

  // --- Simulation constants (reference: Primordis.py) ---

  /// Toroidal world size in pixels.
  static const int worldWidth = 1080;
  static const int worldHeight = 720;

  /// Particle population and per-particle type count.
  static const int particleCount = 24000;
  static const int typeCount = 32;

  /// Max interaction radius; also the uniform-grid bin size.
  static const int maxRadius = 96;
  static const int binSize = maxRadius;

  /// Spatial grid: 1080/96 = 11 by 720/96 = 7 -> 77 bins.
  static const int gridWidth = worldWidth ~/ binSize;
  static const int gridHeight = worldHeight ~/ binSize;
  static const int binCount = gridWidth * gridHeight;

  /// Per-bin particle-index capacity (overflow is dropped, as in the reference).
  static const int maxBinParticles = 512;
}
