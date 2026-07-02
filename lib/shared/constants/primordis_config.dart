/// App-wide configuration constants for Primordis.
///
/// The simulation constants below mirror the `Primordis.py` reference. Their
/// authoritative, typed model (the Freezed `SimParams` with the 32x32 force /
/// min-distance / radius matrices) is introduced in PRIMORDIS-TASK-002; the
/// named constants here let the scaffold and config compile before that lands.
abstract final class PrimordisConfig {
  /// App version. Keep in sync with `pubspec.yaml` `version:`.
  static const String version = '0.10.0';

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

  // --- T4 (web CPU / Dart→WASM) tier policy (PRIMORDIS-ADR-006) ---

  /// Default particle count for the single-thread web CPU fallback tier (T4).
  ///
  /// The T4 ceiling is single-thread WASM: ~3-4k particles hold ~60fps, while
  /// the reference 24k lands at ~1-2.5fps (ADR-006 §1). This is the honest
  /// default the [CpuWasmBackend] seeds at — it is deliberately NOT
  /// [particleCount] (24k). Chosen mid-range of the ADR-006 3-4k band.
  static const int cpuWasmDefaultParticleCount = 3500;

  /// Hard upper bound for the T4 tier. A count slider may go up to this value
  /// but no further (ADR-006 §3: a tier must not exceed its benchmarked
  /// ceiling). Set to the top of the ADR-006 3-4k band.
  static const int cpuWasmMaxParticleCount = 4000;

  /// Rendered point size in logical pixels, matching the reference
  /// `gl_PointSize = 2.0` (`Primordis.py`).
  static const double pointSize = 2;
}
