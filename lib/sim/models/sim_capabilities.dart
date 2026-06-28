import 'package:freezed_annotation/freezed_annotation.dart';

part 'sim_capabilities.freezed.dart';

/// What a concrete `SimBackend` can do — chiefly its particle ceiling.
///
/// Backend selection ([PRIMORDIS-TASK-007] / [PRIMORDIS-TASK-015]) queries this
/// to set the per-tier default/maximum particle count and to drive the
/// "reduced mode" indicator ([PRIMORDIS-ADR-006]). It is intentionally small and
/// platform-neutral so the UI can read it without knowing which backend is live.
@freezed
abstract class SimBackendCapabilities with _$SimBackendCapabilities {
  const factory SimBackendCapabilities({
    /// Whether the backend runs the physics on the GPU (T1/T2) versus a CPU
    /// fallback tier (T3/T4).
    required bool isGpuAccelerated,

    /// Hard upper bound on particle count this backend can sustain.
    required int maxParticles,

    /// Particle count to seed by default for this backend/tier.
    required int defaultParticleCount,

    /// Short human-readable label (e.g. `web-webgpu`, `fake`), for diagnostics.
    required String label,
  }) = _SimBackendCapabilities;
}
