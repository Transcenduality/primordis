/// The `SimBackend` seam.
///
/// This marks the boundary between the standards-compliant Flutter app (UI,
/// Riverpod, Freezed models — above this seam) and the platform-specific GPU /
/// compute code (browser WebGPU via `dart:js_interop`, Dawn/wgpu-over-Metal via
/// `dart:ffi`, a native Metal plugin, or the CPU fallbacks — below this seam).
///
/// Per PRIMORDIS-ADR-001, all GPU / FFI / JS-interop / WGSL code lives BEHIND
/// this interface, deliberately OUTSIDE the standard feature/data/domain
/// layers. That quarantine is the whole point of the architecture: the UI and
/// state layers never import platform specifics and stay fully testable.
///
/// This file is only the placeholder seam established in PRIMORDIS-TASK-001.
/// The concrete interface — `init()`, `seed()`, `setParams()`, `step(dt)`,
/// `present()`, `dispose()`, and a capability/particle-ceiling query — is
/// defined in PRIMORDIS-TASK-002, and concrete backends follow in TASK-004 /
/// TASK-008 / TASK-011 / TASK-013 / TASK-014.
abstract interface class SimBackend {
  // Intentionally empty: members are added in PRIMORDIS-TASK-002.
}
