import 'package:primordis/sim/models/sim_capabilities.dart';
import 'package:primordis/sim/models/sim_params.dart';
import 'package:primordis/sim/models/sim_seed.dart';

/// The `SimBackend` seam.
///
/// This marks the boundary between the standards-compliant Flutter app (UI,
/// Riverpod, Freezed models — above this seam) and the platform-specific GPU /
/// compute code (browser WebGPU via `dart:js_interop`, Dawn/wgpu-over-Metal via
/// `dart:ffi`, a native Metal plugin, or the CPU fallbacks — below this seam).
///
/// Per [PRIMORDIS-ADR-001], all GPU / FFI / JS-interop / WGSL code lives BEHIND
/// this interface, deliberately OUTSIDE the standard feature/data/domain
/// layers. That quarantine is the whole point of the architecture: the UI and
/// state layers never import platform specifics and stay fully testable against
/// a fake backend.
///
/// ## The lifecycle contract
///
/// The method shapes are platform-neutral and must abstract device/pipeline
/// creation, dispatch, parameter upload, and present across both the owned-
/// `<canvas>` web model and the external-`Texture` macOS model
/// ([PRIMORDIS-ADR-005]) without leaking platform specifics:
///
/// - [init] — acquire the device/adapter, build pipelines and persistent
///   buffers. Async because device acquisition is async (e.g. WebGPU
///   `requestAdapter`/`requestDevice`). Call once before any other method.
/// - [seed] — (re)initialize particle state deterministically from a [SimSeed].
///   The backend materializes the buffers via the shared `SimSeeder` (seeding is
///   shared code) and uploads them. Async to allow buffer mapping/upload.
/// - [setParams] — upload the matrices and the live slider/constant uniform
///   block. Synchronous (enqueues a buffer write); called by the frame loop only
///   when [SimParams] changes.
/// - [step] — advance the simulation by `dt` seconds: one Euler tick of the
///   shared kernel (clear bin counts → atomic-scatter binning → 3x3 neighbour
///   interaction + integrate + wrap). Synchronous enqueue; the hot path.
/// - [present] — display the latest frame (point render + composite/texture
///   present). Synchronous enqueue.
/// - [dispose] — release all GPU/native/JS resources. Async to allow device
///   teardown.
/// - [capabilities] — static query for the particle ceiling and whether the
///   backend is GPU-accelerated, used by backend selection and the reduced-mode
///   indicator ([PRIMORDIS-ADR-006]).
///
/// The *meaning* of [step]/[present] differs per backend (3 compute passes + a
/// point render on GPU; a counting-sort + `drawRawPoints` on CPU; an IOSurface
/// `Texture` present on macOS), but the ordering and the dt-driven Euler tick
/// contract are shared and exercised by the frame loop against `FakeSimBackend`.
///
/// Backend *selection* is not part of this seam ([PRIMORDIS-TASK-007] /
/// [PRIMORDIS-TASK-015]); this task only defines the interface and ships the
/// fake so selection can later inject a concrete backend behind the same
/// provider.
abstract interface class SimBackend {
  /// Acquire the device and build pipelines/persistent buffers. Call once.
  Future<void> init();

  /// (Re)initialize particle state deterministically from [seed].
  Future<void> seed(SimSeed seed);

  /// Upload [params] (matrices + live slider/constant uniforms) to the backend.
  void setParams(SimParams params);

  /// Advance the simulation by [dt] seconds (one Euler tick).
  void step(double dt);

  /// Present the latest simulated frame.
  void present();

  /// Release all resources held by this backend.
  Future<void> dispose();

  /// Static capabilities — particle ceiling and GPU/CPU tier.
  SimBackendCapabilities get capabilities;
}
