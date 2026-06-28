import 'package:primordis/sim/models/sim_params.dart';
import 'package:primordis/sim/sim_marshalling.dart';

/// Shared byte-layout of every GPU buffer the kernel binds, so the web
/// ([PRIMORDIS-TASK-004]) and macOS ([PRIMORDIS-TASK-011]) backends size and
/// marshal their buffers identically.
///
/// The element strides below are the WGSL `storage`/`uniform` layout sizes
/// (equivalently `std430` for the storage buffers) of the types declared in
/// `primordis.wgsl`. They are the *contract* the host marshalling
/// (`sim_marshalling.dart`, `SimSeeder`) already writes to; this file states the
/// per-buffer byte sizes derived from [SimParams] so backend buffer allocation
/// is a single shared computation rather than duplicated arithmetic.
///
/// Platform-neutral: depends only on the typed [SimParams] and the uniform-block
/// constant in [SimMarshalling]. No GPU API, FFI, or JS-interop.

/// WGSL element strides (bytes) for the kernel's buffer element types.
///
/// These match how Tint and Naga lay out tightly-packed storage arrays:
/// `array<vec2<f32>>` has stride 8, `array<vec4<f32>>` stride 16, scalar and
/// `atomic<u32>` arrays stride 4 — identical to the `Float32List`/`Uint32List`
/// buffers the host uploads.
abstract final class WgslStride {
  /// `f32` (forces / min-distances / radii element).
  static const int f32 = 4;

  /// `u32` (per-particle type element).
  static const int u32 = 4;

  /// `vec2<f32>` (interleaved x,y position/velocity element).
  static const int vec2f = 8;

  /// `vec4<f32>` (RGBA per-type colour element).
  static const int vec4f = 16;

  /// `atomic<u32>` (bin-counter element).
  static const int atomicU32 = 4;
}

/// The byte size of every kernel buffer for a given [SimParams].
///
/// One value per binding in `primordis.wgsl`; backends allocate GPU buffers of
/// exactly these sizes. The element counts mirror the buffers produced by the
/// `SimSeeder` and `sim_marshalling.dart` (cross-checked in tests), so the
/// layout, the seeded data, and the GPU buffers always agree.
class SimBufferLayout {
  /// Computes all buffer sizes from [params].
  factory SimBufferLayout(SimParams params) {
    final particleCount = params.particleCount;
    final typeCount = params.typeCount;
    final matrixCells = typeCount * typeCount;
    return SimBufferLayout._(
      uniform: SimMarshalling.uniformByteLength,
      positions: particleCount * WgslStride.vec2f,
      velocities: particleCount * WgslStride.vec2f,
      types: particleCount * WgslStride.u32,
      forces: matrixCells * WgslStride.f32,
      minDistances: matrixCells * WgslStride.f32,
      radii: matrixCells * WgslStride.f32,
      typeColors: typeCount * WgslStride.vec4f,
      binCounts: params.binCount * WgslStride.atomicU32,
      binParticles:
          params.binCount * params.maxBinParticles * WgslStride.u32,
    );
  }

  const SimBufferLayout._({
    required this.uniform,
    required this.positions,
    required this.velocities,
    required this.types,
    required this.forces,
    required this.minDistances,
    required this.radii,
    required this.typeColors,
    required this.binCounts,
    required this.binParticles,
  });

  /// Uniform block (`var<uniform> params: Params`) — 64 bytes.
  final int uniform;

  /// `array<vec2<f32>>` positions.
  final int positions;

  /// `array<vec2<f32>>` velocities.
  final int velocities;

  /// `array<u32>` per-particle types.
  final int types;

  /// `array<f32>` row-major forces matrix.
  final int forces;

  /// `array<f32>` row-major min-distances matrix.
  final int minDistances;

  /// `array<f32>` row-major radii matrix.
  final int radii;

  /// `array<vec4<f32>>` per-type colours.
  final int typeColors;

  /// `array<atomic<u32>>` bin counters.
  final int binCounts;

  /// `array<u32>` bin → particle-index list (`binCount * maxBinParticles`).
  final int binParticles;

  /// Total bytes across every buffer (diagnostics / budget checks).
  int get total =>
      uniform +
      positions +
      velocities +
      types +
      forces +
      minDistances +
      radii +
      typeColors +
      binCounts +
      binParticles;
}
