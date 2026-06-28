import 'dart:typed_data';

import 'package:primordis/sim/models/particle_type.dart';
import 'package:primordis/sim/models/sim_params.dart';
import 'package:primordis/sim/models/type_matrix.dart';

/// Packs typed [SimParams] / seed data into the exact `Float32List` /
/// `Uint32List` byte layouts the backends upload.
///
/// This is the **contract between this task and every backend** ([PRIMORDIS-
/// TASK-003] WGSL struct, [PRIMORDIS-TASK-004] WebGPU buffers, the FFI buffers in
/// TASK-011): the field order, offsets, and alignment fixed here are what the
/// shader and the host buffers must agree on. All multi-byte values are written
/// in **little-endian** host order (every target GPU/platform is LE); the typed-
/// list views below inherit host endianness, which matches `writeBuffer`/SSBO
/// upload semantics.
///
/// ## Buffers and their WGSL bindings
///
/// | Buffer            | Dart type     | Layout                                  | WGSL |
/// |-------------------|---------------|-----------------------------------------|------|
/// | uniforms          | `Uint8List`   | 64-byte struct (see [SimMarshalling])   | `var<uniform>` |
/// | forces/min/radii  | `Float32List` | row-major `n*n`, idx `i*n + j`          | `array<f32>` (`std430`) |
/// | type colours      | `Float32List` | `typeCount` * RGBA, 16-byte aligned     | `array<vec4<f32>>` |
/// | positions/vels    | `Float32List` | interleaved `x,y`, `2*N` (from seeder)  | `array<vec2<f32>>` |
/// | types             | `Uint32List`  | `N` (from seeder)                       | `array<u32>` |
/// | bin counts        | `Uint32List`  | `binCount`                              | `array<atomic<u32>>` |
/// | bin particles     | `Uint32List`  | `binCount * maxBinParticles`            | `array<u32>` |
///
/// Bin counts are `Uint32List` specifically because the binning pass scatters
/// into them with `atomicAdd` and reads them via `atomicLoad`, requiring
/// `atomic<u32>` ([PRIMORDIS-ADR-002] / [PRIMORDIS-ADR-003]).
///
/// [SimMarshalling] is the namespace for the uniform-block layout constants; the
/// packing verbs ([packUniforms], [flattenMatrix], …) are top-level functions in
/// this library.
abstract final class SimMarshalling {
  // --- Uniform block: a 64-byte struct of sixteen 4-byte slots ---
  //
  // 16-byte aligned (64 = 4 * 16) so it drops straight into a uniform buffer
  // with no trailing pad. Slots 13..15 are explicit zero padding reserved for
  // future scalars. Each slot is read as f32 or u32 per the map; the WGSL struct
  // in TASK-003 must declare fields in this order with matching types.

  /// `f32` attraction multiplier (`K_attraction`).
  static const int slotAttractionK = 0;

  /// `f32` repulsion multiplier (`K_repulsion`).
  static const int slotRepulsionK = 1;

  /// `f32` drift/friction.
  static const int slotFriction = 2;

  /// `f32` per-tick delta time (seconds).
  static const int slotDt = 3;

  /// `f32` world width (pixels).
  static const int slotWorldWidth = 4;

  /// `f32` world height (pixels).
  static const int slotWorldHeight = 5;

  /// `f32` max interaction radius.
  static const int slotMaxRadius = 6;

  /// `f32` grid bin size.
  static const int slotBinSize = 7;

  /// `u32` grid columns.
  static const int slotGridWidth = 8;

  /// `u32` grid rows.
  static const int slotGridHeight = 9;

  /// `u32` active particle count.
  static const int slotNumParticles = 10;

  /// `u32` total bins.
  static const int slotNumBins = 11;

  /// `u32` particle-type count.
  static const int slotTypeCount = 12;

  /// Number of 4-byte slots in the uniform block.
  static const int uniformSlotCount = 16;

  /// Size of the uniform block in bytes.
  static const int uniformByteLength = uniformSlotCount * 4;

  /// Number of float slots per packed colour (RGBA).
  static const int colorStride = 4;
}

/// Packs [params] plus the per-frame [dt] into the 64-byte uniform block.
Uint8List packUniforms(SimParams params, double dt) {
  final bytes = Uint8List(SimMarshalling.uniformByteLength);
  final f = Float32List.view(bytes.buffer);
  final u = Uint32List.view(bytes.buffer);
  f[SimMarshalling.slotAttractionK] = params.attractionK;
  f[SimMarshalling.slotRepulsionK] = params.repulsionK;
  f[SimMarshalling.slotFriction] = params.friction;
  f[SimMarshalling.slotDt] = dt;
  f[SimMarshalling.slotWorldWidth] = params.worldWidth.toDouble();
  f[SimMarshalling.slotWorldHeight] = params.worldHeight.toDouble();
  f[SimMarshalling.slotMaxRadius] = params.maxRadius;
  f[SimMarshalling.slotBinSize] = params.binSize;
  u[SimMarshalling.slotGridWidth] = params.gridWidth;
  u[SimMarshalling.slotGridHeight] = params.gridHeight;
  u[SimMarshalling.slotNumParticles] = params.particleCount;
  u[SimMarshalling.slotNumBins] = params.binCount;
  u[SimMarshalling.slotTypeCount] = params.typeCount;
  // Slots 13..15 remain zero (Uint8List is zero-initialized) — reserved pad.
  return bytes;
}

/// Reads a [packUniforms] block back into named values (for round-trip tests and
/// diagnostics). [bytes] must be at least [SimMarshalling.uniformByteLength] long.
UniformValues unpackUniforms(Uint8List bytes) {
  assert(
    bytes.lengthInBytes >= SimMarshalling.uniformByteLength,
    'uniform block too short: '
    '${bytes.lengthInBytes} < ${SimMarshalling.uniformByteLength}',
  );
  final f = Float32List.view(
    bytes.buffer,
    bytes.offsetInBytes,
    SimMarshalling.uniformSlotCount,
  );
  final u = Uint32List.view(
    bytes.buffer,
    bytes.offsetInBytes,
    SimMarshalling.uniformSlotCount,
  );
  return UniformValues(
    attractionK: f[SimMarshalling.slotAttractionK],
    repulsionK: f[SimMarshalling.slotRepulsionK],
    friction: f[SimMarshalling.slotFriction],
    dt: f[SimMarshalling.slotDt],
    worldWidth: f[SimMarshalling.slotWorldWidth],
    worldHeight: f[SimMarshalling.slotWorldHeight],
    maxRadius: f[SimMarshalling.slotMaxRadius],
    binSize: f[SimMarshalling.slotBinSize],
    gridWidth: u[SimMarshalling.slotGridWidth],
    gridHeight: u[SimMarshalling.slotGridHeight],
    numParticles: u[SimMarshalling.slotNumParticles],
    numBins: u[SimMarshalling.slotNumBins],
    typeCount: u[SimMarshalling.slotTypeCount],
  );
}

/// Returns the matrix's row-major flat buffer for SSBO upload.
///
/// A defensive copy so callers can't mutate the model's backing store. The flat
/// index of `(i, j)` is `i * dimension + j`.
Float32List flattenMatrix(TypeMatrix matrix) =>
    Float32List.fromList(matrix.values);

/// Packs per-type colours as `typeCount` RGBA tuples (`vec4`, alpha = 1).
///
/// `vec4` (not `vec3`) so the array is unambiguously 16-byte aligned in both
/// `std140` and `std430`; the render layer ([PRIMORDIS-TASK-005]) expands these
/// per particle.
Float32List packTypeColors(List<ParticleType> types) {
  final out = Float32List(types.length * SimMarshalling.colorStride);
  for (var t = 0; t < types.length; t++) {
    final base = t * SimMarshalling.colorStride;
    out[base] = types[t].r;
    out[base + 1] = types[t].g;
    out[base + 2] = types[t].b;
    out[base + 3] = 1.0;
  }
  return out;
}

/// Allocates the zero-initialized bin-count buffer (`array<atomic<u32>>`).
Uint32List newBinCounts(SimParams params) => Uint32List(params.binCount);

/// Allocates the zero-initialized bin-particles buffer (`array<u32>`), sized
/// `binCount * maxBinParticles`.
Uint32List newBinParticles(SimParams params) =>
    Uint32List(params.binCount * params.maxBinParticles);

/// Named, unpacked view of a uniform block (see [unpackUniforms]).
class UniformValues {
  const UniformValues({
    required this.attractionK,
    required this.repulsionK,
    required this.friction,
    required this.dt,
    required this.worldWidth,
    required this.worldHeight,
    required this.maxRadius,
    required this.binSize,
    required this.gridWidth,
    required this.gridHeight,
    required this.numParticles,
    required this.numBins,
    required this.typeCount,
  });

  final double attractionK;
  final double repulsionK;
  final double friction;
  final double dt;
  final double worldWidth;
  final double worldHeight;
  final double maxRadius;
  final double binSize;
  final int gridWidth;
  final int gridHeight;
  final int numParticles;
  final int numBins;
  final int typeCount;
}
