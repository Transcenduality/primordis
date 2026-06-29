import 'dart:typed_data';

import 'package:primordis/sim/kernel/buffer_layout.dart';
import 'package:primordis/sim/models/seeded_sim.dart';
import 'package:primordis/sim/models/sim_params.dart';
import 'package:primordis/sim/sim_marshalling.dart';

/// Composes the shared host marshalling ([sim_marshalling.dart]) into the exact
/// per-buffer byte payloads the web WebGPU backend uploads with
/// `GPUQueue.writeBuffer`.
///
/// This file is deliberately **platform-neutral** (`dart:typed_data` and the
/// project models only — no `dart:js_interop`, no `package:web`), so the
/// marshalling round-trip is unit-testable in `flutter test` even though the
/// backend that consumes it ([web_webgpu_backend.dart]) is web-only. The actual
/// `writeBuffer` upload happens in the backend, which holds the GPU handle; this
/// file only produces the bytes and guarantees they match the [SimBufferLayout]
/// the kernel binds ([PRIMORDIS-TASK-003] / [PRIMORDIS-ADR-003]).
///
/// Two entry points cover the two upload moments:
///
/// - [packSeedBuffers] — the one-time SoA / matrix / colour payload uploaded at
///   `seed()` time.
/// - [packFrameUniform] — the 64-byte uniform block written every frame (sliders
///   + `dt` + counts), the hot-path marshalling entry the frame loop drives
///   ([PRIMORDIS-TASK-006] wires the live slider state into it).
///
/// The bin buffers (`binCounts`, `binParticles`) carry no seed payload — the
/// kernel zeroes and fills them each frame — so they are sized from
/// [SimBufferLayout] by the backend and never marshalled here.

/// The static, per-seed buffer payloads, one typed list per kernel binding that
/// has initial data. Field byte lengths equal the matching [SimBufferLayout]
/// fields (asserted by [verifySeedBuffersMatchLayout]).
class SeedBuffers {
  const SeedBuffers({
    required this.positions,
    required this.velocities,
    required this.types,
    required this.forces,
    required this.minDistances,
    required this.radii,
    required this.typeColors,
  });

  /// Interleaved `x, y` positions (`array<vec2<f32>>`); length `2 * count`.
  final Float32List positions;

  /// Interleaved `x, y` velocities (`array<vec2<f32>>`); length `2 * count`.
  final Float32List velocities;

  /// Per-particle type index (`array<u32>`); length `count`.
  final Uint32List types;

  /// Row-major signed force matrix (`array<f32>`); length `typeCount^2`.
  final Float32List forces;

  /// Row-major repulsion-onset matrix (`array<f32>`); length `typeCount^2`.
  final Float32List minDistances;

  /// Row-major attraction-cutoff matrix (`array<f32>`); length `typeCount^2`.
  final Float32List radii;

  /// Per-type RGBA colours (`array<vec4<f32>>`); length `4 * typeCount`.
  final Float32List typeColors;
}

/// Builds the per-seed upload payloads from a deterministic [seeded] result.
///
/// The matrices are flattened row-major and the colours packed RGBA via the
/// shared [sim_marshalling.dart] verbs, so the web backend and the native
/// backend ([PRIMORDIS-TASK-011]) marshal identically. The position/velocity/
/// type SoA buffers come straight from the seeder (already in the interleaved /
/// `u32` layout the kernel binds).
SeedBuffers packSeedBuffers(SeededSim seeded) => SeedBuffers(
      positions: seeded.positions,
      velocities: seeded.velocities,
      types: seeded.types,
      forces: flattenMatrix(seeded.forces),
      minDistances: flattenMatrix(seeded.minDistances),
      radii: flattenMatrix(seeded.radii),
      typeColors: packTypeColors(seeded.particleTypes),
    );

/// Packs the 64-byte per-frame uniform block (sliders + [dt] + world/grid
/// counts) — the hot-path entry point the frame loop calls every `step`.
///
/// A thin, intentionally-named delegate to [packUniforms] so the backend depends
/// only on this marshalling library, not on the lower-level packer directly.
Uint8List packFrameUniform(SimParams params, double dt) =>
    packUniforms(params, dt);

/// Asserts every [SeedBuffers] payload's byte length equals the corresponding
/// [SimBufferLayout] field, i.e. the marshalled data exactly fills the GPU
/// buffers the backend allocates. A layout/marshalling drift here would silently
/// corrupt the simulation, so this is both a debug guard in the backend and the
/// contract the marshalling round-trip test pins ([PRIMORDIS-TASK-003]).
void verifySeedBuffersMatchLayout(SeedBuffers buffers, SimBufferLayout layout) {
  void check(String name, int actualBytes, int expectedBytes) {
    assert(
      actualBytes == expectedBytes,
      '$name byte length $actualBytes != layout $expectedBytes',
    );
  }

  check('positions', buffers.positions.lengthInBytes, layout.positions);
  check('velocities', buffers.velocities.lengthInBytes, layout.velocities);
  check('types', buffers.types.lengthInBytes, layout.types);
  check('forces', buffers.forces.lengthInBytes, layout.forces);
  check('minDistances', buffers.minDistances.lengthInBytes, layout.minDistances);
  check('radii', buffers.radii.lengthInBytes, layout.radii);
  check('typeColors', buffers.typeColors.lengthInBytes, layout.typeColors);
}
