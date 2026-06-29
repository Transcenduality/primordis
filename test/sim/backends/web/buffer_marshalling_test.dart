import 'package:flutter_test/flutter_test.dart';
import 'package:primordis/sim/backends/web/buffer_marshalling.dart';
import 'package:primordis/sim/kernel/buffer_layout.dart';
import 'package:primordis/sim/models/sim_params.dart';
import 'package:primordis/sim/models/sim_seed.dart';
import 'package:primordis/sim/sim_marshalling.dart';
import 'package:primordis/sim/sim_seeder.dart';

/// The web backend marshals the seed + uniform into GPU buffers; if those bytes
/// don't match `buffer_layout.dart` exactly the sim silently corrupts
/// ([PRIMORDIS-TASK-003]). This pins the composition: every [SeedBuffers]
/// payload fills its [SimBufferLayout] buffer, and the per-frame uniform
/// round-trips. Platform-neutral, so it runs in `flutter test` with no browser.
void main() {
  // A small, fast sim that still exercises every buffer (counts differ so a
  // swapped field would change a length).
  const seed = SimSeed(seed: 7, particleCount: 100, typeCount: 8);
  final seeded = seedSimulation(seed);
  final params = SimParams(
    forces: seeded.forces,
    minDistances: seeded.minDistances,
    radii: seeded.radii,
    particleCount: seeded.particleCount,
    typeCount: seeded.typeCount,
  );
  final layout = SimBufferLayout(params);

  group('packSeedBuffers', () {
    final buffers = packSeedBuffers(seeded);

    test('every payload byte length equals its SimBufferLayout field', () {
      expect(buffers.positions.lengthInBytes, layout.positions);
      expect(buffers.velocities.lengthInBytes, layout.velocities);
      expect(buffers.types.lengthInBytes, layout.types);
      expect(buffers.forces.lengthInBytes, layout.forces);
      expect(buffers.minDistances.lengthInBytes, layout.minDistances);
      expect(buffers.radii.lengthInBytes, layout.radii);
      expect(buffers.typeColors.lengthInBytes, layout.typeColors);
    });

    test('verifySeedBuffersMatchLayout accepts the matched layout', () {
      // Runs the backend's debug-time contract guard; throws on mismatch.
      expect(() => verifySeedBuffersMatchLayout(buffers, layout),
          returnsNormally);
    });

    test('SoA payloads pass the seeded particle data straight through', () {
      expect(buffers.positions, same(seeded.positions));
      expect(buffers.velocities, same(seeded.velocities));
      expect(buffers.types, same(seeded.types));
    });

    test('matrices flatten row-major and colours pack RGBA (alpha 1)', () {
      // forces flattened i*n+j; spot-check a couple of cells against the matrix
      // (float32 storage → compare with tolerance).
      expect(buffers.forces[0], closeTo(seeded.forces.at(0, 0), 1e-6));
      expect(buffers.forces[seeded.typeCount + 1],
          closeTo(seeded.forces.at(1, 1), 1e-6));
      // colours: 4 floats per type, alpha == 1 (exact in float32).
      expect(buffers.typeColors.length, seeded.typeCount * 4);
      expect(buffers.typeColors[3], 1.0);
      expect(buffers.typeColors[0], closeTo(seeded.particleTypes[0].r, 1e-6));
    });
  });

  group('packFrameUniform', () {
    test('round-trips sliders + dt + counts through the 64-byte block', () {
      final live = params.copyWith(
        attractionK: 12.5,
        repulsionK: 3.25,
        friction: 0.5,
      );
      final bytes = packFrameUniform(live, 1 / 60);
      expect(bytes.lengthInBytes, SimMarshalling.uniformByteLength);
      expect(bytes.lengthInBytes, layout.uniform);

      final v = unpackUniforms(bytes);
      expect(v.attractionK, closeTo(12.5, 1e-6));
      expect(v.repulsionK, closeTo(3.25, 1e-6));
      expect(v.friction, closeTo(0.5, 1e-6));
      expect(v.dt, closeTo(1 / 60, 1e-6));
      expect(v.numParticles, 100);
      expect(v.typeCount, 8);
      expect(v.numBins, params.binCount);
    });
  });
}
