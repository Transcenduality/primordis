import 'package:flutter_test/flutter_test.dart';
import 'package:primordis/shared/constants/primordis_config.dart';
import 'package:primordis/sim/kernel/buffer_layout.dart';
import 'package:primordis/sim/models/sim_params.dart';
import 'package:primordis/sim/models/sim_seed.dart';
import 'package:primordis/sim/models/type_matrix.dart';
import 'package:primordis/sim/sim_marshalling.dart';
import 'package:primordis/sim/sim_seeder.dart';

TypeMatrix _matrix(int n, double fill) =>
    TypeMatrix.generate(n, (_, _) => fill);

SimParams _params({
  int particleCount = PrimordisConfig.particleCount,
  int typeCount = PrimordisConfig.typeCount,
}) =>
    SimParams(
      forces: _matrix(typeCount, 0.5),
      minDistances: _matrix(typeCount, 8),
      radii: _matrix(typeCount, 48),
      particleCount: particleCount,
      typeCount: typeCount,
    );

void main() {
  group('WgslStride', () {
    test('matches the WGSL element layout sizes', () {
      expect(WgslStride.f32, 4);
      expect(WgslStride.u32, 4);
      expect(WgslStride.vec2f, 8);
      expect(WgslStride.vec4f, 16);
      expect(WgslStride.atomicU32, 4);
    });
  });

  group('SimBufferLayout (default 24k / 32-type sim)', () {
    final layout = SimBufferLayout(_params());

    test('uniform block is the 64-byte marshalled struct', () {
      expect(layout.uniform, SimMarshalling.uniformByteLength);
      expect(layout.uniform, 64);
    });

    test('positions / velocities are 2 floats per particle', () {
      expect(layout.positions, 24000 * 8);
      expect(layout.velocities, 24000 * 8);
    });

    test('types are one u32 per particle', () {
      expect(layout.types, 24000 * 4);
    });

    test('the three matrices are n*n f32', () {
      expect(layout.forces, 32 * 32 * 4);
      expect(layout.minDistances, 32 * 32 * 4);
      expect(layout.radii, 32 * 32 * 4);
    });

    test('type colours are one vec4 per type', () {
      expect(layout.typeColors, 32 * 16);
    });

    test('bin counts are one atomic<u32> per bin', () {
      expect(layout.binCounts, PrimordisConfig.binCount * 4);
      expect(layout.binCounts, 77 * 4);
    });

    test('bin particles are binCount * maxBinParticles u32', () {
      expect(layout.binParticles, 77 * 512 * 4);
    });

    test('total is the sum of every buffer', () {
      expect(
        layout.total,
        layout.uniform +
            layout.positions +
            layout.velocities +
            layout.types +
            layout.forces +
            layout.minDistances +
            layout.radii +
            layout.typeColors +
            layout.binCounts +
            layout.binParticles,
      );
    });
  });

  group('layout agrees with the seeded buffers and marshalling', () {
    // Cross-check against real seeded/marshalled buffers at a small size so the
    // layout, the SimSeeder, and sim_marshalling can never drift apart.
    const n = 8;
    const count = 200;
    final params = _params(particleCount: count, typeCount: n);
    final layout = SimBufferLayout(params);
    final seeded = seedSimulation(
      const SimSeed(seed: 7, particleCount: count, typeCount: n),
    );

    test('positions / velocities / types byte sizes match SeededSim', () {
      expect(layout.positions, seeded.positions.lengthInBytes);
      expect(layout.velocities, seeded.velocities.lengthInBytes);
      expect(layout.types, seeded.types.lengthInBytes);
    });

    test('matrix byte sizes match flattenMatrix output', () {
      expect(layout.forces, flattenMatrix(seeded.forces).lengthInBytes);
      expect(layout.minDistances, flattenMatrix(seeded.minDistances).lengthInBytes);
      expect(layout.radii, flattenMatrix(seeded.radii).lengthInBytes);
    });

    test('type colours byte size matches packTypeColors output', () {
      expect(layout.typeColors, packTypeColors(seeded.particleTypes).lengthInBytes);
    });

    test('bin buffer byte sizes match the marshalling allocators', () {
      expect(layout.binCounts, newBinCounts(params).lengthInBytes);
      expect(layout.binParticles, newBinParticles(params).lengthInBytes);
    });

    test('uniform byte size matches packUniforms output', () {
      expect(layout.uniform, packUniforms(params, 0.016).lengthInBytes);
    });
  });

  group('layout scales with reduced-mode counts', () {
    test('halving particleCount halves the per-particle buffers', () {
      final full = SimBufferLayout(_params()); // default 24,000
      final half = SimBufferLayout(_params(particleCount: 12000));
      expect(half.positions, full.positions ~/ 2);
      expect(half.velocities, full.velocities ~/ 2);
      expect(half.types, full.types ~/ 2);
      // Matrices/colours depend on typeCount, not particleCount.
      expect(half.forces, full.forces);
      expect(half.typeColors, full.typeColors);
    });
  });
}
