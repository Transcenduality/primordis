import 'package:flutter_test/flutter_test.dart';
import 'package:primordis/sim/cpu/particle_soa.dart';
import 'package:primordis/sim/models/sim_seed.dart';
import 'package:primordis/sim/sim_seeder.dart';

void main() {
  group('ParticleSoa', () {
    test('allocates SoA buffers at the requested sizes', () {
      final soa = ParticleSoa(particleCount: 100, typeCount: 32, binCount: 77);
      expect(soa.positions.length, 200);
      expect(soa.velocities.length, 200);
      expect(soa.types.length, 100);
      expect(soa.renderXY.length, 200);
      expect(soa.binCounts.length, 77);
      expect(soa.binStarts.length, 77);
      expect(soa.sortedIndices.length, 100);
    });

    test('loadFrom copies the seeded working set in place', () {
      const seed = SimSeed(seed: 7, particleCount: 64);
      final seeded = seedSimulation(seed);
      final soa = ParticleSoa(particleCount: 64, typeCount: 32, binCount: 77)
        ..loadFrom(seeded);

      expect(soa.positions, orderedEquals(seeded.positions));
      expect(soa.velocities, orderedEquals(seeded.velocities));
      for (var i = 0; i < 64; i++) {
        expect(soa.types[i], seeded.types[i]);
      }
    });

    test('buffers are reused (same instances) across loadFrom calls', () {
      const seed = SimSeed(particleCount: 32);
      final soa = ParticleSoa(particleCount: 32, typeCount: 32, binCount: 77);
      final positionsRef = soa.positions;
      final typesRef = soa.types;

      soa.loadFrom(seedSimulation(seed));

      expect(identical(soa.positions, positionsRef), isTrue);
      expect(identical(soa.types, typesRef), isTrue);
    });

    test('exposes the injectable SimBuffers interface', () {
      final SimBuffers buffers =
          ParticleSoa(particleCount: 10, typeCount: 32, binCount: 77);
      expect(buffers.particleCount, 10);
      expect(buffers.binCount, 77);
    });
  });
}
