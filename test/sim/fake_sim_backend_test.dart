import 'package:flutter_test/flutter_test.dart';
import 'package:primordis/shared/constants/primordis_config.dart';
import 'package:primordis/sim/fake_sim_backend.dart';
import 'package:primordis/sim/models/sim_params.dart';
import 'package:primordis/sim/models/sim_seed.dart';
import 'package:primordis/sim/models/type_matrix.dart';
import 'package:primordis/sim/sim_seeder.dart';

TypeMatrix _matrix(double fill) =>
    TypeMatrix.generate(PrimordisConfig.typeCount, (_, _) => fill);

void main() {
  group('FakeSimBackend', () {
    test('records the full lifecycle in order', () async {
      final backend = FakeSimBackend();
      await backend.init();
      await backend.seed(const SimSeed(particleCount: 100));
      backend.setParams(SimParams(
        forces: _matrix(0.5),
        minDistances: _matrix(8),
        radii: _matrix(48),
      ));
      backend.step(0.016);
      backend.present();
      await backend.dispose();

      expect(backend.calls, [
        FakeSimCall.init,
        FakeSimCall.seed,
        FakeSimCall.setParams,
        FakeSimCall.step,
        FakeSimCall.present,
        FakeSimCall.dispose,
      ]);
    });

    test('tracks init/dispose state', () async {
      final backend = FakeSimBackend();
      expect(backend.isInitialized, isFalse);
      await backend.init();
      expect(backend.isInitialized, isTrue);
      expect(backend.isDisposed, isFalse);
      await backend.dispose();
      expect(backend.isInitialized, isFalse);
      expect(backend.isDisposed, isTrue);
    });

    test('seed runs the shared seeder deterministically', () async {
      final backend = FakeSimBackend();
      const seed = SimSeed(seed: 7, particleCount: 500);
      await backend.seed(seed);

      expect(backend.lastSeed, seed);
      final seeded = backend.lastSeeded;
      expect(seeded, isNotNull);
      // Matches a direct seeder run (deterministic).
      final expected = seedSimulation(seed);
      expect(seeded!.positions, equals(expected.positions));
      expect(seeded.forces.values, equals(expected.forces.values));
    });

    test('step accumulates simulated time and frame count', () {
      final backend = FakeSimBackend();
      backend.step(0.1);
      backend.step(0.2);
      expect(backend.stepCount, 2);
      expect(backend.lastDt, 0.2);
      expect(backend.simulatedTime, closeTo(0.3, 1e-9));
    });

    test('reports non-GPU capabilities with the reference ceiling', () {
      final caps = FakeSimBackend().capabilities;
      expect(caps.isGpuAccelerated, isFalse);
      expect(caps.maxParticles, PrimordisConfig.particleCount);
      expect(caps.defaultParticleCount, PrimordisConfig.particleCount);
      expect(caps.label, 'fake');
    });
  });
}
