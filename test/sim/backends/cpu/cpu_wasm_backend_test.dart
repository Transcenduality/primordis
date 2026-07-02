import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:primordis/shared/constants/primordis_config.dart';
import 'package:primordis/sim/backends/cpu/cpu_wasm_backend.dart';
import 'package:primordis/sim/models/sim_params.dart';
import 'package:primordis/sim/models/sim_seed.dart';
import 'package:primordis/sim/render/cpu_points_painter.dart';
import 'package:primordis/sim/sim_seeder.dart';

/// A default params block sized for [count] particles / 32 types, using the
/// matrices from a matching seed.
SimParams _paramsFor(int count, {int seed = 1}) {
  final seeded =
      seedSimulation(SimSeed(seed: seed, particleCount: count));
  return SimParams(
    forces: seeded.forces,
    minDistances: seeded.minDistances,
    radii: seeded.radii,
    particleCount: count,
  );
}

void main() {
  group('CpuWasmBackend capabilities', () {
    test('advertises the T4 tier as CPU with the ADR-006 ceiling', () {
      final backend = CpuWasmBackend();
      final caps = backend.capabilities;
      expect(caps.isGpuAccelerated, isFalse);
      expect(caps.maxParticles, PrimordisConfig.cpuWasmMaxParticleCount);
      expect(
        caps.defaultParticleCount,
        PrimordisConfig.cpuWasmDefaultParticleCount,
      );
      expect(caps.label, 'web-cpu-wasm');
    });

    test('default count is honest (3-4k band, never the reference 24k)', () {
      const def = PrimordisConfig.cpuWasmDefaultParticleCount;
      expect(def, inInclusiveRange(3000, 4000));
      expect(def, isNot(PrimordisConfig.particleCount)); // not 24k
      expect(
        PrimordisConfig.cpuWasmMaxParticleCount,
        lessThan(PrimordisConfig.particleCount),
      );
    });
  });

  group('lifecycle', () {
    test('init then dispose flips the state flags', () async {
      final backend = CpuWasmBackend();
      expect(backend.isInitialized, isFalse);
      await backend.init();
      expect(backend.isInitialized, isTrue);
      expect(backend.isDisposed, isFalse);
      await backend.dispose();
      expect(backend.isInitialized, isFalse);
      expect(backend.isDisposed, isTrue);
    });

    test('actualParticleCount is 0 before seeding, honest after', () async {
      final backend = CpuWasmBackend();
      await backend.init();
      expect(backend.actualParticleCount, 0);

      await backend.seed(const SimSeed(particleCount: 500));
      expect(backend.actualParticleCount, 500);
      await backend.dispose();
    });

    test('seeding clamps a 24k request down to the tier ceiling', () async {
      final backend = CpuWasmBackend();
      await backend.init();
      // The default SimSeed count is the reference 24k; the tier must clamp it.
      await backend.seed(const SimSeed());
      expect(
        backend.actualParticleCount,
        PrimordisConfig.cpuWasmMaxParticleCount,
      );
      await backend.dispose();
    });
  });

  group('present / render seam', () {
    test('renderBuffer length matches 2 * actualParticleCount', () async {
      final backend = CpuWasmBackend();
      await backend.init();
      await backend.seed(const SimSeed(particleCount: 300));
      backend.setParams(_paramsFor(300));
      backend.step(1 / 60);
      backend.present();

      expect(backend.renderBuffer.length, 2 * backend.actualParticleCount);
      await backend.dispose();
    });

    test('publishes a frame whose per-type draw count is <= typeCount', () async {
      final backend = CpuWasmBackend();
      await backend.init();
      await backend.seed(const SimSeed(particleCount: 300));
      backend.setParams(_paramsFor(300));
      backend.present();

      final CpuFrame frame = backend.frame.value;
      expect(frame.pointsByType.length, 32);
      // Total points across all per-type buffers == 2 * count.
      final total =
          frame.pointsByType.fold<int>(0, (a, b) => a + b.length);
      expect(total, 2 * backend.actualParticleCount);
      await backend.dispose();
    });

    test('present buffers are stable across a following present (double-buffer)',
        () async {
      final backend = CpuWasmBackend();
      await backend.init();
      await backend.seed(const SimSeed(particleCount: 200));
      backend.setParams(_paramsFor(200));

      backend.present();
      final first = backend.frame.value;
      final firstSnapshot = <Float32List>[
        for (final b in first.pointsByType) Float32List.fromList(b),
      ];

      // Step + present again; the previously published frame's buffers must not
      // have been overwritten (they were the OTHER set of the double-buffer).
      backend.step(1 / 60);
      backend.present();

      for (var t = 0; t < first.pointsByType.length; t++) {
        expect(first.pointsByType[t], orderedEquals(firstSnapshot[t]),
            reason: 'published frame $t mutated by the next present');
      }
      await backend.dispose();
    });
  });

  group('pause (reduced-motion)', () {
    test('pause suppresses stepping and holds the last frame', () async {
      final backend = CpuWasmBackend();
      await backend.init();
      await backend.seed(const SimSeed(particleCount: 200));
      backend.setParams(_paramsFor(200));
      backend.step(1 / 60);
      backend.present();

      final held = Float32List.fromList(backend.renderBuffer);
      backend.pause();
      expect(backend.paused, isTrue);

      // Many steps while paused -> positions unchanged.
      for (var f = 0; f < 10; f++) {
        backend.step(1 / 60);
      }
      backend.present();
      expect(backend.renderBuffer, orderedEquals(held));

      // Resume advances again.
      backend.resume();
      backend.step(1 / 60);
      backend.present();
      expect(backend.renderBuffer, isNot(orderedEquals(held)));
      await backend.dispose();
    });
  });

  group('live params affect the step immediately (no reseed)', () {
    test('changing friction changes the trajectory without reseeding', () async {
      Future<Float32List> runWith(double friction) async {
        final backend = CpuWasmBackend();
        await backend.init();
        await backend.seed(const SimSeed(seed: 5, particleCount: 300));
        backend.setParams(_paramsFor(300, seed: 5).copyWith(friction: friction));
        for (var f = 0; f < 20; f++) {
          backend.step(1 / 60);
        }
        backend.present();
        final out = Float32List.fromList(backend.renderBuffer);
        await backend.dispose();
        return out;
      }

      final slow = await runWith(0.10);
      final fast = await runWith(0.90);
      expect(slow, isNot(orderedEquals(fast)));
    });

    test('applying params after seed does not reseed (count unchanged)',
        () async {
      final backend = CpuWasmBackend();
      await backend.init();
      await backend.seed(const SimSeed(particleCount: 250));
      final before = backend.actualParticleCount;
      backend.setParams(_paramsFor(250).copyWith(attractionK: 99));
      expect(backend.actualParticleCount, before);
      await backend.dispose();
    });
  });

  group('determinism', () {
    test('same seed + params -> identical positions after N frames', () async {
      Future<Float32List> run() async {
        final backend = CpuWasmBackend();
        await backend.init();
        await backend.seed(const SimSeed(seed: 42, particleCount: 400));
        backend.setParams(_paramsFor(400, seed: 42));
        for (var f = 0; f < 30; f++) {
          backend.step(1 / 60);
        }
        backend.present();
        final out = Float32List.fromList(backend.renderBuffer);
        await backend.dispose();
        return out;
      }

      expect(await run(), orderedEquals(await run()));
    });
  });
}
