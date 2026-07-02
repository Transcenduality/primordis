import 'package:flutter_test/flutter_test.dart';
import 'package:primordis/sim/fake_sim_backend.dart';
import 'package:primordis/sim/frame_loop.dart';
import 'package:primordis/sim/models/sim_params.dart';
import 'package:primordis/sim/providers/sim_providers.dart';
import 'package:riverpod/riverpod.dart';

ProviderContainer _container() {
  final container = ProviderContainer();
  addTearDown(container.dispose);
  return container;
}

/// A container with [simBackendProvider] overridden to [backend] — the seam
/// backend selection ([PRIMORDIS-TASK-007] / [PRIMORDIS-TASK-015]) uses to inject
/// a concrete backend. The list literal infers the override type.
ProviderContainer _containerWithBackend(FakeSimBackend backend) {
  final container = ProviderContainer(
    overrides: [simBackendProvider.overrideWith((ref) => backend)],
  );
  addTearDown(container.dispose);
  return container;
}

void main() {
  group('simSeedController', () {
    test('defaults, reseed, and particle-count mutation', () {
      final c = _container();
      expect(c.read(simSeedControllerProvider).seed, 1);

      c.read(simSeedControllerProvider.notifier).reseed(5);
      expect(c.read(simSeedControllerProvider).seed, 5);

      c.read(simSeedControllerProvider.notifier).setParticleCount(1234);
      expect(c.read(simSeedControllerProvider).particleCount, 1234);
      // Seeding flows through.
      expect(c.read(seededSimProvider).particleCount, 1234);
    });
  });

  group('simParamsController', () {
    test('exposes seeded matrices with default sliders', () {
      final c = _container();
      final params = c.read(simParamsControllerProvider);
      final seeded = c.read(seededSimProvider);
      expect(params.attractionK, SimSliders.attractionDefault);
      expect(params.forces, equals(seeded.forces));
    });

    test('slider mutations update SimParams', () {
      final c = _container();
      c.read(simParamsControllerProvider.notifier).setAttractionK(50);
      expect(c.read(simParamsControllerProvider).attractionK, 50);

      c.read(simParamsControllerProvider.notifier).setRepulsionK(10);
      expect(c.read(simParamsControllerProvider).repulsionK, 10);

      c.read(simParamsControllerProvider.notifier).setFriction(0.7);
      expect(c.read(simParamsControllerProvider).friction, 0.7);
    });

    test('slider mutations clamp to SimSliders bounds', () {
      final c = _container();
      final notifier = c.read(simParamsControllerProvider.notifier);

      notifier.setAttractionK(10000);
      expect(c.read(simParamsControllerProvider).attractionK,
          SimSliders.attractionMax);

      notifier.setFriction(0); // below min
      expect(
          c.read(simParamsControllerProvider).friction, SimSliders.frictionMin);
    });

    test('reseeding rebuilds params with fresh matrices', () {
      final c = _container();
      final before = c.read(simParamsControllerProvider).forces;
      c.read(simSeedControllerProvider.notifier).reseed(99);
      final after = c.read(simParamsControllerProvider).forces;
      expect(before, isNot(equals(after)));
    });

    test('resetToDefaults restores slider defaults without touching matrices',
        () {
      final c = _container();
      final notifier = c.read(simParamsControllerProvider.notifier);
      final matricesBefore = c.read(simParamsControllerProvider).forces;

      notifier.setAttractionK(99);
      notifier.setRepulsionK(1);
      notifier.setFriction(0.9);
      notifier.resetToDefaults();

      final params = c.read(simParamsControllerProvider);
      expect(params.attractionK, SimSliders.attractionDefault);
      expect(params.repulsionK, SimSliders.repulsionDefault);
      expect(params.friction, SimSliders.frictionDefault);
      // Reset is uniform-only: the matrices are untouched (contrast with
      // reseed, which regenerates them).
      expect(params.forces, equals(matricesBefore));
    });
  });

  group('simBackend', () {
    test('defaults to a FakeSimBackend, stable across reads', () {
      final c = _container();
      final a = c.read(simBackendProvider);
      final b = c.read(simBackendProvider);
      expect(a, isA<FakeSimBackend>());
      expect(a, same(b));
    });

    test('can be overridden to inject a concrete backend', () {
      final injected = FakeSimBackend();
      final c = _containerWithBackend(injected);
      expect(c.read(simBackendProvider), same(injected));
    });

    test('frameLoop binds to the active backend', () {
      final injected = FakeSimBackend();
      final c = _containerWithBackend(injected);
      final loop = c.read(frameLoopProvider);
      expect(loop, isA<FrameLoop>());
      // Driving the loop reaches the injected backend.
      loop.tick(
        dt: 0.016,
        params: c.read(simParamsControllerProvider),
        paused: false,
      );
      expect(injected.stepCount, 1);
    });
  });

  group('runStateController', () {
    test('default state is running, not paused', () {
      final c = _container();
      final s = c.read(runStateControllerProvider);
      expect(s.isRunning, isTrue);
      expect(s.isPaused, isFalse);
      expect(s.frame, 0);
    });

    test('pause / resume / toggle transitions', () {
      final c = _container();
      final notifier = c.read(runStateControllerProvider.notifier);

      notifier.pause();
      expect(c.read(runStateControllerProvider).isPaused, isTrue);

      notifier.resume();
      expect(c.read(runStateControllerProvider).isPaused, isFalse);

      notifier.togglePause();
      expect(c.read(runStateControllerProvider).isPaused, isTrue);
    });

    test('frame advance and reset', () {
      final c = _container();
      final notifier = c.read(runStateControllerProvider.notifier);

      notifier.markFrameStepped();
      notifier.markFrameStepped();
      expect(c.read(runStateControllerProvider).frame, 2);

      notifier.stop();
      expect(c.read(runStateControllerProvider).isRunning, isFalse);

      notifier.reset();
      final s = c.read(runStateControllerProvider);
      expect(s.frame, 0);
      expect(s.isRunning, isTrue);
    });
  });
}
