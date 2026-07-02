import 'package:flutter_test/flutter_test.dart';
import 'package:primordis/features/simulation/providers/sim_params_provider.dart';
import 'package:primordis/sim/fake_sim_backend.dart';
import 'package:primordis/sim/models/sim_seed.dart';
import 'package:primordis/sim/providers/sim_providers.dart';
import 'package:riverpod/riverpod.dart';

ProviderContainer _containerWithBackend(FakeSimBackend backend) {
  final container = ProviderContainer(
    overrides: [simBackendProvider.overrideWith((ref) => backend)],
  );
  addTearDown(container.dispose);
  return container;
}

void main() {
  group('reducedMotionController', () {
    test('defaults to false', () {
      final container = ProviderContainer();
      addTearDown(container.dispose);
      expect(container.read(reducedMotionControllerProvider), isFalse);
    });

    test('set() updates the flag', () {
      final container = ProviderContainer();
      addTearDown(container.dispose);
      container.read(reducedMotionControllerProvider.notifier).set(true);
      expect(container.read(reducedMotionControllerProvider), isTrue);
    });

    test('set() is idempotent for an unchanged value (no rebuild storm)', () {
      final container = ProviderContainer();
      addTearDown(container.dispose);
      var rebuilds = 0;
      container.listen(
        reducedMotionControllerProvider,
        (prev, next) => rebuilds++,
      );
      container.read(reducedMotionControllerProvider.notifier).set(false);
      expect(rebuilds, 0);
    });
  });

  group('simRunnerController', () {
    test('brings the backend up through init() then seed()', () async {
      final backend = FakeSimBackend();
      final container = _containerWithBackend(backend);

      // Force the async build to run and complete.
      await container.read(simRunnerControllerProvider.future);

      expect(backend.calls, [FakeSimCall.init, FakeSimCall.seed]);
      expect(backend.isInitialized, isTrue);
    });

    test('reseed() re-runs only seed(), not another init()', () async {
      final backend = FakeSimBackend();
      final container = _containerWithBackend(backend);
      await container.read(simRunnerControllerProvider.future);

      const newSeed = SimSeed(seed: 42);
      await container
          .read(simRunnerControllerProvider.notifier)
          .reseed(newSeed);

      expect(backend.calls, [FakeSimCall.init, FakeSimCall.seed, FakeSimCall.seed]);
      expect(backend.lastSeed, newSeed);
    });

    test(
        'mutating the seed controller (the ControlPanel reseed path) never '
        're-runs build()/init()', () async {
      final backend = FakeSimBackend();
      final container = _containerWithBackend(backend);
      await container.read(simRunnerControllerProvider.future);
      expect(backend.initCount, 1);

      // Mirror ControlPanel._reseed: mutate the seed provider, then call the
      // runner's reseed action with the new seed.
      container.read(simSeedControllerProvider.notifier).reseed(77);
      final seed = container.read(simSeedControllerProvider);
      await container.read(simRunnerControllerProvider.notifier).reseed(seed);
      // Let any (incorrect) provider rebuild settle before asserting.
      await container.read(simRunnerControllerProvider.future);

      expect(
        backend.initCount,
        1,
        reason: 'init() must run exactly once per backend lifecycle — '
            'a reseed is a seed()-only operation',
      );
      expect(backend.seedCount, 2);
      expect(backend.lastSeed, seed);
    });

    test('exposes AsyncData once bring-up completes', () async {
      final backend = FakeSimBackend();
      final container = _containerWithBackend(backend);
      await container.read(simRunnerControllerProvider.future);

      expect(
        container.read(simRunnerControllerProvider).hasValue,
        isTrue,
      );
    });
  });
}
