import 'dart:async';
import 'dart:math';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:primordis/features/simulation/providers/sim_params_provider.dart';
import 'package:primordis/features/simulation/widgets/labeled_slider.dart';
import 'package:primordis/sim/models/sim_params.dart';
import 'package:primordis/sim/providers/sim_providers.dart';

/// The simulation's chrome: the three live sliders plus reset/seed and
/// play/pause controls ([PRIMORDIS-TASK-006]).
///
/// Fully backend-agnostic: every control reads/writes Riverpod providers
/// backed by [SimParamsController] / [SimSeedController] /
/// [RunStateController] / [SimRunnerController]; it never imports a concrete
/// [SimBackend] implementation, so the identical widget drives web GPU, web
/// CPU, native GPU, and native CPU backends alike ([PRIMORDIS-ADR-001]).
///
/// A [ConsumerWidget] (not `StatefulWidget`): there is no local mutable state
/// here — every value comes from `ref.watch`, and every mutation goes through
/// a provider method (house standard — no `setState` for business logic).
class ControlPanel extends ConsumerWidget {
  const ControlPanel({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final params = ref.watch(simParamsControllerProvider);
    final runState = ref.watch(runStateControllerProvider);
    final paramsController = ref.read(simParamsControllerProvider.notifier);
    final theme = Theme.of(context);

    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Controls', style: theme.textTheme.titleMedium),
        const SizedBox(height: 8),
        LabeledSlider(
          label: 'Attraction K',
          value: params.attractionK,
          min: SimSliders.attractionMin,
          max: SimSliders.attractionMax,
          tooltip: 'Scales how strongly particles pull toward attractors '
              'within their interaction radius.',
          onChanged: paramsController.setAttractionK,
        ),
        LabeledSlider(
          label: 'Repulsion K',
          value: params.repulsionK,
          min: SimSliders.repulsionMin,
          max: SimSliders.repulsionMax,
          tooltip: 'Scales short-range repulsion strength when particles '
              'get closer than their minimum distance.',
          onChanged: paramsController.setRepulsionK,
        ),
        LabeledSlider(
          label: 'Drift',
          value: params.friction,
          min: SimSliders.frictionMin,
          max: SimSliders.frictionMax,
          tooltip: 'Per-step velocity retention (friction). Lower values '
              'damp motion faster.',
          onChanged: paramsController.setFriction,
        ),
        const SizedBox(height: 12),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            Tooltip(
              message: 'Reset the sliders above to their default values '
                  'without changing particles or colours.',
              child: OutlinedButton.icon(
                onPressed: paramsController.resetToDefaults,
                icon: const Icon(Icons.restart_alt),
                label: const Text('Reset params'),
              ),
            ),
            Tooltip(
              message: 'Reseed: regenerate particles, colours, and the '
                  'force/distance/radius matrices.',
              child: OutlinedButton.icon(
                onPressed: () => _reseed(ref),
                icon: const Icon(Icons.shuffle),
                label: const Text('Reseed'),
              ),
            ),
            Tooltip(
              message: runState.isPaused
                  ? 'Resume the simulation.'
                  : 'Pause the simulation (holds the last frame). This is '
                      'the reduced-motion affordance.',
              child: FilledButton.icon(
                onPressed: ref.read(runStateControllerProvider.notifier).togglePause,
                icon: Icon(runState.isPaused ? Icons.play_arrow : Icons.pause),
                label: Text(runState.isPaused ? 'Play' : 'Pause'),
              ),
            ),
          ],
        ),
      ],
    );
  }

  void _reseed(WidgetRef ref) {
    final nextSeed = Random().nextInt(1 << 31);
    ref.read(simSeedControllerProvider.notifier).reseed(nextSeed);
    final seed = ref.read(simSeedControllerProvider);
    // Reseeding the backend is async (storage-buffer rewrite); the tap
    // handler itself is not. A failed reseed leaves the previous particles
    // live — a safe degraded state — rather than crashing the UI; a
    // user-visible error surface is tracked as a follow-up.
    unawaited(
      ref.read(simRunnerControllerProvider.notifier).reseed(seed),
    );
  }
}
