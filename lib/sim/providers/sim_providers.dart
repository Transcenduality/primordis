import 'dart:async';

import 'package:primordis/sim/fake_sim_backend.dart';
import 'package:primordis/sim/frame_loop.dart';
import 'package:primordis/sim/models/run_state.dart';
import 'package:primordis/sim/models/seeded_sim.dart';
import 'package:primordis/sim/models/sim_params.dart';
import 'package:primordis/sim/models/sim_seed.dart';
import 'package:primordis/sim/sim_backend.dart';
import 'package:primordis/sim/sim_seeder.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'sim_providers.g.dart';

/// Riverpod providers that expose the shared sim core to the UI.
///
/// All state is managed with plain `Ref` Riverpod — no `setState`/
/// `ChangeNotifier` for business logic (house standard, [PRIMORDIS-ADR-001]).
/// The active backend sits behind [simBackendProvider] so backend selection
/// ([PRIMORDIS-TASK-007] / [PRIMORDIS-TASK-015]) can swap a concrete GPU/CPU
/// implementation in via `overrideWith` without any UI change.

/// The current [SimSeed]. Mutating it (reseed / particle-count change)
/// re-derives [seededSimProvider] and, in turn, [simParamsControllerProvider].
@riverpod
class SimSeedController extends _$SimSeedController {
  @override
  SimSeed build() => const SimSeed();

  /// Reseeds from [seed] (new particles, matrices, and colours).
  void reseed(int seed) => state = state.copyWith(seed: seed);

  /// Sets the particle count (reduced-mode tiers, [PRIMORDIS-ADR-006]).
  void setParticleCount(int particleCount) =>
      state = state.copyWith(particleCount: particleCount);
}

/// Deterministic seeding output for the current [SimSeed]. Pure/derived.
@riverpod
SeededSim seededSim(Ref ref) {
  final seed = ref.watch(simSeedControllerProvider);
  return seedSimulation(seed);
}

/// The current [SimParams]: the seeded matrices plus the live sliders.
///
/// Built from [seededSimProvider], so a reseed rebuilds it with fresh matrices
/// and slider defaults. Slider mutations clamp to [SimSliders] bounds.
@riverpod
class SimParamsController extends _$SimParamsController {
  @override
  SimParams build() {
    final seeded = ref.watch(seededSimProvider);
    return SimParams(
      forces: seeded.forces,
      minDistances: seeded.minDistances,
      radii: seeded.radii,
      particleCount: seeded.particleCount,
      typeCount: seeded.typeCount,
    );
  }

  /// Sets the attraction multiplier, clamped to [SimSliders].
  void setAttractionK(double value) => state = state.copyWith(
        attractionK:
            value.clamp(SimSliders.attractionMin, SimSliders.attractionMax),
      );

  /// Sets the repulsion multiplier, clamped to [SimSliders].
  void setRepulsionK(double value) => state = state.copyWith(
        repulsionK:
            value.clamp(SimSliders.repulsionMin, SimSliders.repulsionMax),
      );

  /// Sets drift/friction, clamped to [SimSliders].
  void setFriction(double value) => state = state.copyWith(
        friction: value.clamp(SimSliders.frictionMin, SimSliders.frictionMax),
      );
}

/// The active [SimBackend] handle.
///
/// Defaults to [FakeSimBackend] so the app and frame loop run with no GPU; a
/// real backend is injected later by overriding this provider. Kept alive so the
/// backend persists for the app's lifetime, and disposed with the container.
///
/// **Lifecycle bring-up is not done here.** This provider only constructs and
/// exposes the handle; it deliberately does NOT call `init()`/`seed()` because
/// those are async and out of a synchronous provider's reach. The run driver
/// (the `Ticker` owner added in [PRIMORDIS-TASK-005] / [PRIMORDIS-TASK-006]) must
/// `await backend.init()` then `await backend.seed(seed)` (and apply the initial
/// params) **before** the first [FrameLoop.tick] — the order documented on
/// [SimBackend]. The fake tolerates being stepped un-initialized; a real
/// GPU/CPU backend will not, so the driver owns that sequencing.
@Riverpod(keepAlive: true)
SimBackend simBackend(Ref ref) {
  final backend = FakeSimBackend();
  ref.onDispose(() => unawaited(backend.dispose()));
  return backend;
}

/// The [FrameLoop] bound to the active [simBackendProvider].
///
/// The UI's `Ticker` drives [FrameLoop.tick] each frame ([PRIMORDIS-TASK-005] /
/// [PRIMORDIS-TASK-006]). The loop assumes the backend has already been
/// `init()`/`seed()`ed by that driver — see [simBackendProvider].
@Riverpod(keepAlive: true)
FrameLoop frameLoop(Ref ref) {
  final backend = ref.watch(simBackendProvider);
  return FrameLoop(backend: backend);
}

/// Run state: running/paused and the frame counter. [pause] is the reduced-
/// motion forward-hook ([PRIMORDIS-ADR-006]).
@riverpod
class RunStateController extends _$RunStateController {
  @override
  RunState build() => const RunState();

  /// Suppresses stepping (holds the last frame).
  void pause() => state = state.copyWith(isPaused: true);

  /// Resumes stepping.
  void resume() => state = state.copyWith(isPaused: false);

  /// Flips the paused flag.
  void togglePause() => state = state.copyWith(isPaused: !state.isPaused);

  /// Marks the simulation live.
  void start() => state = state.copyWith(isRunning: true);

  /// Marks the simulation stopped/torn down.
  void stop() => state = state.copyWith(isRunning: false);

  /// Records that one frame stepped (advances the counter).
  void markFrameStepped() => state = state.copyWith(frame: state.frame + 1);

  /// Resets to the initial run state.
  void reset() => state = const RunState();
}
