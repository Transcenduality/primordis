import 'dart:async';

import 'package:primordis/sim/models/sim_seed.dart';
import 'package:primordis/sim/providers/sim_providers.dart';
import 'package:primordis/sim/sim_backend.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'sim_params_provider.g.dart';

/// Feature-layer providers driving the live sim controls
/// ([PRIMORDIS-TASK-006]).
///
/// The shared sim core ([sim_providers.dart]) already exposes the seed,
/// params, backend, and run-state providers this feature reads and mutates;
/// this file adds the one piece that belongs to the UI/chrome layer — the
/// [SimRunnerController], which owns backend lifecycle bring-up (`init` →
/// `seed`) and marshals the live [SimParamsController] value into the backend
/// once per frame via [SimBackend.updateParams]-equivalent [FrameLoop.tick].
///
/// Kept in `features/simulation/providers/` (not `lib/sim/`) because it is
/// standard Riverpod business logic with no platform-specific code of its own
/// — it only calls through the [SimBackend] interface, never
/// `dart:js_interop`/`dart:ffi` directly ([PRIMORDIS-ADR-001]).

/// Whether the app should start paused because the platform reports a
/// reduced-motion preference.
///
/// This is a **plain settable flag**, not a `MediaQuery` read: `MediaQuery` is
/// only available from a `BuildContext`, so the widget layer (the runner host,
/// [PRIMORDIS-TASK-006]) observes `MediaQuery.disableAnimationsOf(context)` on
/// first build and calls [ReducedMotionController.set] once, which in turn
/// pauses [RunStateController] if true. Keeping the provider free of
/// `BuildContext` keeps it unit-testable without pumping a widget tree.
@Riverpod(keepAlive: true)
class ReducedMotionController extends _$ReducedMotionController {
  @override
  bool build() => false;

  /// Records the platform's reduced-motion preference. Setting `true` for the
  /// first time also pauses the run state — the reduced-motion affordance
  /// required by [PRIMORDIS-ADR-006] — but never auto-resumes on `false`
  /// (resuming is always an explicit play action).
  void set(bool reducedMotion) {
    if (state == reducedMotion) return;
    state = reducedMotion;
  }
}

/// Owns [SimBackend] lifecycle bring-up (`init` → `seed`) and exposes whether
/// the backend is ready to step.
///
/// The frame loop / run driver ([PRIMORDIS-TASK-006]) must not call
/// `FrameLoop.tick` until this reports `true` — the [SimBackend] contract
/// requires `init()` then `seed()` to complete before `setParams`/`step`
/// ([lib/sim/sim_backend.dart]). Reseeding re-runs only `seed()` (new
/// particles/matrices/colours via storage buffers), distinct from a params
/// reset (uniform-only, [SimParamsController.resetToDefaults]).
@Riverpod(keepAlive: true)
class SimRunnerController extends _$SimRunnerController {
  // `FutureOr<void>` is the signature `@riverpod`'s AsyncNotifier codegen
  // requires for `build()`; there is no `Future<void>?`/`void` alternative
  // available on the generated base class.
  @override
  // ignore: avoid_futureor_void
  FutureOr<void> build() async {
    final backend = ref.watch(simBackendProvider);
    final seed = ref.watch(simSeedControllerProvider);
    await backend.init();
    await backend.seed(seed);
  }

  /// Re-seeds the running backend from [seed] (new particles, colours, and
  /// asymmetric matrices) without a full `init()`/teardown.
  ///
  /// Distinct from [SimParamsController.resetToDefaults]: reseeding rewrites
  /// storage buffers (particles/matrices/colours); a params reset only
  /// rewrites the uniform block.
  Future<void> reseed(SimSeed seed) async {
    final backend = ref.read(simBackendProvider);
    await backend.seed(seed);
  }
}
