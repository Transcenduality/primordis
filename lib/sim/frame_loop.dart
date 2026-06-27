import 'package:primordis/sim/models/sim_params.dart';
import 'package:primordis/sim/sim_backend.dart';

/// Backend-agnostic per-frame driver.
///
/// It sequences the shared frame contract — upload [SimParams] only when they
/// change, then `step(dt)`, then `present()` — without knowing or caring which
/// concrete [SimBackend] is live ([PRIMORDIS-ADR-001]). The *meaning* of
/// step/present differs per backend (GPU compute passes + point render; CPU
/// counting-sort + `drawRawPoints`; macOS texture present), but the ordering and
/// the dt-driven Euler tick are shared and tested here against `FakeSimBackend`.
///
/// It holds no render surface and no timer, so a test can [tick] it directly. A
/// real driver (a `Ticker` in the UI; [PRIMORDIS-TASK-005] / [PRIMORDIS-TASK-006])
/// supplies wall-clock `dt` and the current params/pause state each frame.
///
/// Pausing (the reduced-motion forward-hook, [PRIMORDIS-ADR-006]) suppresses
/// stepping: a paused tick advances nothing and holds the last composited frame.
class FrameLoop {
  FrameLoop({required SimBackend backend}) : _backend = backend;

  final SimBackend _backend;

  /// The params last uploaded via `setParams`, used to detect changes. Null
  /// until the first non-paused tick, which always uploads.
  SimParams? _lastApplied;

  int _frame = 0;

  /// Frames stepped since construction or the last [reset].
  int get frame => _frame;

  /// The most recently uploaded params (null before the first step).
  SimParams? get lastAppliedParams => _lastApplied;

  /// Drives one frame.
  ///
  /// When [paused], returns `false` immediately and does nothing (no upload, no
  /// step, no present). Otherwise: uploads [params] iff they differ from the
  /// last uploaded params, then `step(dt)`, then `present()`, and returns `true`.
  bool tick({
    required double dt,
    required SimParams params,
    required bool paused,
  }) {
    if (paused) return false;
    if (_lastApplied != params) {
      _backend.setParams(params);
      _lastApplied = params;
    }
    _backend.step(dt);
    _backend.present();
    _frame++;
    return true;
  }

  /// Resets the frame counter and forces the next tick to re-upload params.
  void reset() {
    _lastApplied = null;
    _frame = 0;
  }
}
