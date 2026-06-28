import 'package:freezed_annotation/freezed_annotation.dart';

part 'run_state.freezed.dart';

/// The frame loop's run state: whether the simulation is active, paused, and how
/// many frames have advanced.
///
/// [isPaused] is the accessibility forward-hook required by [PRIMORDIS-ADR-006]:
/// because the whole canvas is motion, the app must offer a paused/static state
/// for reduced-motion users. The frame loop suppresses stepping while paused
/// (holding the last composited frame); [PRIMORDIS-TASK-015] /
/// [PRIMORDIS-TASK-018] wire it to the reduced-motion preference and UI.
@freezed
abstract class RunState with _$RunState {
  const factory RunState({
    /// Whether the simulation is live at all (false ⇒ stopped/torn down).
    @Default(true) bool isRunning,

    /// Whether stepping is suppressed (paused/static frame).
    @Default(false) bool isPaused,

    /// Frames advanced since the last reset (advances only while stepping).
    @Default(0) int frame,
  }) = _RunState;
}
