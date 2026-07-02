import 'dart:typed_data';

import 'package:primordis/sim/parity/frame_metrics.dart';
import 'package:primordis/sim/parity/parity_fingerprint.dart';
import 'package:primordis/sim/parity/parity_metrics.dart';

/// A backend the parity harness can drive to produce a fingerprint.
///
/// This is the **pluggability seam**: adding the web WebGPU, Dawn/Metal FFI, or
/// MSL backend to the parity suite requires only implementing this three-method
/// interface and registering it with [runParity] — no new metric code, no new
/// band definitions ([PRIMORDIS-TASK-017] plugs the Dawn/Naga comparison in
/// exactly this way). The CPU tier's adapter ([CpuParityBackend]) is the
/// reference implementation.
///
/// The contract:
/// - [label] names the backend in diagnostics.
/// - [seed] loads the deterministic initial condition (identical across
///   backends — same seed, same params) and resets step count to zero.
/// - [step] advances the simulation by one frame of [dt] seconds.
/// - the SoA accessors expose the *current* particle state for metric capture.
abstract interface class ParityBackend {
  /// Human-readable backend name (`cpu`, `wgsl-kernel`, `dawn`, `web-webgpu`).
  String get label;

  /// Number of live particles (constant across a run).
  int get particleCount;

  /// Number of particle types.
  int get typeCount;

  /// Loads the deterministic seeded initial state, ready to [step].
  void seed();

  /// Advances the simulation by one frame of [dt] seconds.
  void step(double dt);

  /// Current interleaved `x, y` positions; length `2 * particleCount`.
  Float32List get positions;

  /// Current interleaved `x, y` velocities; length `2 * particleCount`.
  Float32List get velocities;

  /// Per-particle type index; length `particleCount`.
  Int32List get types;
}

/// Drives [backend] for [totalSteps] frames of [dt] each, capturing a
/// [FrameMetrics] at every step named in [checkpoints], and returns the
/// resulting [ParityFingerprint].
///
/// [checkpoints] maps a checkpoint name to the (0-based) step index *after
/// which* the metrics are sampled — e.g. `{'early': 20, 'mid': 100, 'steady':
/// 300}`. Sampling `early` at step 0 captures the seeded state before any
/// dynamics. All named steps must be `< totalSteps`.
///
/// The [attractionK]/[repulsionK]/[friction] values are recorded on the
/// fingerprint purely for self-description (so a non-default-parameter capture
/// is only ever compared against a matching reference); the backend itself owns
/// how those values drive its physics.
ParityFingerprint runParity({
  required ParityBackend backend,
  required int seed,
  required MetricGrid grid,
  required int totalSteps,
  required Map<String, int> checkpoints,
  required double dt,
  required double attractionK,
  required double repulsionK,
  required double friction,
}) {
  final maxCheckpoint =
      checkpoints.values.fold<int>(-1, (m, v) => v > m ? v : m);
  if (maxCheckpoint >= totalSteps) {
    throw ArgumentError(
      'checkpoint step $maxCheckpoint is beyond totalSteps $totalSteps',
    );
  }
  // Invert to look up which checkpoint name(s) fire at a given step.
  final atStep = <int, List<String>>{};
  for (final entry in checkpoints.entries) {
    (atStep[entry.value] ??= <String>[]).add(entry.key);
  }

  backend.seed();
  final captured = <String, FrameMetrics>{};

  FrameMetrics snapshot() => FrameMetrics.from(
        positions: backend.positions,
        velocities: backend.velocities,
        types: backend.types,
        particleCount: backend.particleCount,
        typeCount: backend.typeCount,
        grid: grid,
      );

  // A checkpoint at step 0 captures the seeded state before any step runs.
  for (final name in atStep[0] ?? const <String>[]) {
    captured[name] = snapshot();
  }
  for (var s = 1; s < totalSteps; s++) {
    backend.step(dt);
    for (final name in atStep[s] ?? const <String>[]) {
      captured[name] = snapshot();
    }
  }

  // Re-key the captured checkpoints back into the caller's declared order.
  final ordered = <String, FrameMetrics>{};
  for (final name in checkpoints.keys) {
    final m = captured[name];
    if (m != null) ordered[name] = m;
  }

  return ParityFingerprint(
    label: backend.label,
    seed: seed,
    particleCount: backend.particleCount,
    typeCount: backend.typeCount,
    attractionK: attractionK,
    repulsionK: repulsionK,
    friction: friction,
    checkpoints: ordered,
  );
}
