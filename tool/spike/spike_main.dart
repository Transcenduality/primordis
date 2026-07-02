// PRIMORDIS-TASK-801 spike — Dawn/wgpu-over-Metal via minigpu (dart:ffi).
//
// Proves the experimental FFI WebGPU layer survives the REAL workload: the
// exact 3-pass atomic-binning kernel (`lib/sim/kernel/primordis.wgsl`,
// unmodified on disk) at 24,000 particles / 32 types, advanced ≥1000 frames
// with no device-lost, no NaN/Inf positions, and bin counts consistent with
// the MAX_BIN_PARTICLES cap.
//
// Run:  flutter run -d macos -t tool/spike/spike_main.dart
//
// This is a spike harness, not app code: it uses CPU readback for validation,
// which must never leak into the shipped frame loop (present path is
// PRIMORDIS-TASK-802). It reuses the shared seeding/marshalling/layout code so
// what it validates is what the real backend will upload.

import 'dart:io';

import 'package:flutter/widgets.dart';

import 'spike_runner.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final report = await runSpike(frames: 1000);
  stdout.writeln(report.summary());
  exit(report.passed ? 0 : 1);
}
