// Regenerates the committed CPU reference fingerprints
// (`test/parity/fixtures/cpu_reference_default.json` and
// `cpu_reference_highdrift.json`) from the deterministic CPU tier.
//
// The CPU tier's counting-sort binning is bit-stable per seed+params, so this
// needs no GPU and produces identical output every run — which is exactly why
// the CPU tier is the harness's primary anchor. Run this only when the shared
// seed/params or the metric definitions intentionally change; commit the
// regenerated JSON and review the diff.
//
// Usage:
//   dart run tool/parity/generate_cpu_reference.dart
//
// Writes both fixtures in place (label "reference").

import 'dart:convert';
import 'dart:io';

import 'package:primordis/sim/parity/parity_fingerprint.dart';
import 'package:primordis/sim/parity/parity_runner.dart';

// This tool depends on the test-only harness config, imported by relative path
// because it lives under test/ (not shipped in the app). The analyzer's
// always_use_package_imports lint does not apply to files under tool/.
import '../../test/parity/parity_harness_support.dart';

ParityFingerprint _run(ParityHarnessConfig cfg) => runParity(
      backend: cfg.buildCpuBackend(),
      seed: cfg.seed,
      grid: cfg.grid,
      totalSteps: cfg.totalSteps,
      checkpoints: cfg.checkpoints,
      dt: cfg.dt,
      attractionK: cfg.attractionK,
      repulsionK: cfg.repulsionK,
      friction: cfg.friction,
    );

void _write(String path, ParityFingerprint fp) {
  // Relabel to "reference": these are the baselines the CPU/GPU tiers are
  // measured against, so diagnostics read "cpu vs reference".
  final json = fp.toJson()..['label'] = 'reference';
  File(path).writeAsStringSync(
    '${const JsonEncoder.withIndent('  ').convert(json)}\n',
  );
  stdout.writeln('Wrote $path');
}

void main() {
  _write(
    'test/parity/fixtures/cpu_reference_default.json',
    _run(ParityHarnessConfig.defaults()),
  );
  _write(
    'test/parity/fixtures/cpu_reference_highdrift.json',
    _run(ParityHarnessConfig.highDrift()),
  );
}
