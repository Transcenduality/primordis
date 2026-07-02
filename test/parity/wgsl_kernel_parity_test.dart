import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:primordis/sim/parity/parity_bands.dart';
import 'package:primordis/sim/parity/parity_fingerprint.dart';
import 'package:primordis/sim/parity/parity_metrics.dart';
import 'package:primordis/sim/parity/raw_snapshot.dart';

import 'parity_harness_support.dart';

/// Ingests the **standalone WGSL kernel** validation output
/// ([PRIMORDIS-TASK-003]) and checks it against the reference under the *looser*
/// cross-backend band. The kernel runs on a real WebGPU device under Node
/// (`test/sim/kernel/harness/`), which is not available in the pure-Dart
/// `analyze-test` CI job, so the kernel's raw checkpoint snapshots are exported
/// once (via `export_fingerprint.mjs`) and committed as a fixture. When the
/// fixture is absent (e.g. a fresh checkout before the GPU export is run) the
/// ingestion test **skips** rather than fails — CI stays green without a GPU,
/// and the parity assertion runs wherever the fixture exists.
void main() {
  group('RawSnapshot ingestion', () {
    test('parses flat SoA arrays from JSON', () {
      final snap = RawSnapshot.fromJson(<String, dynamic>{
        'particleCount': 2,
        'positions': <double>[1, 2, 3, 4],
        'velocities': <double>[0.1, 0.2, 0.3, 0.4],
        'types': <int>[0, 1],
      });
      expect(snap.particleCount, 2);
      expect(snap.positions, isA<Float32List>());
      expect(snap.positions.length, 4);
      expect(snap.types.length, 2);
      expect(snap.types[1], 1);
    });

    test('builds a fingerprint from snapshots using the shared metric code', () {
      // Two hand-built checkpoints; the point is that FrameMetrics is computed
      // by the SAME Dart code that judges the CPU tier, so cross-backend
      // comparison is apples-to-apples.
      final grid = MetricGrid(worldWidth: 1080, worldHeight: 720, binSize: 96);
      final positions = Float32List(200);
      final velocities = Float32List(200);
      final types = Int32List(100);
      for (var i = 0; i < 100; i++) {
        positions[i * 2] = (i * 10.0) % 1080;
        positions[i * 2 + 1] = (i * 7.0) % 720;
        types[i] = i % 32;
      }
      final snap = RawSnapshot(
        positions: positions,
        velocities: velocities,
        types: types,
        particleCount: 100,
      );
      final fp = buildFingerprintFromSnapshots(
        label: 'wgsl-kernel',
        seed: 42,
        typeCount: 32,
        attractionK: 32,
        repulsionK: 32,
        friction: 0.25,
        grid: grid,
        snapshots: <String, RawSnapshot>{'early': snap, 'steady': snap},
      );
      expect(fp.label, 'wgsl-kernel');
      expect(fp.checkpoints.keys, containsAll(<String>['early', 'steady']));
      expect(fp.checkpoints['early']!.particleCount, 100);
    });
  });

  group('WGSL kernel vs reference (cross-backend band)', () {
    const fixturePath = 'test/parity/fixtures/wgsl_kernel_snapshots.json';

    test('kernel snapshots fall within the cross-backend band', () {
      final file = File(fixturePath);
      if (!file.existsSync()) {
        // The GPU export has not been run in this environment; skip cleanly.
        // Regenerate with: cd test/sim/kernel/harness && node export_fingerprint.mjs
        markTestSkipped('WGSL kernel snapshot fixture not present ('
            '$fixturePath) — run export_fingerprint.mjs on a WebGPU host');
        return;
      }
      final raw = jsonDecode(file.readAsStringSync()) as Map<String, dynamic>;
      final snapshots = <String, RawSnapshot>{
        for (final entry in (raw['checkpoints'] as Map<String, dynamic>).entries)
          entry.key:
              RawSnapshot.fromJson(entry.value as Map<String, dynamic>),
      };
      final cfg = ParityHarnessConfig.defaults();
      final observed = buildFingerprintFromSnapshots(
        label: 'wgsl-kernel',
        seed: (raw['seed'] as num?)?.toInt() ?? cfg.seed,
        typeCount: (raw['typeCount'] as num?)?.toInt() ?? cfg.typeCount,
        attractionK:
            (raw['attractionK'] as num?)?.toDouble() ?? cfg.attractionK,
        repulsionK: (raw['repulsionK'] as num?)?.toDouble() ?? cfg.repulsionK,
        friction: (raw['friction'] as num?)?.toDouble() ?? cfg.friction,
        grid: cfg.grid,
        snapshots: snapshots,
      );
      final reference = ParityFingerprint.fromJson(
        jsonDecode(
          File('test/parity/fixtures/cpu_reference_default.json')
              .readAsStringSync(),
        ) as Map<String, dynamic>,
      );
      final violations =
          observed.violationsAgainst(reference, ParityBands.crossBackend);
      expect(violations, isEmpty, reason: violations.join('\n'));
    });
  });
}
