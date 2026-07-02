import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:primordis/sim/parity/frame_metrics.dart';
import 'package:primordis/sim/parity/parity_bands.dart';
import 'package:primordis/sim/parity/parity_fingerprint.dart';
import 'package:primordis/sim/parity/parity_metrics.dart';

FrameMetrics _frame(double speedMean) => FrameMetrics(
      particleCount: 3000,
      clusterCount: 3,
      occupiedBinFraction: 0.8,
      occupancy: const Moments(
        count: 77,
        mean: 39,
        variance: 80,
        min: 20,
        max: 70,
      ),
      nearestNeighbour: const Moments(
        count: 3000,
        mean: 8,
        variance: 15,
        min: 0.3,
        max: 27,
      ),
      segregationIndex: 0.35,
      speed: Moments(
        count: 3000,
        mean: speedMean,
        variance: 0.02,
        min: 0,
        max: 1,
      ),
      kineticEnergy: 0.03,
      maxEdgeBinExcess: 0.1,
    );

ParityFingerprint _fingerprint(String label, double steadySpeed) =>
    ParityFingerprint(
      label: label,
      seed: 42,
      particleCount: 3000,
      typeCount: 32,
      attractionK: 32,
      repulsionK: 32,
      friction: 0.25,
      checkpoints: <String, FrameMetrics>{
        'early': _frame(6.0),
        'mid': _frame(0.24),
        'steady': _frame(steadySpeed),
      },
    );

void main() {
  group('ParityFingerprint JSON round-trip', () {
    test('serialises and deserialises losslessly', () {
      final fp = _fingerprint('cpu', 0.22);
      final json = jsonDecode(fp.toPrettyJson()) as Map<String, dynamic>;
      final restored = ParityFingerprint.fromJson(json);
      expect(restored.label, 'cpu');
      expect(restored.seed, 42);
      expect(restored.friction, 0.25);
      expect(restored.checkpoints.keys.toList(), <String>['early', 'mid', 'steady']);
      expect(restored.checkpoints['steady']!.speed.mean, 0.22);
      expect(restored.checkpoints['early']!.speed.mean, 6.0);
    });
  });

  group('ParityFingerprint.violationsAgainst', () {
    test('a matching fingerprint yields no violations', () {
      final ref = _fingerprint('reference', 0.22);
      final obs = _fingerprint('cpu', 0.23); // within the 25% speed band
      expect(obs.violationsAgainst(ref, ParityBands.vsReference), isEmpty);
    });

    test('violations are prefixed with the checkpoint and source labels', () {
      final ref = _fingerprint('reference', 0.22);
      final obs = _fingerprint('cpu', 5.0); // steady speed way out of band
      final v = obs.violationsAgainst(ref, ParityBands.vsReference);
      expect(v, isNotEmpty);
      expect(v.first, startsWith('[steady] cpu vs reference'));
    });

    test('a missing checkpoint is reported as a structural mismatch', () {
      final ref = _fingerprint('reference', 0.22);
      final partial = ParityFingerprint(
        label: 'cpu',
        seed: 42,
        particleCount: 3000,
        typeCount: 32,
        attractionK: 32,
        repulsionK: 32,
        friction: 0.25,
        checkpoints: <String, FrameMetrics>{'early': _frame(6.0)},
      );
      final v = partial.violationsAgainst(ref, ParityBands.vsReference);
      expect(v.any((s) => s.contains('missing checkpoint')), isTrue);
    });

    test('an extra checkpoint is reported as a structural mismatch', () {
      final ref = ParityFingerprint(
        label: 'reference',
        seed: 42,
        particleCount: 3000,
        typeCount: 32,
        attractionK: 32,
        repulsionK: 32,
        friction: 0.25,
        checkpoints: <String, FrameMetrics>{'early': _frame(6.0)},
      );
      final obs = _fingerprint('cpu', 0.22); // has mid + steady too
      final v = obs.violationsAgainst(ref, ParityBands.vsReference);
      expect(v.any((s) => s.contains('extra checkpoint')), isTrue);
    });
  });
}
