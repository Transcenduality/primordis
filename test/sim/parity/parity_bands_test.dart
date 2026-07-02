import 'package:flutter_test/flutter_test.dart';
import 'package:primordis/sim/parity/frame_metrics.dart';
import 'package:primordis/sim/parity/parity_bands.dart';
import 'package:primordis/sim/parity/parity_metrics.dart';

// Helper defaults are deliberately arbitrary; each test passes the specific
// values it asserts on, so no test relies on a default matching a magic number.
FrameMetrics _frame({
  int particleCount = 2500,
  int clusterCount = 3,
  double occupiedBinFraction = 0.8,
  double occMean = 39,
  double occVar = 80,
  double nnMean = 9,
  double nnVar = 15,
  double segregation = 0.35,
  double speedMean = 0.5,
  double speedVar = 0.02,
  double kineticEnergy = 0.03,
  double edgeExcess = 0.1,
}) =>
    FrameMetrics(
      particleCount: particleCount,
      clusterCount: clusterCount,
      occupiedBinFraction: occupiedBinFraction,
      occupancy: Moments(
        count: 77,
        mean: occMean,
        variance: occVar,
        min: 20,
        max: 70,
      ),
      nearestNeighbour: Moments(
        count: particleCount,
        mean: nnMean,
        variance: nnVar,
        min: 0.3,
        max: 27,
      ),
      segregationIndex: segregation,
      speed: Moments(
        count: particleCount,
        mean: speedMean,
        variance: speedVar,
        min: 0,
        max: 1,
      ),
      kineticEnergy: kineticEnergy,
      maxEdgeBinExcess: edgeExcess,
    );

void main() {
  group('Band', () {
    test('effective allowance is the larger of absolute and relative', () {
      const band = Band(absolute: 1, relative: 0.1);
      // For a small reference the absolute dominates.
      expect(band.allowanceFor(5), 1.0);
      // For a large reference the relative dominates (0.1 * 100 = 10).
      expect(band.allowanceFor(100), 10.0);
    });

    test('exact band admits only an equal value', () {
      expect(Band.exact.contains(42, 42), isTrue);
      expect(Band.exact.contains(42, 43), isFalse);
    });

    test('contains respects the band edges', () {
      const band = Band(absolute: 2);
      expect(band.contains(10, 12), isTrue);
      expect(band.contains(10, 8), isTrue);
      expect(band.contains(10, 12.001), isFalse);
    });
  });

  group('ParityBands.violations', () {
    test('an identical frame has no violations', () {
      final f = _frame();
      expect(ParityBands.vsReference.violations(f, f), isEmpty);
    });

    test('a within-band perturbation passes', () {
      final ref = _frame(speedMean: 0.2);
      // 20% band on speedMean; 0.23 is +15%.
      final obs = _frame(speedMean: 0.23);
      expect(ParityBands.vsReference.violations(ref, obs), isEmpty);
    });

    test('an out-of-band speed change is flagged with a diagnostic', () {
      final ref = _frame(speedMean: 0.2);
      final obs = _frame(speedMean: 0.8); // +300%
      final v = ParityBands.vsReference.violations(ref, obs);
      expect(v, isNotEmpty);
      expect(v.single, contains('speed.mean'));
      expect(v.single, contains('observed'));
      expect(v.single, contains('reference'));
    });

    test('a lost particle fails the exact population band', () {
      final ref = _frame(particleCount: 3000);
      final obs = _frame(particleCount: 2999);
      final v = ParityBands.vsReference.violations(ref, obs);
      expect(v.any((s) => s.startsWith('particleCount')), isTrue);
    });

    test('cross-backend bands are strictly looser than vs-reference', () {
      // A change that violates the tighter band but not the looser one.
      final ref = _frame(nnMean: 8);
      final obs = _frame(nnMean: 10.5); // +31%
      expect(ParityBands.vsReference.violations(ref, obs), isNotEmpty);
      expect(ParityBands.crossBackend.violations(ref, obs), isEmpty);
    });

    test('cross-backend still enforces exact population conservation', () {
      final ref = _frame(particleCount: 3000);
      final obs = _frame(particleCount: 2990);
      expect(
        ParityBands.crossBackend
            .violations(ref, obs)
            .any((s) => s.startsWith('particleCount')),
        isTrue,
      );
    });
  });
}
