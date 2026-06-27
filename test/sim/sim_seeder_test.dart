import 'package:flutter_test/flutter_test.dart';
import 'package:primordis/shared/constants/primordis_config.dart';
import 'package:primordis/sim/models/sim_seed.dart';
import 'package:primordis/sim/models/type_matrix.dart';
import 'package:primordis/sim/sim_seeder.dart';

/// True if [m] has at least one off-diagonal pair where `at(i, j) != at(j, i)`.
bool _isAsymmetric(TypeMatrix m) {
  for (var i = 0; i < m.dimension; i++) {
    for (var j = i + 1; j < m.dimension; j++) {
      if (m.at(i, j) != m.at(j, i)) return true;
    }
  }
  return false;
}

void main() {
  group('seedSimulation', () {
    test('produces SoA buffers sized for the seed', () {
      const seed = SimSeed(particleCount: 1000, typeCount: 8);
      final s = seedSimulation(seed);
      expect(s.particleCount, 1000);
      expect(s.typeCount, 8);
      expect(s.positions.length, 2000);
      expect(s.velocities.length, 2000);
      expect(s.types.length, 1000);
      expect(s.particleTypes.length, 8);
      expect(s.forces.dimension, 8);
      expect(s.minDistances.dimension, 8);
      expect(s.radii.dimension, 8);
    });

    test('is deterministic: same seed yields identical output', () {
      const seed = SimSeed(particleCount: 2000);
      final a = seedSimulation(seed);
      final b = seedSimulation(seed);
      expect(a.positions, equals(b.positions));
      expect(a.velocities, equals(b.velocities));
      expect(a.types, equals(b.types));
      expect(a.forces.values, equals(b.forces.values));
      expect(a.minDistances.values, equals(b.minDistances.values));
      expect(a.radii.values, equals(b.radii.values));
      expect(a.particleTypes, equals(b.particleTypes));
    });

    test('different seeds yield different output', () {
      final a = seedSimulation(const SimSeed(seed: 11, particleCount: 2000));
      final b = seedSimulation(const SimSeed(seed: 22, particleCount: 2000));
      expect(a.positions, isNot(equals(b.positions)));
      expect(a.forces.values, isNot(equals(b.forces.values)));
    });

    test('the three matrices are asymmetric (directed per-type-pair)', () {
      final s = seedSimulation(const SimSeed());
      expect(_isAsymmetric(s.forces), isTrue);
      expect(_isAsymmetric(s.minDistances), isTrue);
      expect(_isAsymmetric(s.radii), isTrue);
    });

    test('values fall within the reference ranges', () {
      const seed = SimSeed(particleCount: 3000);
      final s = seedSimulation(seed);
      const eps = 1e-5;

      for (var i = 0; i < s.particleCount; i++) {
        final x = s.positions[i * 2];
        final y = s.positions[i * 2 + 1];
        expect(x, inInclusiveRange(0, PrimordisConfig.worldWidth + eps));
        expect(y, inInclusiveRange(0, PrimordisConfig.worldHeight + eps));
        expect(s.velocities[i * 2], inInclusiveRange(-8 - eps, 8 + eps));
        expect(s.velocities[i * 2 + 1], inInclusiveRange(-8 - eps, 8 + eps));
        expect(s.types[i], inInclusiveRange(0, PrimordisConfig.typeCount - 1));
      }

      for (final t in s.particleTypes) {
        expect(t.r, inInclusiveRange(0, 1));
        expect(t.g, inInclusiveRange(0, 1));
        expect(t.b, inInclusiveRange(0, 1));
      }

      final n = s.typeCount;
      for (var i = 0; i < n; i++) {
        for (var j = 0; j < n; j++) {
          final force = s.forces.at(i, j);
          // Forces are signed; magnitude is uniform in 0.1..0.8.
          expect(force.abs(), inInclusiveRange(0.1 - eps, 0.8 + eps));
          expect(s.minDistances.at(i, j), inInclusiveRange(4 - eps, 12 + eps));
          expect(
            s.radii.at(i, j),
            inInclusiveRange(20 - eps, PrimordisConfig.maxRadius + eps),
          );
        }
      }
    });

    test('forces carry both signs (mask applied)', () {
      final s = seedSimulation(const SimSeed());
      var positive = 0;
      var negative = 0;
      final n = s.typeCount;
      for (var i = 0; i < n; i++) {
        for (var j = 0; j < n; j++) {
          if (s.forces.at(i, j) > 0) {
            positive++;
          } else {
            negative++;
          }
        }
      }
      expect(positive, greaterThan(0));
      expect(negative, greaterThan(0));
    });
  });
}
