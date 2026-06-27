import 'package:flutter_test/flutter_test.dart';
import 'package:primordis/shared/constants/primordis_config.dart';
import 'package:primordis/sim/models/particle_type.dart';
import 'package:primordis/sim/models/run_state.dart';
import 'package:primordis/sim/models/sim_capabilities.dart';
import 'package:primordis/sim/models/sim_params.dart';
import 'package:primordis/sim/models/sim_seed.dart';
import 'package:primordis/sim/models/type_matrix.dart';

/// A constant-filled matrix the size of the simulation's type count.
TypeMatrix _matrix(double fill) =>
    TypeMatrix.generate(PrimordisConfig.typeCount, (_, _) => fill);

SimParams _params() => SimParams(
      forces: _matrix(0.5),
      minDistances: _matrix(8),
      radii: _matrix(48),
    );

void main() {
  group('SimSeed', () {
    test('defaults mirror the reference workload', () {
      const seed = SimSeed();
      expect(seed.seed, 1);
      expect(seed.particleCount, PrimordisConfig.particleCount);
      expect(seed.typeCount, PrimordisConfig.typeCount);
    });

    test('copyWith and equality', () {
      const a = SimSeed();
      final b = a.copyWith(seed: 42);
      expect(b.seed, 42);
      expect(b.particleCount, a.particleCount);
      expect(a, isNot(equals(b)));
      expect(a, equals(const SimSeed()));
    });
  });

  group('ParticleType', () {
    test('construction, equality, copyWith', () {
      const t = ParticleType(index: 3, r: 0.1, g: 0.2, b: 0.3);
      expect(t.index, 3);
      expect(t, equals(const ParticleType(index: 3, r: 0.1, g: 0.2, b: 0.3)));
      expect(t.copyWith(r: 0.9).r, 0.9);
      expect(t, isNot(equals(t.copyWith(index: 4))));
    });
  });

  group('RunState', () {
    test('defaults to running, not paused, frame 0', () {
      const s = RunState();
      expect(s.isRunning, isTrue);
      expect(s.isPaused, isFalse);
      expect(s.frame, 0);
    });

    test('copyWith', () {
      const s = RunState();
      expect(s.copyWith(isPaused: true).isPaused, isTrue);
      expect(s.copyWith(frame: 5).frame, 5);
    });
  });

  group('SimBackendCapabilities', () {
    test('construction and equality', () {
      const c = SimBackendCapabilities(
        isGpuAccelerated: true,
        maxParticles: 100000,
        defaultParticleCount: 24000,
        label: 'web-webgpu',
      );
      expect(c.isGpuAccelerated, isTrue);
      expect(c.maxParticles, 100000);
      expect(
        c,
        equals(const SimBackendCapabilities(
          isGpuAccelerated: true,
          maxParticles: 100000,
          defaultParticleCount: 24000,
          label: 'web-webgpu',
        )),
      );
    });
  });

  group('SimParams', () {
    test('slider defaults match the reference', () {
      final p = _params();
      expect(p.attractionK, SimSliders.attractionDefault);
      expect(p.repulsionK, SimSliders.repulsionDefault);
      expect(p.friction, SimSliders.frictionDefault);
    });

    test('world/grid defaults mirror PrimordisConfig (drift guard)', () {
      final p = _params();
      expect(p.particleCount, PrimordisConfig.particleCount);
      expect(p.typeCount, PrimordisConfig.typeCount);
      expect(p.worldWidth, PrimordisConfig.worldWidth);
      expect(p.worldHeight, PrimordisConfig.worldHeight);
      expect(p.maxRadius, PrimordisConfig.maxRadius.toDouble());
      expect(p.binSize, PrimordisConfig.binSize.toDouble());
      expect(p.gridWidth, PrimordisConfig.gridWidth);
      expect(p.gridHeight, PrimordisConfig.gridHeight);
      expect(p.binCount, PrimordisConfig.binCount);
      expect(p.maxBinParticles, PrimordisConfig.maxBinParticles);
      // Grid geometry is internally consistent.
      expect(p.gridWidth * p.gridHeight, p.binCount);
    });

    test('value equality includes the matrices', () {
      // Two independently-built params with content-equal matrices are equal,
      // which the frame loop relies on to skip redundant setParams calls.
      final a = _params();
      final b = _params();
      expect(a, equals(b));
      expect(a.hashCode, equals(b.hashCode));
    });

    test('copyWith on a slider breaks equality', () {
      final a = _params();
      final b = a.copyWith(attractionK: 99);
      expect(b.attractionK, 99);
      expect(a, isNot(equals(b)));
    });

    test('a differing matrix breaks equality', () {
      final a = _params();
      final b = a.copyWith(forces: _matrix(0.6));
      expect(a, isNot(equals(b)));
    });
  });
}
