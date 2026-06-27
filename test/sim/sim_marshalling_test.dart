import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:primordis/shared/constants/primordis_config.dart';
import 'package:primordis/sim/models/particle_type.dart';
import 'package:primordis/sim/models/sim_params.dart';
import 'package:primordis/sim/models/type_matrix.dart';
import 'package:primordis/sim/sim_marshalling.dart';

TypeMatrix _matrix(double fill) =>
    TypeMatrix.generate(PrimordisConfig.typeCount, (_, _) => fill);

SimParams _params() => SimParams(
      forces: _matrix(0.5),
      minDistances: _matrix(8),
      radii: _matrix(48),
      attractionK: 12.5,
      repulsionK: 64,
      friction: 0.5,
    );

void main() {
  group('uniform block', () {
    test('is a 64-byte, 16-slot, 16-byte-aligned struct', () {
      expect(SimMarshalling.uniformSlotCount, 16);
      expect(SimMarshalling.uniformByteLength, 64);
      expect(SimMarshalling.uniformByteLength % 16, 0);
    });

    test('packs each field at its documented byte offset', () {
      final bytes = packUniforms(_params(), 0.5);
      expect(bytes.length, 64);

      final bd = ByteData.view(bytes.buffer);
      // f32 fields (slot * 4 == byte offset).
      expect(bd.getFloat32(0, Endian.host), closeTo(12.5, 1e-6));
      expect(bd.getFloat32(4, Endian.host), closeTo(64, 1e-6));
      expect(bd.getFloat32(8, Endian.host), closeTo(0.5, 1e-6));
      expect(bd.getFloat32(12, Endian.host), closeTo(0.5, 1e-6)); // dt
      expect(bd.getFloat32(16, Endian.host),
          closeTo(PrimordisConfig.worldWidth.toDouble(), 1e-6));
      expect(bd.getFloat32(20, Endian.host),
          closeTo(PrimordisConfig.worldHeight.toDouble(), 1e-6));
      expect(bd.getFloat32(24, Endian.host),
          closeTo(PrimordisConfig.maxRadius.toDouble(), 1e-6));
      expect(bd.getFloat32(28, Endian.host),
          closeTo(PrimordisConfig.binSize.toDouble(), 1e-6));
      // u32 fields.
      expect(bd.getUint32(32, Endian.host), PrimordisConfig.gridWidth);
      expect(bd.getUint32(36, Endian.host), PrimordisConfig.gridHeight);
      expect(bd.getUint32(40, Endian.host), PrimordisConfig.particleCount);
      expect(bd.getUint32(44, Endian.host), PrimordisConfig.binCount);
      expect(bd.getUint32(48, Endian.host), PrimordisConfig.typeCount);
      // Reserved padding slots are zero.
      expect(bd.getUint32(52, Endian.host), 0);
      expect(bd.getUint32(56, Endian.host), 0);
      expect(bd.getUint32(60, Endian.host), 0);
    });

    test('round-trips through unpackUniforms', () {
      final p = _params();
      final v = unpackUniforms(packUniforms(p, 0.016));
      expect(v.attractionK, closeTo(p.attractionK, 1e-5));
      expect(v.repulsionK, closeTo(p.repulsionK, 1e-5));
      expect(v.friction, closeTo(p.friction, 1e-6));
      expect(v.dt, closeTo(0.016, 1e-6));
      expect(v.worldWidth, closeTo(p.worldWidth.toDouble(), 1e-6));
      expect(v.worldHeight, closeTo(p.worldHeight.toDouble(), 1e-6));
      expect(v.maxRadius, closeTo(p.maxRadius, 1e-6));
      expect(v.binSize, closeTo(p.binSize, 1e-6));
      expect(v.gridWidth, p.gridWidth);
      expect(v.gridHeight, p.gridHeight);
      expect(v.numParticles, p.particleCount);
      expect(v.numBins, p.binCount);
      expect(v.typeCount, p.typeCount);
    });
  });

  group('flattenMatrix', () {
    test('is row-major with idx = i * n + j and the right length', () {
      final m = TypeMatrix.generate(5, (r, c) => (r * 5 + c).toDouble());
      final flat = flattenMatrix(m);
      expect(flat.length, 25);
      for (var i = 0; i < 5; i++) {
        for (var j = 0; j < 5; j++) {
          expect(flat[i * 5 + j], m.at(i, j));
        }
      }
    });

    test('returns a defensive copy', () {
      final m = TypeMatrix.generate(3, (_, _) => 1);
      final flat = flattenMatrix(m);
      flat[0] = 999;
      expect(m.at(0, 0), 1); // original is untouched
    });
  });

  group('packTypeColors', () {
    test('packs typeCount RGBA tuples with alpha = 1', () {
      final colors = packTypeColors(const [
        ParticleType(index: 0, r: 0.1, g: 0.2, b: 0.3),
        ParticleType(index: 1, r: 0.4, g: 0.5, b: 0.6),
      ]);
      expect(colors.length, 2 * SimMarshalling.colorStride);
      expect(colors[0], closeTo(0.1, 1e-6));
      expect(colors[1], closeTo(0.2, 1e-6));
      expect(colors[2], closeTo(0.3, 1e-6));
      expect(colors[3], 1.0);
      expect(colors[4], closeTo(0.4, 1e-6));
      expect(colors[7], 1.0);
    });
  });

  group('bin buffers', () {
    test('bin counts are Uint32List sized binCount', () {
      final p = _params();
      final counts = newBinCounts(p);
      expect(counts, isA<Uint32List>());
      expect(counts.length, p.binCount);
      expect(counts.length, PrimordisConfig.binCount);
      expect(counts.every((c) => c == 0), isTrue);
    });

    test('bin particles are Uint32List sized binCount * maxBinParticles', () {
      final p = _params();
      final buf = newBinParticles(p);
      expect(buf, isA<Uint32List>());
      expect(buf.length, p.binCount * p.maxBinParticles);
      expect(buf.length, PrimordisConfig.binCount * PrimordisConfig.maxBinParticles);
    });
  });
}
