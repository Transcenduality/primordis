import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:primordis/sim/models/type_matrix.dart';

void main() {
  group('TypeMatrix', () {
    test('stores values row-major and at(i, j) reads i * n + j', () {
      final m = TypeMatrix.fromRows([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      expect(m.dimension, 3);
      expect(m.at(0, 0), 1);
      expect(m.at(0, 2), 3);
      expect(m.at(2, 0), 7);
      expect(m.at(1, 1), 5);
      // Flat backing store is row-major.
      expect(m.values, Float32List.fromList(<double>[1, 2, 3, 4, 5, 6, 7, 8, 9]));
    });

    test('generate fills in row-major order', () {
      final m = TypeMatrix.generate(2, (row, col) => (row * 10 + col).toDouble());
      expect(m.values, Float32List.fromList(<double>[0, 1, 10, 11]));
    });

    test('supports asymmetry: at(i, j) is independent of at(j, i)', () {
      final m = TypeMatrix.fromRows([
        [0, 1],
        [2, 0],
      ]);
      expect(m.at(0, 1), isNot(m.at(1, 0)));
    });

    test('has value equality over contents', () {
      final a = TypeMatrix.generate(4, (r, c) => (r * 4 + c).toDouble());
      final b = TypeMatrix.generate(4, (r, c) => (r * 4 + c).toDouble());
      final c = TypeMatrix.generate(4, (r, c) => (r * 4 + c + 1).toDouble());
      expect(a, equals(b));
      expect(a.hashCode, equals(b.hashCode));
      expect(a, isNot(equals(c)));
    });

    test('defensively copies the source buffer (stays immutable)', () {
      final source = Float32List.fromList(<double>[1, 2, 3, 4]);
      final m = TypeMatrix(2, source);
      source[0] = 999; // mutate the caller's buffer after construction
      expect(m.at(0, 0), 1); // matrix is unaffected
    });

    test('asserts the backing store matches dimension^2', () {
      expect(
        () => TypeMatrix(2, Float32List(3)),
        throwsA(isA<AssertionError>()),
      );
    });
  });
}
