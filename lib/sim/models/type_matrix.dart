import 'dart:typed_data';

/// An immutable, row-major NxN matrix of `float32` values.
///
/// Primordis encodes per-type-pair behaviour in three **asymmetric** 32x32
/// matrices — `forces`, `minDistances`, `radii` — where `m.at(i, j)` (the effect
/// of type `i` reacting to type `j`) is independent of `m.at(j, i)`. Faithfully
/// reproducing that asymmetry is load-bearing for the simulation, so this type
/// exists to make the shape explicit and give the matrices proper **value
/// equality** (two matrices with identical contents compare equal), which the
/// Freezed [SimParams] and the frame loop's "params changed?" check rely on.
///
/// The backing store is a single [Float32List] in row-major order, so the flat
/// index of cell `(row, col)` is `row * dimension + col` — the exact layout the
/// WGSL kernel ([PRIMORDIS-ADR-003]) and CPU fallbacks index with
/// `my_type * NUM_TYPES + other_type`. See [values] for the marshalling-ready
/// flat buffer.
///
/// This file is platform-neutral (only `dart:typed_data`); it compiles
/// identically on web and native, per [PRIMORDIS-ADR-001].
class TypeMatrix {
  /// Wraps a [dimension]x[dimension] row-major matrix.
  ///
  /// [values] is **defensively copied** so the matrix is truly immutable: a
  /// caller may reuse or mutate their buffer afterwards without affecting this
  /// matrix's contents, equality, or hashCode (which the frame loop's
  /// change-detection relies on). Use [TypeMatrix.fromRows] /
  /// [TypeMatrix.generate] to build one from scratch.
  TypeMatrix(int dimension, Float32List values)
      : this._owned(dimension, Float32List.fromList(values));

  /// Internal: takes ownership of [values] without copying. Used only by the
  /// factories, which build a fresh buffer that is never exposed before wrapping.
  TypeMatrix._owned(this.dimension, this.values)
      : assert(dimension > 0, 'dimension must be positive'),
        assert(
          values.length == dimension * dimension,
          'values.length must equal dimension^2',
        );

  /// Builds a matrix by calling [cell] for every `(row, col)` in row-major order.
  factory TypeMatrix.generate(
    int dimension,
    double Function(int row, int col) cell,
  ) {
    final data = Float32List(dimension * dimension);
    var k = 0;
    for (var row = 0; row < dimension; row++) {
      for (var col = 0; col < dimension; col++) {
        data[k++] = cell(row, col);
      }
    }
    return TypeMatrix._owned(dimension, data);
  }

  /// Builds a matrix from a list of equal-length rows.
  factory TypeMatrix.fromRows(List<List<double>> rows) {
    final dimension = rows.length;
    final data = Float32List(dimension * dimension);
    var k = 0;
    for (var row = 0; row < dimension; row++) {
      assert(
        rows[row].length == dimension,
        'row $row has length ${rows[row].length}, expected $dimension',
      );
      for (var col = 0; col < dimension; col++) {
        data[k++] = rows[row][col];
      }
    }
    return TypeMatrix._owned(dimension, data);
  }

  /// Side length of the (square) matrix — the simulation's type count.
  final int dimension;

  /// Row-major flat backing store, length `dimension * dimension`.
  ///
  /// This is the exact buffer uploaded to a backend's storage buffer; do not
  /// mutate it. See [TypeMatrix] for the index contract.
  final Float32List values;

  /// The value at row [row], column [col] (`values[row * dimension + col]`).
  double at(int row, int col) {
    assert(row >= 0 && row < dimension, 'row out of range: $row');
    assert(col >= 0 && col < dimension, 'col out of range: $col');
    return values[row * dimension + col];
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    if (other is! TypeMatrix || other.dimension != dimension) return false;
    final a = values;
    final b = other.values;
    for (var i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  @override
  int get hashCode => Object.hash(dimension, Object.hashAll(values));

  @override
  String toString() => 'TypeMatrix($dimension x $dimension)';
}
