import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:primordis/sim/ffi/wgsl_pass_adapter.dart';
import 'package:primordis/sim/kernel/kernel_source.dart';

/// Tests run from the package root, so the canonical kernel is read straight
/// off disk — the adapter must work against the real source, not a fixture.
final String kernelSource =
    File(KernelConfig.assetPath).readAsStringSync();

void main() {
  group('adaptKernelForMinigpu', () {
    for (final pass in KernelEntryPoints.computePasses) {
      group('pass $pass', () {
        late String adapted;

        setUp(() => adapted = adaptKernelForMinigpu(kernelSource, pass));

        test('renames exactly that pass to `main`', () {
          expect(RegExp(r'\bfn\s+main\s*\(').allMatches(adapted), hasLength(1));
          expect(RegExp('\\bfn\\s+$pass\\s*\\(').hasMatch(adapted), isFalse);
          // The other two passes keep their names (same module, unused).
          for (final other in KernelEntryPoints.computePasses) {
            if (other == pass) continue;
            expect(
              RegExp('\\bfn\\s+$other\\s*\\(').hasMatch(adapted),
              isTrue,
              reason: 'pass $other should remain declared',
            );
          }
        });

        test(
            'rewrites every @group(0) binding to read_write storage and '
            'stays within the 8-storage-buffer stage limit', () {
          final group0Vars = RegExp(
            r'@group\(0\)\s*@binding\(\d+\)\s*var<([^>]+)>',
          ).allMatches(adapted).map((m) => m[1]).toList();
          // 7 bindings after the matrix merge (params, positions, velocities,
          // types, mergedMatrices, binCounts, binParticles) — ≤ 8, the WebGPU
          // default maxStorageBuffersPerShaderStage.
          expect(group0Vars, hasLength(7));
          expect(group0Vars, everyElement('storage, read_write'));
        });

        test('merges the three matrices into one binding-4 buffer', () {
          // One merged declaration at the forces slot; originals gone.
          expect(
            RegExp(
              r'@group\(0\)\s*@binding\(4\)\s*var<storage, read_write>\s*'
              r'mergedMatrices\s*:\s*array<f32>;',
            ).allMatches(adapted),
            hasLength(1),
          );
          for (final gone in ['forces', 'minDistances', 'radii']) {
            expect(
              RegExp('var<[^>]+>\\s*$gone\\s*:').hasMatch(adapted),
              isFalse,
              reason: '$gone declaration should be merged away',
            );
          }
          // Indexed reads target the merged buffer at 0 / n² / 2n².
          expect(adapted, contains('mergedMatrices[idx]'));
          expect(
            adapted,
            contains(
              'mergedMatrices[params.typeCount * params.typeCount + (idx)]',
            ),
          );
          expect(
            adapted,
            contains(
              'mergedMatrices[2u * params.typeCount * params.typeCount + '
              '(idx)]',
            ),
          );
        });

        test('leaves @group(1) render declarations untouched', () {
          expect(
            RegExp(
              r'@group\(1\)\s*@binding\(0\)\s*var<uniform>',
            ).hasMatch(adapted),
            isTrue,
          );
          final group1ReadOnly = RegExp(
            r'@group\(1\)\s*@binding\([123]\)\s*var<storage, read>',
          ).allMatches(adapted);
          expect(group1ReadOnly, hasLength(3));
        });

        test('does not touch the physics bodies', () {
          // Spot-check load-bearing physics lines survive verbatim.
          expect(
            adapted,
            contains('abs(forceStrength) * 5.0 * (1.0 - dist / mind)'),
          );
          expect(adapted, contains('atomicAdd(&binCounts[binIdx], 1u)'));
          expect(adapted, contains('v = v * params.friction;'));
        });
      });
    }

    test('mergedMatrices concatenates in kernel-index order', () {
      final merged = mergedMatrices(
        Float32List.fromList([1, 2]),
        Float32List.fromList([3, 4]),
        Float32List.fromList([5, 6]),
      );
      expect(merged, orderedEquals([1, 2, 3, 4, 5, 6]));
    });

    test('rejects unknown entry points', () {
      expect(
        () => adaptKernelForMinigpu(kernelSource, 'vs_main'),
        throwsArgumentError,
      );
      expect(
        () => adaptKernelForMinigpu(kernelSource, 'nope'),
        throwsArgumentError,
      );
    });

    test('rejects sources that already declare fn main', () {
      const collision = '''
@group(0) @binding(0) var<storage, read_write> x : array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {}
@compute @workgroup_size(64)
fn clearBins(@builtin(global_invocation_id) gid: vec3<u32>) {}
''';
      expect(
        () => adaptKernelForMinigpu(collision, KernelEntryPoints.clearBins),
        throwsStateError,
      );
    });

    test('rejects a pass name missing from the source', () {
      const noSuchPass = '''
@group(0) @binding(0) var<uniform> params : Params;
@compute @workgroup_size(64)
fn clearBins(@builtin(global_invocation_id) gid: vec3<u32>) {}
''';
      expect(
        () => adaptKernelForMinigpu(noSuchPass, KernelEntryPoints.interact),
        throwsArgumentError,
      );
    });
  });
}
