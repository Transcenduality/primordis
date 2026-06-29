import 'package:flutter_test/flutter_test.dart';
import 'package:primordis/shared/constants/primordis_config.dart';
import 'package:primordis/sim/kernel/kernel_source.dart';

/// Extracts `name -> (group, binding)` for every `@group(g) @binding(b) var …`
/// declaration in the WGSL source, formatting-independently.
Map<String, ({int group, int binding})> _bindings(String src) {
  final re = RegExp(
    r'@group\((\d+)\)\s*@binding\((\d+)\)\s*var[^;]*?\b(\w+)\s*:',
  );
  return {
    for (final m in re.allMatches(src))
      m.group(3)!: (group: int.parse(m.group(1)!), binding: int.parse(m.group(2)!)),
  };
}

/// Strips `//` line comments so token checks inspect WGSL *code*, not the
/// header documentation (which intentionally names dart:js_interop / dart:ffi to
/// describe the source-of-truth contract). WGSL has no block comments here.
String _stripComments(String src) => src
    .split('\n')
    .map((line) {
      final i = line.indexOf('//');
      return i >= 0 ? line.substring(0, i) : line;
    })
    .join('\n');

void main() {
  // rootBundle needs an initialized binding to read the declared asset.
  TestWidgetsFlutterBinding.ensureInitialized();

  late String src;

  setUpAll(() async {
    resetKernelSourceCache();
    src = await loadKernelSource();
  });

  group('loadKernelSource', () {
    test('loads a non-trivial WGSL string from the asset', () {
      expect(src, isNotEmpty);
      expect(src.length, greaterThan(1000));
    });

    test('is memoised (same instance on repeat calls)', () async {
      final a = await loadKernelSource();
      final b = await loadKernelSource();
      expect(identical(a, b), isTrue);
    });
  });

  group('entry points', () {
    test('every declared entry-point name is defined in the source', () {
      for (final name in const [
        KernelEntryPoints.clearBins,
        KernelEntryPoints.scatterBins,
        KernelEntryPoints.interact,
        KernelEntryPoints.vertexMain,
        KernelEntryPoints.fragmentMain,
      ]) {
        expect(src, contains('fn $name('), reason: 'missing fn $name');
      }
    });

    test('computePasses is the three compute passes in dispatch order', () {
      expect(KernelEntryPoints.computePasses, const [
        KernelEntryPoints.clearBins,
        KernelEntryPoints.scatterBins,
        KernelEntryPoints.interact,
      ]);
    });

    test('the three compute passes are @compute stages', () {
      // Each compute entry point is immediately preceded by @compute
      // @workgroup_size(...).
      for (final name in KernelEntryPoints.computePasses) {
        expect(
          src,
          matches(RegExp(r'@compute\s*@workgroup_size\([^)]*\)\s*fn ' + name)),
          reason: '$name is not a @compute @workgroup_size entry point',
        );
      }
    });
  });

  group('shared constants mirror the shader', () {
    test('WORKGROUP_SIZE override default == KernelConfig.workgroupSize', () {
      final m = RegExp(r'override\s+WORKGROUP_SIZE\s*:\s*u32\s*=\s*(\d+)u')
          .firstMatch(src);
      expect(m, isNotNull, reason: 'WORKGROUP_SIZE override not found');
      expect(int.parse(m!.group(1)!), KernelConfig.workgroupSize);
    });

    test('MAX_BIN_PARTICLES override default == KernelConfig.maxBinParticles', () {
      final m = RegExp(r'override\s+MAX_BIN_PARTICLES\s*:\s*u32\s*=\s*(\d+)u')
          .firstMatch(src);
      expect(m, isNotNull, reason: 'MAX_BIN_PARTICLES override not found');
      expect(int.parse(m!.group(1)!), KernelConfig.maxBinParticles);
    });

    test('KernelConfig.maxBinParticles mirrors PrimordisConfig', () {
      expect(KernelConfig.maxBinParticles, PrimordisConfig.maxBinParticles);
    });

    test('@workgroup_size uses the WORKGROUP_SIZE override (no drift)', () {
      expect(src, contains('@workgroup_size(WORKGROUP_SIZE)'));
    });
  });

  group('WGSL atomics rules (ADR-003)', () {
    test('bin counters are array<atomic<u32>>', () {
      expect(src, contains('array<atomic<u32>>'));
    });

    test('binCounts is touched only through atomic builtins', () {
      // Every direct index of binCounts must be inside an atomic builtin, i.e.
      // preceded by `&` (atomicAdd/atomicLoad/atomicStore(&binCounts[...])).
      // No non-atomic alias — the exact class of bug Tint/Naga diverge on.
      final withoutAtomic = src.replaceAll('&binCounts[', '');
      expect(
        withoutAtomic.contains('binCounts['),
        isFalse,
        reason: 'binCounts is indexed without an atomic builtin somewhere',
      );
      expect(src, contains('atomicAdd(&binCounts['));
      expect(src, contains('atomicLoad(&binCounts['));
      expect(src, contains('atomicStore(&binCounts['));
    });
  });

  group('backend-agnostic source (no platform code)', () {
    test('contains no JS-interop / FFI / device-creation tokens (code only)', () {
      final code = _stripComments(src);
      for (final banned in const [
        'dart:',
        'package:',
        'js_interop',
        'navigator',
        'requestDevice',
        'requestAdapter',
        'createBuffer',
        'createComputePipeline',
      ]) {
        expect(
          code.contains(banned),
          isFalse,
          reason: 'kernel code must not contain "$banned"',
        );
      }
    });
  });

  group('bind-group / binding map matches KernelBindings', () {
    test('compute group (0) bindings line up with the shader', () {
      final b = _bindings(src);
      void check(String name, int binding) {
        expect(b[name]?.group, KernelBindings.computeGroup, reason: '$name group');
        expect(b[name]?.binding, binding, reason: '$name binding');
      }

      check('params', KernelBindings.params);
      check('positions', KernelBindings.positions);
      check('velocities', KernelBindings.velocities);
      check('types', KernelBindings.types);
      check('forces', KernelBindings.forces);
      check('minDistances', KernelBindings.minDistances);
      check('radii', KernelBindings.radii);
      check('binCounts', KernelBindings.binCounts);
      check('binParticles', KernelBindings.binParticles);
    });

    test('render group (1) bindings line up with the shader', () {
      final b = _bindings(src);
      void check(String name, int binding) {
        expect(b[name]?.group, KernelBindings.renderGroup, reason: '$name group');
        expect(b[name]?.binding, binding, reason: '$name binding');
      }

      check('renderParams', KernelBindings.renderParams);
      check('renderPositions', KernelBindings.renderPositions);
      check('renderTypes', KernelBindings.renderTypes);
      check('typeColors', KernelBindings.typeColors);
    });

    test('compute bindings are unique', () {
      const indices = [
        KernelBindings.params,
        KernelBindings.positions,
        KernelBindings.velocities,
        KernelBindings.types,
        KernelBindings.forces,
        KernelBindings.minDistances,
        KernelBindings.radii,
        KernelBindings.binCounts,
        KernelBindings.binParticles,
      ];
      expect(indices.toSet().length, indices.length);
    });
  });

  group('computeWorkgroups', () {
    test('is ceil(itemCount / workgroupSize)', () {
      expect(computeWorkgroups(0), 0);
      expect(computeWorkgroups(1), 1);
      expect(computeWorkgroups(256), 1);
      expect(computeWorkgroups(257), 2);
      // 24,000 particles at workgroup size 256 -> 94 groups.
      expect(computeWorkgroups(PrimordisConfig.particleCount), 94);
      // 77 bins -> a single group.
      expect(computeWorkgroups(PrimordisConfig.binCount), 1);
    });

    test('honours a custom workgroup size', () {
      expect(computeWorkgroups(100, workgroupSize: 64), 2);
      expect(computeWorkgroups(128, workgroupSize: 64), 2);
    });
  });
}
