import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:primordis/shared/constants/primordis_config.dart';
import 'package:primordis/sim/backends/macos/macos_dawn_backend.dart';
import 'package:primordis/sim/ffi/dawn_gpu.dart';
import 'package:primordis/sim/ffi/wgsl_pass_adapter.dart';
import 'package:primordis/sim/kernel/kernel_source.dart';
import 'package:primordis/sim/models/sim_params.dart';
import 'package:primordis/sim/models/sim_seed.dart';
import 'package:primordis/sim/sim_marshalling.dart';
import 'package:primordis/sim/sim_seeder.dart';

/// Contract tests for [MacosDawnBackend] over a recording fake [DawnGpu] —
/// no GPU device required (the real-device soak lives in `tool/spike/`).

void main() {
  final kernel = File(KernelConfig.assetPath).readAsStringSync();
  const seed = SimSeed(particleCount: 100);

  late FakeDawnGpu gpu;
  late MacosDawnBackend backend;

  SimParams paramsFor(SimSeed s) {
    final seeded = seedSimulation(s);
    return SimParams(
      forces: seeded.forces,
      minDistances: seeded.minDistances,
      radii: seeded.radii,
      particleCount: s.particleCount,
    );
  }

  setUp(() {
    gpu = FakeDawnGpu();
    backend = MacosDawnBackend(gpu, kernelLoader: () async => kernel);
  });

  group('init', () {
    test('builds the three adapted single-entry pipelines + params buffer',
        () async {
      await backend.init();
      expect(gpu.initCalled, isTrue);
      expect(gpu.passes, hasLength(3));
      for (final pass in gpu.passes) {
        expect(RegExp(r'\bfn\s+main\s*\(').allMatches(pass.wgsl), hasLength(1));
      }
      // Pass order mirrors the frame: clear → scatter → interact.
      expect(gpu.passes[0].wgsl, isNot(contains('fn clearBins')));
      expect(gpu.passes[1].wgsl, isNot(contains('fn scatterBins')));
      expect(gpu.passes[2].wgsl, isNot(contains('fn interact')));
      expect(gpu.buffers, hasLength(1));
      expect(gpu.buffers.single.byteSize, SimMarshalling.uniformByteLength);
    });

    test('surfaces device failure so selection can fall back', () async {
      gpu.failInit = true;
      await expectLater(backend.init(), throwsStateError);
    });

    test('is single-use: init after dispose throws', () async {
      await backend.init();
      await backend.dispose();
      await expectLater(backend.init(), throwsStateError);
    });
  });

  group('seed', () {
    test('before init throws', () {
      expect(() => backend.seed(seed), throwsStateError);
    });

    test('allocates byte-sized buffers and uploads the seeded state',
        () async {
      await backend.init();
      await backend.seed(seed);
      final n = seed.particleCount;
      final byBytes = {for (final b in gpu.buffers) b.byteSize: b};
      expect(byBytes[2 * n * 4]!.lastF32, isNotNull); // positions/velocities
      expect(byBytes[n * 4]!.lastU32, isNotNull); // types
      // Bin counts zeroed; bin particles allocated (binCount × cap × 4).
      final binCounts = byBytes[PrimordisConfig.binCount * 4]!;
      expect(binCounts.lastU32, everyElement(0));
      expect(
        byBytes.keys,
        contains(
          PrimordisConfig.binCount * PrimordisConfig.maxBinParticles * 4,
        ),
      );
      final seeded = seedSimulation(seed);
      final positions = gpu.buffers
          .firstWhere((b) => b.byteSize == 2 * n * 4)
          .lastF32;
      expect(positions, orderedEquals(seeded.positions));
    });
  });

  group('setParams', () {
    test('uploads the merged matrices buffer and binds sparse slots',
        () async {
      await backend.init();
      await backend.seed(seed);
      final params = paramsFor(seed);
      backend.setParams(params);
      await pumpQueue();

      final n2 = params.typeCount * params.typeCount;
      final matrices =
          gpu.buffers.firstWhere((b) => b.byteSize == 3 * n2 * 4);
      expect(
        matrices.lastF32,
        orderedEquals(
          mergedMatrices(
            flattenMatrix(params.forces),
            flattenMatrix(params.minDistances),
            flattenMatrix(params.radii),
          ),
        ),
      );
      // Every pass sees the same sparse slot set (5/6 vacant).
      for (final pass in gpu.passes) {
        expect(
          pass.bound.keys.toSet(),
          {0, 1, 2, 3, kMergedMatricesBinding, 7, 8},
        );
      }
    });
  });

  group('step', () {
    test('writes dt into the uniform then dispatches clear/scatter/interact',
        () async {
      await backend.init();
      await backend.seed(seed);
      backend.setParams(paramsFor(seed));
      await pumpQueue();

      backend.step(1 / 60);
      await pumpQueue();

      final uniform = gpu.buffers.single7bytes(
        SimMarshalling.uniformByteLength,
      );
      final unpacked = unpackUniforms(
        uniform.lastU32!.buffer.asUint8List(),
      );
      expect(unpacked.dt, closeTo(1 / 60, 1e-7));
      expect(unpacked.numParticles, seed.particleCount);

      expect(gpu.dispatches, [
        (0, computeWorkgroups(PrimordisConfig.binCount)),
        (1, computeWorkgroups(seed.particleCount)),
        (2, computeWorkgroups(seed.particleCount)),
      ]);
    });

    test('is a no-op before init/seed/params and after a device error',
        () async {
      backend.step(1 / 60); // un-initialized: silently ignored
      await pumpQueue();
      expect(gpu.dispatches, isEmpty);

      await backend.init();
      await backend.seed(seed);
      backend.setParams(paramsFor(seed));
      await pumpQueue();
      gpu.failDispatch = true;
      backend.step(1 / 60);
      await pumpQueue();
      expect(backend.deviceError, isNotNull);
      gpu.dispatches.clear();
      backend.step(1 / 60); // parked
      await pumpQueue();
      expect(gpu.dispatches, isEmpty);
    });

    test('drops ticks while a frame is in flight', () async {
      await backend.init();
      await backend.seed(seed);
      backend.setParams(paramsFor(seed));
      await pumpQueue();

      gpu.holdDispatches = true;
      backend.step(1 / 60);
      backend.step(1 / 60); // dropped — first frame still in flight
      gpu.releaseHeldDispatches();
      await pumpQueue();
      expect(gpu.dispatches, hasLength(3)); // one frame, not two
    });
  });

  test('dispose destroys all passes, buffers, and the device', () async {
    await backend.init();
    await backend.seed(seed);
    backend.setParams(paramsFor(seed));
    await pumpQueue();
    await backend.dispose();
    expect(gpu.destroyed, isTrue);
    expect(gpu.passes.every((p) => p.isDestroyed), isTrue);
    expect(gpu.buffers.every((b) => b.isDestroyed), isTrue);
  });

  test('capabilities report the GPU tier at the reference workload', () {
    expect(backend.capabilities.isGpuAccelerated, isTrue);
    expect(backend.capabilities.maxParticles, PrimordisConfig.particleCount);
    expect(backend.capabilities.label, 'macos-dawn');
  });
}

/// Drains microtasks/futures queued by the backend's internal chains.
Future<void> pumpQueue() => Future<void>.delayed(Duration.zero);

extension on List<FakeDawnBuffer> {
  FakeDawnBuffer single7bytes(int byteSize) =>
      firstWhere((b) => b.byteSize == byteSize);
}

final class FakeDawnGpu implements DawnGpu {
  bool initCalled = false;
  bool failInit = false;
  bool destroyed = false;
  bool failDispatch = false;
  bool holdDispatches = false;

  final List<FakeDawnPass> passes = [];
  final List<FakeDawnBuffer> buffers = [];

  /// `(passIndex, groupsX)` in dispatch order.
  final List<(int, int)> dispatches = [];
  final List<Completer<void>> _held = [];

  void releaseHeldDispatches() {
    holdDispatches = false;
    for (final c in _held) {
      c.complete();
    }
    _held.clear();
  }

  @override
  Future<void> init() async {
    if (failInit) throw StateError('fake Dawn init failure');
    initCalled = true;
  }

  @override
  DawnGpuBuffer createF32Buffer(int byteSize) => _create(byteSize);

  @override
  DawnGpuBuffer createU32Buffer(int byteSize) => _create(byteSize);

  FakeDawnBuffer _create(int byteSize) {
    final buffer = FakeDawnBuffer(byteSize);
    buffers.add(buffer);
    return buffer;
  }

  @override
  DawnGpuPass createPass(String wgsl) {
    final pass = FakeDawnPass(wgsl, this, passes.length);
    passes.add(pass);
    return pass;
  }

  @override
  Future<void> destroy() async => destroyed = true;
}

final class FakeDawnPass implements DawnGpuPass {
  FakeDawnPass(this.wgsl, this._gpu, this._index);

  final String wgsl;
  final FakeDawnGpu _gpu;
  final int _index;
  final Map<int, DawnGpuBuffer> bound = {};
  bool isDestroyed = false;

  @override
  void bind(int slot, DawnGpuBuffer buffer) => bound[slot] = buffer;

  @override
  Future<void> dispatch(int groupsX) async {
    if (_gpu.failDispatch) throw StateError('fake device lost');
    if (_gpu.holdDispatches) {
      final completer = Completer<void>();
      _gpu._held.add(completer);
      await completer.future;
    }
    _gpu.dispatches.add((_index, groupsX));
  }

  @override
  void destroy() => isDestroyed = true;
}

final class FakeDawnBuffer implements DawnGpuBuffer {
  FakeDawnBuffer(this.byteSize);

  @override
  final int byteSize;
  Float32List? lastF32;
  Uint32List? lastU32;
  bool isDestroyed = false;

  @override
  Future<void> writeF32(Float32List data) async {
    _check(data);
    lastF32 = Float32List.fromList(data);
  }

  @override
  Future<void> writeU32(Uint32List data) async {
    _check(data);
    lastU32 = Uint32List.fromList(data);
  }

  @override
  Future<void> readF32(Float32List out) async => out.setAll(0, lastF32!);

  @override
  Future<void> readU32(Uint32List out) async => out.setAll(0, lastU32!);

  @override
  void destroy() => isDestroyed = true;

  void _check(TypedData data) {
    if (data.lengthInBytes != byteSize) {
      throw ArgumentError(
        'size mismatch: $byteSize vs ${data.lengthInBytes}',
      );
    }
  }
}
