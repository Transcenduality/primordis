import 'dart:async';
import 'dart:typed_data';

import 'package:primordis/shared/constants/primordis_config.dart';
import 'package:primordis/sim/ffi/dawn_gpu.dart';
import 'package:primordis/sim/ffi/wgsl_pass_adapter.dart';
import 'package:primordis/sim/kernel/kernel_source.dart';
import 'package:primordis/sim/models/sim_capabilities.dart';
import 'package:primordis/sim/models/sim_params.dart';
import 'package:primordis/sim/models/sim_seed.dart';
import 'package:primordis/sim/sim_backend.dart';
import 'package:primordis/sim/sim_marshalling.dart';
import 'package:primordis/sim/sim_seeder.dart';

/// The native macOS [SimBackend]: the shared WGSL kernel on **Dawn/wgpu over
/// Metal**, reached through [DawnGpu] (minigpu via `dart:ffi`) —
/// PRIMORDIS-TASK-801, PRIMORDIS-ADR-004 Approach (a).
///
/// De-risked by the TASK-801 spike (`tool/spike/`): the exact 3-pass
/// atomic-binning kernel at 24k/32-types ran 1000 frames on Dawn-over-Metal
/// (Apple M1 Max) with no device-lost, no NaN/Inf, exact bin conservation,
/// ~75 fps *including* per-pass awaits and validation readback.
///
/// `flutter_gpu` re-check (ADR-004 gate): against the pinned Flutter 3.44.0
/// (engine 4c525dac5e) `flutter_gpu` still exposes **no compute API** — its
/// Dart surface has no `ComputePass`; this backend remains necessary.
///
/// Scope boundary: this class ends at "compute runs and the storage buffers
/// hold correct results". [present] is a no-op until the IOSurface-backed
/// Metal texture present path lands (PRIMORDIS-TASK-802 / ADR-005). The
/// point-render stages and per-type colours are therefore not uploaded here.
///
/// Frame pacing: [step] is synchronous-enqueue per the [SimBackend] contract,
/// but minigpu dispatch completion is a `Future`. Passes within one frame are
/// strictly ordered on an internal chain; if a new tick arrives while the
/// previous frame's chain is still in flight the tick is DROPPED (the GPU is
/// the bottleneck — queueing would only grow latency unboundedly).
///
/// Single-use, like the CPU tier: construct a fresh instance per session;
/// [init] after [dispose] throws.
final class MacosDawnBackend implements SimBackend {
  MacosDawnBackend(this._gpu, {Future<String> Function()? kernelLoader})
      : _kernelLoader = kernelLoader ?? loadKernelSource;

  final DawnGpu _gpu;
  final Future<String> Function() _kernelLoader;

  DawnGpuPass? _clearBins;
  DawnGpuPass? _scatterBins;
  DawnGpuPass? _interact;

  DawnGpuBuffer? _paramsBuf;
  DawnGpuBuffer? _positions;
  DawnGpuBuffer? _velocities;
  DawnGpuBuffer? _types;
  DawnGpuBuffer? _matrices;
  DawnGpuBuffer? _binCounts;
  DawnGpuBuffer? _binParticles;

  SimParams? _params;
  int _particleCount = 0;
  bool _initialized = false;
  bool _disposed = false;
  bool _frameInFlight = false;
  Object? _deviceError;

  @override
  Future<void> init() async {
    if (_disposed) {
      throw StateError('MacosDawnBackend is single-use: init after dispose');
    }
    if (_initialized) return;
    await _gpu.init();
    final source = await _kernelLoader();
    _clearBins = _gpu.createPass(
      adaptKernelForMinigpu(source, KernelEntryPoints.clearBins),
    );
    _scatterBins = _gpu.createPass(
      adaptKernelForMinigpu(source, KernelEntryPoints.scatterBins),
    );
    _interact = _gpu.createPass(
      adaptKernelForMinigpu(source, KernelEntryPoints.interact),
    );
    _paramsBuf = _gpu.createU32Buffer(SimMarshalling.uniformByteLength);
    _initialized = true;
  }

  @override
  Future<void> seed(SimSeed seed) async {
    _requireInit('seed');
    final seeded = seedSimulation(seed);

    // (Re)allocate the per-particle and grid buffers when the population
    // changes (first seed allocates everything).
    if (seeded.particleCount != _particleCount) {
      _positions?.destroy();
      _velocities?.destroy();
      _types?.destroy();
      _binCounts?.destroy();
      _binParticles?.destroy();
      _particleCount = seeded.particleCount;
      final n = _particleCount;
      _positions = _gpu.createF32Buffer(2 * n * 4);
      _velocities = _gpu.createF32Buffer(2 * n * 4);
      _types = _gpu.createU32Buffer(n * 4);
      _binCounts = _gpu.createU32Buffer(PrimordisConfig.binCount * 4);
      _binParticles = _gpu.createU32Buffer(
        PrimordisConfig.binCount * PrimordisConfig.maxBinParticles * 4,
      );
      _bindAll();
    }

    await _positions!.writeF32(seeded.positions);
    await _velocities!.writeF32(seeded.velocities);
    await _types!.writeU32(seeded.types);
    // Fresh grid state; matrices arrive via [setParams] (the seeded matrices
    // are carried inside SimParams by the shared model, PRIMORDIS-TASK-002).
    await _binCounts!.writeU32(Uint32List(PrimordisConfig.binCount));
  }

  @override
  void setParams(SimParams params) {
    _requireInit('setParams');
    _params = params;
    // Merged [forces, minDistances, radii] upload — cheap (3 × typeCount²
    // floats) and setParams runs only on change. The uniform block itself is
    // written per-step (it carries dt).
    final merged = mergedMatrices(
      flattenMatrix(params.forces),
      flattenMatrix(params.minDistances),
      flattenMatrix(params.radii),
    );
    final matrices = _matrices ??= _gpu.createF32Buffer(merged.lengthInBytes);
    _bindAll();
    unawaited(matrices.writeF32(merged));
  }

  @override
  void step(double dt) {
    final params = _params;
    if (!_initialized ||
        _disposed ||
        params == null ||
        _positions == null ||
        _matrices == null ||
        _deviceError != null) {
      return;
    }
    if (_frameInFlight) return; // GPU is behind — drop the tick, don't queue.
    _frameInFlight = true;

    final uniform = packUniforms(params, dt);
    final binGroups = computeWorkgroups(PrimordisConfig.binCount);
    final particleGroups = computeWorkgroups(_particleCount);
    unawaited(() async {
      try {
        await _paramsBuf!.writeU32(uniform.buffer.asUint32List());
        await _clearBins!.dispatch(binGroups);
        await _scatterBins!.dispatch(particleGroups);
        await _interact!.dispatch(particleGroups);
      } catch (e) {
        // Device lost / validation failure: park the backend; selection
        // (TASK-805) reads a dead backend as "fall back".
        _deviceError = e;
      } finally {
        _frameInFlight = false;
      }
    }());
  }

  @override
  void present() {
    // No-op until the IOSurface-backed Metal texture present path
    // (PRIMORDIS-TASK-802). Compute state stays GPU-resident; no readback.
  }

  @override
  Future<void> dispose() async {
    if (_disposed) return;
    _disposed = true;
    for (final pass in [_clearBins, _scatterBins, _interact]) {
      pass?.destroy();
    }
    for (final buf in [
      _paramsBuf,
      _positions,
      _velocities,
      _types,
      _matrices,
      _binCounts,
      _binParticles,
    ]) {
      buf?.destroy();
    }
    await _gpu.destroy();
  }

  @override
  SimBackendCapabilities get capabilities => const SimBackendCapabilities(
        isGpuAccelerated: true,
        // The reference workload; Apple-Silicon headroom beyond 24k is
        // quantified by the TASK-809 benchmark harness, not assumed here.
        maxParticles: PrimordisConfig.particleCount,
        defaultParticleCount: PrimordisConfig.particleCount,
        label: 'macos-dawn',
      );

  /// The first device error [step] encountered, if any (selection/diagnostic
  /// seam; a non-null value means the backend has parked itself).
  Object? get deviceError => _deviceError;

  /// Rebinds every live buffer on all three passes (idempotent; minigpu
  /// rebuilds bind groups lazily on next dispatch).
  void _bindAll() {
    final bindings = <int, DawnGpuBuffer?>{
      KernelBindings.params: _paramsBuf,
      KernelBindings.positions: _positions,
      KernelBindings.velocities: _velocities,
      KernelBindings.types: _types,
      kMergedMatricesBinding: _matrices,
      KernelBindings.binCounts: _binCounts,
      KernelBindings.binParticles: _binParticles,
    };
    for (final pass in [_clearBins!, _scatterBins!, _interact!]) {
      for (final entry in bindings.entries) {
        final buffer = entry.value;
        if (buffer != null) pass.bind(entry.key, buffer);
      }
    }
  }

  void _requireInit(String op) {
    if (_disposed) throw StateError('MacosDawnBackend used after dispose');
    if (!_initialized) throw StateError('$op called before init()');
  }
}
