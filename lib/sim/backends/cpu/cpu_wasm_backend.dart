import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:primordis/shared/constants/primordis_config.dart';
import 'package:primordis/sim/cpu/counting_sort_binning.dart';
import 'package:primordis/sim/cpu/cpu_sim_step.dart';
import 'package:primordis/sim/cpu/particle_soa.dart';
import 'package:primordis/sim/models/sim_capabilities.dart';
import 'package:primordis/sim/models/sim_params.dart';
import 'package:primordis/sim/models/sim_seed.dart';
import 'package:primordis/sim/render/cpu_points_painter.dart';
import 'package:primordis/sim/sim_backend.dart';
import 'package:primordis/sim/sim_seeder.dart';

/// The web CPU fallback tier (T4): a pure-Dart, single-threaded [SimBackend]
/// that runs entirely inside the `dart2wasm` build with **no GPU and no
/// FFI/JS-interop** ([PRIMORDIS-ADR-006], [PRIMORDIS-TASK-008]).
///
/// It is selected when `navigator.gpu` is absent or adapter/device acquisition
/// fails ([PRIMORDIS-TASK-007]), so WebGPU-less browsers still render a working
/// simulation. Its dependency tree is deliberately confined to
/// `dart:typed_data`, `dart:math`, `dart:ui`, and `package:flutter` foundation
/// — **zero** `dart:ffi`, `dart:js_interop`, `package:web`, `dart:html`, or
/// `dart:js_util` — so it compiles under `flutter build web --wasm`
/// ([PRIMORDIS-ADR-007]).
///
/// ## What it reproduces, and how it diverges
///
/// The per-frame physics ([cpuSimStep]) is faithful to the reference 3+1-pass
/// GPU pipeline (`Primordis.py`), but binning uses a **deterministic sequential
/// counting sort** ([countingSortBinning]) instead of `atomicAdd` scatter, and
/// rendering is a single `Canvas.drawRawPoints`-per-type blit rather than a GPU
/// point render. The 512-particle per-bin cap is intentionally not ported (see
/// [countingSortBinning]).
///
/// ## Honest particle count and pause
///
/// The tier ceiling is ~3-4k @ 60fps ([PrimordisConfig.cpuWasmDefaultParticleCount]
/// / [PrimordisConfig.cpuWasmMaxParticleCount]); it never silently runs 24k.
/// [actualParticleCount] reports the count actually being simulated so the
/// reduced-mode indicator ([PRIMORDIS-TASK-015]) can show the truth. [paused]
/// (the reduced-motion forward-hook, [PRIMORDIS-ADR-006]) suppresses stepping
/// and holds the last frame.
///
/// ## Present / render seam
///
/// [present] fills the reused packed [renderBuffer] (`2 * particleCount` floats,
/// the single-draw contract) and publishes an immutable [CpuFrame] on [frame]
/// for a [CpuPointsPainter] to blit. The physics never runs inside `paint`.
class CpuWasmBackend implements SimBackend {
  /// Creates a T4 backend. [maxParticles] defaults to the ADR-006 tier ceiling
  /// and clamps the seed count; the caller (backend selection) may lower it.
  CpuWasmBackend({
    int maxParticles = PrimordisConfig.cpuWasmMaxParticleCount,
    int defaultParticleCount = PrimordisConfig.cpuWasmDefaultParticleCount,
  })  : assert(maxParticles > 0, 'maxParticles must be positive'),
        assert(
          defaultParticleCount > 0 && defaultParticleCount <= maxParticles,
          'defaultParticleCount must be in (0, maxParticles]',
        ),
        _maxParticles = maxParticles,
        _defaultParticleCount = defaultParticleCount;

  final int _maxParticles;
  final int _defaultParticleCount;

  bool _initialized = false;
  bool _disposed = false;
  bool _paused = false;

  ParticleSoa? _buffers;
  GridGeometry? _grid;
  SimParams? _params;

  /// Per-type colours (parallel to per-type index groupings), built at seed.
  List<ui.Color> _colors = const <ui.Color>[];

  /// Per-type lists of particle indices, built once at seed so [present] can
  /// group render coordinates by type without re-scanning types every frame.
  List<Int32List> _indicesByType = const <Int32List>[];

  /// Double-buffered per-type packed `x, y` render buffers. [present] writes the
  /// back set and publishes it, then flips — so the frame it published last
  /// stays intact (a paused hold or an in-flight paint reads stable data) while
  /// the next present fills the other set. No per-frame allocation.
  List<Float32List> _pointsByTypeA = const <Float32List>[];
  List<Float32List> _pointsByTypeB = const <Float32List>[];
  bool _useA = true;

  /// The latest published frame for the painter. Repaints on identity change.
  final ValueNotifier<CpuFrame> frame = ValueNotifier<CpuFrame>(CpuFrame.empty);

  /// Whether [init] has run (and [dispose] has not).
  bool get isInitialized => _initialized;

  /// Whether [dispose] has run.
  bool get isDisposed => _disposed;

  /// Whether stepping is currently suppressed (reduced-motion pause).
  bool get paused => _paused;

  /// The count actually being simulated (0 before the first [seed]). Honest per
  /// [PRIMORDIS-ADR-006]; never the reference 24k.
  int get actualParticleCount => _buffers?.particleCount ?? 0;

  /// The reused single packed `x, y` render buffer (`2 * particleCount` floats).
  /// Filled by [present]; the single-draw contract's buffer.
  Float32List get renderBuffer =>
      _buffers?.renderXY ?? Float32List(0);

  @override
  SimBackendCapabilities get capabilities => SimBackendCapabilities(
        isGpuAccelerated: false,
        maxParticles: _maxParticles,
        defaultParticleCount: _defaultParticleCount,
        label: 'web-cpu-wasm',
      );

  @override
  Future<void> init() async {
    // The backend is single-use: dispose() permanently disposes the [frame]
    // ValueNotifier, so re-initing a disposed backend would leave it looking
    // alive while any present() crashes on the disposed notifier. Backend
    // selection must construct a fresh instance rather than reviving one.
    if (_disposed) {
      throw StateError('CpuWasmBackend is single-use; init after dispose');
    }
    _initialized = true;
  }

  @override
  Future<void> seed(SimSeed seed) async {
    if (!_initialized || _disposed) {
      throw StateError('seed() requires an initialized, non-disposed backend');
    }
    // Honour the tier ceiling: never simulate more than this tier can sustain.
    final count = seed.particleCount > _maxParticles
        ? _maxParticles
        : seed.particleCount;
    final effectiveSeed = seed.copyWith(particleCount: count);
    final seeded = seedSimulation(effectiveSeed);

    final grid = GridGeometry(
      worldWidth: PrimordisConfig.worldWidth.toDouble(),
      worldHeight: PrimordisConfig.worldHeight.toDouble(),
      binSize: PrimordisConfig.binSize.toDouble(),
    );
    final buffers = ParticleSoa(
      particleCount: count,
      typeCount: seeded.typeCount,
      binCount: grid.binCount,
    )..loadFrom(seeded);

    _grid = grid;
    _buffers = buffers;
    _colors = particleTypeColors(seeded.particleTypes);

    // Group particle indices by type once; buffers are reused every present.
    final byType = List<List<int>>.generate(
      seeded.typeCount,
      (_) => <int>[],
      growable: false,
    );
    for (var i = 0; i < count; i++) {
      byType[buffers.types[i]].add(i);
    }
    _indicesByType = <Int32List>[
      for (final list in byType) Int32List.fromList(list),
    ];
    _pointsByTypeA = <Float32List>[
      for (final list in _indicesByType) Float32List(list.length * 2),
    ];
    _pointsByTypeB = <Float32List>[
      for (final list in _indicesByType) Float32List(list.length * 2),
    ];
    _useA = true;

    // Publish an initial frame so a paused-from-start canvas has content.
    present();
  }

  @override
  void setParams(SimParams params) {
    _params = params;
  }

  @override
  void step(double dt) {
    if (_paused) return;
    final buffers = _buffers;
    final grid = _grid;
    final params = _params;
    if (buffers == null || grid == null || params == null) return;
    cpuSimStep(buffers, params, grid, dt);
  }

  @override
  void present() {
    final buffers = _buffers;
    if (buffers == null) return;

    // Single packed buffer (the "one draw call" contract's Float32List).
    fillRenderBuffer(buffers);

    // Fill the back set of per-type packed buffers for the coloured painter
    // (<=32 draws), then publish it and flip so the just-published set is not
    // overwritten by the next present.
    final target = _useA ? _pointsByTypeA : _pointsByTypeB;
    final positions = buffers.positions;
    for (var t = 0; t < _indicesByType.length; t++) {
      final indices = _indicesByType[t];
      final out = target[t];
      for (var k = 0; k < indices.length; k++) {
        final i = indices[k];
        out[k * 2] = positions[i * 2];
        out[k * 2 + 1] = positions[i * 2 + 1];
      }
    }
    _useA = !_useA;

    frame.value = CpuFrame(
      pointsByType: target,
      colors: _colors,
      pointSize: PrimordisConfig.pointSize,
    );
  }

  /// Pauses stepping and holds the last presented frame (reduced-motion).
  void pause() => _paused = true;

  /// Resumes stepping.
  void resume() => _paused = false;

  @override
  Future<void> dispose() async {
    // Idempotent: lifecycle/selection code may dispose defensively, and a
    // ValueNotifier must not be disposed twice.
    if (_disposed) return;
    _initialized = false;
    _disposed = true;
    _buffers = null;
    _grid = null;
    _params = null;
    _colors = const <ui.Color>[];
    _indicesByType = const <Int32List>[];
    _pointsByTypeA = const <Float32List>[];
    _pointsByTypeB = const <Float32List>[];
    frame
      ..value = CpuFrame.empty
      ..dispose();
  }
}
