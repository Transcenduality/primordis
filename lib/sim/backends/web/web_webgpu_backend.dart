// The web `SimBackend`: runs the shared WGSL kernel on browser WebGPU.
//
// Acquires `navigator.gpu` → adapter → device, builds the storage/uniform
// buffers, the three compute pipelines (clear / scatter-bin / interact+integrate)
// and the point-render pipeline — all from the ONE shared kernel source
// ([PRIMORDIS-TASK-003]) — and dispatches the four passes per frame onto a
// canvas it owns ([web_canvas_handle.dart]).
//
// This is the GPU path that delivers the full 24,000+ particles at 60fps where
// WebGPU is present ([PRIMORDIS-ADR-002]). It is quarantined below the
// `SimBackend` seam ([PRIMORDIS-ADR-001]); the UI never sees a WebGPU type.
//
// Web-only: `dart:js_interop` + `package:web`, reachable only via the
// conditional facade ([web_backend.dart]). No `dart:html`, no `dart:js_util`
// ([PRIMORDIS-ADR-007]).
@JS()
library;

import 'dart:async';
import 'dart:js_interop';

import 'package:primordis/shared/constants/primordis_config.dart';
import 'package:primordis/sim/backends/web/buffer_marshalling.dart';
import 'package:primordis/sim/backends/web/web_canvas_handle.dart';
import 'package:primordis/sim/backends/web/webgpu_interop.dart';
import 'package:primordis/sim/backends/web/webgpu_support.dart';
import 'package:primordis/sim/kernel/buffer_layout.dart';
import 'package:primordis/sim/kernel/kernel_source.dart';
import 'package:primordis/sim/models/sim_capabilities.dart';
import 'package:primordis/sim/models/sim_params.dart';
import 'package:primordis/sim/models/sim_seed.dart';
import 'package:primordis/sim/sim_backend.dart';
import 'package:primordis/sim/sim_marshalling.dart';
import 'package:primordis/sim/sim_seeder.dart';

/// Thrown by [WebWebGpuBackend.init] when device acquisition fails after the
/// feature-detect said WebGPU was present (adapter/device became null or the
/// device was lost during bring-up). Typed so the selector ([PRIMORDIS-TASK-007])
/// can catch it and fall back to the CPU tier instead of crashing the app.
class WebGpuUnavailableException implements Exception {
  const WebGpuUnavailableException(this.message);
  final String message;
  @override
  String toString() => 'WebGpuUnavailableException: $message';
}

/// Browser-WebGPU [SimBackend].
///
/// Lifecycle (the contract on [SimBackend]): [init] (device + pipelines) →
/// [seed] (sized buffers + bind groups + initial upload) → per frame
/// [setParams] (on change) → [step] (3 compute passes) → [present] (point
/// render). [dispose] releases every GPU resource.
///
/// The backend is inherently **single-step driven** — it holds no internal
/// animation loop; it advances only when its driver calls [step]. That directly
/// satisfies the reduced-motion / pause requirement ([PRIMORDIS-ADR-006]): a
/// paused driver simply stops calling [step]/[present] and the last frame holds.
class WebWebGpuBackend implements SimBackend {
  /// Probes WebGPU availability without constructing the backend. Never throws —
  /// returns a [WebGpuSupport] the selector ([PRIMORDIS-TASK-007]) branches on.
  ///
  /// Reads `navigator.gpu` and attempts `requestAdapter()`; the pure
  /// classification ([classifyWebGpuProbe]) maps the two observations onto the
  /// result so the decision table is unit-tested separately from this shim.
  static Future<WebGpuSupport> probe() async {
    try {
      final gpu = navigatorGpu;
      if (gpu == null) {
        return classifyWebGpuProbe(hasGpuApi: false, hasAdapter: false);
      }
      final adapter = await gpu.requestAdapter().toDart;
      return classifyWebGpuProbe(hasGpuApi: true, hasAdapter: adapter != null);
    } catch (_) {
      return WebGpuSupport.error;
    }
  }

  @override
  SimBackendCapabilities get capabilities => const SimBackendCapabilities(
        isGpuAccelerated: true,
        maxParticles: PrimordisConfig.particleCount,
        defaultParticleCount: PrimordisConfig.particleCount,
        label: 'web-webgpu',
      );

  // --- Device-scoped handles (built in init, fixed for the backend's life) ---
  GPUDevice? _device;
  GPUQueue? _queue;
  /// The device's preferred canvas format, computed once in [init] and shared by
  /// the render pipeline's colour target and the canvas context configuration so
  /// they can never diverge.
  String? _canvasFormat;
  GPUBindGroupLayout? _computeLayout;
  GPUBindGroupLayout? _renderLayout;
  GPUComputePipeline? _clearPipeline;
  GPUComputePipeline? _scatterPipeline;
  GPUComputePipeline? _interactPipeline;
  GPURenderPipeline? _renderPipeline;
  GPUBuffer? _uniformBuffer;

  // --- Seed-scoped handles (rebuilt on every seed; sized from SimParams) ---
  WebCanvasHandle? _canvas;
  GPUBuffer? _positions;
  GPUBuffer? _velocities;
  GPUBuffer? _types;
  GPUBuffer? _forces;
  GPUBuffer? _minDistances;
  GPUBuffer? _radii;
  GPUBuffer? _typeColors;
  GPUBuffer? _binCounts;
  GPUBuffer? _binParticles;
  GPUBindGroup? _computeBindGroup;
  GPUBindGroup? _renderBindGroup;

  SimParams? _params;
  bool _disposed = false;
  bool _deviceLost = false;

  /// True once [init] has acquired a device and built the pipelines.
  bool get isInitialized => _device != null && !_disposed;

  /// True once [seed] has built the buffers/bind groups and the sim can step.
  bool get isSeeded => _computeBindGroup != null;

  /// The owned canvas element, for TASK-005 to stack/size (null before [seed]).
  Object? get canvasElement => _canvas?.canvas;

  @override
  Future<void> init() async {
    if (_disposed) {
      throw StateError('WebWebGpuBackend used after dispose');
    }
    final gpu = navigatorGpu;
    if (gpu == null) {
      throw const WebGpuUnavailableException('navigator.gpu is unavailable');
    }
    final adapter = await gpu.requestAdapter().toDart;
    if (adapter == null) {
      throw const WebGpuUnavailableException('requestAdapter() returned null');
    }
    final GPUDevice device;
    try {
      device = await adapter.requestDevice().toDart;
    } catch (e) {
      throw WebGpuUnavailableException('requestDevice() failed: $e');
    }
    // Surface device-loss so an in-flight sim degrades instead of throwing on
    // every subsequent encode (the selector can re-probe / fall back).
    unawaited(device.lost.toDart.then((_) => _deviceLost = true));

    final source = await loadKernelSource();
    final module = device.createShaderModule(
      GPUShaderModuleDescriptor(code: source),
    );

    final computeLayout = device.createBindGroupLayout(
      _computeLayoutDescriptor(),
    );
    final renderLayout = device.createBindGroupLayout(
      _renderLayoutDescriptor(),
    );
    final computePipelineLayout = device.createPipelineLayout(
      GPUPipelineLayoutDescriptor(
        bindGroupLayouts: <GPUBindGroupLayout>[computeLayout].toJS,
      ),
    );
    // The render pipeline references @group(1); pipeline-layout indices are
    // positional, so group 0 is the (reused) compute layout and is bound — but
    // unused — by the render shader.
    final renderPipelineLayout = device.createPipelineLayout(
      GPUPipelineLayoutDescriptor(
        bindGroupLayouts:
            <GPUBindGroupLayout>[computeLayout, renderLayout].toJS,
      ),
    );

    final constants = GPUPipelineConstants(
      WORKGROUP_SIZE: KernelConfig.workgroupSize,
      MAX_BIN_PARTICLES: KernelConfig.maxBinParticles,
    );
    GPUComputePipeline computePipe(String entryPoint) =>
        device.createComputePipeline(
          GPUComputePipelineDescriptor(
            layout: computePipelineLayout,
            compute: GPUProgrammableStage(
              module: module,
              entryPoint: entryPoint,
              constants: constants,
            ),
          ),
        );

    _clearPipeline = computePipe(KernelEntryPoints.clearBins);
    _scatterPipeline = computePipe(KernelEntryPoints.scatterBins);
    _interactPipeline = computePipe(KernelEntryPoints.interact);
    // Compute the preferred canvas format ONCE; the same value feeds the render
    // pipeline's colour target here and the canvas configuration in seed(), so
    // a present() format mismatch can't arise from two independent queries.
    final canvasFormat = gpu.getPreferredCanvasFormat();
    _renderPipeline = device.createRenderPipeline(
      GPURenderPipelineDescriptor(
        layout: renderPipelineLayout,
        vertex: GPUVertexState(
          module: module,
          entryPoint: KernelEntryPoints.vertexMain,
        ),
        fragment: GPUFragmentState(
          module: module,
          entryPoint: KernelEntryPoints.fragmentMain,
          targets: <GPUColorTargetState>[
            GPUColorTargetState(format: canvasFormat),
          ].toJS,
        ),
        primitive: GPUPrimitiveState(topology: 'point-list'),
      ),
    );

    _uniformBuffer = device.createBuffer(
      GPUBufferDescriptor(
        size: SimMarshalling.uniformByteLength,
        usage: GpuBufferUsage.uniform | GpuBufferUsage.copyDst,
      ),
    );

    _device = device;
    _queue = device.queue;
    _canvasFormat = canvasFormat;
    _computeLayout = computeLayout;
    _renderLayout = renderLayout;
  }

  @override
  Future<void> seed(SimSeed seed) async {
    final device = _device;
    final queue = _queue;
    if (device == null || queue == null || _canvasFormat == null) {
      throw StateError('seed() called before init()');
    }
    final seeded = seedSimulation(seed);
    final params = SimParams(
      forces: seeded.forces,
      minDistances: seeded.minDistances,
      radii: seeded.radii,
      particleCount: seeded.particleCount,
      typeCount: seeded.typeCount,
    );
    _params = params;
    final layout = SimBufferLayout(params);

    // Re-seed releases the previous seed-scoped buffers before reallocating.
    _disposeSeedBuffers();

    GPUBuffer storage(int size) => device.createBuffer(
          GPUBufferDescriptor(
            size: size,
            usage: GpuBufferUsage.storage | GpuBufferUsage.copyDst,
          ),
        );

    _positions = storage(layout.positions);
    _velocities = storage(layout.velocities);
    _types = storage(layout.types);
    _forces = storage(layout.forces);
    _minDistances = storage(layout.minDistances);
    _radii = storage(layout.radii);
    _typeColors = storage(layout.typeColors);
    // Bin buffers carry no seed payload — WebGPU zero-initializes them and the
    // kernel clears/fills them each frame.
    _binCounts = storage(layout.binCounts);
    _binParticles = storage(layout.binParticles);

    // Upload the deterministic seed (SoA + matrices + colours).
    final buffers = packSeedBuffers(seeded);
    assert(() {
      verifySeedBuffersMatchLayout(buffers, layout);
      return true;
    }());
    queue
      ..writeBuffer(_positions!, 0, buffers.positions.toJS)
      ..writeBuffer(_velocities!, 0, buffers.velocities.toJS)
      ..writeBuffer(_types!, 0, buffers.types.toJS)
      ..writeBuffer(_forces!, 0, buffers.forces.toJS)
      ..writeBuffer(_minDistances!, 0, buffers.minDistances.toJS)
      ..writeBuffer(_radii!, 0, buffers.radii.toJS)
      ..writeBuffer(_typeColors!, 0, buffers.typeColors.toJS);

    _buildBindGroups(device);
    _ensureCanvas(device, params);
    _writeUniform(0); // a valid uniform before the first step (dt = 0).
  }

  @override
  void setParams(SimParams params) {
    final queue = _queue;
    if (queue == null) return;
    _params = params;
    // Matrices are seeded data and don't change with the sliders, but re-upload
    // them so an externally-supplied matrix set (e.g. a future editor) takes
    // effect; cheap (3 × typeCount² floats), and setParams runs only on change.
    // The live sliders + dt reach the GPU through the per-frame uniform write in
    // [step]; nothing to upload here for them.
    if (_forces != null) {
      queue
        ..writeBuffer(_forces!, 0, flattenMatrix(params.forces).toJS)
        ..writeBuffer(_minDistances!, 0, flattenMatrix(params.minDistances).toJS)
        ..writeBuffer(_radii!, 0, flattenMatrix(params.radii).toJS);
    }
  }

  @override
  void step(double dt) {
    final device = _device;
    final queue = _queue;
    final params = _params;
    final bindGroup = _computeBindGroup;
    if (device == null || queue == null || params == null ||
        bindGroup == null || _deviceLost) {
      return;
    }
    // Slider + dt + counts uniform, written every frame (TASK-006 drives the
    // live slider state into [_params]).
    _writeUniform(dt);

    final encoder = device.createCommandEncoder();
    final pass = encoder.beginComputePass();
    final binGroups = computeWorkgroups(params.binCount);
    final partGroups = computeWorkgroups(params.particleCount);

    pass
      ..setPipeline(_clearPipeline!)
      ..setBindGroup(KernelBindings.computeGroup, bindGroup)
      ..dispatchWorkgroups(binGroups)
      ..setPipeline(_scatterPipeline!)
      ..setBindGroup(KernelBindings.computeGroup, bindGroup)
      ..dispatchWorkgroups(partGroups)
      ..setPipeline(_interactPipeline!)
      ..setBindGroup(KernelBindings.computeGroup, bindGroup)
      ..dispatchWorkgroups(partGroups)
      ..end();
    queue.submit(<GPUCommandBuffer>[encoder.finish()].toJS);
  }

  @override
  void present() {
    final device = _device;
    final queue = _queue;
    final params = _params;
    final canvas = _canvas;
    final renderBindGroup = _renderBindGroup;
    final computeBindGroup = _computeBindGroup;
    if (device == null || queue == null || params == null || canvas == null ||
        renderBindGroup == null || computeBindGroup == null || _deviceLost) {
      return;
    }
    final encoder = device.createCommandEncoder();
    final pass = encoder.beginRenderPass(
      GPURenderPassDescriptor(
        colorAttachments: <GPURenderPassColorAttachment>[
          GPURenderPassColorAttachment(
            view: canvas.currentView(),
            // Transparent clear so the canvas composites under the Flutter
            // glass-pane (TASK-005 / ADR-005).
            clearValue: GPUColorDict(r: 0, g: 0, b: 0, a: 0),
            loadOp: 'clear',
            storeOp: 'store',
          ),
        ].toJS,
      ),
    );
    pass
      ..setPipeline(_renderPipeline!)
      // Group 0 is unused by the render shader but present in the pipeline
      // layout; bind the compatible compute group to satisfy the positional
      // layout, then bind the render views at group 1.
      ..setBindGroup(KernelBindings.computeGroup, computeBindGroup)
      ..setBindGroup(KernelBindings.renderGroup, renderBindGroup)
      ..draw(params.particleCount)
      ..end();
    queue.submit(<GPUCommandBuffer>[encoder.finish()].toJS);
  }

  @override
  Future<void> dispose() async {
    _disposeSeedBuffers();
    _canvas?.dispose();
    _canvas = null;
    _uniformBuffer?.destroy();
    _uniformBuffer = null;
    _device?.destroy();
    _device = null;
    _queue = null;
    _canvasFormat = null;
    _computeLayout = null;
    _renderLayout = null;
    _clearPipeline = null;
    _scatterPipeline = null;
    _interactPipeline = null;
    _renderPipeline = null;
    _params = null;
    _disposed = true;
  }

  // --- internals -----------------------------------------------------------

  void _writeUniform(double dt) {
    final queue = _queue;
    final uniform = _uniformBuffer;
    final params = _params;
    if (queue == null || uniform == null || params == null) return;
    queue.writeBuffer(uniform, 0, packFrameUniform(params, dt).toJS);
  }

  void _buildBindGroups(GPUDevice device) {
    GPUBindGroupEntry entry(int binding, GPUBuffer buffer) =>
        GPUBindGroupEntry(
          binding: binding,
          resource: GPUBufferBinding(buffer: buffer),
        );

    _computeBindGroup = device.createBindGroup(
      GPUBindGroupDescriptor(
        layout: _computeLayout!,
        entries: <GPUBindGroupEntry>[
          entry(KernelBindings.params, _uniformBuffer!),
          entry(KernelBindings.positions, _positions!),
          entry(KernelBindings.velocities, _velocities!),
          entry(KernelBindings.types, _types!),
          entry(KernelBindings.forces, _forces!),
          entry(KernelBindings.minDistances, _minDistances!),
          entry(KernelBindings.radii, _radii!),
          entry(KernelBindings.binCounts, _binCounts!),
          entry(KernelBindings.binParticles, _binParticles!),
        ].toJS,
      ),
    );
    _renderBindGroup = device.createBindGroup(
      GPUBindGroupDescriptor(
        layout: _renderLayout!,
        entries: <GPUBindGroupEntry>[
          entry(KernelBindings.renderParams, _uniformBuffer!),
          entry(KernelBindings.renderPositions, _positions!),
          entry(KernelBindings.renderTypes, _types!),
          entry(KernelBindings.typeColors, _typeColors!),
        ].toJS,
      ),
    );
  }

  void _ensureCanvas(GPUDevice device, SimParams params) {
    if (_canvas != null) return;
    final canvas = WebCanvasHandle.create(
      device: device,
      format: _canvasFormat!,
      width: params.worldWidth,
      height: params.worldHeight,
    );
    if (canvas == null) {
      // The feature-detect already reported WebGPU present, but the canvas
      // refused a `webgpu` context — surface it as an availability failure so
      // the selector (PRIMORDIS-TASK-007) can fall back, rather than the backend
      // silently never presenting (a null `_canvas` no-ops `present()` forever).
      throw const WebGpuUnavailableException(
        'could not obtain a webgpu canvas context',
      );
    }
    _canvas = canvas;
  }

  void _disposeSeedBuffers() {
    for (final b in <GPUBuffer?>[
      _positions, _velocities, _types, _forces, _minDistances, _radii,
      _typeColors, _binCounts, _binParticles,
    ]) {
      b?.destroy();
    }
    _positions = null;
    _velocities = null;
    _types = null;
    _forces = null;
    _minDistances = null;
    _radii = null;
    _typeColors = null;
    _binCounts = null;
    _binParticles = null;
    _computeBindGroup = null;
    _renderBindGroup = null;
  }

  GPUBindGroupLayoutDescriptor _computeLayoutDescriptor() {
    GPUBindGroupLayoutEntry e(int binding, String type) =>
        GPUBindGroupLayoutEntry(
          binding: binding,
          visibility: GpuShaderStage.compute,
          buffer: GPUBufferBindingLayout(type: type),
        );
    const uniform = GpuBufferBindingType.uniform;
    const rw = GpuBufferBindingType.storage;
    const ro = GpuBufferBindingType.readOnlyStorage;
    return GPUBindGroupLayoutDescriptor(
      entries: <GPUBindGroupLayoutEntry>[
        e(KernelBindings.params, uniform),
        e(KernelBindings.positions, rw),
        e(KernelBindings.velocities, rw),
        e(KernelBindings.types, ro),
        e(KernelBindings.forces, ro),
        e(KernelBindings.minDistances, ro),
        e(KernelBindings.radii, ro),
        e(KernelBindings.binCounts, rw),
        e(KernelBindings.binParticles, rw),
      ].toJS,
    );
  }

  GPUBindGroupLayoutDescriptor _renderLayoutDescriptor() {
    GPUBindGroupLayoutEntry e(int binding, String type) =>
        GPUBindGroupLayoutEntry(
          binding: binding,
          visibility: GpuShaderStage.vertex,
          buffer: GPUBufferBindingLayout(type: type),
        );
    const uniform = GpuBufferBindingType.uniform;
    const ro = GpuBufferBindingType.readOnlyStorage;
    return GPUBindGroupLayoutDescriptor(
      entries: <GPUBindGroupLayoutEntry>[
        e(KernelBindings.renderParams, uniform),
        e(KernelBindings.renderPositions, ro),
        e(KernelBindings.renderTypes, ro),
        e(KernelBindings.typeColors, ro),
      ].toJS,
    );
  }
}
