// `dart:js_interop` extension-type bindings for the slice of the browser WebGPU
// API the web backend uses, plus the `navigator.gpu` entry point via
// `package:web`.
//
// WHY HAND-WRITTEN: `package:web` (1.1.x) ships only the WebGPU *flag constants*
// (`GPUBufferUsage`, `GPUTextureUsage`, …) — not the object API (`GPUDevice`,
// `GPUBuffer`, pipelines, encoders). So those object types are modelled here as
// `extension type`s over `JSObject`, exactly as [PRIMORDIS-TASK-004] specifies.
//
// `--wasm` CONSTRAINT ([PRIMORDIS-ADR-007]): everything here is expressed with
// `dart:js_interop` + `package:web` only. There is no `dart:html` and no
// `dart:js_util` — both are banned anywhere in the `flutter build web --wasm`
// dependency tree. Descriptors are created as JS object literals via `external
// factory` constructors; typed-array uploads use `Float32List.toJS` /
// `Uint32List.toJS`.
//
// This file is bindings only: no device acquisition, no dispatch, no buffer
// sizing logic. The backend ([web_webgpu_backend.dart]) drives these; the canvas
// wrapper ([web_canvas_handle.dart]) owns the `GPUCanvasContext`.
@JS()
library;

import 'dart:js_interop';

import 'package:web/web.dart' as web;

// ---------------------------------------------------------------------------
// Usage / visibility bit flags.
//
// These mirror the WebGPU spec values (identical to `package:web`'s
// `$GPUBufferUsage` etc.) but are restated as plain `int` constants so callers
// read `GpuBufferUsage.storage` instead of the `$`-prefixed generated holders.
// They are immutable spec constants; a const-only holder class is exempt from
// the `avoid_classes_with_only_static_members` lint.
// ---------------------------------------------------------------------------

/// `GPUBufferUsage` flags.
abstract final class GpuBufferUsage {
  static const int copySrc = 0x0004;
  static const int copyDst = 0x0008;
  static const int uniform = 0x0040;
  static const int storage = 0x0080;
}

/// `GPUTextureUsage` flags.
abstract final class GpuTextureUsage {
  static const int copySrc = 0x01;
  static const int renderAttachment = 0x10;
}

/// `GPUShaderStage` visibility flags (bind-group-layout entry visibility).
abstract final class GpuShaderStage {
  static const int vertex = 0x1;
  static const int fragment = 0x2;
  static const int compute = 0x4;
}

/// Bind-group-layout buffer `type` strings.
abstract final class GpuBufferBindingType {
  static const String uniform = 'uniform';
  static const String storage = 'storage';
  static const String readOnlyStorage = 'read-only-storage';
}

// ---------------------------------------------------------------------------
// Entry point: navigator.gpu.
// ---------------------------------------------------------------------------

/// The `GPU` object (`navigator.gpu`) — the WebGPU entry point.
extension type GPU._(JSObject _) implements JSObject {
  /// Requests an adapter; resolves to null when no usable adapter exists.
  external JSPromise<GPUAdapter?> requestAdapter();

  /// The optimal swap-chain texture format for the canvas on this device.
  external String getPreferredCanvasFormat();
}

extension type _NavigatorGpu._(JSObject _) implements JSObject {
  external GPU? get gpu;
}

/// `navigator.gpu`, or null when the browser/platform exposes no WebGPU API.
///
/// Reinterprets the real `package:web` navigator (an `extension type` over
/// `JSObject`) as one carrying a `gpu` getter — `package:web`'s `Navigator`
/// does not declare it, but the underlying JS object does where WebGPU exists.
GPU? get navigatorGpu => _NavigatorGpu._(web.window.navigator).gpu;

// ---------------------------------------------------------------------------
// Device, queue, adapter.
// ---------------------------------------------------------------------------

extension type GPUAdapter._(JSObject _) implements JSObject {
  external JSPromise<GPUDevice> requestDevice();
}

extension type GPUDevice._(JSObject _) implements JSObject {
  external GPUQueue get queue;

  /// Resolves when the device is lost (used to surface device-lost gracefully).
  external JSPromise<GPUDeviceLostInfo> get lost;

  external GPUBuffer createBuffer(GPUBufferDescriptor descriptor);
  external GPUShaderModule createShaderModule(GPUShaderModuleDescriptor d);
  external GPUBindGroupLayout createBindGroupLayout(
    GPUBindGroupLayoutDescriptor d,
  );
  external GPUPipelineLayout createPipelineLayout(GPUPipelineLayoutDescriptor d);
  external GPUComputePipeline createComputePipeline(
    GPUComputePipelineDescriptor d,
  );
  external GPURenderPipeline createRenderPipeline(
    GPURenderPipelineDescriptor d,
  );
  external GPUBindGroup createBindGroup(GPUBindGroupDescriptor d);
  external GPUCommandEncoder createCommandEncoder();
  external void destroy();
}

extension type GPUDeviceLostInfo._(JSObject _) implements JSObject {
  external String get reason;
  external String get message;
}

extension type GPUQueue._(JSObject _) implements JSObject {
  /// Uploads [data] (a `JSArrayBuffer`/typed-array view) into [buffer] at
  /// [bufferOffset] bytes.
  external void writeBuffer(GPUBuffer buffer, int bufferOffset, JSAny data);
  external void submit(JSArray<GPUCommandBuffer> commandBuffers);
}

extension type GPUBuffer._(JSObject _) implements JSObject {
  external void destroy();
}

extension type GPUShaderModule._(JSObject _) implements JSObject {}

extension type GPUBindGroupLayout._(JSObject _) implements JSObject {}

extension type GPUPipelineLayout._(JSObject _) implements JSObject {}

extension type GPUComputePipeline._(JSObject _) implements JSObject {}

extension type GPURenderPipeline._(JSObject _) implements JSObject {}

extension type GPUBindGroup._(JSObject _) implements JSObject {}

extension type GPUCommandBuffer._(JSObject _) implements JSObject {}

// ---------------------------------------------------------------------------
// Command encoding.
// ---------------------------------------------------------------------------

extension type GPUCommandEncoder._(JSObject _) implements JSObject {
  external GPUComputePassEncoder beginComputePass();
  external GPURenderPassEncoder beginRenderPass(GPURenderPassDescriptor d);
  external GPUCommandBuffer finish();
}

extension type GPUComputePassEncoder._(JSObject _) implements JSObject {
  external void setPipeline(GPUComputePipeline pipeline);
  external void setBindGroup(int index, GPUBindGroup bindGroup);
  external void dispatchWorkgroups(int workgroupCountX);
  external void end();
}

extension type GPURenderPassEncoder._(JSObject _) implements JSObject {
  external void setPipeline(GPURenderPipeline pipeline);
  external void setBindGroup(int index, GPUBindGroup bindGroup);
  external void draw(int vertexCount);
  external void end();
}

// ---------------------------------------------------------------------------
// Canvas context (configured by `web_canvas_handle.dart`).
// ---------------------------------------------------------------------------

extension type GPUCanvasContext._(JSObject _) implements JSObject {
  external void configure(GPUCanvasConfiguration configuration);
  external void unconfigure();
  external GPUTexture getCurrentTexture();
}

extension type GPUTexture._(JSObject _) implements JSObject {
  external GPUTextureView createView();
}

extension type GPUTextureView._(JSObject _) implements JSObject {}

// ---------------------------------------------------------------------------
// Descriptors — JS object literals built via `external factory` constructors.
// ---------------------------------------------------------------------------

extension type GPUBufferDescriptor._(JSObject _) implements JSObject {
  external factory GPUBufferDescriptor({int size, int usage});
}

extension type GPUShaderModuleDescriptor._(JSObject _) implements JSObject {
  external factory GPUShaderModuleDescriptor({String code});
}

extension type GPUBufferBindingLayout._(JSObject _) implements JSObject {
  external factory GPUBufferBindingLayout({String type});
}

extension type GPUBindGroupLayoutEntry._(JSObject _) implements JSObject {
  external factory GPUBindGroupLayoutEntry({
    int binding,
    int visibility,
    GPUBufferBindingLayout buffer,
  });
}

extension type GPUBindGroupLayoutDescriptor._(JSObject _) implements JSObject {
  external factory GPUBindGroupLayoutDescriptor({
    JSArray<GPUBindGroupLayoutEntry> entries,
  });
}

extension type GPUPipelineLayoutDescriptor._(JSObject _) implements JSObject {
  external factory GPUPipelineLayoutDescriptor({
    JSArray<GPUBindGroupLayout> bindGroupLayouts,
  });
}

/// WGSL `override` constants set at pipeline creation (`WORKGROUP_SIZE`,
/// `MAX_BIN_PARTICLES`). Keys are the exact WGSL identifiers, so the JS field
/// names are upper snake case by necessity.
// ignore: non_constant_identifier_names
extension type GPUPipelineConstants._(JSObject _) implements JSObject {
  external factory GPUPipelineConstants({
    // ignore: non_constant_identifier_names
    int WORKGROUP_SIZE,
    // ignore: non_constant_identifier_names
    int MAX_BIN_PARTICLES,
  });
}

extension type GPUProgrammableStage._(JSObject _) implements JSObject {
  external factory GPUProgrammableStage({
    GPUShaderModule module,
    String entryPoint,
    GPUPipelineConstants constants,
  });
}

extension type GPUComputePipelineDescriptor._(JSObject _) implements JSObject {
  external factory GPUComputePipelineDescriptor({
    GPUPipelineLayout layout,
    GPUProgrammableStage compute,
  });
}

extension type GPUColorTargetState._(JSObject _) implements JSObject {
  external factory GPUColorTargetState({String format});
}

extension type GPUVertexState._(JSObject _) implements JSObject {
  external factory GPUVertexState({
    GPUShaderModule module,
    String entryPoint,
    GPUPipelineConstants constants,
  });
}

extension type GPUFragmentState._(JSObject _) implements JSObject {
  external factory GPUFragmentState({
    GPUShaderModule module,
    String entryPoint,
    JSArray<GPUColorTargetState> targets,
  });
}

extension type GPUPrimitiveState._(JSObject _) implements JSObject {
  external factory GPUPrimitiveState({String topology});
}

extension type GPURenderPipelineDescriptor._(JSObject _) implements JSObject {
  external factory GPURenderPipelineDescriptor({
    GPUPipelineLayout layout,
    GPUVertexState vertex,
    GPUFragmentState fragment,
    GPUPrimitiveState primitive,
  });
}

extension type GPUBufferBinding._(JSObject _) implements JSObject {
  external factory GPUBufferBinding({GPUBuffer buffer});
}

extension type GPUBindGroupEntry._(JSObject _) implements JSObject {
  external factory GPUBindGroupEntry({int binding, GPUBufferBinding resource});
}

extension type GPUBindGroupDescriptor._(JSObject _) implements JSObject {
  external factory GPUBindGroupDescriptor({
    GPUBindGroupLayout layout,
    JSArray<GPUBindGroupEntry> entries,
  });
}

extension type GPUColorDict._(JSObject _) implements JSObject {
  external factory GPUColorDict({double r, double g, double b, double a});
}

extension type GPURenderPassColorAttachment._(JSObject _)
    implements JSObject {
  external factory GPURenderPassColorAttachment({
    GPUTextureView view,
    GPUColorDict clearValue,
    String loadOp,
    String storeOp,
  });
}

extension type GPURenderPassDescriptor._(JSObject _) implements JSObject {
  external factory GPURenderPassDescriptor({
    JSArray<GPURenderPassColorAttachment> colorAttachments,
  });
}

extension type GPUCanvasConfiguration._(JSObject _) implements JSObject {
  external factory GPUCanvasConfiguration({
    GPUDevice device,
    String format,
    String alphaMode,
  });
}
