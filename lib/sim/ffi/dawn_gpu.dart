/// The minimal GPU surface [MacosDawnBackend] drives — a thin, testable seam
/// over the minigpu (Dawn/wgpu-over-Metal via `dart:ffi`) layer.
///
/// Why this exists (PRIMORDIS-TASK-801): the backend must be contract-testable
/// on the Dart VM with no GPU device, and raw `Pointer`s / minigpu types must
/// never thread into backend logic. The production implementation is
/// [`MinigpuDawnGpu`]; tests inject a recording fake.
///
/// Error-containment invariant: minigpu's native layer `abort()`s the whole
/// process on a mis-sized upload (`std::runtime_error` crossing the FFI
/// boundary is uncatchable from Dart — observed in the TASK-801 spike).
/// Implementations MUST therefore validate sizes Dart-side and throw
/// [ArgumentError]/[StateError] before any native call can see bad input.
library;

import 'dart:typed_data';

/// One GPU storage buffer of fixed byte size.
abstract interface class DawnGpuBuffer {
  /// Allocated size in bytes (fixed at creation).
  int get byteSize;

  /// Uploads [data] (its full length). Throws [ArgumentError] if
  /// `data.lengthInBytes != byteSize` — never lets a mismatch reach native.
  Future<void> writeF32(Float32List data);

  /// See [writeF32].
  Future<void> writeU32(Uint32List data);

  /// Reads the full buffer back into [out] (same size contract as writes).
  /// Spike/diagnostic use only — never on the shipped frame path.
  Future<void> readF32(Float32List out);

  /// See [readF32].
  Future<void> readU32(Uint32List out);

  /// Releases the buffer. Further use throws [StateError].
  void destroy();
}

/// One compute pipeline (a single-entry-point WGSL module, per the minigpu
/// constraint that the entry point is always `main`).
abstract interface class DawnGpuPass {
  /// Binds [buffer] at `@group(0) @binding(slot)`. Slots may be sparse.
  void bind(int slot, DawnGpuBuffer buffer);

  /// Dispatches `groupsX × 1 × 1` workgroups.
  Future<void> dispatch(int groupsX);

  /// Releases the pipeline. Further use throws [StateError].
  void destroy();
}

/// The device: init once, create resources, destroy once (single-use, like
/// the CPU tier's backend — construct a fresh instance per session).
abstract interface class DawnGpu {
  /// Acquires the adapter/device. Throws [StateError] (with the underlying
  /// cause) if Dawn init fails, so backend selection can fall back cleanly.
  Future<void> init();

  /// Allocates a zero-initialized storage buffer of [byteSize] bytes holding
  /// f32 data.
  DawnGpuBuffer createF32Buffer(int byteSize);

  /// Allocates a zero-initialized storage buffer of [byteSize] bytes holding
  /// u32 data (also used for `atomic<u32>`).
  DawnGpuBuffer createU32Buffer(int byteSize);

  /// Compiles [wgsl] (already adapted — entry point `main`) into a pipeline.
  DawnGpuPass createPass(String wgsl);

  /// Tears down the device and all native resources.
  Future<void> destroy();
}
