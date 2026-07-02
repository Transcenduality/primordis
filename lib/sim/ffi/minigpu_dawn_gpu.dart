import 'dart:typed_data';

import 'package:minigpu/minigpu.dart';

import 'package:primordis/sim/ffi/dawn_gpu.dart';

/// Production [DawnGpu] over the minigpu package (Dawn/wgpu-over-Metal via
/// `dart:ffi`, PRIMORDIS-ADR-004 Approach (a)).
///
/// minigpu API facts this wrapper encodes (verified against minigpu 1.5.x
/// source during the TASK-801 spike; see `wgsl_pass_adapter.dart` for the
/// shader-side constraints):
/// - `createBuffer` takes a BYTE size; `Buffer.read`/`write` take ELEMENT
///   counts.
/// - A size mismatch throws `std::runtime_error` in C++, which crosses the
///   FFI boundary uncatchably and `abort()`s the process — hence the strict
///   Dart-side checks in [_MinigpuBuffer].
/// - Dawn pin: minigpu 1.5.4 builds Dawn @ 7bd3e6712cde5f69b2053839ab949313e194a57c
///   (dawn.googlesource.com) — the Tint revision relevant to TASK-807.
final class MinigpuDawnGpu implements DawnGpu {
  MinigpuDawnGpu({void Function(int level, String message)? onLog}) {
    if (onLog != null) {
      // 2 = warn: surfaces Dawn validation errors without frame-rate spam.
      Minigpu.setLogCallback(onLog, level: 2);
    }
  }

  final Minigpu _gpu = Minigpu();
  bool _destroyed = false;

  @override
  Future<void> init() async {
    _checkLive();
    try {
      await _gpu.init();
    } catch (e) {
      throw StateError('Dawn/minigpu device init failed: $e');
    }
  }

  @override
  DawnGpuBuffer createF32Buffer(int byteSize) => _createBuffer(byteSize);

  @override
  DawnGpuBuffer createU32Buffer(int byteSize) =>
      _createBuffer(byteSize, dataType: BufferDataType.uint32);

  DawnGpuBuffer _createBuffer(
    int byteSize, {
    BufferDataType dataType = BufferDataType.float32,
  }) {
    _checkLive();
    if (byteSize <= 0 || byteSize % 4 != 0) {
      throw ArgumentError.value(
        byteSize,
        'byteSize',
        'must be a positive multiple of 4',
      );
    }
    return _MinigpuBuffer(_gpu.createBuffer(byteSize, dataType), byteSize);
  }

  @override
  DawnGpuPass createPass(String wgsl) {
    _checkLive();
    return _MinigpuPass(_gpu.createComputeShader()..loadKernelString(wgsl));
  }

  @override
  Future<void> destroy() async {
    if (_destroyed) return;
    _destroyed = true;
    await _gpu.destroy();
  }

  void _checkLive() {
    if (_destroyed) throw StateError('MinigpuDawnGpu already destroyed');
  }
}

final class _MinigpuBuffer implements DawnGpuBuffer {
  _MinigpuBuffer(this._buffer, this.byteSize);

  final Buffer _buffer;
  @override
  final int byteSize;

  @override
  Future<void> writeF32(Float32List data) {
    _checkSize(data);
    return _buffer.write(data, data.length);
  }

  @override
  Future<void> writeU32(Uint32List data) {
    _checkSize(data);
    return _buffer.write(data, data.length, dataType: BufferDataType.uint32);
  }

  @override
  Future<void> readF32(Float32List out) {
    _checkSize(out);
    return _buffer.read(out, out.length);
  }

  @override
  Future<void> readU32(Uint32List out) {
    _checkSize(out);
    return _buffer.read(out, out.length, dataType: BufferDataType.uint32);
  }

  @override
  void destroy() => _buffer.destroy();

  /// The abort()-prevention gate — see the library doc of `dawn_gpu.dart`.
  void _checkSize(TypedData data) {
    if (data.lengthInBytes != byteSize) {
      throw ArgumentError(
        'buffer size mismatch: buffer is $byteSize bytes, '
        'data is ${data.lengthInBytes} bytes',
      );
    }
  }
}

final class _MinigpuPass implements DawnGpuPass {
  _MinigpuPass(this._shader);

  final ComputeShader _shader;

  @override
  void bind(int slot, DawnGpuBuffer buffer) =>
      _shader.setBufferAtSlot(slot, (buffer as _MinigpuBuffer)._buffer);

  @override
  Future<void> dispatch(int groupsX) => _shader.dispatch(groupsX, 1, 1);

  @override
  void destroy() => _shader.destroy();
}
