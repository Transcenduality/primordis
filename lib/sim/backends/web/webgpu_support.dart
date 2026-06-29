/// Platform-neutral WebGPU availability model and feature-detect decision logic.
///
/// This file holds the *decision* — given what `navigator.gpu` /
/// `requestAdapter()` returned, is the web GPU backend usable? — as pure Dart so
/// it is unit-testable in `flutter test` (the Dart VM) without a browser. The
/// thin web shim that actually reads `navigator.gpu` and awaits `requestAdapter`
/// lives in `web_webgpu_backend.dart` (behind `dart.library.js_interop`) and
/// feeds its observations into [classifyWebGpuProbe].
///
/// Backend *selection* — what to do with an [WebGpuSupport] result — is owned by
/// [PRIMORDIS-TASK-007]; this file only reports support so that task can switch
/// to the Dart→WASM CPU fallback ([PRIMORDIS-ADR-006]) when WebGPU is absent.
library;

/// Whether browser WebGPU is usable, and if not, *why* — so the selector
/// ([PRIMORDIS-TASK-007]) can log a precise reason and route to the fallback.
enum WebGpuSupport {
  /// `navigator.gpu` is present and an adapter was acquired. The backend can be
  /// constructed and `init()`ed.
  supported,

  /// No `navigator.gpu` — the browser/platform has no WebGPU API at all
  /// (Firefox Linux/Android, Intel Macs, pre-26 Safari/iOS, or any non-web
  /// build). This is the common, expected "route to fallback" case.
  unsupportedNoApi,

  /// `navigator.gpu` exists but `requestAdapter()` returned null — WebGPU is
  /// exposed but no usable adapter is available (e.g. a blocklisted GPU).
  unsupportedNoAdapter,

  /// Probing threw unexpectedly. Treated as unsupported, but distinguished so
  /// the selector can surface a diagnostic rather than a silent fallback.
  error;

  /// True only for [supported]; the single check the selector branches on.
  bool get isSupported => this == WebGpuSupport.supported;
}

/// Maps the two observations a WebGPU probe makes — is there a `navigator.gpu`
/// API, and did `requestAdapter()` yield an adapter — onto a [WebGpuSupport].
///
/// Pure and total (every input combination has a defined result), so the web
/// shim stays a trivial "observe, then classify" and the branch table is the
/// thing covered by unit tests:
///
/// - no API            → [WebGpuSupport.unsupportedNoApi]
/// - API, no adapter   → [WebGpuSupport.unsupportedNoAdapter]
/// - API, adapter      → [WebGpuSupport.supported]
///
/// [hasAdapter] is ignored when [hasGpuApi] is false (you cannot request an
/// adapter without the API), keeping the contract unambiguous.
WebGpuSupport classifyWebGpuProbe({
  required bool hasGpuApi,
  required bool hasAdapter,
}) {
  if (!hasGpuApi) return WebGpuSupport.unsupportedNoApi;
  if (!hasAdapter) return WebGpuSupport.unsupportedNoAdapter;
  return WebGpuSupport.supported;
}
