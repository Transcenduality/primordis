import 'package:flutter_test/flutter_test.dart';
import 'package:primordis/sim/backends/web/webgpu_support.dart';

/// The feature-detect decision table ([classifyWebGpuProbe]) is the unit-
/// testable half of the WebGPU probe ([PRIMORDIS-TASK-004] / TASK-007): the web
/// shim only observes `navigator.gpu` + `requestAdapter()`, then this pure
/// function classifies — so every branch (no API / no adapter / supported) and
/// the graceful-failure path are covered here without a browser.
void main() {
  group('classifyWebGpuProbe', () {
    test('no navigator.gpu → unsupportedNoApi (the route-to-fallback case)', () {
      expect(
        classifyWebGpuProbe(hasGpuApi: false, hasAdapter: false),
        WebGpuSupport.unsupportedNoApi,
      );
      // hasAdapter is ignored when there is no API.
      expect(
        classifyWebGpuProbe(hasGpuApi: false, hasAdapter: true),
        WebGpuSupport.unsupportedNoApi,
      );
    });

    test('API present but requestAdapter() null → unsupportedNoAdapter', () {
      expect(
        classifyWebGpuProbe(hasGpuApi: true, hasAdapter: false),
        WebGpuSupport.unsupportedNoAdapter,
      );
    });

    test('API present and adapter acquired → supported', () {
      expect(
        classifyWebGpuProbe(hasGpuApi: true, hasAdapter: true),
        WebGpuSupport.supported,
      );
    });
  });

  group('WebGpuSupport.isSupported', () {
    test('true only for supported', () {
      expect(WebGpuSupport.supported.isSupported, isTrue);
      expect(WebGpuSupport.unsupportedNoApi.isSupported, isFalse);
      expect(WebGpuSupport.unsupportedNoAdapter.isSupported, isFalse);
      expect(WebGpuSupport.error.isSupported, isFalse);
    });
  });
}
