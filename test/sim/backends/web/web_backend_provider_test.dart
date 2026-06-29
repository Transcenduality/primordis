import 'package:flutter_test/flutter_test.dart';
import 'package:primordis/sim/backends/web/web_backend.dart';
import 'package:primordis/sim/providers/web_backend_provider.dart';
import 'package:riverpod/riverpod.dart';

/// On the Dart VM (`flutter test`) the conditional facade resolves to the
/// non-web stub ([web_backend_stub.dart]), so the web providers must report
/// "unsupported / null" and pull in NO WebGPU code. This is the graceful
/// off-web path the selector ([PRIMORDIS-TASK-007]) relies on.
void main() {
  ProviderContainer container() {
    final c = ProviderContainer();
    addTearDown(c.dispose);
    return c;
  }

  group('facade (stub on the VM)', () {
    test('probeWebGpu reports unsupportedNoApi off-web', () async {
      expect(await probeWebGpu(), WebGpuSupport.unsupportedNoApi);
    });

    test('createWebSimBackend returns null off-web', () {
      expect(createWebSimBackend(), isNull);
    });
  });

  group('providers', () {
    test('webGpuSupportProvider resolves to unsupportedNoApi off-web', () async {
      final c = container();
      final support = await c.read(webGpuSupportProvider.future);
      expect(support, WebGpuSupport.unsupportedNoApi);
      expect(support.isSupported, isFalse);
    });

    test('webSimBackendProvider is null off-web', () {
      final c = container();
      expect(c.read(webSimBackendProvider), isNull);
    });
  });
}
