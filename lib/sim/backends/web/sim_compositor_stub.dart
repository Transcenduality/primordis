// Non-web default for the compositor facade ([sim_compositor_factory.dart]).
//
// Selected on every target WITHOUT `dart.library.js_interop` — the Dart VM
// (`flutter test`), macOS, and any native build. It pulls in NO web-only code,
// so non-web builds never compile the DOM compositor. There is no sibling canvas
// off-web, so the compositor is always the [NoopSimCompositor].
import 'package:primordis/sim/backends/web/sim_compositor.dart';
import 'package:primordis/sim/sim_backend.dart';

/// Always the no-op compositor off-web.
SimCompositor createSimCompositor(SimBackend backend) =>
    const NoopSimCompositor();
