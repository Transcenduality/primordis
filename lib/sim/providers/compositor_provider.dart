import 'package:primordis/sim/backends/web/sim_compositor_factory.dart';
import 'package:primordis/sim/providers/sim_providers.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'compositor_provider.g.dart';

/// The active [SimCompositor], bound to the live [simBackendProvider].
///
/// This is how the UI ([features/simulation]) reaches the present-surface
/// placement contract without importing any web-only code: the conditional
/// facade ([createSimCompositor]) yields the DOM canvas compositor on web when a
/// WebGPU backend is live, and a [NoopSimCompositor] off-web / on the CPU tier
/// ([PRIMORDIS-ADR-005] / [PRIMORDIS-ADR-001]). Because the compositor is exposed
/// through Riverpod (plain `Ref`) rather than held in widget `setState`, the
/// "no `setState`-driven business logic" requirement of [PRIMORDIS-TASK-005] is
/// satisfied.
///
/// Kept alive for the app's lifetime and detached with the container. It
/// re-derives if the backend is swapped (e.g. the selector overriding
/// [simBackendProvider], [PRIMORDIS-TASK-007]).
@Riverpod(keepAlive: true)
SimCompositor simCompositor(Ref ref) {
  final compositor = createSimCompositor(ref.watch(simBackendProvider));
  ref.onDispose(compositor.detach);
  return compositor;
}
