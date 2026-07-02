import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:primordis/features/simulation/providers/sim_params_provider.dart';
import 'package:primordis/features/simulation/widgets/control_panel.dart';
import 'package:primordis/shared/constants/primordis_config.dart';
import 'package:primordis/sim/backends/web/web_pointer_router.dart';
import 'package:primordis/sim/backends/web/world_viewport.dart';
import 'package:primordis/sim/providers/compositor_provider.dart';
import 'package:primordis/sim/providers/sim_providers.dart';

/// The transparent **glass-pane** host for the simulation ([PRIMORDIS-ADR-005]).
///
/// On web the GPU point field renders into a sibling `<canvas>` stacked BEHIND
/// this widget tree (placed by the compositor read from
/// [simCompositorProvider]); on macOS it is an IOSurface `Texture`
/// ([PRIMORDIS-TASK-012]). Either way this view paints nothing opaque over the
/// field, so the simulation shows through, while the chrome (and the sliders
/// landing in [PRIMORDIS-TASK-006]) overlay on top.
///
/// Two web-specific jobs are wired here, both driven from pure helpers so the
/// fragile parts are unit-tested away from the DOM:
///
/// 1. **DPR / resize sync.** A [LayoutBuilder] + `MediaQuery.devicePixelRatio`
///    feed a [WorldViewport]; on every layout the resulting [CompositorLayout]
///    is pushed to the compositor, which sizes the canvas backing store to
///    `field × dpr` and lines it up under this region.
/// 2. **Explicit pointer routing.** The sibling canvas is `pointer-events: none`,
///    so this glass-pane owns all input. A full-region [Listener] applies
///    [PointerRouter]: taps over the chrome stay with Flutter; taps over the open
///    field are forwarded (as world coordinates) via [onFieldPointer].
///
/// No `setState`-driven business logic: the compositor handle comes from a
/// Riverpod provider, and layout/DPR come from the framework, not local mutable
/// state ([PRIMORDIS-TASK-005]).
class SimulationView extends ConsumerStatefulWidget {
  const SimulationView({super.key, this.onFieldPointer});

  /// Invoked with the world-space position (`0..1080`, `0..720`) when a pointer
  /// goes down over the open field. The backend perturbation sink wires in with
  /// the slider/uniform work ([PRIMORDIS-TASK-006]); until then this is the
  /// tested routing seam and defaults to a no-op.
  final void Function(Offset world)? onFieldPointer;

  @override
  ConsumerState<SimulationView> createState() => _SimulationViewState();
}

class _SimulationViewState extends ConsumerState<SimulationView>
    with SingleTickerProviderStateMixin {
  static const PointerRouter _router = PointerRouter();

  /// Identifies the chrome panel so its bounds can be excluded from field
  /// routing (the control region the router keeps with Flutter).
  final GlobalKey _chromeKey = GlobalKey();

  /// Drives the per-frame `FrameLoop.tick` at the platform's vsync rate. Holds
  /// no simulation state itself — every value it reads (`params`, `paused`)
  /// comes from Riverpod on each callback ([PRIMORDIS-TASK-006]); this is
  /// purely the "when" of stepping, not the "what".
  late final Ticker _ticker;
  Duration _lastElapsed = Duration.zero;

  /// True once the reduced-motion preference has been read and applied for
  /// this widget's lifetime, so it is only auto-applied once (a later
  /// explicit play/pause tap always wins).
  bool _appliedReducedMotion = false;

  @override
  void initState() {
    super.initState();
    _ticker = createTicker(_onTick)..start();
  }

  @override
  void dispose() {
    _ticker.dispose();
    super.dispose();
  }

  void _onTick(Duration elapsed) {
    if (!mounted) return;
    final dt = (elapsed - _lastElapsed).inMicroseconds / Duration.microsecondsPerSecond;
    _lastElapsed = elapsed;
    // Guard against a stray first-frame spike or a debugger pause producing a
    // huge dt that would visibly jump the simulation.
    final clampedDt = dt.clamp(0.0, 1 / 15);

    final runnerReady = ref.read(simRunnerControllerProvider).hasValue;
    if (!runnerReady) return;

    final runState = ref.read(runStateControllerProvider);
    final loop = ref.read(frameLoopProvider);
    final stepped = loop.tick(
      dt: clampedDt,
      params: ref.read(simParamsControllerProvider),
      paused: runState.isPaused || !runState.isRunning,
    );
    if (stepped) {
      ref.read(runStateControllerProvider.notifier).markFrameStepped();
    }
  }

  /// Applies the platform's reduced-motion preference once: starting paused
  /// satisfies the accessibility requirement that a full-screen-motion app
  /// offer a static state by default for `prefers-reduced-motion` /
  /// `MediaQuery.disableAnimations` users ([PRIMORDIS-ADR-006]). Only ever
  /// pauses — it never resumes, so it can't fight an explicit play tap.
  ///
  /// Reads `MediaQuery` during `build` (a pure read, establishing the
  /// dependency normally) but defers the provider **writes** to a
  /// post-frame callback: Riverpod disallows mutating a provider while the
  /// widget tree is building, which a synchronous write here would violate.
  void _syncReducedMotion(BuildContext context) {
    if (_appliedReducedMotion) return;
    final reduced = MediaQuery.disableAnimationsOf(context);
    _appliedReducedMotion = true;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      ref.read(reducedMotionControllerProvider.notifier).set(reduced);
      if (reduced) {
        ref.read(runStateControllerProvider.notifier).pause();
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final compositor = ref.watch(simCompositorProvider);
    _syncReducedMotion(context);
    // Watching keeps this widget alive while bring-up (`init`/`seed`) runs;
    // the ticker callback reads the provider directly so a rebuild never
    // stops frames from being scheduled.
    ref.watch(simRunnerControllerProvider);

    return LayoutBuilder(
      builder: (layoutContext, constraints) {
        final viewport = WorldViewport(
          regionWidth: constraints.maxWidth,
          regionHeight: constraints.maxHeight,
          devicePixelRatio: MediaQuery.devicePixelRatioOf(layoutContext),
          worldWidth: PrimordisConfig.worldWidth.toDouble(),
          worldHeight: PrimordisConfig.worldHeight.toDouble(),
        );

        // Push the backing-store/placement to the compositor after this frame
        // lays out, so the region's global origin is known. Idempotent: the
        // compositor skips an unchanged layout (holds a paused last frame).
        WidgetsBinding.instance.addPostFrameCallback((_) {
          if (!mounted) return;
          final box = layoutContext.findRenderObject() as RenderBox?;
          if (box == null || !box.hasSize) return;
          compositor.syncLayout(
            viewport.toCompositorLayout(box.localToGlobal(Offset.zero)),
          );
        });

        return Stack(
          fit: StackFit.expand,
          children: [
            // The glass-pane: a full-region pointer sink that paints nothing,
            // so the field shows the simulation behind it. `translucent` lets it
            // see pointers even though it is visually empty; the router decides
            // whether each one is a field interaction.
            Listener(
              behavior: HitTestBehavior.translucent,
              onPointerDown: (event) =>
                  _routeFieldPointer(layoutContext, event.localPosition, viewport),
              child: const SizedBox.expand(),
            ),
            Positioned(
              left: 16,
              top: 16,
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 320),
                child: _Chrome(key: _chromeKey),
              ),
            ),
          ],
        );
      },
    );
  }

  /// Routes a field-region pointer-down to [SimulationView.onFieldPointer],
  /// skipping it when it falls over the chrome panel (which the overlay owns).
  void _routeFieldPointer(
    BuildContext regionContext,
    Offset regionLocal,
    WorldViewport viewport,
  ) {
    final chrome = _chromeRect(regionContext);
    final routing = _router.routeAt(
      regionLocal,
      controlRects: <Rect>[if (chrome != null) chrome],
      viewport: viewport,
    );
    if (routing.route == PointerRoute.field && routing.world != null) {
      widget.onFieldPointer?.call(routing.world!);
    }
  }

  /// The chrome panel's bounds in region-local coordinates (same space as the
  /// [Listener]'s `localPosition`), or null before it has laid out.
  Rect? _chromeRect(BuildContext regionContext) {
    final chromeBox =
        _chromeKey.currentContext?.findRenderObject() as RenderBox?;
    final regionBox = regionContext.findRenderObject() as RenderBox?;
    if (chromeBox == null ||
        regionBox == null ||
        !chromeBox.hasSize ||
        !regionBox.hasSize) {
      return null;
    }
    final topLeft =
        regionBox.globalToLocal(chromeBox.localToGlobal(Offset.zero));
    return topLeft & chromeBox.size;
  }
}

/// The overlay chrome: title/tagline/version plus the live [ControlPanel]
/// ([PRIMORDIS-TASK-006]). Background is translucent (never opaque) so the
/// glass-pane stays see-through.
class _Chrome extends StatelessWidget {
  const _Chrome({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return ColoredBox(
      color: Colors.black.withValues(alpha: 0.35),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Primordis', style: theme.textTheme.headlineSmall),
            const SizedBox(height: 4),
            Text(
              'GPU particle-life — Flutter Web + macOS',
              style: theme.textTheme.bodySmall,
            ),
            const SizedBox(height: 8),
            Text('v${PrimordisConfig.version}',
                style: theme.textTheme.labelSmall),
            const SizedBox(height: 12),
            const Divider(height: 1),
            const SizedBox(height: 12),
            const ControlPanel(),
          ],
        ),
      ),
    );
  }
}
