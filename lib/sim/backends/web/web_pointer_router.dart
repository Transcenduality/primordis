// The explicit pointer-routing policy at the glass-pane / field seam.
//
// Because the WebGPU `<canvas>` lives OUTSIDE Flutter's hit-test tree
// ([PRIMORDIS-ADR-005]), the canvas is made `pointer-events: none` and Flutter
// (on top, transparent) owns all input. This class is the decision the field
// listener applies: for a given pointer position, does the event belong to a
// control (let the Flutter widget handle it) or to the open field (forward the
// world coordinate to the backend)? Exactly one destination is returned, so the
// seam can never double-handle or drop an event.
//
// Pure Dart (`dart:ui` value types only) so the policy is unit-tested directly,
// independent of the DOM wiring — the bespoke-and-fragile risk called out in the
// ADR's Negative consequences.
import 'dart:ui';

import 'package:primordis/sim/backends/web/world_viewport.dart';

/// The two sides of the seam an event can be routed to.
enum PointerRoute {
  /// Handled by the Flutter overlay (a slider/chrome widget consumes it, or it
  /// is an inert tap on a control region / letterbox margin).
  flutter,

  /// Forwarded to the simulation backend as a field interaction, carrying the
  /// world coordinate.
  field,
}

/// A routing decision: the [route] and, when it is [PointerRoute.field], the
/// [world] coordinate the pointer maps to (otherwise null).
class PointerRouting {
  const PointerRouting(this.route, [this.world])
      : assert(
          route == PointerRoute.field || world == null,
          'world is only meaningful for a field route',
        );

  final PointerRoute route;

  /// The world-space position (`0 <= x < worldWidth`, `0 <= y < worldHeight`;
  /// half-open at the max edges, matching [WorldViewport.worldFromRegionLocal])
  /// for a field route; null for a Flutter route.
  final Offset? world;

  @override
  String toString() => 'PointerRouting($route, $world)';
}

/// Decides, for each pointer position, which side of the seam owns it.
///
/// Policy (deterministic, single-destination):
/// 1. Inside any control rect → [PointerRoute.flutter] (the overlay widget owns
///    it; the field forwarder ignores it, so no double-dispatch).
/// 2. Otherwise inside the fitted world field → [PointerRoute.field] with the
///    mapped world coordinate.
/// 3. Otherwise (a letterbox margin with no control) → [PointerRoute.flutter];
///    margin taps are inert rather than spurious field interactions.
class PointerRouter {
  const PointerRouter();

  /// Routes a pointer at [regionLocal] (region-local CSS pixels) given the
  /// current [controlRects] (also region-local) and [viewport].
  PointerRouting routeAt(
    Offset regionLocal, {
    required Iterable<Rect> controlRects,
    required WorldViewport viewport,
  }) {
    for (final rect in controlRects) {
      if (rect.contains(regionLocal)) {
        return const PointerRouting(PointerRoute.flutter);
      }
    }
    final world = viewport.worldFromRegionLocal(regionLocal);
    if (world == null) return const PointerRouting(PointerRoute.flutter);
    return PointerRouting(PointerRoute.field, world);
  }
}
