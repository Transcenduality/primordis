import 'package:flutter/material.dart';

/// A Material 3 [Slider] with a label, live value display, tooltip, and full
/// [Semantics] — the shared control shape for every live sim parameter
/// ([PRIMORDIS-TASK-006]).
///
/// Kept presentation-only and backend-agnostic on purpose: callers supply
/// [value] and [onChanged] from a Riverpod provider (see
/// `sim_params_provider.dart`); this widget holds no business state of its own,
/// so there is no `setState` anywhere in the control path.
///
/// Accessibility (house standard — accessibility is a top-level goal, not
/// polish): the [tooltip] wraps the whole control via [Tooltip]; the
/// decorative label/value texts are excluded from the semantics tree, and a
/// [Semantics] label node wraps the [Slider] (the platform-conventional
/// labelled-control structure: a labelling parent node whose child is the
/// slider's own node). Crucially the [Slider]'s **own** semantics node is
/// never excluded, so its native increase/decrease actions, value, and
/// keyboard operability (arrow keys once focused) all remain available to
/// assistive tech.
class LabeledSlider extends StatelessWidget {
  const LabeledSlider({
    super.key,
    required this.label,
    required this.value,
    required this.min,
    required this.max,
    required this.tooltip,
    required this.onChanged,
    this.valueFormatter,
  });

  /// The control's display label (e.g. "Attraction K").
  final String label;

  /// Current value, within `[min, max]`.
  final double value;

  /// Lower bound of the slider's range.
  final double min;

  /// Upper bound of the slider's range.
  final double max;

  /// Tooltip text describing the effect this control has on the simulation.
  final String tooltip;

  /// Invoked with the new value on every drag update.
  final ValueChanged<double> onChanged;

  /// Formats [value] for display; defaults to two decimal places.
  final String Function(double value)? valueFormatter;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final formatted = (valueFormatter ?? _defaultFormat)(value);

    return Tooltip(
      message: tooltip,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          // The visual label/value row is decorative for assistive tech: the
          // same information is exposed on the slider's semantics node below,
          // so exclude ONLY this row — never the Slider, whose built-in
          // increase/decrease actions are what make the control adjustable
          // for screen-reader users.
          ExcludeSemantics(
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(label, style: theme.textTheme.labelLarge),
                Text(formatted, style: theme.textTheme.labelMedium),
              ],
            ),
          ),
          // The labelling node wraps — never replaces — the Slider's own
          // semantics node, so the control keeps its native increase/decrease
          // actions and stays adjustable for screen-reader users.
          Semantics(
            label: label,
            child: Slider(
              value: value.clamp(min, max),
              min: min,
              max: max,
              label: formatted,
              semanticFormatterCallback: (_) => formatted,
              onChanged: onChanged,
            ),
          ),
        ],
      ),
    );
  }

  static String _defaultFormat(double value) => value.toStringAsFixed(2);
}
