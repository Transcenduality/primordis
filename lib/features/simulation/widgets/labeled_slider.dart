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
/// polish): the [tooltip] wraps the whole control via [Tooltip], and an
/// explicit [Semantics] node exposes [label] plus a formatted [value] so
/// screen readers announce both the control's purpose and its current
/// reading, independent of the tooltip (which is not screen-reader-visible on
/// every platform). The underlying [Slider] is natively keyboard-operable
/// (arrow keys) once focused.
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
    final semanticLabel = '$label: $formatted';

    return Tooltip(
      message: tooltip,
      child: Semantics(
        label: label,
        value: formatted,
        slider: true,
        child: ExcludeSemantics(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(label, style: theme.textTheme.labelLarge),
                  Text(formatted, style: theme.textTheme.labelMedium),
                ],
              ),
              Slider(
                value: value.clamp(min, max),
                min: min,
                max: max,
                label: formatted,
                semanticFormatterCallback: (_) => semanticLabel,
                onChanged: onChanged,
              ),
            ],
          ),
        ),
      ),
    );
  }

  static String _defaultFormat(double value) => value.toStringAsFixed(2);
}
