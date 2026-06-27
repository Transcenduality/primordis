import 'package:flutter/material.dart';
import 'package:primordis/shared/constants/primordis_config.dart';

/// The home screen.
///
/// The simulation surface composites into the reserved area below — on web a
/// stacked WebGPU `<canvas>` behind a transparent Flutter glass-pane
/// (PRIMORDIS-TASK-005), on macOS an IOSurface-backed `Texture`
/// (PRIMORDIS-TASK-012). The scaffold introduces no continuous motion, so it
/// stays reduced-motion safe (PRIMORDIS-ADR-006).
class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      body: SafeArea(
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text('Primordis', style: theme.textTheme.headlineMedium),
              const SizedBox(height: 8),
              Text(
                'GPU particle-life — Flutter Web + macOS',
                style: theme.textTheme.bodyMedium,
              ),
              const SizedBox(height: 24),
              Container(
                width: 360,
                height: 220,
                alignment: Alignment.center,
                decoration: BoxDecoration(
                  border: Border.all(color: theme.colorScheme.outline),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Text(
                  'simulation surface\n(backend lands in TASK-004 / TASK-012)',
                  textAlign: TextAlign.center,
                  style: theme.textTheme.bodySmall,
                ),
              ),
              const SizedBox(height: 24),
              Text('v${PrimordisConfig.version}',
                  style: theme.textTheme.labelSmall),
            ],
          ),
        ),
      ),
    );
  }
}
