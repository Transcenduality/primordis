import 'package:flutter/material.dart';
import 'package:primordis/features/simulation/widgets/simulation_view.dart';

/// The home screen: the full-bleed simulation surface.
///
/// The [Scaffold] is **transparent** so the simulation composites through the
/// glass-pane — on web a stacked WebGPU `<canvas>` behind a transparent Flutter
/// view (PRIMORDIS-TASK-005), on macOS an IOSurface-backed `Texture`
/// (PRIMORDIS-TASK-012). The simulation is full-screen motion with the chrome
/// (and sliders, PRIMORDIS-TASK-006) overlaid by [SimulationView]; pausing the
/// frame loop holds the last composited frame (PRIMORDIS-ADR-006).
class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      backgroundColor: Colors.transparent,
      body: SimulationView(),
    );
  }
}
