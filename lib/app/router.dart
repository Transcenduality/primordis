import 'package:go_router/go_router.dart';
import 'package:primordis/features/home/presentation/screens/home_screen.dart';

/// Route path constants. Screens/widgets reference these instead of hardcoded
/// strings (house standard).
abstract final class Routes {
  static const String home = '/';
}

/// The application router (GoRouter; DGROUP_WEB-ADR-018).
///
/// Note: no `initialLocation` is set — on web that would override the browser
/// URL and break deep links (DGROUP_WEB-ADR-021). GoRouter defaults to `/`.
final GoRouter appRouter = GoRouter(
  routes: [
    GoRoute(
      path: Routes.home,
      builder: (context, state) => const HomeScreen(),
    ),
  ],
);
