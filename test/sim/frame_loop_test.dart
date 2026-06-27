import 'package:flutter_test/flutter_test.dart';
import 'package:primordis/shared/constants/primordis_config.dart';
import 'package:primordis/sim/fake_sim_backend.dart';
import 'package:primordis/sim/frame_loop.dart';
import 'package:primordis/sim/models/sim_params.dart';
import 'package:primordis/sim/models/type_matrix.dart';

TypeMatrix _matrix(double fill) =>
    TypeMatrix.generate(PrimordisConfig.typeCount, (_, _) => fill);

SimParams _params() => SimParams(
      forces: _matrix(0.5),
      minDistances: _matrix(8),
      radii: _matrix(48),
    );

void main() {
  late FakeSimBackend backend;
  late FrameLoop loop;

  setUp(() {
    backend = FakeSimBackend();
    loop = FrameLoop(backend: backend);
  });

  test('first tick uploads params, then steps, then presents — in order', () {
    final stepped = loop.tick(dt: 0.016, params: _params(), paused: false);
    expect(stepped, isTrue);
    expect(backend.calls, [
      FakeSimCall.setParams,
      FakeSimCall.step,
      FakeSimCall.present,
    ]);
    expect(loop.frame, 1);
  });

  test('setParams is called only when params change', () {
    final p = _params();
    loop.tick(dt: 0.016, params: p, paused: false);
    // Same value (value-equal) → no second setParams.
    loop.tick(dt: 0.016, params: _params(), paused: false);
    expect(backend.setParamsCount, 1);
    expect(backend.stepCount, 2);
    expect(backend.presentCount, 2);

    // A changed slider → setParams again.
    loop.tick(dt: 0.016, params: p.copyWith(attractionK: 99), paused: false);
    expect(backend.setParamsCount, 2);
    expect(backend.stepCount, 3);
  });

  test('every stepped frame presents, even when params are unchanged', () {
    final p = _params();
    loop.tick(dt: 0.016, params: p, paused: false);
    loop.tick(dt: 0.016, params: p, paused: false);
    loop.tick(dt: 0.016, params: p, paused: false);
    expect(backend.stepCount, 3);
    expect(backend.presentCount, 3);
    expect(loop.frame, 3);
  });

  test('pause suppresses stepping entirely', () {
    final p = _params();
    final stepped = loop.tick(dt: 0.016, params: p, paused: true);
    expect(stepped, isFalse);
    expect(backend.calls, isEmpty);
    expect(backend.stepCount, 0);
    expect(loop.frame, 0);

    // Resuming steps normally.
    loop.tick(dt: 0.016, params: p, paused: false);
    expect(backend.stepCount, 1);
  });

  test('reset forces the next tick to re-upload params', () {
    final p = _params();
    loop.tick(dt: 0.016, params: p, paused: false);
    expect(backend.setParamsCount, 1);

    loop.reset();
    expect(loop.frame, 0);
    expect(loop.lastAppliedParams, isNull);

    loop.tick(dt: 0.016, params: p, paused: false);
    expect(backend.setParamsCount, 2); // re-uploaded after reset
  });

  test('dt is forwarded to the backend step', () {
    loop.tick(dt: 0.25, params: _params(), paused: false);
    expect(backend.lastDt, 0.25);
  });
}
