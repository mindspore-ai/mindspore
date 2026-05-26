# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
P0 regression verification for multi-threaded backward (CPU).

These checks exercise concurrent forward/backward overlap, targeting the
A1 (process-global local_auto_diff_engine) and A3 (process-global GradState
/ OpRunStatus) state-leak bugs. With the P1-P3 TLS fixes and P5 per-task
context propagation in place, the A1/A3 checks pass and stay green as a
regression gate.

What it exercises that the earlier repro did not:
  * mb2 graph is backward-able -- retains the main-thread forward loss and
    backwards it AFTER the worker finished, comparing gradients. Exposes A3.
  * autodiff engine isolation -- probes get_current_autodiff_engine_id(); if
    the main-thread forward reads the worker backward's engine id, the
    process-global local_auto_diff_engine has leaked. Exposes A1.
  * runtime state restoration -- enable_grad / grad_flag around the overlap
    region must be restored afterwards. Exposes A3.
  * stress -- many layers/hooks, 1000 iterations, randomised sleeps.

Each check runs in an ISOLATED CHILD PROCESS: a state-leak data race can
SEGFAULT the process, so an in-process run would crash the whole pytest /
CI runner. A child crash is observed by the parent as an ordinary failure.

test_no_autodiff_engine_leak / test_runtime_state_restored are live
(markerless) regression checks now that the P1-P3 / P5 fixes have landed.
test_mb2_graph_backwardable and test_stress_overlap stay xfail(strict=False):
they exercise real concurrent shared-weight graph mutation, which can hit
tensor autograd-metadata races that are out of scope for the alternating /
TLS phase (plan 1.2), so their result is informational, not a gate.

Run standalone:  python tests/st/pynative/grad/test_multi_thread_backward.py
Or under pytest from the repo root. Internal: `--check <name>` runs one
check in-process (this is what the subprocess wrappers invoke).

Not covered here: P0 item 5 (C++ TSAN) needs a thread-sanitizer build of
the engine -- a separate task.
"""
# pylint: disable=protected-access
import random
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, _Function
from mindspore._c_expression import run_backward
from mindspore.graph.api import _pynative_executor

try:
    from tests.mark_utils import arg_mark
except ImportError:                          # standalone run, outside pytest
    def arg_mark(**_kwargs):
        return lambda fn: fn

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
_pynative_executor.set_grad_flag(True)
np.random.seed(0)
random.seed(0)


# --- tensor.backward() setup, replicated from
#     tests/st/pynative/grad/test_backward.py -----------------------------
@property
def requires_grad(self):
    return self._requires_grad


@requires_grad.setter
def requires_grad(self, value=True):
    self._requires_grad = value


@property
def grad(self):
    return self._grad


@grad.setter
def grad(self, value):
    self._grad = value


def backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None):
    return run_backward((self,), gradient, retain_graph, create_graph, inputs,
                        allow_unreachable=True, accumulate_grad=True)


Tensor.requires_grad = requires_grad
Tensor.grad = grad
Tensor.backward = backward
# ----------------------------------------------------------------------

# get_current_autodiff_engine_id() returns this when no AutoDiff engine is
# active on the current context; a real engine id is >= 0.
NO_ENGINE = -1


def sample_state():
    """Runtime-state probe. Sample OUTSIDE the _Function hooks only -- the
    NoGradGuard inside _Function.forward() makes enable_grad misleading
    there. Returns (enable_grad, grad_flag, current_autodiff_engine_id)."""
    return (_pynative_executor.enable_grad(),
            _pynative_executor.grad_flag(),
            _pynative_executor.get_current_autodiff_engine_id())


def reset_runtime_state():
    """Force global PyNative runtime state to a known-good baseline.
    A3 corruption is process-global and otherwise cascades across checks,
    so every graph-building primitive resets before it runs."""
    _pynative_executor.set_enable_grad(True)
    _pynative_executor.set_grad_flag(True)


class HookCoordinator:
    """Barrier(2) shared by the forward thread and the backward thread."""

    def __init__(self, timeout=10.0):
        self._timeout = timeout
        self._barrier = threading.Barrier(2, timeout=timeout)
        self._enabled = False
        self.jitter = 0.0

    def enable(self, jitter=0.0):
        self._barrier = threading.Barrier(2, timeout=self._timeout)
        self._enabled = True
        self.jitter = jitter

    def disable(self):
        self._enabled = False

    def abort(self):
        self._barrier.abort()

    def sync(self):
        """Barrier rendezvous between the forward and backward threads."""
        if not self._enabled:
            return
        if self.jitter:
            time.sleep(random.uniform(0.0, self.jitter))
        try:
            self._barrier.wait()
        except threading.BrokenBarrierError:
            pass


coordinator = HookCoordinator()


class SyncHook(_Function):
    """Identity _Function; forward() and backward() each hit the barrier."""

    @staticmethod
    def forward(ctx, x):
        coordinator.sync()
        return x

    @staticmethod
    def backward(ctx, grad_out):
        coordinator.sync()
        return grad_out


def make_layers(n_layers, dim):
    """A list of (w1, w2) leaf weights; one SyncHook is applied per layer."""
    layers = []
    for _ in range(n_layers):
        w1 = Tensor(np.random.randn(dim, dim).astype(np.float32))
        w2 = Tensor(np.random.randn(dim, dim).astype(np.float32))
        w1.requires_grad = True
        w2.requires_grad = True
        layers.append((w1, w2))
    return layers


def zero_grads(layers):
    for w1, w2 in layers:
        w1.grad = None
        w2.grad = None


def forward_loss(x, layers, use_hook, engine_probe=None):
    h = x
    for w1, w2 in layers:
        h = ms.ops.relu(ms.ops.matmul(h, w1))
        # Probe the AutoDiff engine id in PLAIN main-thread forward code
        # (not inside the _Function hook -- the _Function machinery masks
        # the engine context there). During an overlap this should stay
        # NO_ENGINE; a real id means the worker backward's engine leaked.
        if engine_probe is not None:
            engine_probe.append(
                _pynative_executor.get_current_autodiff_engine_id())
        if use_hook:
            h = SyncHook.apply(h)
        h = ms.ops.matmul(h, w2)
    return (h * h).sum()


def snapshot_grads(layers):
    out = []
    for w1, w2 in layers:
        out.append(None if w1.grad is None else w1.grad.asnumpy().copy())
        out.append(None if w2.grad is None else w2.grad.asnumpy().copy())
    return out


def reference_grads(x, layers):
    """Single-threaded forward+backward; isolated gradient snapshot."""
    reset_runtime_state()
    zero_grads(layers)
    forward_loss(x, layers, use_hook=False).backward()
    g = snapshot_grads(layers)
    zero_grads(layers)
    return g


def grads_match(got, ref, atol=1e-4):
    if len(got) != len(ref):
        return False
    for g, r in zip(got, ref):
        if g is None or r is None or not np.allclose(g, r, atol=atol):
            return False
    return True


def assert_overlap_clean(res):
    """Common precondition: the overlap completed without a hang and
    without either side raising, so the A1/A3-specific check that follows
    is not confounded or misattributed."""
    assert not res["worker_hung"], "worker backward hung"
    assert res["bwd_exc"] is None, (
        f"worker backward raised: {res['bwd_exc'][0]!r}")
    assert res["fwd_exc"] is None, (
        f"main-thread forward raised: {res['fwd_exc'][0]!r}")


def overlap_step(layers, x_bwd, x_fwd, jitter=0.0):
    """
    worker thread: backward(x_bwd's graph)   ||   main thread: forward(x_fwd)
    lock-stepped through the SyncHook barrier. The main-thread forward loss
    is RETAINED and returned so the caller can backward it afterwards.
    """
    reset_runtime_state()                                     # clean baseline
    coordinator.disable()
    zero_grads(layers)
    loss_bwd = forward_loss(x_bwd, layers, use_hook=True)      # mb1 graph

    bwd_box = []

    def run_bwd():
        try:
            loss_bwd.backward()
        except BaseException as e:                            # pylint: disable=broad-exception-caught
            bwd_box.append((e, traceback.format_exc()))
            coordinator.abort()

    coordinator.enable(jitter=jitter)
    t = threading.Thread(target=run_bwd, name="bwd-thread", daemon=True)
    t.start()

    fwd_box = []
    fwd_engine_ids = []
    loss_fwd = None
    try:
        loss_fwd = forward_loss(x_fwd, layers, use_hook=True,  # mb2 graph
                                engine_probe=fwd_engine_ids)
    except BaseException as e:                                # pylint: disable=broad-exception-caught
        fwd_box.append((e, traceback.format_exc()))
        coordinator.abort()
    finally:
        coordinator.disable()
        t.join(timeout=coordinator._timeout)
        if t.is_alive():
            # Unstick a worker parked on the barrier, then give it a moment
            # to unwind, so a hang is less likely to leak a live thread.
            coordinator.abort()
            t.join(timeout=2.0)

    return {
        "loss_fwd": loss_fwd,
        "worker_hung": t.is_alive(),
        "bwd_exc": bwd_box[0] if bwd_box else None,
        "fwd_exc": fwd_box[0] if fwd_box else None,
        "fwd_engine_ids": fwd_engine_ids,
    }


# --- the checks (run in a child process; see the subprocess section) ------

def _check_mb2_graph_backwardable():
    """
    The main-thread forward graph built DURING the overlap must be
    backward-able afterwards -- DualPipe's core semantic. Exposes A3: the
    worker backward's AutoGradGuard disables grad globally, so the
    concurrent forward may stop recording grad -> mb2 graph non-diff.

    The corruption is intermittent, so this loops with jitter; a single
    shot is too flaky to be a reliable regression check.
    """
    dim = 8
    n_iters = 200
    layers = make_layers(3, dim)
    x_mb1 = Tensor(np.random.randn(4, dim).astype(np.float32))
    x_mb2 = Tensor(np.random.randn(4, dim).astype(np.float32))
    ref_mb2 = reference_grads(x_mb2, layers)

    bad = []
    for i in range(n_iters):
        res = overlap_step(layers, x_mb1, x_mb2, jitter=0.0005)
        if res["worker_hung"]:
            bad.append(f"iter {i}: worker hung")
            break
        if res["bwd_exc"] is not None:
            bad.append(f"iter {i}: worker backward raised "
                       f"{res['bwd_exc'][0]!r}")
            continue
        if res["loss_fwd"] is None:
            bad.append(f"iter {i}: forward raised {res['fwd_exc'][0]!r}")
            continue
        zero_grads(layers)
        try:
            res["loss_fwd"].backward()
        except Exception as e:                            # pylint: disable=broad-exception-caught
            bad.append(f"iter {i}: mb2 graph not backward-able ({e!r})")
            continue
        if not grads_match(snapshot_grads(layers), ref_mb2):
            bad.append(f"iter {i}: mb2 gradients wrong")
    assert not bad, (
        f"mb2 graph corrupted in {len(bad)}/{n_iters} overlaps "
        f"-- A3: concurrent forward grad recording corrupted; "
        f"first: {bad[0]}")


def _check_no_autodiff_engine_leak():
    """
    The main-thread forward must not observe a worker backward's AutoDiff
    engine. Exposes A1: local_auto_diff_engine is a process-global, held by
    AutoDiffGuard for the worker's whole RunBackward scope.

    The probe is in plain main-thread forward code (forward_loss), not
    inside a _Function hook -- the _Function machinery masks the engine
    context, so an in-hook probe gives a false pass.
    """
    dim = 8
    layers = make_layers(3, dim)
    x_mb1 = Tensor(np.random.randn(4, dim).astype(np.float32))
    x_mb2 = Tensor(np.random.randn(4, dim).astype(np.float32))

    # Baseline: a single-threaded forward sees no active engine.
    reset_runtime_state()
    base_probe = []
    forward_loss(x_mb2, layers, use_hook=False, engine_probe=base_probe)
    assert all(e == NO_ENGINE for e in base_probe), (
        f"precondition failed: non-overlap forward already saw engine ids "
        f"{base_probe} (expected all {NO_ENGINE})")

    res = overlap_step(layers, x_mb1, x_mb2)
    assert_overlap_clean(res)

    probe = res["fwd_engine_ids"]
    print(f"   (A1 probe: main-thread forward engine ids during overlap = "
          f"{probe})")
    leaked = sorted({e for e in probe if e != NO_ENGINE})
    assert not leaked, (
        f"main-thread forward observed AutoDiff engine id(s) {leaked} "
        f"during the overlap -- A1: the worker backward's "
        f"local_auto_diff_engine leaked across threads")


def _check_runtime_state_restored():
    """
    After the overlap region, global enable_grad / grad_flag must be back
    at the known-good baseline. Exposes A3: AutoGradGuard's save/restore of
    process-global runtime state races with the concurrent thread, so the
    overlap can leave grad globally disabled.

    Asserted against the baseline, not against a before-sample: if a prior
    operation already left the state dirty, a before/after comparison would
    falsely pass.
    """
    dim = 8
    layers = make_layers(3, dim)
    x_mb1 = Tensor(np.random.randn(4, dim).astype(np.float32))
    x_mb2 = Tensor(np.random.randn(4, dim).astype(np.float32))

    reset_runtime_state()
    baseline = sample_state()[:2]
    res = overlap_step(layers, x_mb1, x_mb2)  # resets at entry, may corrupt
    after = sample_state()[:2]
    assert_overlap_clean(res)
    assert after == baseline, (
        f"runtime state not restored after overlap: enable_grad/grad_flag "
        f"baseline={baseline} after={after} -- A3: worker backward left "
        f"global grad state corrupted")


def _check_stress_overlap():
    """
    Many iterations with randomised sleeps to widen the race windows.
    On the current engine this is expected to fail -- and the A1 race can
    SEGFAULT the process here, which is why every check is subprocess-
    isolated.
    """
    dim = 8
    n_iters = 1000
    layers = make_layers(2, dim)
    x_mb1 = Tensor(np.random.randn(4, dim).astype(np.float32))
    x_mb2 = Tensor(np.random.randn(4, dim).astype(np.float32))
    ref_mb2 = reference_grads(x_mb2, layers)
    reset_runtime_state()
    baseline = sample_state()[:2]

    fail = {"hang": 0, "worker_exc": 0, "fwd_exc": 0, "mb2_not_diff": 0,
            "mb2_wrong": 0, "engine_leak": 0, "state_not_restored": 0}
    first_err = None
    done = 0

    for i in range(n_iters):
        done = i + 1
        res = overlap_step(layers, x_mb1, x_mb2, jitter=0.0005)
        if res["worker_hung"]:
            fail["hang"] += 1
            first_err = first_err or f"iter {i}: worker hung"
            break                       # a hang repeats; stop early
        if res["bwd_exc"]:
            fail["worker_exc"] += 1
            first_err = first_err or f"iter {i}: worker {res['bwd_exc'][0]!r}"
        if res["fwd_exc"]:
            fail["fwd_exc"] += 1
            first_err = first_err or f"iter {i}: forward {res['fwd_exc'][0]!r}"
        if any(e != NO_ENGINE for e in res["fwd_engine_ids"]):
            fail["engine_leak"] += 1
        if sample_state()[:2] != baseline:
            fail["state_not_restored"] += 1
        if res["loss_fwd"] is not None and not res["bwd_exc"]:
            zero_grads(layers)
            try:
                res["loss_fwd"].backward()
                if not grads_match(snapshot_grads(layers), ref_mb2):
                    fail["mb2_wrong"] += 1
            except Exception:                                 # pylint: disable=broad-exception-caught
                fail["mb2_not_diff"] += 1
                first_err = first_err or f"iter {i}: mb2 not backward-able"

    total = sum(fail.values())
    print(f"   stress: {done} iters run, failures={fail}")
    assert total == 0, (
        f"stress exposed {total} failures over {done} iters: {fail}; "
        f"first: {first_err}")


# --- subprocess isolation -------------------------------------------------
# Each check runs in its own child process. The A1 data race can SEGFAULT
# the process (not just misbehave), so an in-process pytest run would crash
# the whole runner. Running each check in a child confines a crash to
# "child exited on signal N", which the parent observes as an ordinary
# assertion failure.

CHECKS = {
    "mb2": _check_mb2_graph_backwardable,
    "leak": _check_no_autodiff_engine_leak,
    "state": _check_runtime_state_restored,
    "stress": _check_stress_overlap,
}

CHECK_TIMEOUTS = {"mb2": 300, "leak": 120, "state": 120, "stress": 420}


def _run_one_check(name):
    """Child-process entry point: run a single check, then exit with
    0 = passed, 1 = failed (bug exposed), 2 = unexpected error. The verdict
    is emitted as a single-line `VERDICT|name|status|message` marker so the
    parent can extract it past MindSpore's shutdown log noise."""
    check = CHECKS.get(name)
    if check is None:
        print(f"VERDICT|{name}|ERROR|unknown check name")
        sys.exit(2)
    try:
        check()
    except AssertionError as e:
        print(f"VERDICT|{name}|FAIL|{' '.join(str(e).split())}")
        sys.exit(1)
    except BaseException as e:                                # pylint: disable=broad-exception-caught
        traceback.print_exc()
        print(f"VERDICT|{name}|ERROR|{type(e).__name__}: "
              f"{' '.join(str(e).split())}")
        sys.exit(2)
    print(f"VERDICT|{name}|pass|ok")
    sys.exit(0)


def _output_tail(proc, limit=800):
    text = (proc.stdout or "") + (proc.stderr or "")
    return text[-limit:].strip()


def _verdict(proc):
    """Extract the single-line VERDICT marker emitted by _run_one_check."""
    for line in (proc.stdout or "").splitlines():
        if line.startswith("VERDICT|"):
            parts = line.split("|", 3)
            if len(parts) == 4:
                return parts[3]
    return _output_tail(proc, 300) or "(no verdict; see child output)"


def _signal_desc(rc):
    """Human label for a negative returncode (process killed by signal)."""
    name = {6: "SIGABRT", 9: "SIGKILL", 11: "SIGSEGV"}.get(-rc, "")
    return f"killed by signal {-rc}" + (f" ({name})" if name else "")


def _run_check_subprocess(name):
    """Run check `name` in a child process; return its CompletedProcess.
    Raises subprocess.TimeoutExpired on a hang."""
    return subprocess.run(
        [sys.executable, __file__, "--check", name],
        capture_output=True, text=True, timeout=CHECK_TIMEOUTS[name],
        check=False)


def _assert_check_subprocess(name):
    """pytest body: the check runs in a child process so a SEGFAULT or hang
    is contained. A child crash becomes a controlled assertion failure here,
    so the pytest runner survives instead of dying with the child."""
    try:
        proc = _run_check_subprocess(name)
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            f"{name}: check subprocess hung (> {CHECK_TIMEOUTS[name]}s)") from exc
    rc = proc.returncode
    if rc < 0:
        raise AssertionError(
            f"{name}: check subprocess {_signal_desc(rc)} -- a hard crash "
            f"(the A1/A3 race corrupting memory)\n"
            f"--- child output tail ---\n{_output_tail(proc)}")
    assert rc == 0, f"{name}: check subprocess exit {rc} -- {_verdict(proc)}"


@pytest.mark.xfail(strict=False,
                   reason="real-concurrency shared-weight graph mutation; "
                          "tensor autograd-metadata races are out of scope "
                          "this phase (plan 1.2)")
def test_mb2_graph_backwardable():
    """
    Feature: PyNative multi-thread backward - real-concurrency regression
             (A3, concurrent grad recording).
    Description: A worker thread runs tensor.backward() concurrently with the
                 main thread's forward (mb2), lock-stepped through a Barrier;
                 mb2 is backward-ed afterwards. Runs in an isolated subprocess
                 (see _check_mb2_graph_backwardable).
    Expectation: The mb2 graph is backward-able and its gradients match the
                 single-threaded reference.
    """
    _assert_check_subprocess("mb2")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_no_autodiff_engine_leak():
    """
    Feature: PyNative multi-thread backward - real-concurrency regression
             (A1, local_auto_diff_engine).
    Description: The main-thread forward runs concurrently with a worker
                 backward and probes get_current_autodiff_engine_id(). Runs in
                 an isolated subprocess (see _check_no_autodiff_engine_leak).
    Expectation: The main-thread forward never observes the worker's AutoDiff
                 engine id (P1 isolates local_auto_diff_engine per OS thread).
    """
    _assert_check_subprocess("leak")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_runtime_state_restored():
    """
    Feature: PyNative multi-thread backward - real-concurrency regression
             (A3, global grad runtime state).
    Description: A concurrent forward/backward overlap runs, then global
                 enable_grad / grad_flag are sampled against the known-good
                 baseline. Runs in an isolated subprocess (see
                 _check_runtime_state_restored).
    Expectation: The grad runtime state is restored to the baseline after the
                 overlap (P2/P3 isolate GradState / OpRunStatus per thread).
    """
    _assert_check_subprocess("state")


@pytest.mark.xfail(strict=False,
                   reason="real-concurrency stress; tensor autograd-metadata "
                          "races are out of scope this phase (plan 1.2)")
def test_stress_overlap():
    """
    Feature: PyNative multi-thread backward - real-concurrency stress
             (A1/A3 races).
    Description: 1000 concurrent forward/backward overlap iterations with
                 randomised jitter to widen the race windows. Runs in an
                 isolated subprocess (see _check_stress_overlap).
    Expectation: No failures over all iterations. Non-strict xfail: this
                 1000-iteration real-concurrency stress also hits tensor
                 autograd-metadata races out of scope this phase (plan 1.2).
    """
    _assert_check_subprocess("stress")


def main():
    print("=== P0: multi-thread backward regression verification ===")
    print(f"mindspore {ms.__version__}  PYNATIVE  CPU")
    print("(each check runs in an isolated subprocess)\n")
    results = []
    for name in CHECKS:
        try:
            proc = _run_check_subprocess(name)
            rc = proc.returncode
            if rc == 0:
                status, detail = "pass", ""
            elif rc == 1:
                status, detail = "FAIL", _verdict(proc)
            elif rc < 0:
                status, detail = "CRASH", _signal_desc(rc)
            else:
                status, detail = "ERROR", f"exit {rc} -- {_verdict(proc)}"
        except subprocess.TimeoutExpired:
            status = "HANG"
            detail = f"subprocess timed out (> {CHECK_TIMEOUTS[name]}s)"
        results.append((name, status, detail))
        print(f"  [{status:>5s}]  {name}")

    print("\n=== summary ===")
    for name, status, detail in results:
        print(f"  {status:>5s}  {name}")
        for line in detail.splitlines():
            print(f"         {line}")
    exposed = [n for n, s, _ in results if s != "pass"]
    print()
    if exposed:
        print(f"P0 ACCEPTANCE MET: {len(exposed)}/{len(results)} checks "
              f"fail/crash on the current engine -- A1/A3 are visible; "
              f"these become regressions for P1/P2.")
    else:
        print("P0 ACCEPTANCE NOT MET: every check passed.")
    # Non-zero exit when any check did not pass, so this can gate automation.
    sys.exit(1 if exposed else 0)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--check":
        _run_one_check(sys.argv[2])
    else:
        main()
