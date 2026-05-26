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
Alternating (cooperative) multi-thread backward verification (CPU).

P0 of the alternating-support plan. The scope here is deliberately small:
support COOPERATIVE / ALTERNATING execution -- two Python threads may have
overlapping logical scopes (thread A parked inside RunBackward while
thread B runs forward), but at any moment only ONE thread actually runs
MindSpore code. No real concurrency, no overlap performance.

The strict alternation is enforced by `Handoff`: the worker thread parks
inside its `SyncHook.backward` and blocks; the main thread then runs while
the worker is provably suspended; then the worker is released.

Three properties this phase must achieve (all currently A1/A3 bugs):
  1. while the worker's backward scope is alive, the main thread's forward
     must not observe the worker's AutoDiff engine          (A1)
  2. the forward graph the main thread builds in that interval must still
     be backward-able afterwards, with correct gradients    (A3)
  3. one thread's grad-mode change must not bleed into another thread (A3)

These are PLAIN tests (not xfail): they DEFINE this phase's goal. They
FAIL on the current, unmodified engine -- that is P0's purpose, to prove
the current code fails -- and go GREEN once the TLS fixes land (plan
P1: local_auto_diff_engine, P2: GradState, P3: OpRunStatus). They are
distinct from test_multi_thread_backward.py, which is the xfail stress
file for the harder real-concurrency problems left to a later phase.

Run standalone:  python tests/st/pynative/grad/test_alternating_thread_backward.py
Or under pytest from the repo root.
"""
# pylint: disable=protected-access
import sys
import threading

import numpy as np
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
np.random.seed(0)

# get_current_autodiff_engine_id() returns this when no engine is active.
NO_ENGINE = -1


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


def reset_thread_grad_state():
    """Plan §4.1/§4.5: every Python thread that builds graphs must init its
    own PyNative grad state at entry. With today's process-global state
    this is effectively a no-op; once GradState becomes thread-local a new
    thread starts from defaults and MUST call this."""
    _pynative_executor.set_enable_grad(True)
    _pynative_executor.set_grad_flag(True)


class Handoff:
    """Strict-alternation coordinator. The worker parks inside its backward
    and blocks; the main thread runs while the worker is suspended; only
    one thread runs MindSpore code at a time (no real concurrency)."""

    def __init__(self):
        self._enabled = False
        self._worker_parked = threading.Event()
        self._main_go = threading.Event()

    def arm(self):
        self._enabled = True
        self._worker_parked.clear()
        self._main_go.clear()

    def disarm(self):
        self._enabled = False

    def worker_park(self):
        """Called from the worker's SyncHook.backward: announce that the
        worker is parked inside RunBackward, then block until released."""
        if not self._enabled:
            return
        self._worker_parked.set()
        if not self._main_go.wait(timeout=20.0):
            raise AssertionError(
                "main thread did not release the parked backward worker")

    def wait_worker_parked(self, timeout=20.0):
        return self._worker_parked.wait(timeout)

    def release_worker(self):
        self._main_go.set()

    def notify_worker_failed(self):
        """Unblock a main thread waiting for the worker to park."""
        self._worker_parked.set()


handoff = Handoff()


class SyncHook(_Function):
    """Identity _Function. forward() is a plain pass-through; backward()
    parks the worker thread inside RunBackward (when the handoff is armed)
    so the main thread can run while the worker's backward scope is alive."""

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_out):
        handoff.worker_park()
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
        # Probe the AutoDiff engine id in PLAIN forward code (not inside the
        # _Function hook -- the _Function machinery masks the engine there).
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
    reset_thread_grad_state()
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


def alternating_overlap(layers, x_bwd, x_fwd):
    """
    Strict alternation. The worker thread runs backward(x_bwd's graph) and
    parks at its SyncHook; while it is parked -- and only then -- the main
    thread runs forward(x_fwd) and probes the AutoDiff engine id. Then the
    worker is released and joined.

    Returns (mb2_loss, engine_probe, worker_exc).
    """
    handoff.disarm()
    zero_grads(layers)
    loss_bwd = forward_loss(x_bwd, layers, use_hook=True)      # mb1, on main

    box = []

    def run_bwd():
        reset_thread_grad_state()              # plan §4.5: init thread state
        try:
            loss_bwd.backward()
        except BaseException as e:             # pylint: disable=broad-exception-caught
            box.append(e)
            handoff.notify_worker_failed()     # unblock the main thread

    handoff.arm()
    worker = threading.Thread(target=run_bwd, name="bwd-thread", daemon=True)
    worker.start()

    if not handoff.wait_worker_parked():
        handoff.release_worker()
        worker.join(timeout=5.0)
        handoff.disarm()
        assert not worker.is_alive(), (
            "worker backward never parked and did not exit after release")
        raise AssertionError("worker backward never parked at the SyncHook")

    if box:
        handoff.release_worker()
        worker.join(timeout=5.0)
        handoff.disarm()
        assert not worker.is_alive(), (
            "worker backward raised before parking and did not exit")
        return None, [], box[0]

    # The worker is parked inside RunBackward; only this thread runs now.
    probe = []
    loss_fwd = None
    fwd_exc = None
    try:
        loss_fwd = forward_loss(x_fwd, layers, use_hook=True,
                                engine_probe=probe)
    except BaseException as e:                 # pylint: disable=broad-exception-caught
        fwd_exc = e
    finally:
        handoff.release_worker()
        worker.join(timeout=20.0)
        worker_alive = worker.is_alive()
        handoff.disarm()

    assert not worker_alive, (
        "worker backward did not exit after the main thread released it")
    if fwd_exc is not None:
        raise AssertionError(f"main-thread forward raised: {fwd_exc!r}")
    return loss_fwd, probe, (box[0] if box else None)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_alternating_forward_does_not_see_worker_engine():
    """
    Feature: PyNative alternating multi-thread backward - AutoDiff engine
             thread isolation (plan P1, local_auto_diff_engine).
    Description: A worker thread runs tensor.backward() and parks inside
                 SyncHook.backward; while it is parked the main thread runs a
                 forward pass and samples get_current_autodiff_engine_id().
    Expectation: The main-thread forward never observes the worker backward's
                 AutoDiff engine id; every sample stays NO_ENGINE (-1).
    """
    reset_thread_grad_state()
    layers = make_layers(1, 8)
    x_mb1 = Tensor(np.random.randn(4, 8).astype(np.float32))
    x_mb2 = Tensor(np.random.randn(4, 8).astype(np.float32))

    _, probe, worker_exc = alternating_overlap(layers, x_mb1, x_mb2)
    assert worker_exc is None, f"worker backward raised: {worker_exc!r}"

    leaked = sorted({e for e in probe if e != NO_ENGINE})
    assert not leaked, (
        f"main-thread forward saw the worker backward's AutoDiff engine "
        f"id(s) {leaked} (probe={probe}) -- the process-global "
        f"local_auto_diff_engine leaked across threads")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_alternating_mb2_graph_backwardable():
    """
    Feature: PyNative alternating multi-thread backward - grad-state thread
             isolation (plan P2/P3, GradState / OpRunStatus).
    Description: While a worker thread's backward scope is alive (parked at a
                 hook), the main thread builds a forward graph (mb2); after the
                 worker finishes, mb2 is backward-ed.
    Expectation: The mb2 graph is differentiable and its gradients match the
                 single-threaded reference.
    """
    reset_thread_grad_state()
    layers = make_layers(1, 8)
    x_mb1 = Tensor(np.random.randn(4, 8).astype(np.float32))
    x_mb2 = Tensor(np.random.randn(4, 8).astype(np.float32))
    ref_mb2 = reference_grads(x_mb2, layers)

    loss_fwd, _, worker_exc = alternating_overlap(layers, x_mb1, x_mb2)
    assert worker_exc is None, f"worker backward raised: {worker_exc!r}"
    assert loss_fwd is not None

    zero_grads(layers)
    try:
        loss_fwd.backward()
    except Exception as e:                                    # pylint: disable=broad-exception-caught
        raise AssertionError(
            f"mb2 graph built during the alternating region is not "
            f"backward-able ({e!r}) -- the worker backward disabled grad "
            f"globally") from e
    assert grads_match(snapshot_grads(layers), ref_mb2), (
        "mb2 gradients differ from the single-threaded reference -- the "
        "concurrent-scope forward recorded a corrupted graph")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_grad_mode_is_thread_local():
    """
    Feature: PyNative alternating multi-thread backward - GradState thread
             isolation (plan P2).
    Description: Thread A enters no-grad mode (enable_grad/grad_flag=False) and
                 parks; thread B sets its own grad mode; both threads then
                 sample their enable_grad / grad_flag.
    Expectation: Each thread observes its own grad mode - A stays no-grad and B
                 stays grad-enabled; no cross-thread leakage.
    """
    reset_thread_grad_state()                       # this (B) thread inits

    a_in_no_grad = threading.Event()
    b_done = threading.Event()
    a_seen = {}

    def thread_a():
        reset_thread_grad_state()
        _pynative_executor.set_enable_grad(False)   # A enters no-grad mode
        _pynative_executor.set_grad_flag(False)
        a_in_no_grad.set()
        if not b_done.wait(timeout=20.0):
            a_seen["timeout"] = True
        a_seen["enable_grad"] = _pynative_executor.enable_grad()
        a_seen["grad_flag"] = _pynative_executor.grad_flag()

    ta = threading.Thread(target=thread_a, name="thread-a", daemon=True)
    ta.start()
    assert a_in_no_grad.wait(timeout=20.0), "thread A never set its grad mode"

    # A is parked with grad disabled; B sets its own mode and samples.
    _pynative_executor.set_enable_grad(True)
    _pynative_executor.set_grad_flag(True)
    b_enable = _pynative_executor.enable_grad()
    b_grad_flag = _pynative_executor.grad_flag()
    b_done.set()
    ta.join(timeout=20.0)

    assert not ta.is_alive(), "thread A did not exit after thread B completed"
    assert not a_seen.get("timeout"), "thread A timed out waiting for thread B"
    assert b_enable is True, (
        f"thread B enable_grad={b_enable}, expected True -- thread A's "
        f"no-grad mode leaked into B")
    assert b_grad_flag is True, (
        f"thread B grad_flag={b_grad_flag}, expected True -- thread A's "
        f"grad_flag leaked into B")
    assert a_seen.get("enable_grad") is False, (
        f"thread A enable_grad={a_seen.get('enable_grad')}, expected False "
        f"-- thread B's grad mode clobbered A; grad mode is not thread-local")
    assert a_seen.get("grad_flag") is False, (
        f"thread A grad_flag={a_seen.get('grad_flag')}, expected False "
        f"-- thread B's grad_flag clobbered A; grad mode is not thread-local")


def main():
    print("=== alternating multi-thread backward verification (CPU) ===")
    print(f"mindspore {ms.__version__}  PYNATIVE\n")
    checks = [
        ("A1  forward does not see worker engine",
         test_alternating_forward_does_not_see_worker_engine),
        ("A3  mb2 graph backward-able",
         test_alternating_mb2_graph_backwardable),
        ("A3  grad mode is thread-local",
         test_grad_mode_is_thread_local),
    ]
    failed = []
    for name, fn in checks:
        try:
            fn()
            print(f"  [pass]   {name}")
        except AssertionError as e:
            failed.append((name, str(e)))
            print(f"  [FAIL]   {name}")
        except BaseException as e:                            # pylint: disable=broad-exception-caught
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(f"  [ERROR]  {name}")

    print()
    for name, detail in failed:
        print(f"  {name}\n    {detail}\n")
    if failed:
        print(f"{len(failed)}/{len(checks)} checks fail on the current "
              f"engine. This is P0 -- proving the current code fails. They "
              f"go green after the TLS fixes (plan P1-P3).")
    else:
        print("all checks pass -- the alternating-mode TLS fixes are in.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
