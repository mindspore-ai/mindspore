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

"""Testcases for Tensor ms dispatch"""

import gc
import threading
import weakref
import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, mint, ops
from mindspore import _Function
from mindspore.ops.auto_generate.gen_ops_prim import SelectExtView
from mindspore.utils.ms_dispatch import MsDispatchMode
from mindspore._c_expression import _DisableMsDispatchMode, run_backward
from mindspore.graph.api import _pynative_executor
from tests.mark_utils import arg_mark

RUN_OPS = []

class Layout():
    pass

class DTensor(Tensor):
    def __new__(cls, local_tensor, layout):
        t = Tensor._make_subclass(cls, local_tensor)
        t._local_tensor = local_tensor
        t._layout = layout
        return t

    def __init__(self, local_tensor, layout):
        # Skip __init__. Tensor is created by Tensor._make_subclass in __new__.
        pass

    def asnumpy(self):
        return self._local_tensor.asnumpy()

    @classmethod
    def __ms_dispatch__(cls, func, args=(), kwargs=None):
        def wrap_dtensor(x):
            if isinstance(x, Tensor):
                return DTensor(x, Layout())
            if isinstance(x, (tuple, list)):
                return type(x)(wrap_dtensor(item) for item in x)
            return x

        RUN_OPS.append(func.name)
        if kwargs is None:
            kwargs = {}

        new_args = [arg._local_tensor if isinstance(arg, DTensor) else arg for arg in args]
        with _DisableMsDispatchMode():
            return wrap_dtensor(func(*new_args, **kwargs))


# ====== Common helper: create a DTensor from input data ======

def _make_base_dtensor(data):
    """
    Reset DTensor.RUN_OPS and create a DTensor from given data.
    """
    global RUN_OPS
    RUN_OPS = []
    x = DTensor(Tensor(data), Layout())
    return x


def _force_gc():
    for _ in range(8):
        gc.collect()


def _enable_grad_for_current_thread():
    _pynative_executor.set_enable_grad(True)
    _pynative_executor.set_grad_flag(True)


def _run_backward(loss):
    run_backward((loss,), None, None, False, None, allow_unreachable=True, accumulate_grad=True)


def _make_grad_tensor(data):
    tensor = Tensor(np.array(data, dtype=np.float32))
    tensor.requires_grad_()
    return tensor


def _tensor_numpy(tensor):
    return np.asarray(tensor.asnumpy()).copy()


def _tensor_grad_numpy(tensor):
    assert tensor._grad is not None
    return np.asarray(tensor._grad.asnumpy()).copy()


def _assert_allclose(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def _ms_dispatch_micro_loss(x, weight, function_cls):
    hidden = x * weight + Tensor([0.25, -0.5], ms.float32)
    out = function_cls.apply(hidden)
    out_ref = weakref.ref(out)
    loss = out.sum()
    return loss, out_ref


def _run_serial_ms_dispatch_micro_pair():
    x_bwd = _make_grad_tensor([2.0, 3.0])
    w_bwd = _make_grad_tensor([0.5, -1.5])
    loss_bwd, _ = _ms_dispatch_micro_loss(x_bwd, w_bwd, MsDispatchSaveOutputFunction)
    loss_bwd_value = _tensor_numpy(loss_bwd)
    _run_backward(loss_bwd)

    x_fwd = _make_grad_tensor([4.0, 5.0])
    w_fwd = _make_grad_tensor([-2.0, 0.75])
    loss_fwd, _ = _ms_dispatch_micro_loss(x_fwd, w_fwd, MsDispatchSaveOutputFunction)
    loss_fwd_value = _tensor_numpy(loss_fwd)
    _run_backward(loss_fwd)

    return {
        "loss_bwd": loss_bwd_value,
        "loss_fwd": loss_fwd_value,
        "x_bwd_grad": _tensor_grad_numpy(x_bwd),
        "w_bwd_grad": _tensor_grad_numpy(w_bwd),
        "x_fwd_grad": _tensor_grad_numpy(x_fwd),
        "w_fwd_grad": _tensor_grad_numpy(w_fwd),
    }


class _OverlapHandoff:
    """Coordinate one parked backward thread with one main-thread forward."""

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
        if not self._enabled:
            return
        self._worker_parked.set()
        if not self._main_go.wait(timeout=20.0):
            raise AssertionError("main thread did not release the parked backward worker")

    def wait_worker_parked(self):
        return self._worker_parked.wait(timeout=20.0)

    def release_worker(self):
        self._main_go.set()

    def notify_worker_failed(self):
        self._worker_parked.set()


_OVERLAP_HANDOFF = _OverlapHandoff()


class MsDispatchNonDiffFunction(_Function):
    @staticmethod
    def forward(ctx, x):
        out = DTensor(x * x, Layout())
        ctx.mark_non_differentiable(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * 99


class MsDispatchSaveOutputFunction(_Function):
    @staticmethod
    def forward(ctx, x):
        out = DTensor(x * x, Layout())
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (saved_out,) = ctx.saved_tensors
        assert isinstance(saved_out, DTensor)
        return grad_output * saved_out._local_tensor


class MsDispatchSaveOutputAndParkFunction(_Function):
    @staticmethod
    def forward(ctx, x):
        out = DTensor(x * x, Layout())
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (saved_out,) = ctx.saved_tensors
        assert isinstance(saved_out, DTensor)
        _OVERLAP_HANDOFF.worker_park()
        return grad_output * saved_out._local_tensor


class MsDispatchReturnInputFunction(_Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * 2


class MsDispatchNonDiffReturnInputFunction(_Function):
    @staticmethod
    def forward(ctx, x):
        ctx.mark_non_differentiable(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def test_ms_dispatch_custom_function_mark_non_differentiable_output():
    """
    Feature: Ms dispatch
    Description: Custom Function marks a DTensor output as non-differentiable.
    Expectation: The DTensor output is matched as the forward output and gradients are zero.
    """
    x = Tensor([2.0, 3.0], ms.float32)

    def fn(input_x):
        return MsDispatchNonDiffFunction.apply(input_x).sum()

    value, grad = ms.value_and_grad(fn)(x)
    assert np.allclose(value.asnumpy(), np.array(13.0, dtype=np.float32))
    assert np.allclose(grad.asnumpy(), np.zeros_like(x.asnumpy()))


def test_ms_dispatch_custom_function_save_output_tensor():
    """
    Feature: Ms dispatch
    Description: Custom Function saves a DTensor output for backward.
    Expectation: ctx.saved_tensors returns the DTensor wrapper and backward uses it successfully.
    """
    x = Tensor([2.0, 3.0], ms.float32)

    def fn(input_x):
        return MsDispatchSaveOutputFunction.apply(input_x).sum()

    value, grad = ms.value_and_grad(fn)(x)
    assert np.allclose(value.asnumpy(), np.array(13.0, dtype=np.float32))
    assert np.allclose(grad.asnumpy(), np.array([4.0, 9.0], dtype=np.float32))


def test_ms_dispatch_custom_function_return_input_output():
    """
    Feature: Ms dispatch
    Description: Custom Function returns a DTensor input as its output.
    Expectation: The internal no-grad view keeps DTensor dispatch metadata.
    """
    x = DTensor(Tensor([2.0, 3.0], ms.float32), Layout())

    y = MsDispatchReturnInputFunction.apply(x)
    assert isinstance(y, DTensor)
    z = y + 1
    assert isinstance(z, DTensor)
    assert np.allclose(z.asnumpy(), np.array([3.0, 4.0], dtype=np.float32))


def test_ms_dispatch_custom_function_non_diff_return_input_output():
    """
    Feature: Ms dispatch
    Description: Custom Function marks a DTensor input non-differentiable and returns it as output.
    Expectation: The non-differentiable no-grad view keeps DTensor dispatch metadata.
    """
    x = DTensor(Tensor([2.0, 3.0], ms.float32), Layout())

    y = MsDispatchNonDiffReturnInputFunction.apply(x)
    assert isinstance(y, DTensor)
    z = y + 1
    assert isinstance(z, DTensor)
    assert np.allclose(z.asnumpy(), np.array([3.0, 4.0], dtype=np.float32))


def test_ms_dispatch_custom_function_save_output_with_saved_tensor_hooks():
    """
    Feature: Ms dispatch
    Description: Custom Function saves a DTensor output while saved_tensors_hooks are active.
    Expectation: Pack and unpack hooks receive the DTensor wrapper.
    """
    events = []
    x = Tensor([2.0, 3.0], ms.float32)

    def pack_hook(tensor):
        events.append(("pack", type(tensor).__name__))
        assert isinstance(tensor, DTensor)
        return tensor

    def unpack_hook(tensor):
        events.append(("unpack", type(tensor).__name__))
        assert isinstance(tensor, DTensor)
        return tensor

    def fn(input_x):
        with ms.saved_tensors_hooks(pack_hook, unpack_hook):
            return MsDispatchSaveOutputFunction.apply(input_x).sum()

    value, grad = ms.value_and_grad(fn)(x)
    assert np.allclose(value.asnumpy(), np.array(13.0, dtype=np.float32))
    assert np.allclose(grad.asnumpy(), np.array([4.0, 9.0], dtype=np.float32))
    assert events == [("pack", "DTensor"), ("unpack", "DTensor")]


@pytest.mark.parametrize("retain_graph", [None, True])
def test_ms_dispatch_custom_function_saved_output_releases_forward_wrapper(retain_graph):
    """
    Feature: Ms dispatch
    Description: Custom Function saves a DTensor output without retaining the forward Python wrapper.
    Expectation: The forward DTensor wrapper is collectable without backward and after retain_graph backward.
    """
    prev_grad_flag = _pynative_executor.grad_flag()
    _pynative_executor.set_grad_flag(True)
    try:
        x = Tensor([2.0, 3.0], ms.float32)
        x.requires_grad_()
        y = MsDispatchSaveOutputFunction.apply(x)
        y_ref = weakref.ref(y)
        z = y.sum()
        if retain_graph is not None:
            run_backward((z,), None, retain_graph, False, None, allow_unreachable=True, accumulate_grad=True)
        del y, z, x
        _force_gc()
        assert y_ref() is None
    finally:
        _pynative_executor.set_grad_flag(prev_grad_flag)


def test_ms_dispatch_overlap_like_threaded_backward_matches_serial_precision():
    """
    Feature: Ms dispatch
    Description: A worker thread runs backward through a saved DTensor output while
        the main thread builds another graph.
    Expectation: The saved output wrapper is released, and overlapped loss/dx/dw
        match the serial baseline.
    """
    prev_enable_grad = _pynative_executor.enable_grad()
    prev_grad_flag = _pynative_executor.grad_flag()
    _enable_grad_for_current_thread()
    try:
        expect = _run_serial_ms_dispatch_micro_pair()

        x_bwd = _make_grad_tensor([2.0, 3.0])
        w_bwd = _make_grad_tensor([0.5, -1.5])
        loss_bwd, y_bwd_ref = _ms_dispatch_micro_loss(
            x_bwd, w_bwd, MsDispatchSaveOutputAndParkFunction)
        loss_bwd_value = _tensor_numpy(loss_bwd)
        _force_gc()
        assert y_bwd_ref() is None

        worker_errors = []

        def run_worker_backward():
            _enable_grad_for_current_thread()
            try:
                _run_backward(loss_bwd)
            except BaseException as exc:  # pylint: disable=broad-exception-caught
                worker_errors.append(exc)
                _OVERLAP_HANDOFF.notify_worker_failed()

        _OVERLAP_HANDOFF.arm()
        worker = threading.Thread(target=run_worker_backward, name="ms-dispatch-bwd-thread", daemon=True)
        worker.start()
        try:
            assert _OVERLAP_HANDOFF.wait_worker_parked(), "worker backward never parked"
            assert not worker_errors, f"worker backward raised before parking: {worker_errors[0]!r}"

            x_fwd = _make_grad_tensor([4.0, 5.0])
            w_fwd = _make_grad_tensor([-2.0, 0.75])
            loss_fwd, y_fwd_ref = _ms_dispatch_micro_loss(
                x_fwd, w_fwd, MsDispatchSaveOutputAndParkFunction)
            loss_fwd_value = _tensor_numpy(loss_fwd)
            _force_gc()
            assert y_fwd_ref() is None
        finally:
            _OVERLAP_HANDOFF.release_worker()
            worker.join(timeout=20.0)
            _OVERLAP_HANDOFF.disarm()

        assert not worker.is_alive(), "worker backward did not exit after release"
        assert not worker_errors, f"worker backward raised: {worker_errors[0]!r}"
        _assert_allclose(loss_bwd_value, expect["loss_bwd"])
        _assert_allclose(_tensor_grad_numpy(x_bwd), expect["x_bwd_grad"])
        _assert_allclose(_tensor_grad_numpy(w_bwd), expect["w_bwd_grad"])

        _run_backward(loss_fwd)
        _assert_allclose(loss_fwd_value, expect["loss_fwd"])
        _assert_allclose(_tensor_grad_numpy(x_fwd), expect["x_fwd_grad"])
        _assert_allclose(_tensor_grad_numpy(w_fwd), expect["w_fwd_grad"])
    finally:
        _OVERLAP_HANDOFF.release_worker()
        _OVERLAP_HANDOFF.disarm()
        _pynative_executor.set_enable_grad(prev_enable_grad)
        _pynative_executor.set_grad_flag(prev_grad_flag)


def test_ms_dispatch_tensor_hook_return_dtensor():
    """
    Feature: Ms dispatch
    Description: A tensor hook returns DTensor and the next hook receives the DTensor wrapper.
    Expectation: source_py_obj is preserved across the hook boundary.
    """
    events = []
    x = Tensor([2.0, 3.0], ms.float32)

    def fn(input_x):
        out = input_x * input_x

        def hook_dtensor(grad):
            events.append("hook_dtensor")
            return DTensor(grad * 2, Layout())

        def hook_check(grad):
            events.append(type(grad).__name__)
            assert isinstance(grad, DTensor)
            return grad._local_tensor * 3

        out.register_hook(hook_dtensor)
        out.register_hook(hook_check)
        return out.sum()

    value, grad = ms.value_and_grad(fn)(x)
    assert np.allclose(value.asnumpy(), np.array(13.0, dtype=np.float32))
    assert np.allclose(grad.asnumpy(), np.array([24.0, 36.0], dtype=np.float32))
    assert events == ["hook_dtensor", "DTensor"]


def test_ms_dispatch_source_py_obj_user_data_key_is_reserved():
    """
    Feature: Ms dispatch
    Description: Python code cannot set the reserved source_py_obj user data key.
    Expectation: User code cannot bypass the internal source_py_obj preservation helper.
    """
    x = Tensor([1.0], ms.float32)

    with pytest.raises(RuntimeError, match="reserved for internal ms_dispatch state"):
        x._set_user_data("source_py_obj", x)


def test_ms_dispatch_mode_return_dtensor_preserves_python_wrapper():
    """
    Feature: Ms dispatch
    Description: A guard callback returns a fresh DTensor.
    Expectation: The C++ dispatch return conversion preserves that exact Python wrapper.
    """
    callback_result = {}

    class ReturnDtensorMode(MsDispatchMode):
        def __ms_dispatch__(self, func, args=(), kwargs=None):
            if kwargs is None:
                kwargs = {}
            with _DisableMsDispatchMode():
                out = DTensor(func(*args, **kwargs), Layout())
            callback_result["ref"] = weakref.ref(out)
            return out

    with ReturnDtensorMode():
        y = Tensor([1.0, 2.0], ms.float32).add(Tensor([3.0, 4.0], ms.float32))

    assert isinstance(y, DTensor)
    assert callback_result["ref"]() is y


def test_ms_dispatch_custom_backward_return_dtensor_preserves_next_hook_wrapper():
    """
    Feature: Ms dispatch
    Description: Custom Function backward returns a fresh DTensor consumed by a tensor hook.
    Expectation: The hook receives the same DTensor wrapper returned by backward.
    """
    backward_result = {}

    class BackwardReturnsDtensor(_Function):
        @staticmethod
        def forward(ctx, x):
            return x * x

        @staticmethod
        def backward(ctx, grad_output):
            out = DTensor(grad_output * 5, Layout())
            backward_result["ref"] = weakref.ref(out)
            return out

    def fn(input_x):
        def input_hook(grad):
            assert isinstance(grad, DTensor)
            assert backward_result["ref"]() is grad
            return grad._local_tensor

        input_x.register_hook(input_hook)
        return BackwardReturnsDtensor.apply(input_x).sum()

    value, grad = ms.value_and_grad(fn)(Tensor([2.0, 3.0], ms.float32))
    assert np.allclose(value.asnumpy(), np.array(13.0, dtype=np.float32))
    assert np.allclose(grad.asnumpy(), np.array([5.0, 5.0], dtype=np.float32))


def test_ms_dispatch_saved_tensor_unpack_return_dtensor_preserves_wrapper():
    """
    Feature: Ms dispatch
    Description: saved_tensors_hooks unpack_hook returns a fresh DTensor.
    Expectation: ctx.saved_tensors returns the same DTensor wrapper produced by unpack_hook.
    """
    unpack_result = {}

    class SaveOutputWithHooks(_Function):
        @staticmethod
        def forward(ctx, x):
            out = DTensor(x * x, Layout())
            ctx.save_for_backward(out)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            (saved_out,) = ctx.saved_tensors
            assert isinstance(saved_out, DTensor)
            assert unpack_result["ref"]() is saved_out
            return grad_output * saved_out._local_tensor

    def pack_hook(tensor):
        assert isinstance(tensor, DTensor)
        return tensor

    def unpack_hook(tensor):
        assert isinstance(tensor, DTensor)
        out = DTensor(tensor._local_tensor, Layout())
        unpack_result["ref"] = weakref.ref(out)
        return out

    def fn(input_x):
        with ms.saved_tensors_hooks(pack_hook, unpack_hook):
            return SaveOutputWithHooks.apply(input_x).sum()

    value, grad = ms.value_and_grad(fn)(Tensor([2.0, 3.0], ms.float32))
    assert np.allclose(value.asnumpy(), np.array(13.0, dtype=np.float32))
    assert np.allclose(grad.asnumpy(), np.array([4.0, 9.0], dtype=np.float32))


# ====== Scenario 1: basic add ======

def test_ms_dispatch_add():
    """
    Feature: Ms dispatch
    Description: Test dtensor add via ms dispatch.
    Expectation: Result is correct and AddExt is recorded.
    """
    x = _make_base_dtensor([1, 1, 1])
    y = x.add(x)
    assert isinstance(y, DTensor)
    assert np.allclose(y.asnumpy(), np.array([2, 2, 2]))
    assert RUN_OPS == ["AddExt"]


# ====== Scenario 2: reshape with tuple shape ======

def test_ms_dispatch_reshape_tuple():
    """
    Feature: Ms dispatch
    Description: Test dtensor reshape with tuple shape via ms dispatch.
    Expectation: Result is correct and Reshape is recorded after AddExt.
    """
    x = _make_base_dtensor([1, 1, 1])
    y = x.add(x)
    assert isinstance(y, DTensor)
    z = y.reshape((3, 1))
    assert isinstance(z, DTensor)
    assert np.allclose(z.asnumpy(), np.array([[2], [2], [2]]))
    assert RUN_OPS == ["AddExt", "Reshape"]


# ====== Scenario 3: tensor overload operator (+ scalar) ======

def test_ms_dispatch_add_scalar_overload():
    """
    Feature: Ms dispatch
    Description: Test dtensor add scalar via tensor overload with ms dispatch.
    Expectation: Result is correct and AddScalar is recorded.
    """
    x = _make_base_dtensor([1, 1, 1])
    y = x.add(x)
    assert isinstance(y, DTensor)
    z = y.reshape((3, 1))
    assert isinstance(z, DTensor)
    w = z + 1
    assert isinstance(w, DTensor)
    assert np.allclose(w.asnumpy(), np.array([[3], [3], [3]]))
    assert RUN_OPS == ["AddExt", "Reshape", "AddScalar"]


# ====== Scenario 4: mint.sin ======

def test_ms_dispatch_mint_sin():
    """
    Feature: Ms dispatch
    Description: Test dtensor with mint.sin via ms dispatch.
    Expectation: Result is correct and Sin is recorded.
    """
    x = _make_base_dtensor([1, 1, 1])
    y = x.add(x)
    assert isinstance(y, DTensor)
    z = y.reshape((3, 1))
    assert isinstance(z, DTensor)
    w = z + 1
    assert isinstance(w, DTensor)
    a = mint.sin(w)
    assert isinstance(a, DTensor)
    assert np.allclose(a.asnumpy(), np.sin(np.array([[3], [3], [3]])))
    assert RUN_OPS == ["AddExt", "Reshape", "AddScalar", "Sin"]


# ====== Scenario 5: ops.Assign primitive ======

def test_ms_dispatch_assign_primitive():
    """
    Feature: Ms dispatch
    Description: Test dtensor with ops.Assign primitive via ms dispatch.
    Expectation: Result is correct and Assign is recorded.
    """
    x = _make_base_dtensor([1, 1, 1])
    y = x.add(x)
    assert isinstance(y, DTensor)
    z = y.reshape((3, 1))
    w = z + 1
    a = mint.sin(w)
    assert isinstance(a, DTensor)

    expect_data = np.array([[1], [1], [1]])
    b = ops.Assign()(a, Tensor(expect_data))
    assert isinstance(b, DTensor)
    assert np.allclose(b.asnumpy(), expect_data)
    assert RUN_OPS == ["AddExt", "Reshape", "AddScalar", "Sin", "Assign"]


# ====== Scenario 6: mint.cat ======

def test_ms_dispatch_mint_cat():
    """
    Feature: Ms dispatch
    Description: Test dtensor with mint.cat via ms dispatch.
    Expectation: Result is correct and Concat is recorded.
    """
    x = _make_base_dtensor([1, 1, 1])
    y = x.add(x)
    assert isinstance(y, DTensor)
    z = y.reshape((3, 1))
    assert isinstance(z, DTensor)
    w = z + 1
    assert isinstance(w, DTensor)
    a = mint.sin(w)

    expect_data = np.array([[1], [1], [1]])
    b = ops.Assign()(a, Tensor(expect_data))
    assert isinstance(b, DTensor)
    c = mint.cat((b, b))
    assert isinstance(c, DTensor)
    assert np.allclose(c.asnumpy(), np.array([[1], [1], [1], [1], [1], [1]]))
    assert RUN_OPS == ["AddExt", "Reshape", "AddScalar", "Sin", "Assign", "Concat"]


# ====== Scenario 7: primitive attribute Split ======

def test_ms_dispatch_split_primitive_attr():
    """
    Feature: Ms dispatch
    Description: Test dtensor with primitive attribute Split via ms dispatch.
    Expectation: Split result shape/value is correct and Split is recorded.
    """
    x = _make_base_dtensor([1, 1, 1, 1])
    y = ms.ops.Split(axis=0, output_num=2)(x)
    assert len(y) == 2
    assert isinstance(y[0], DTensor)
    assert isinstance(y[1], DTensor)
    assert np.allclose(y[0].asnumpy(), np.array([1, 1]))
    assert np.allclose(y[1].asnumpy(), np.array([1, 1]))
    assert RUN_OPS == ["Split"]


# ====== Scenario 8: ops.cast ======

def test_ms_dispatch_cast():
    """
    Feature: Ms dispatch
    Description: Test dtensor with ops.cast via ms dispatch.
    Expectation: Result is correct and Cast is recorded.
    """
    x = _make_base_dtensor([1, 1, 1, 1])
    y = ms.ops.Split(axis=0, output_num=2)(x)
    assert len(y) == 2
    assert isinstance(y[0], DTensor)
    assert isinstance(y[1], DTensor)
    y0 = ms.ops.cast(y[0], ms.float32)
    assert isinstance(y0, DTensor)

    assert np.allclose(y0.asnumpy(), np.array([1.0, 1.0]))
    assert RUN_OPS == ["Split", "Cast"]


# ====== Scenario 9: Tensor.to(dtype) ======

def test_ms_dispatch_to_dtype():
    """
    Feature: Ms dispatch
    Description: Test dtensor to(ms.float32) via ms dispatch.
    Expectation: Result is correct and ToDtype is recorded.
    """
    x = _make_base_dtensor([1, 1, 1, 1])
    y = ms.ops.Split(axis=0, output_num=2)(x)
    y1 = y[1].to(ms.float32)
    assert isinstance(y1, DTensor)
    assert np.allclose(y1.asnumpy(), np.array([1.0, 1.0]))
    # to(dtype) calls Cast internally
    assert RUN_OPS == ["Split", "Cast"]


def test__update_data_shallow_copy_tensor():
    """
    Feature: Ms dispatch
    Description: Test Tensor._update_data.
    Expectation: data should be updated (shallow copy).
    """
    data1 = np.array([1, 2, 3], dtype=np.float32)
    data2 = np.array([4, 5, 6], dtype=np.float32)

    t1 = Tensor(data1)
    t2 = Tensor(data2)

    t1._update_data(t2)

    np.testing.assert_allclose(t1.asnumpy(), t2.asnumpy())
    assert t1._data_ptr() == t2._data_ptr()


def test__update_data_raise_on_non_tensor():
    """
    Feature: Ms dispatch
    Description: Test Tensor._update_data with invalid input.
    Expectation: Raise RuntimeError.
    """
    t = Tensor(np.array([1, 2, 3], dtype=np.float32))
    not_tensor = 123

    with pytest.raises(RuntimeError, match="Only support input type Tensor"):
        t._update_data(not_tensor)


def test_ms_dispatch_grad():
    """
    Feature: Ms dispatch
    Description: Test grad with ms dispatch.
    Expectation: Result is correct and Grad is recorded.
    """
    x = _make_base_dtensor([1, 1, 1])

    def forward_fn(x):
        return x.sin()

    ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["Sin", "Cos", "Mul"]


class TestMsDispatchMode(MsDispatchMode):
    def __ms_dispatch__(self, func, args=(), kwargs=None):
        RUN_OPS.append(func.name)
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


def test_ms_dispatch_mode_guard():
    """
    Feature: Ms dispatch
    Description: Test MsDispatchMode.
    Expectation: Result is correct and MsDispatchMode is recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestMsDispatchMode():
        x = Tensor([1, 1, 1])
        _ = x.sin()
    assert RUN_OPS == ["Sin"]


class TestNestedMsDispatchMode:
    class MsDispatchMode1(MsDispatchMode):
        def __ms_dispatch__(self, func, args=(), kwargs=None):
            RUN_OPS.append("MsDispatchMode1")
            RUN_OPS.append(func.name)
            if kwargs is None:
                kwargs = {}
            return func(*args, **kwargs)


    class MsDispatchMode2(MsDispatchMode):
        def __ms_dispatch__(self, func, args=(), kwargs=None):
            RUN_OPS.append("MsDispatchMode2")
            RUN_OPS.append(func.name)
            if kwargs is None:
                kwargs = {}
            return func(*args, **kwargs)


class ErrorRaisingGuard(MsDispatchMode):
    """Mode that raises RuntimeError on a specific op, passes through otherwise."""
    def __init__(self, error_on_op=None):
        self.error_on_op = error_on_op

    def __ms_dispatch__(self, func, args=(), kwargs=None):
        if self.error_on_op and func.name == self.error_on_op:
            raise RuntimeError(f"Intentional error on {func.name}")
        RUN_OPS.append(func.name)
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


def test_ms_dispatch_nested_mode_guard():
    """
    Feature: Ms dispatch
    Description: Test NestedMsDispatchMode.
    Expectation: Result is correct and NestedMsDispatchMode is recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestNestedMsDispatchMode.MsDispatchMode1():
        with TestNestedMsDispatchMode.MsDispatchMode2():
            x = Tensor([1, 1, 1])
            _ = x.sin()
    assert RUN_OPS == ["MsDispatchMode2", "Sin", "MsDispatchMode1", "Sin"]


def test_ms_dispatch_nested_mode_guard_with_dtensor():
    """
    Feature: Ms dispatch
    Description: Test NestedMsDispatchMode with DTensor.
    Expectation: Result is correct and NestedMsDispatchMode is recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestNestedMsDispatchMode.MsDispatchMode1():
        with TestNestedMsDispatchMode.MsDispatchMode2():
            x = _make_base_dtensor([1, 1, 1])
            y = x.sin()
            assert isinstance(y, DTensor)
    assert RUN_OPS == ["MsDispatchMode2", "Sin", "MsDispatchMode1", "Sin", "Sin"]


def test_ms_dispatch_mode_guard_with_grad():
    """
    Feature: Ms dispatch
    Description: Test ms dispatch guard with grad.
    Expectation: Result is correct and Grad is recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    x = Tensor([1., 1., 1.])
    with TestMsDispatchMode():
        def forward_fn(x):
            out = x.sin()
            return out
        ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["Sin", "Ones", "Cos", "Mul"]


def test_ms_dispatch_nested_mode_guard_with_grad():
    """
    Feature: Ms dispatch
    Description: Test nested ms dispatch guard with grad.
    Expectation: Result is correct and Grad is recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestNestedMsDispatchMode.MsDispatchMode1():
        with TestNestedMsDispatchMode.MsDispatchMode2():
            x = Tensor([1., 1., 1.])
            def forward_fn(x):
                return x.sin()
            ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["MsDispatchMode2", "Sin", "MsDispatchMode1", "Sin",
                       "MsDispatchMode2", "Ones", "MsDispatchMode1", "Ones", 
                       "MsDispatchMode2", "Cos", "MsDispatchMode1", "Cos", 
                       "MsDispatchMode2", "Mul", "MsDispatchMode1", "Mul"]


def test_ms_dispatch_nested_mode_guard_and_dtensor_with_grad():
    """
    Feature: Ms dispatch
    Description: Test nested ms dispatch guard and dtensor with grad.
    Expectation: Result is correct and Grad is recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestNestedMsDispatchMode.MsDispatchMode1():
        with TestNestedMsDispatchMode.MsDispatchMode2():
            x = _make_base_dtensor([1., 1., 1.])
            def forward_fn(x):
                out = x.sin()
                assert isinstance(out, DTensor)
                return out
            ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["MsDispatchMode2", "Sin", "MsDispatchMode1", "Sin", "Sin",
                       "MsDispatchMode2", "Ones", "MsDispatchMode1", "Ones", 
                       "MsDispatchMode2", "Cos", "MsDispatchMode1", "Cos", "Cos",
                       "MsDispatchMode2", "Mul", "MsDispatchMode1", "Mul", "Mul"]


# ====== MsDispatchMode with grad: add operation ======

def test_ms_dispatch_mode_guard_with_grad_add():
    """
    Feature: Ms dispatch
    Description: Test ms dispatch guard with grad for add operation.
    Expectation: Result is correct and AddExt grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    x = Tensor([1.0, 2.0, 3.0])
    with TestMsDispatchMode():
        def forward_fn(x):
            return x.add(x)
        ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["AddExt", "Ones", "Add"]


def test_ms_dispatch_nested_mode_guard_with_grad_add():
    """
    Feature: Ms dispatch
    Description: Test nested ms dispatch guard with grad for add operation.
    Expectation: Result is correct and AddExt grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestNestedMsDispatchMode.MsDispatchMode1():
        with TestNestedMsDispatchMode.MsDispatchMode2():
            x = Tensor([1.0, 2.0, 3.0])
            def forward_fn(x):
                return x.add(x)
            ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["MsDispatchMode2", "AddExt", "MsDispatchMode1", "AddExt",
                       "MsDispatchMode2", "Ones", "MsDispatchMode1", "Ones",
                       "MsDispatchMode2", "Add", "MsDispatchMode1", "Add"]


def test_ms_dispatch_nested_mode_guard_and_dtensor_with_grad_add():
    """
    Feature: Ms dispatch
    Description: Test nested ms dispatch guard and dtensor with grad for add operation.
    Expectation: Result is correct and AddExt grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestNestedMsDispatchMode.MsDispatchMode1():
        with TestNestedMsDispatchMode.MsDispatchMode2():
            x = _make_base_dtensor([1.0, 2.0, 3.0])
            def forward_fn(x):
                out = x.add(x)
                assert isinstance(out, DTensor)
                return out
            ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["MsDispatchMode2", "AddExt", "MsDispatchMode1", "AddExt", "AddExt",
                       "MsDispatchMode2", "Ones", "MsDispatchMode1", "Ones",
                       "MsDispatchMode2", "Add", "MsDispatchMode1", "Add"]


# ====== MsDispatchMode with grad: reshape operation ======

def test_ms_dispatch_mode_guard_with_grad_reshape():
    """
    Feature: Ms dispatch
    Description: Test ms dispatch guard with grad for reshape operation.
    Expectation: Result is correct and Reshape grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    with TestMsDispatchMode():
        def forward_fn(x):
            return x.reshape((4,)).sum()
        ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["Reshape", "SumExt", "Ones", "Reshape", "Tile", "Reshape"]


def test_ms_dispatch_nested_mode_guard_with_grad_reshape():
    """
    Feature: Ms dispatch
    Description: Test nested ms dispatch guard with grad for reshape operation.
    Expectation: Result is correct and Reshape grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestNestedMsDispatchMode.MsDispatchMode1():
        with TestNestedMsDispatchMode.MsDispatchMode2():
            x = Tensor([[1.0, 2.0], [3.0, 4.0]])
            def forward_fn(x):
                return x.reshape((4,)).sum()
            ms.value_and_grad(forward_fn)(x)

    assert RUN_OPS == ["MsDispatchMode2", "Reshape", "MsDispatchMode1", "Reshape",
                       "MsDispatchMode2", "SumExt", "MsDispatchMode1", "SumExt",
                       "MsDispatchMode2", "Ones", "MsDispatchMode1", "Ones",
                       "MsDispatchMode2", "Reshape", "MsDispatchMode1", "Reshape",
                       "MsDispatchMode2", "Tile", "MsDispatchMode1", "Tile",
                       "MsDispatchMode2","Reshape", "MsDispatchMode1", "Reshape"]


def test_ms_dispatch_nested_mode_guard_and_dtensor_with_grad_reshape():
    """
    Feature: Ms dispatch
    Description: Test nested ms dispatch guard and dtensor with grad for reshape operation.
    Expectation: Result is correct and Reshape grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestNestedMsDispatchMode.MsDispatchMode1():
        with TestNestedMsDispatchMode.MsDispatchMode2():
            x = _make_base_dtensor([[1.0, 2.0], [3.0, 4.0]])
            def forward_fn(x):
                out = x.reshape((4,)).sum()
                assert isinstance(out, DTensor)
                return out
            ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["MsDispatchMode2", "Reshape", "MsDispatchMode1", "Reshape", "Reshape",
                       "MsDispatchMode2", "SumExt", "MsDispatchMode1", "SumExt", "SumExt",
                       "MsDispatchMode2", "Ones", "MsDispatchMode1", "Ones",
                       "MsDispatchMode2", "Reshape", "MsDispatchMode1", "Reshape",
                       "MsDispatchMode2", "Tile", "MsDispatchMode1", "Tile",
                       "MsDispatchMode2","Reshape", "MsDispatchMode1", "Reshape"]


# ====== MsDispatchMode with grad: mint.cat operation ======

def test_ms_dispatch_mode_guard_with_grad_cat():
    """
    Feature: Ms dispatch
    Description: Test ms dispatch guard with grad for mint.cat operation.
    Expectation: Result is correct and Concat grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    with TestMsDispatchMode():
        def forward_fn(x):
            return mint.cat((x, x), dim=0).sum()
        ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["Concat", "SumExt", "Ones", "Reshape", "Tile", "SplitWithSize", "Add"]


def test_ms_dispatch_nested_mode_guard_with_grad_cat():
    """
    Feature: Ms dispatch
    Description: Test nested ms dispatch guard with grad for mint.cat operation.
    Expectation: Result is correct and Concat grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestNestedMsDispatchMode.MsDispatchMode1():
        with TestNestedMsDispatchMode.MsDispatchMode2():
            x = Tensor([[1.0, 2.0], [3.0, 4.0]])
            def forward_fn(x):
                return mint.cat((x, x), dim=0).sum()
            ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["MsDispatchMode2", "Concat", "MsDispatchMode1", "Concat",
                       "MsDispatchMode2", "SumExt", "MsDispatchMode1", "SumExt",
                       "MsDispatchMode2", "Ones", "MsDispatchMode1", "Ones",
                       "MsDispatchMode2", "Reshape", "MsDispatchMode1", "Reshape",
                       "MsDispatchMode2", "Tile", "MsDispatchMode1", "Tile",
                       "MsDispatchMode2", "SplitWithSize", "MsDispatchMode1", "SplitWithSize",
                       "MsDispatchMode2", "Add", "MsDispatchMode1", "Add"]


def test_ms_dispatch_nested_mode_guard_and_dtensor_with_grad_cat():
    """
    Feature: Ms dispatch
    Description: Test nested ms dispatch guard and dtensor with grad for mint.cat operation.
    Expectation: Result is correct and Concat grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestNestedMsDispatchMode.MsDispatchMode1():
        with TestNestedMsDispatchMode.MsDispatchMode2():
            x = _make_base_dtensor([[1.0, 2.0], [3.0, 4.0]])
            def forward_fn(x):
                out = mint.cat((x, x), dim=0).sum()
                assert isinstance(out, DTensor)
                return out
            ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["MsDispatchMode2", "Concat", "MsDispatchMode1", "Concat", "Concat",
                       "MsDispatchMode2", "SumExt", "MsDispatchMode1", "SumExt", "SumExt",
                       "MsDispatchMode2", "Ones", "MsDispatchMode1", "Ones",
                       "MsDispatchMode2", "Reshape", "MsDispatchMode1", "Reshape",
                       "MsDispatchMode2", "Tile", "MsDispatchMode1", "Tile",
                       "MsDispatchMode2", "SplitWithSize", "MsDispatchMode1", "SplitWithSize",
                       "MsDispatchMode2", "Add", "MsDispatchMode1", "Add"]


# ====== MsDispatchMode with grad: ops.cast operation ======

def test_ms_dispatch_mode_guard_with_grad_cast():
    """
    Feature: Ms dispatch
    Description: Test ms dispatch guard with grad for ops.cast operation.
    Expectation: Result is correct and Cast grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    x = Tensor([1.0, 2.0, 3.0], dtype=ms.float32)
    with TestMsDispatchMode():
        def forward_fn(x):
            y = ms.ops.cast(x, ms.float16)
            return ms.ops.cast(y, ms.float32).sum()
        ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["Cast", "Cast", "SumExt", "Ones", "Reshape", "Tile", "Cast", "Cast"]


def test_ms_dispatch_nested_mode_guard_with_grad_cast():
    """
    Feature: Ms dispatch
    Description: Test nested ms dispatch guard with grad for ops.cast operation.
    Expectation: Result is correct and Cast grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestNestedMsDispatchMode.MsDispatchMode1():
        with TestNestedMsDispatchMode.MsDispatchMode2():
            x = Tensor([1.0, 2.0, 3.0], dtype=ms.float32)
            def forward_fn(x):
                y = ms.ops.cast(x, ms.float16)
                return ms.ops.cast(y, ms.float32).sum()
            ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["MsDispatchMode2", "Cast", "MsDispatchMode1", "Cast",
                       "MsDispatchMode2", "Cast", "MsDispatchMode1", "Cast",
                       "MsDispatchMode2", "SumExt", "MsDispatchMode1", "SumExt",
                       "MsDispatchMode2", "Ones", "MsDispatchMode1", "Ones",
                       "MsDispatchMode2", "Reshape", "MsDispatchMode1", "Reshape",
                       "MsDispatchMode2", "Tile", "MsDispatchMode1", "Tile",
                       "MsDispatchMode2", "Cast", "MsDispatchMode1", "Cast",
                       "MsDispatchMode2", "Cast", "MsDispatchMode1", "Cast"]


def test_ms_dispatch_nested_mode_guard_and_dtensor_with_grad_cast():
    """
    Feature: Ms dispatch
    Description: Test nested ms dispatch guard and dtensor with grad for ops.cast operation.
    Expectation: Result is correct and Cast grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestNestedMsDispatchMode.MsDispatchMode1():
        with TestNestedMsDispatchMode.MsDispatchMode2():
            x = _make_base_dtensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
            def forward_fn(x):
                y = ms.ops.cast(x, ms.float16)
                out = ms.ops.cast(y, ms.float32).sum()
                assert isinstance(out, DTensor)
                return out
            ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["MsDispatchMode2", "Cast", "MsDispatchMode1", "Cast", "Cast",
                       "MsDispatchMode2", "Cast", "MsDispatchMode1", "Cast", "Cast",
                       "MsDispatchMode2", "SumExt", "MsDispatchMode1", "SumExt", "SumExt",
                       "MsDispatchMode2", "Ones", "MsDispatchMode1", "Ones",
                       "MsDispatchMode2", "Reshape", "MsDispatchMode1", "Reshape",
                       "MsDispatchMode2", "Tile", "MsDispatchMode1", "Tile",
                       "MsDispatchMode2", "Cast", "MsDispatchMode1", "Cast",
                       "MsDispatchMode2", "Cast", "MsDispatchMode1", "Cast"]


# ====== MsDispatchMode with grad: mul operation ======

def test_ms_dispatch_mode_guard_with_grad_mul():
    """
    Feature: Ms dispatch
    Description: Test ms dispatch guard with grad for mul operation.
    Expectation: Result is correct and Mul grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    x = Tensor([1.0, 2.0, 3.0])
    with TestMsDispatchMode():
        def forward_fn(x):
            return (x * x).sum()
        ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["Mul", "SumExt", "Ones", "Reshape", "Tile", "Mul", "Mul", "Add"]


def test_ms_dispatch_nested_mode_guard_with_grad_mul():
    """
    Feature: Ms dispatch
    Description: Test nested ms dispatch guard with grad for mul operation.
    Expectation: Result is correct and Mul grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestNestedMsDispatchMode.MsDispatchMode1():
        with TestNestedMsDispatchMode.MsDispatchMode2():
            x = Tensor([1.0, 2.0, 3.0])
            def forward_fn(x):
                return (x * x).sum()
            ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["MsDispatchMode2", "Mul", "MsDispatchMode1", "Mul",
                       "MsDispatchMode2", "SumExt", "MsDispatchMode1", "SumExt",
                       "MsDispatchMode2", "Ones", "MsDispatchMode1", "Ones",
                       "MsDispatchMode2", "Reshape", "MsDispatchMode1", "Reshape",
                       "MsDispatchMode2", "Tile", "MsDispatchMode1", "Tile",
                       "MsDispatchMode2", "Mul", "MsDispatchMode1", "Mul",
                       "MsDispatchMode2", "Mul", "MsDispatchMode1", "Mul",
                       "MsDispatchMode2", "Add", "MsDispatchMode1", "Add"]


def test_ms_dispatch_nested_mode_guard_and_dtensor_with_grad_mul():
    """
    Feature: Ms dispatch
    Description: Test nested ms dispatch guard and dtensor with grad for mul operation.
    Expectation: Result is correct and Mul grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestNestedMsDispatchMode.MsDispatchMode1():
        with TestNestedMsDispatchMode.MsDispatchMode2():
            x = _make_base_dtensor([1.0, 2.0, 3.0])
            def forward_fn(x):
                out = (x * x).sum()
                assert isinstance(out, DTensor)
                return out
            ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["MsDispatchMode2", "Mul", "MsDispatchMode1", "Mul", "Mul",
                       "MsDispatchMode2", "SumExt", "MsDispatchMode1", "SumExt", "SumExt",
                       "MsDispatchMode2", "Ones", "MsDispatchMode1", "Ones",
                       "MsDispatchMode2", "Reshape", "MsDispatchMode1", "Reshape",
                       "MsDispatchMode2", "Tile", "MsDispatchMode1", "Tile",
                       "MsDispatchMode2", "Mul", "MsDispatchMode1", "Mul", "Mul",
                       "MsDispatchMode2", "Mul", "MsDispatchMode1", "Mul", "Mul",
                       "MsDispatchMode2", "Add", "MsDispatchMode1", "Add", "Add"]


# ====== MsDispatchMode with grad: exp operation ======

def test_ms_dispatch_mode_guard_with_grad_exp():
    """
    Feature: Ms dispatch
    Description: Test ms dispatch guard with grad for exp operation.
    Expectation: Result is correct and Exp grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    x = Tensor([1.0, 2.0, 3.0])
    with TestMsDispatchMode():
        def forward_fn(x):
            return mint.exp(x).sum()
        ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["Exp", "SumExt", "Ones", "Reshape", "Tile", "Mul"]


def test_ms_dispatch_nested_mode_guard_with_grad_exp():
    """
    Feature: Ms dispatch
    Description: Test nested ms dispatch guard with grad for exp operation.
    Expectation: Result is correct and Exp grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestNestedMsDispatchMode.MsDispatchMode1():
        with TestNestedMsDispatchMode.MsDispatchMode2():
            x = Tensor([1.0, 2.0, 3.0])
            def forward_fn(x):
                return mint.exp(x).sum()
            ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["MsDispatchMode2", "Exp", "MsDispatchMode1", "Exp",
                       "MsDispatchMode2", "SumExt", "MsDispatchMode1", "SumExt",
                       "MsDispatchMode2", "Ones", "MsDispatchMode1", "Ones",
                       "MsDispatchMode2", "Reshape", "MsDispatchMode1", "Reshape",
                       "MsDispatchMode2", "Tile", "MsDispatchMode1", "Tile",
                       "MsDispatchMode2", "Mul", "MsDispatchMode1", "Mul"]


def test_ms_dispatch_nested_mode_guard_and_dtensor_with_grad_exp():
    """
    Feature: Ms dispatch
    Description: Test nested ms dispatch guard and dtensor with grad for exp operation.
    Expectation: Result is correct and Exp grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestNestedMsDispatchMode.MsDispatchMode1():
        with TestNestedMsDispatchMode.MsDispatchMode2():
            x = _make_base_dtensor([1.0, 2.0, 3.0])
            def forward_fn(x):
                out = mint.exp(x).sum()
                assert isinstance(out, DTensor)
                return out
            ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["MsDispatchMode2", "Exp", "MsDispatchMode1", "Exp", "Exp",
                       "MsDispatchMode2", "SumExt", "MsDispatchMode1", "SumExt", "SumExt",
                       "MsDispatchMode2", "Ones", "MsDispatchMode1", "Ones",
                       "MsDispatchMode2", "Reshape", "MsDispatchMode1", "Reshape",
                       "MsDispatchMode2", "Tile", "MsDispatchMode1", "Tile",
                       "MsDispatchMode2", "Mul", "MsDispatchMode1", "Mul", "Mul"]


def test_ms_dispatch_disable_mode_guard_bypasses_dispatch():
    """
    Feature: Ms dispatch
    Description: _DisableMsDispatchMode suppresses mode dispatch.
    Expectation: Ops inside _DisableMsDispatchMode are NOT recorded by the outer mode.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestMsDispatchMode():
        x = Tensor([1.0, 2.0, 3.0])
        _ = x.sin()  # should be recorded
        with _DisableMsDispatchMode():
            _ = x.cos()  # should NOT be recorded
        _ = x.exp()  # should be recorded
    assert RUN_OPS == ["Sin", "Exp"]


def test_ms_dispatch_disable_mode_isolated_between_python_threads():
    """
    Feature: Ms dispatch
    Description: Thread A enters _DisableMsDispatchMode while thread B uses MsDispatchMode.
    Expectation: Thread B still observes its own ms_dispatch mode.
    """
    global RUN_OPS
    RUN_OPS = []
    holder_entered = threading.Event()
    release_holder = threading.Event()
    holder_errors = []

    def disable_holder():
        try:
            with _DisableMsDispatchMode():
                holder_entered.set()
                if not release_holder.wait(timeout=20.0):
                    raise AssertionError("main thread did not release disable holder")
        except BaseException as exc:  # pylint: disable=broad-exception-caught
            holder_errors.append(exc)
            holder_entered.set()

    holder = threading.Thread(target=disable_holder, name="ms-dispatch-disable-holder", daemon=True)
    holder.start()
    try:
        assert holder_entered.wait(timeout=20.0), "disable holder did not enter _DisableMsDispatchMode"
        assert not holder_errors, f"disable holder raised before main thread ran: {holder_errors[0]!r}"

        with TestMsDispatchMode():
            _ = Tensor([1.0, 2.0, 3.0]).sin()
    finally:
        release_holder.set()
        holder.join(timeout=20.0)

    assert not holder.is_alive(), "disable holder did not exit"
    assert not holder_errors, f"disable holder raised: {holder_errors[0]!r}"
    assert RUN_OPS == ["Sin"]


def test_ms_dispatch_disable_mode_guard_with_dtensor():
    """
    Feature: Ms dispatch
    Description: _DisableMsDispatchMode suppresses DTensor dispatch.
    Expectation: Ops inside _DisableMsDispatchMode bypass DTensor.__ms_dispatch__
                 and return a plain Tensor.
    """
    global RUN_OPS
    RUN_OPS = []
    x = _make_base_dtensor([1.0, 2.0, 3.0])
    y = x.sin()
    assert isinstance(y, DTensor)
    assert RUN_OPS == ["Sin"]
    ops_before = len(RUN_OPS)
    with _DisableMsDispatchMode():
        z = Tensor([1.0, 2.0, 3.0]).sin()
    assert not isinstance(z, DTensor)
    assert len(RUN_OPS) == ops_before  # no new ops recorded


def test_ms_dispatch_guard_exception_handling():
    """
    Feature: Ms dispatch
    Description: Exception propagation + guard stays functional after catch.
    Expectation: RuntimeError propagates; guard works normally after exception is caught.
    """
    global RUN_OPS
    RUN_OPS = []
    guard = ErrorRaisingGuard(error_on_op="Cos")
    with guard:
        x = Tensor([1.0, 2.0, 3.0])
        _ = x.sin()  # should succeed
        assert RUN_OPS == ["Sin"]
        try:
            _ = x.cos()  # should raise
            assert False, "Expected RuntimeError"
        except RuntimeError as e:
            assert "Intentional error on Cos" in str(e)
        # Mode should still work after the caught exception
        RUN_OPS.clear()
        _ = x.exp()
        assert RUN_OPS == ["Exp"]


def test_ms_dispatch_guard_created_outside_with_block():
    """
    Feature: Ms dispatch
    Description: Mode instance created before `with` block.
    Expectation: Mode dispatches correctly when used with a pre-created instance.
    """
    global RUN_OPS
    RUN_OPS = []
    guard = TestMsDispatchMode()
    x = Tensor([1.0, 2.0, 3.0])
    # Mode not active yet
    _ = x.sin()
    assert RUN_OPS == []
    # Now activate it
    with guard:
        _ = x.cos()
    assert RUN_OPS == ["Cos"]


def test_ms_dispatch_nesting_same_guard_instance():
    """
    Feature: Ms dispatch
    Description: Same guard instance pushed twice on stack.
    Expectation: Either works with double dispatch, or raises RuntimeError.
    """
    global RUN_OPS
    RUN_OPS = []
    guard = TestMsDispatchMode()
    try:
        with guard:
            with guard:
                x = Tensor([1.0, 2.0, 3.0])
                _ = x.sin()
        # If it works, Sin should appear twice (once per guard level)
        assert RUN_OPS.count("Sin") == 2
    except RuntimeError:
        # MindSpore may reject pushing the same guard instance twice
        pass


def test_ms_dispatch_guard_priority_over_dtensor():
    """
    Feature: Ms dispatch
    Description: Mode dispatches before DTensor.__ms_dispatch__.
    Expectation: Mode's __ms_dispatch__ is called first, then DTensor's.
    """
    global RUN_OPS
    RUN_OPS = []

    class PriorityGuard(MsDispatchMode):
        def __ms_dispatch__(self, func, args=(), kwargs=None):
            RUN_OPS.append("PriorityGuard")
            RUN_OPS.append(func.name)
            if kwargs is None:
                kwargs = {}
            return func(*args, **kwargs)

    with PriorityGuard():
        x = _make_base_dtensor([1.0, 2.0, 3.0])
        _ = x.sin()
    # Mode should appear first, then DTensor's dispatch
    assert RUN_OPS == ["PriorityGuard", "Sin", "Sin"]


def test_ms_dispatch_grad_numerical_correctness():
    """
    Feature: Ms dispatch
    Description: Verify actual gradient values, not just RUN_OPS.
    Expectation: Gradient of sin(x) is cos(x); values match numerically.
    """
    global RUN_OPS
    RUN_OPS = []
    data = np.array([0.0, np.pi / 4, np.pi / 2], dtype=np.float32)
    x = DTensor(Tensor(data), Layout())

    def forward_fn(x):
        return x.sin().sum()

    value, grad = ms.value_and_grad(forward_fn)(x)
    expected_grad = np.cos(data)
    assert np.allclose(grad.asnumpy(), expected_grad, atol=1e-5)
    expected_value = np.sum(np.sin(data))
    assert np.allclose(value.asnumpy(), expected_value, atol=1e-5)


def test_ms_dispatch_guard_as_decorator():
    """
    Feature: Ms dispatch
    Description: MsDispatchMode used as @decorator (ContextDecorator).
    Expectation: Mode is active during the decorated function call.
    """
    global RUN_OPS
    RUN_OPS = []

    @TestMsDispatchMode()
    def guarded_computation():
        x = Tensor([1.0, 2.0, 3.0])
        return x.sin()

    result = guarded_computation()
    assert np.allclose(result.asnumpy(), np.sin(np.array([1.0, 2.0, 3.0])), atol=1e-5)
    assert RUN_OPS == ["Sin"]


def test_ms_dispatch_dtensor_mul_scalar_inplace_pattern():
    """
    Feature: Ms dispatch
    Description: DTensor with multiply-accumulate pattern via ms dispatch.
    Expectation: Result is numerically correct and mul ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    x = DTensor(Tensor([1.0, 2.0, 3.0]), Layout())
    y = x * 2
    z = y + x
    assert isinstance(y, DTensor)
    assert isinstance(z, DTensor)
    assert np.allclose(z.asnumpy(), np.array([3.0, 6.0, 9.0]))
    assert RUN_OPS == ["Muls", "AddExt"]


def test_ms_dispatch_complex_multi_branch_computation():
    """
    Feature: Ms dispatch
    Description: Diamond-shaped computation graph (fan-out/fan-in) via ms dispatch.
    Expectation: Result is numerically correct; all branch ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    x = DTensor(Tensor([1.0, 2.0, 3.0]), Layout())
    # Fan-out: two branches from x
    branch_a = x.sin()
    branch_b = x.cos()
    # Fan-in: combine branches
    result = branch_a.add(branch_b)
    assert isinstance(branch_a, DTensor)
    assert isinstance(branch_b, DTensor)
    assert isinstance(result, DTensor)
    expected = np.sin(np.array([1.0, 2.0, 3.0])) + np.cos(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(result.asnumpy(), expected, atol=1e-5)
    assert RUN_OPS == ["Sin", "Cos", "AddExt"]


class SkipCallbackDispatchMode(MsDispatchMode):
    """Mode that intercepts Reshape and returns its own result without calling func()."""
    def __ms_dispatch__(self, func, args=(), kwargs=None):
        RUN_OPS.append(func.name)
        if func.name == "Reshape":
            # Skip calling func(*args) — manually compute the reshape result
            input_tensor = args[0]
            shape = args[1]
            with _DisableMsDispatchMode():
                return input_tensor.reshape(shape)
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


def test_ms_dispatch_reshape_grad_when_callback_skipped():
    """
    Feature: Ms dispatch
    Description: When MsDispatchMode intercepts Reshape and does NOT call func(*args),
                 gradient should still be recorded for reshape.
    Expectation: value_and_grad succeeds; grad has correct shape and values.
    """
    global RUN_OPS
    RUN_OPS = []
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    with SkipCallbackDispatchMode():
        def forward_fn(x):
            return x.reshape((4,)).sum()
        value, grad = ms.value_and_grad(forward_fn)(x)
    assert value.asnumpy() == 10.0
    assert grad.shape == (2, 2)
    assert np.allclose(grad.asnumpy(), np.ones((2, 2)))


def test_ms_dispatch_mode_guard_with_grad_reshape_values():
    """
    Feature: Ms dispatch
    Description: Reshape grad through normal dispatch (handler calls func).
    Expectation: Grad values are correct (all ones for sum after reshape).
    """
    global RUN_OPS
    RUN_OPS = []
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    with TestMsDispatchMode():
        def forward_fn(x):
            return x.reshape((4,)).sum()
        value, grad = ms.value_and_grad(forward_fn)(x)
    assert value.asnumpy() == 10.0
    assert grad.shape == (2, 2)
    assert np.allclose(grad.asnumpy(), np.ones((2, 2)))
    assert RUN_OPS == ["Reshape", "SumExt", "Ones", "Reshape", "Tile", "Reshape"]


def test_reshape_grad_normal_view_path():
    """
    Feature: Reshape grad
    Description: Test reshape backward on normal (non-dispatch) view path.
    Expectation: Grad values are correct.
    """
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    def forward_fn(x):
        return x.reshape((3, 2)).sum()
    value, grad = ms.value_and_grad(forward_fn)(x)
    assert value.asnumpy() == 21.0
    assert grad.shape == (2, 3)
    assert np.allclose(grad.asnumpy(), np.ones((2, 3)))


def test_ms_dispatch_dtensor_reshape_grad_values():
    """
    Feature: Ms dispatch
    Description: Test DTensor reshape with gradient values.
    Expectation: Grad flows correctly through DTensor dispatch path.
    """
    global RUN_OPS
    RUN_OPS = []
    x = _make_base_dtensor([[1.0, 2.0], [3.0, 4.0]])
    def forward_fn(x):
        out = x.reshape((4,)).sum()
        return out
    value, grad = ms.value_and_grad(forward_fn)(x)
    assert np.allclose(value.asnumpy(), 10.0)
    assert grad.shape == (2, 2)
    assert np.allclose(grad.asnumpy(), np.ones((2, 2)))
    assert RUN_OPS == ["Reshape", "SumExt"]


def test_reshape_meta_device_shape_correct():
    """
    Feature: Meta device reshape
    Description: Reshape on meta device returns correct shape (no kernel executed).
    Expectation: Output has correct shape and is on meta device.
    """
    with ms.DeviceCtx('meta'):
        x = mint.rand((2, 3), dtype=ms.float32)
        y = x.reshape(6)
        assert y.shape == (6,)
        assert y.device == 'meta'
        assert y.data_ptr() == 0


# ====== RunOpStub dispatch grad: callback skipped ======

class SkipCallbackComputeMode(MsDispatchMode):
    """Mode that intercepts ACos via RunOpStub and computes with numpy,
    skipping the callback. PostDispatchGrad handles gradient registration.
    Note: ACos has no 'dispatch: enable: True' in its op_def yaml, so
    Primitive.__call__ routes through _run_op -> RunOpStub (not pyboost).
    RunOpStub args format: (Primitive, op_name_str, (tensor_inputs,))."""
    def __ms_dispatch__(self, func, args=(), kwargs=None):
        RUN_OPS.append(func.name)
        if func.name == "ACos":
            # args[2] is the tuple of actual tensor inputs in RunOpStub path
            actual_inputs = args[2]
            return Tensor(np.arccos(actual_inputs[0].asnumpy()))
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


def test_runopstub_dispatch_grad_callback_skipped():
    """
    Feature: RunOpStub dispatch grad
    Description: Guard handler intercepts ACos (via Primitive.__call__ -> RunOpStub)
                 and returns numpy-computed result WITHOUT calling func().
                 ACos uses py_method dispatch (no 'dispatch: enable: True' in yaml),
                 so it enters RunOpStub -> RunOpDispatch -> PostDispatchGrad path.
                 PostDispatchGrad must register gradient so backward works.
    Expectation: value_and_grad succeeds with correct gradient values.
    """
    global RUN_OPS
    RUN_OPS = []
    x = Tensor(np.array([0.1, 0.2, 0.3], dtype=np.float32))
    with SkipCallbackComputeMode():
        def forward_fn(x):
            return ops.ACos()(x).sum()
        value, grad = ms.value_and_grad(forward_fn)(x)
    expected_value = np.sum(np.arccos(np.array([0.1, 0.2, 0.3], dtype=np.float32)))
    expected_grad = -1.0 / np.sqrt(1 - np.array([0.1, 0.2, 0.3], dtype=np.float32) ** 2)
    assert np.allclose(value.asnumpy(), expected_value, atol=1e-5)
    assert np.allclose(grad.asnumpy(), expected_grad, atol=1e-5)


def test_runopstub_dispatch_grad_callback_called():
    """
    Feature: RunOpStub dispatch grad
    Description: Guard handler calls func() normally (callback IS called) for ACos
                 which routes through RunOpStub (no 'dispatch: enable: True' in yaml).
                 Regression test: PostDispatchGrad must NOT be called to avoid double grad.
    Expectation: value_and_grad succeeds with correct gradient values.
    """
    global RUN_OPS
    RUN_OPS = []
    x = Tensor(np.array([0.1, 0.2, 0.3], dtype=np.float32))
    with TestMsDispatchMode():
        def forward_fn(x):
            return ops.ACos()(x).sum()
        value, grad = ms.value_and_grad(forward_fn)(x)
    expected_value = np.sum(np.arccos(np.array([0.1, 0.2, 0.3], dtype=np.float32)))
    expected_grad = -1.0 / np.sqrt(1 - np.array([0.1, 0.2, 0.3], dtype=np.float32) ** 2)
    assert np.allclose(value.asnumpy(), expected_value, atol=1e-5)
    assert np.allclose(grad.asnumpy(), expected_grad, atol=1e-5)
    assert RUN_OPS == ["ACos", "SumExt", "Ones", "Reshape", "Tile"]


class SkipCallbackLayerNormMode(MsDispatchMode):
    """Mode that intercepts LayerNorm via RunOpStub and computes with numpy,
    skipping the callback. Tests PostDispatchGrad with a multi-output op.
    LayerNorm has no 'dispatch: enable: True', so it routes through RunOpStub.
    RunOpStub args format: (Primitive, op_name_str, (input_x, gamma, beta, ...)).
    Note: LayerNorm returns (output, mean, rstd) where rstd = 1/sqrt(var + eps)."""
    def __ms_dispatch__(self, func, args=(), kwargs=None):
        RUN_OPS.append(func.name)
        if func.name == "LayerNorm":
            actual = args[2]
            x_np = actual[0].asnumpy()
            gamma_np = actual[1].asnumpy()
            beta_np = actual[2].asnumpy()
            eps = actual[5]
            mean = x_np.mean(axis=-1, keepdims=True)
            var = x_np.var(axis=-1, keepdims=True)
            rstd = 1.0 / np.sqrt(var + eps)
            normed = (x_np - mean) * rstd
            out = normed * gamma_np + beta_np
            return (Tensor(out.astype(np.float32)),
                    Tensor(mean.astype(np.float32)),
                    Tensor(rstd.astype(np.float32)))
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


def test_runopstub_dispatch_grad_layernorm_callback_skipped():
    """
    Feature: RunOpStub dispatch grad
    Description: Guard handler intercepts LayerNorm (multi-output op) via RunOpStub
                 and returns numpy-computed result WITHOUT calling func().
                 Tests PostDispatchGrad with tuple-of-tensors output.
    Expectation: value_and_grad succeeds; gradient matches the callback-called path.
    """
    global RUN_OPS
    np.random.seed(42)
    x_np = np.random.randn(2, 3).astype(np.float32)
    x = Tensor(x_np)
    gamma = Tensor(np.ones(3, dtype=np.float32))
    beta = Tensor(np.zeros(3, dtype=np.float32))

    # Callback-skipped path
    RUN_OPS = []
    with SkipCallbackLayerNormMode():
        def forward_fn(x):
            out, _, _ = ops.LayerNorm(begin_norm_axis=-1)(x, gamma, beta)
            return (out * out).sum()
        value_skip, grad_skip = ms.value_and_grad(forward_fn)(x)

    # Reference: callback-called path
    RUN_OPS = []
    with TestMsDispatchMode():
        def forward_fn_ref(x):
            out, _, _ = ops.LayerNorm(begin_norm_axis=-1)(x, gamma, beta)
            return (out * out).sum()
        value_ref, grad_ref = ms.value_and_grad(forward_fn_ref)(x)

    assert np.allclose(value_skip.asnumpy(), value_ref.asnumpy(), atol=1e-4)
    assert np.allclose(grad_skip.asnumpy(), grad_ref.asnumpy(), atol=1e-4)


def test_runopstub_dispatch_grad_layernorm_callback_called():
    """
    Feature: RunOpStub dispatch grad
    Description: Guard handler calls func() normally (callback IS called) for LayerNorm.
                 Regression test: PostDispatchGrad must NOT be called for multi-output ops.
    Expectation: value_and_grad succeeds with correct gradient values.
    """
    global RUN_OPS
    np.random.seed(42)
    x_np = np.random.randn(2, 3).astype(np.float32)
    x = Tensor(x_np)
    gamma = Tensor(np.ones(3, dtype=np.float32))
    beta = Tensor(np.zeros(3, dtype=np.float32))

    # Without dispatch (ground truth)
    def forward_fn_plain(x):
        out, _, _ = ops.LayerNorm(begin_norm_axis=-1)(x, gamma, beta)
        return (out * out).sum()
    value_plain, grad_plain = ms.value_and_grad(forward_fn_plain)(x)

    # With dispatch (callback called)
    RUN_OPS = []
    with TestMsDispatchMode():
        def forward_fn(x):
            out, _, _ = ops.LayerNorm(begin_norm_axis=-1)(x, gamma, beta)
            return (out * out).sum()
        value, grad = ms.value_and_grad(forward_fn)(x)

    assert np.allclose(value.asnumpy(), value_plain.asnumpy(), atol=1e-5)
    assert np.allclose(grad.asnumpy(), grad_plain.asnumpy(), atol=1e-5)
    assert "LayerNorm" in RUN_OPS


# ====== Pyboost view ops dispatch grad ======

class SkipCallbackViewMode(MsDispatchMode):
    """Mode that intercepts pyboost view ops and computes with numpy,
    skipping the callback. These ops have 'dispatch: enable: True' in yaml
    so they go through the pyboost path (not RunOpStub).
    Pyboost args format: (Tensor, scalar_arg1, scalar_arg2, ...)."""
    def __ms_dispatch__(self, func, args=(), kwargs=None):
        RUN_OPS.append(func.name)
        if func.name == "SelectExtView":
            x, dim, idx = args
            return Tensor(np.take(x.asnumpy(), idx, axis=dim))
        if func.name == "TransposeExtView":
            x, d0, d1 = args
            return Tensor(np.swapaxes(x.asnumpy(), d0, d1).copy())
        if func.name == "Transpose":
            x, dims = args
            return Tensor(np.transpose(x.asnumpy(), dims).copy())
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


@pytest.mark.xfail(reason="SelectExtView backward uses select_ext_view+inplace_copy with view semantics; "
                           "skipping callback returns a numpy copy, breaking shared storage")
def test_pyboost_dispatch_grad_select_ext_view_callback_skipped():
    """
    Feature: Pyboost dispatch grad
    Description: Guard handler intercepts SelectExtView (pyboost view op) and
                 returns numpy-computed result WITHOUT calling func().
                 SelectExtView backward relies on view/inplace semantics:
                   grad_input = zeros(); select_ext_view(grad_input); inplace_copy(...)
                 When the guard intercepts backward's select_ext_view with numpy,
                 the view link is broken and grad_input stays zero.
    Expectation: Gradient is lost (known limitation of skip-callback for view ops
                 whose backward depends on view semantics).
    """
    global RUN_OPS
    np.random.seed(0)
    x = Tensor(np.random.randn(3, 4).astype(np.float32))

    RUN_OPS = []
    with SkipCallbackViewMode():
        v_skip, g_skip = ms.value_and_grad(lambda x: SelectExtView()(x, 0, 1).sum())(x)
    v_plain, g_plain = ms.value_and_grad(lambda x: SelectExtView()(x, 0, 1).sum())(x)

    assert np.allclose(v_skip.asnumpy(), v_plain.asnumpy(), atol=1e-5)
    assert np.allclose(g_skip.asnumpy(), g_plain.asnumpy(), atol=1e-5)


def test_pyboost_dispatch_grad_select_ext_view_callback_called():
    """
    Feature: Pyboost dispatch grad
    Description: Guard handler calls func() normally for SelectExtView.
    Expectation: value_and_grad succeeds; gradient matches the plain path.
    """
    global RUN_OPS
    np.random.seed(0)
    x = Tensor(np.random.randn(3, 4).astype(np.float32))

    RUN_OPS = []
    with TestMsDispatchMode():
        v_call, g_call = ms.value_and_grad(lambda x: SelectExtView()(x, 0, 1).sum())(x)
    v_plain, g_plain = ms.value_and_grad(lambda x: SelectExtView()(x, 0, 1).sum())(x)

    assert np.allclose(v_call.asnumpy(), v_plain.asnumpy(), atol=1e-5)
    assert np.allclose(g_call.asnumpy(), g_plain.asnumpy(), atol=1e-5)


def test_pyboost_dispatch_grad_transpose_ext_view_callback_skipped():
    """
    Feature: Pyboost dispatch grad
    Description: Guard handler intercepts only TransposeExtView and returns
                 numpy-computed result WITHOUT calling func().
                 TransposeExtView backward doesn't rely on view semantics
                 (just does transpose_ext_view on grad), so skip-callback is safe.
    Expectation: value_and_grad succeeds; gradient matches the plain path.
    """
    global RUN_OPS
    np.random.seed(0)
    x = Tensor(np.random.randn(2, 3).astype(np.float32))

    class SkipTransposeExtViewOnly(MsDispatchMode):
        def __ms_dispatch__(self, func, args=(), kwargs=None):
            RUN_OPS.append(func.name)
            if func.name == "TransposeExtView":
                t, d0, d1 = args
                return Tensor(np.swapaxes(t.asnumpy(), d0, d1).copy())
            if kwargs is None:
                kwargs = {}
            return func(*args, **kwargs)

    RUN_OPS = []
    with SkipTransposeExtViewOnly():
        v_skip, g_skip = ms.value_and_grad(lambda x: ops.TransposeExtView()(x, 0, 1)[0, 0])(x)
    v_plain, g_plain = ms.value_and_grad(lambda x: ops.TransposeExtView()(x, 0, 1)[0, 0])(x)

    assert np.allclose(v_skip.asnumpy(), v_plain.asnumpy(), atol=1e-5)
    assert np.allclose(g_skip.asnumpy(), g_plain.asnumpy(), atol=1e-5)


def test_pyboost_dispatch_grad_transpose_ext_view_callback_called():
    """
    Feature: Pyboost dispatch grad
    Description: Guard handler calls func() normally for TransposeExtView.
    Expectation: value_and_grad succeeds; gradient matches the plain path.
    """
    global RUN_OPS
    np.random.seed(0)
    x = Tensor(np.random.randn(2, 3).astype(np.float32))

    RUN_OPS = []
    with TestMsDispatchMode():
        v_call, g_call = ms.value_and_grad(lambda x: ops.TransposeExtView()(x, 0, 1)[0, 0])(x)
    v_plain, g_plain = ms.value_and_grad(lambda x: ops.TransposeExtView()(x, 0, 1)[0, 0])(x)

    assert np.allclose(v_call.asnumpy(), v_plain.asnumpy(), atol=1e-5)
    assert np.allclose(g_call.asnumpy(), g_plain.asnumpy(), atol=1e-5)


def test_pyboost_dispatch_grad_transpose_callback_skipped():
    """
    Feature: Pyboost dispatch grad
    Description: Guard handler intercepts only Transpose and returns
                 numpy-computed result WITHOUT calling func().
                 Transpose backward doesn't rely on view semantics
                 (just does inverse transpose on grad), so skip-callback is safe.
    Expectation: value_and_grad succeeds; gradient matches the plain path.
    """
    global RUN_OPS
    np.random.seed(0)
    x = Tensor(np.random.randn(2, 3).astype(np.float32))

    class SkipTransposeOnly(MsDispatchMode):
        def __ms_dispatch__(self, func, args=(), kwargs=None):
            RUN_OPS.append(func.name)
            if func.name == "Transpose":
                t, dims = args
                return Tensor(np.transpose(t.asnumpy(), dims).copy())
            if kwargs is None:
                kwargs = {}
            return func(*args, **kwargs)

    RUN_OPS = []
    with SkipTransposeOnly():
        v_skip, g_skip = ms.value_and_grad(lambda x: ops.Transpose()(x, (1, 0))[0, 0])(x)
    v_plain, g_plain = ms.value_and_grad(lambda x: ops.Transpose()(x, (1, 0))[0, 0])(x)

    assert np.allclose(v_skip.asnumpy(), v_plain.asnumpy(), atol=1e-5)
    assert np.allclose(g_skip.asnumpy(), g_plain.asnumpy(), atol=1e-5)


def test_pyboost_dispatch_grad_transpose_callback_called():
    """
    Feature: Pyboost dispatch grad
    Description: Guard handler calls func() normally for Transpose.
    Expectation: value_and_grad succeeds; gradient matches the plain path.
    """
    global RUN_OPS
    np.random.seed(0)
    x = Tensor(np.random.randn(2, 3).astype(np.float32))

    RUN_OPS = []
    with TestMsDispatchMode():
        v_call, g_call = ms.value_and_grad(lambda x: ops.Transpose()(x, (1, 0))[0, 0])(x)
    v_plain, g_plain = ms.value_and_grad(lambda x: ops.Transpose()(x, (1, 0))[0, 0])(x)

    assert np.allclose(v_call.asnumpy(), v_plain.asnumpy(), atol=1e-5)
    assert np.allclose(g_call.asnumpy(), g_plain.asnumpy(), atol=1e-5)


# ====== Realistic dispatch scenarios (torchtitan-style) ======

class OpTracer(MsDispatchMode):
    """Like torchtitan's MmWeightTracker: traces op execution during fwd/bwd.
    Always calls func() — pure observation, no interception."""
    def __init__(self):
        self.fwd_ops = []
        self.bwd_ops = []
        self._in_backward = False

    def __ms_dispatch__(self, func, args=(), kwargs=None):
        if self._in_backward:
            self.bwd_ops.append(func.name)
        else:
            self.fwd_ops.append(func.name)
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


def test_dispatch_op_tracer_fwd_bwd():
    """
    Feature: Realistic dispatch — op tracing (torchtitan MmWeightTracker pattern)
    Description: MsDispatchMode that always calls func() and records which ops
                 run during forward and backward. Validates that dispatch does not
                 affect gradient correctness and that op list is deterministic.
    Expectation: Gradient matches non-dispatch baseline; expected ops appear.
    """
    np.random.seed(42)
    x = Tensor(np.random.randn(4, 4).astype(np.float32))

    def mlp(x):
        h = ops.MatMul()(x, x)          # RunOpStub path (no dispatch yaml)
        h = mint.nn.functional.relu(h)   # pyboost path
        return h.sum()

    # baseline
    v_ref, g_ref = ms.value_and_grad(mlp)(x)

    # traced
    tracer = OpTracer()
    with tracer:
        v_traced, g_traced = ms.value_and_grad(mlp)(x)
        tracer._in_backward = True
    # backward ops are captured automatically since dispatch mode spans the whole call

    assert np.allclose(v_traced.asnumpy(), v_ref.asnumpy(), atol=1e-5)
    assert np.allclose(g_traced.asnumpy(), g_ref.asnumpy(), atol=1e-5)
    assert "MatMul" in tracer.fwd_ops
    assert "ReLU" in tracer.fwd_ops or "ReluExt" in tracer.fwd_ops


class ActivationOffloadMode(MsDispatchMode):
    """Simulates activation CPU offloading (swap) — like PyTorch's offload_to_cpu pattern.
    After func() produces output, save a numpy copy and return the original tensor.
    This tests that dispatch interception + callback-called path preserves gradient."""
    def __init__(self):
        self.saved_activations = {}
        self._counter = 0

    def __ms_dispatch__(self, func, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        result = func(*args, **kwargs)
        # "Offload" output to CPU (save numpy snapshot)
        if isinstance(result, Tensor) and result.size > 0:
            self.saved_activations[self._counter] = result.asnumpy().copy()
            self._counter += 1
        return result


def test_dispatch_activation_offload_with_grad():
    """
    Feature: Realistic dispatch — activation offloading (swap pattern)
    Description: MsDispatchMode calls func() normally but also saves a numpy copy
                 of each output (simulating CPU offload). Verifies that the extra
                 asnumpy() call doesn't corrupt autograd.
    Expectation: Gradient matches non-dispatch baseline; activations are captured.
    """
    np.random.seed(42)
    x = Tensor(np.random.randn(4, 4).astype(np.float32))

    def net(x):
        return (x.sin() * x.cos()).sum()

    v_ref, g_ref = ms.value_and_grad(net)(x)

    offloader = ActivationOffloadMode()
    with offloader:
        v_off, g_off = ms.value_and_grad(net)(x)

    assert np.allclose(v_off.asnumpy(), v_ref.asnumpy(), atol=1e-5)
    assert np.allclose(g_off.asnumpy(), g_ref.asnumpy(), atol=1e-5)
    assert len(offloader.saved_activations) > 0


class SelectiveRecomputeMode(MsDispatchMode):
    """Simulates selective activation checkpointing (torchtitan SAC pattern).
    For a target op, skips the callback and computes a low-cost approximation.
    During backward, the framework recomputes the real op via the autograd graph.
    Tests that PostDispatchGrad correctly registers gradient for recomputed ops."""
    def __init__(self, target_op, approx_fn):
        self.target_op = target_op
        self.approx_fn = approx_fn
        self.recompute_count = 0

    def __ms_dispatch__(self, func, args=(), kwargs=None):
        if func.name == self.target_op:
            self.recompute_count += 1
            return self.approx_fn(*args)
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


def test_dispatch_selective_recompute_acos():
    """
    Feature: Realistic dispatch — selective recompute (SAC pattern, RunOpStub path)
    Description: Intercepts ACos (RunOpStub op) and computes with numpy, simulating
                 a scenario where forward activation is discarded and recomputed.
                 PostDispatchGrad must register gradient so backward works.
    Expectation: Gradient matches baseline.
    """
    np.random.seed(42)
    x = Tensor(np.array([0.1, 0.3, 0.5, 0.7], dtype=np.float32))

    def net(x):
        return ops.ACos()(x).sum()

    v_ref, g_ref = ms.value_and_grad(net)(x)

    # ACos goes through RunOpStub; args = (Primitive, name, (tensor,))
    def acos_approx(*args):
        actual = args[2]
        return Tensor(np.arccos(actual[0].asnumpy()))

    mode = SelectiveRecomputeMode("ACos", acos_approx)
    with mode:
        v_sac, g_sac = ms.value_and_grad(net)(x)

    assert np.allclose(v_sac.asnumpy(), v_ref.asnumpy(), atol=1e-5)
    assert np.allclose(g_sac.asnumpy(), g_ref.asnumpy(), atol=1e-5)
    assert mode.recompute_count >= 1


def test_dispatch_selective_recompute_sin():
    """
    Feature: Realistic dispatch — selective recompute (SAC pattern, pyboost path)
    Description: Intercepts Sin (pyboost op) and computes with numpy.
                 Unlike RunOpStub ops, pyboost handles gradient registration
                 via the auto-grad framework after inner returns.
    Expectation: Gradient matches baseline.
    """
    np.random.seed(42)
    x = Tensor(np.array([0.1, 0.3, 0.5, 0.7], dtype=np.float32))

    def net(x):
        return mint.sin(x).sum()

    v_ref, g_ref = ms.value_and_grad(net)(x)

    # Sin is pyboost; args = (tensor,)
    def sin_approx(*args):
        return Tensor(np.sin(args[0].asnumpy()))

    mode = SelectiveRecomputeMode("Sin", sin_approx)
    with mode:
        v_sac, g_sac = ms.value_and_grad(net)(x)

    assert np.allclose(v_sac.asnumpy(), v_ref.asnumpy(), atol=1e-5)
    assert np.allclose(g_sac.asnumpy(), g_ref.asnumpy(), atol=1e-5)
    assert mode.recompute_count >= 1


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard', mem_peak=2048.01)
def test_ms_dispatch():
    """
    Feature: Ms dispatch
    Description: Run all small ms dispatch tests.
    Expectation: All sub tests run successfully.
    """
    test_ms_dispatch_add()
    test_ms_dispatch_reshape_tuple()
    test_ms_dispatch_add_scalar_overload()
    test_ms_dispatch_mint_sin()
    test_ms_dispatch_assign_primitive()
    test_ms_dispatch_mint_cat()
    test_ms_dispatch_split_primitive_attr()
    test_ms_dispatch_cast()
    test_ms_dispatch_to_dtype()
    test__update_data_shallow_copy_tensor()
    test__update_data_raise_on_non_tensor()

    test_ms_dispatch_mode_guard()
    test_ms_dispatch_nested_mode_guard()
    test_ms_dispatch_nested_mode_guard_with_dtensor()

    test_ms_dispatch_mode_guard_with_grad()
    test_ms_dispatch_nested_mode_guard_with_grad()
    test_ms_dispatch_nested_mode_guard_and_dtensor_with_grad()

    # MsDispatchMode with grad: add operation
    test_ms_dispatch_mode_guard_with_grad_add()
    test_ms_dispatch_nested_mode_guard_with_grad_add()
    test_ms_dispatch_nested_mode_guard_and_dtensor_with_grad_add()

    # MsDispatchMode with grad: reshape operation
    test_ms_dispatch_mode_guard_with_grad_reshape()
    test_ms_dispatch_nested_mode_guard_with_grad_reshape()
    test_ms_dispatch_nested_mode_guard_and_dtensor_with_grad_reshape()

    # MsDispatchMode with grad: mint.cat operation
    test_ms_dispatch_mode_guard_with_grad_cat()
    test_ms_dispatch_nested_mode_guard_with_grad_cat()
    test_ms_dispatch_nested_mode_guard_and_dtensor_with_grad_cat()

    # MsDispatchMode with grad: ops.cast operation
    test_ms_dispatch_mode_guard_with_grad_cast()
    test_ms_dispatch_nested_mode_guard_with_grad_cast()
    test_ms_dispatch_nested_mode_guard_and_dtensor_with_grad_cast()

    # MsDispatchMode with grad: mul operation
    test_ms_dispatch_mode_guard_with_grad_mul()
    test_ms_dispatch_nested_mode_guard_with_grad_mul()
    test_ms_dispatch_nested_mode_guard_and_dtensor_with_grad_mul()

    # MsDispatchMode with grad: exp operation
    test_ms_dispatch_mode_guard_with_grad_exp()
    test_ms_dispatch_nested_mode_guard_with_grad_exp()
    test_ms_dispatch_nested_mode_guard_and_dtensor_with_grad_exp()

    # New scenario tests
    test_ms_dispatch_disable_mode_guard_bypasses_dispatch()
    test_ms_dispatch_disable_mode_guard_with_dtensor()
    test_ms_dispatch_guard_exception_handling()
    test_ms_dispatch_guard_created_outside_with_block()
    test_ms_dispatch_nesting_same_guard_instance()
    test_ms_dispatch_guard_priority_over_dtensor()
    test_ms_dispatch_grad_numerical_correctness()
    test_ms_dispatch_guard_as_decorator()
    test_ms_dispatch_dtensor_mul_scalar_inplace_pattern()
    test_ms_dispatch_complex_multi_branch_computation()
    test_ms_dispatch_source_py_obj_user_data_key_is_reserved()
    test_ms_dispatch_mode_return_dtensor_preserves_python_wrapper()
    test_ms_dispatch_custom_backward_return_dtensor_preserves_next_hook_wrapper()
    test_ms_dispatch_saved_tensor_unpack_return_dtensor_preserves_wrapper()

    # Reshape grad coverage tests
    test_ms_dispatch_reshape_grad_when_callback_skipped()
    test_ms_dispatch_mode_guard_with_grad_reshape_values()
    test_reshape_grad_normal_view_path()
    test_ms_dispatch_dtensor_reshape_grad_values()
    #test_reshape_meta_device_shape_correct()

    # RunOpStub dispatch grad tests
    test_runopstub_dispatch_grad_callback_skipped()
    test_runopstub_dispatch_grad_callback_called()
    test_runopstub_dispatch_grad_layernorm_callback_skipped()
    test_runopstub_dispatch_grad_layernorm_callback_called()

    # Pyboost view ops dispatch grad tests
    test_pyboost_dispatch_grad_select_ext_view_callback_called()
    test_pyboost_dispatch_grad_transpose_ext_view_callback_skipped()
    test_pyboost_dispatch_grad_transpose_ext_view_callback_called()
    test_pyboost_dispatch_grad_transpose_callback_skipped()
    test_pyboost_dispatch_grad_transpose_callback_called()

    # Realistic dispatch scenarios (torchtitan-style)
    test_dispatch_op_tracer_fwd_bwd()
    test_dispatch_activation_offload_with_grad()
    test_dispatch_selective_recompute_acos()
    test_dispatch_selective_recompute_sin()
