# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
""" test_auto_grad """
# pylint: disable=protected-access

import numpy as np
import pytest
import mindspore as ms
from mindspore import _Function, recompute
from mindspore import Tensor, jit, _no_grad
from mindspore.common.recompute import _clear_recompute_session, _recompute_session_ctx, _recompute_handle_collector_ctx
from mindspore.graph.api import _pynative_executor, _GradientEdge
from mindspore.ops import stop_gradient
from mindspore._c_expression import run_backward, pyboost_detach
from tests.mark_utils import arg_mark

_pynative_executor.set_grad_flag(True)


@property
def requires_grad(self):
    """
    Return whether the parameter requires gradient.
    """
    return self._requires_grad


@requires_grad.setter
def requires_grad(self, value=True):
    if not isinstance(value, bool):
        raise TypeError("The argument `requires_grad` must be bool type")
    self._requires_grad = value


@property
def grad(self):
    """
    Return whether the parameter requires gradient.
    """
    return self._grad


@grad.setter
def grad(self, value):
    self._grad = value


@property
def is_leaf(self):
    """
    Return whether the parameter requires gradient.
    """
    return self._is_leaf


@property
def retains_grad(self):
    """
    Return whether the tensor retains gradient.
    """
    return self._retains_grad


@property
def grad_fn(self):
    return self._grad_node


@property
def output_nr(self):
    return self._output_index


def retain_grad(self):
    """
    set the tensor retains gradient.
    """
    return self._retain_grad()


def detach(self):
    """
    detach the tensor.
    """
    return pyboost_detach(self)


def backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None):
    if isinstance(inputs, list):
        inputs = tuple(inputs)
    if isinstance(gradient, list):
        gradient = tuple(gradient)
    return run_backward((self,), gradient, retain_graph, create_graph, inputs, allow_unreachable=True,
                        accumulate_grad=True)


def run_grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, allow_unreachable=True):
    if isinstance(inputs, list):
        inputs = tuple(inputs)
    if isinstance(grad_outputs, list):
        grad_outputs = tuple(grad_outputs)
    return run_backward(outputs, grad_outputs, retain_graph, create_graph, inputs, allow_unreachable=allow_unreachable,
                        accumulate_grad=False)


Tensor.requires_grad = requires_grad
Tensor.grad = grad
Tensor.backward = backward
Tensor.is_leaf = is_leaf
Tensor.retains_grad = retains_grad
Tensor.retain_grad = retain_grad
Tensor.grad_fn = grad_fn
Tensor.output_nr = output_nr
Tensor.detach = detach


def test_backward_normal():
    """
    Feature: Test backward normal
    Description: Test backward api
    Expectation: Success.
    """
    x = Tensor([3], ms.float32)
    x.requires_grad = True
    y = Tensor([5], ms.float32)
    y.requires_grad = True
    z = x * y
    z.backward()
    assert np.allclose(x.grad.asnumpy(), np.array([5], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(y.grad.asnumpy(), np.array([3], dtype=np.float32), 0.00001, 0.00001)


def test_backward_normal2():
    """
    Feature: Test backward api
    Description: Test backward api requires_grad
    Expectation: Success.
    """
    x = Tensor([3], ms.float32)
    x.requires_grad_()
    y = Tensor([5], ms.float32)
    y.requires_grad_()
    z = x * y
    h = stop_gradient(z)
    h.requires_grad_()
    z = h * h
    z += h
    z.backward()
    assert np.allclose(h.grad.asnumpy(), np.array([31], dtype=np.float32), 0.00001, 0.00001)
    assert x.grad is None
    assert y.grad is None


@jit
def jit_net(x):
    y = x * x
    z = y + y
    return z


def test_backward_with_jit():
    """
    Feature: Test backward api
    Description: Test backward api requires_grad
    Expectation: Success.
    """
    x = Tensor([3], ms.float32)
    x.requires_grad_()
    y = Tensor([5], ms.float32)
    y.requires_grad_()
    z = x * y
    h = jit_net(z)
    z = h * h
    z += h
    z.backward()
    assert np.allclose(x.grad.asnumpy(), np.array([270300], dtype=np.float32), 0.00001, 0.00001)


def test_backward_accumulate():
    """
    Feature: Test backward api
    Description: Test backward api accumulate
    Expectation: Success.
    """
    x = Tensor([3], ms.float32)
    x.requires_grad_()
    y = Tensor([5], ms.float32)
    y.requires_grad_()
    z = x * y
    z.backward()
    h = x + y
    h.backward()
    assert np.allclose(x.grad.asnumpy(), np.array([6], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(y.grad.asnumpy(), np.array([4], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard')
def test_backward_constant_tensor():
    """
    Feature: Test backward api
    Description: Test backward api accumulate
    Expectation: Success.
    """
    x = Tensor([3], ms.float32)
    x.requires_grad_()
    y = Tensor([5], ms.float32)
    y.requires_grad_()
    with _no_grad():
        z = x * y
    with pytest.raises(RuntimeError) as err:
        z.backward()
    assert "The output tensor you provided doesn't requires grad" in str(err.value)


def test_backward_get_grad():
    """
    Feature: Test backward api
    Description: Test backward api accumulate
    Expectation: Success.
    """
    x = Tensor([3], ms.float32)
    assert x.grad is None
    x.requires_grad_()
    assert x.grad is None
    y = Tensor([5], ms.float32)
    y.requires_grad_()
    z = x * y
    z.backward()
    assert np.allclose(x.grad.asnumpy(), np.array([5.], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(y.grad.asnumpy(), np.array([3.], dtype=np.float32), 0.00001, 0.00001)


def test_backward_set_grad_none():
    """
    Feature: Test backward api
    Description: Test backward api set grad none
    Expectation: Success.
    """
    x = Tensor([3], ms.float32)
    x.grad = None
    x.requires_grad_()
    y = Tensor([5], ms.float32)
    y.requires_grad_()
    z = x * y
    z.backward()
    x.grad = None
    y.grad = None
    h = x * y
    h.backward()
    assert np.allclose(x.grad.asnumpy(), np.array([5], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(y.grad.asnumpy(), np.array([3], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard')
def test_backward_set_grad_err_shape():
    """
    Feature: Test backward api
    Description: Test backward api set grad none
    Expectation: Success.
    """
    x = Tensor([3], ms.float32)
    x.requires_grad_()
    y = Tensor([5], ms.float32)
    y.requires_grad_()
    z = x * y
    z.backward()
    with pytest.raises(RuntimeError) as err:
        x.grad = Tensor([3], ms.int64)
    assert "The grad dtype and shape should be same" in str(err.value)
    with pytest.raises(RuntimeError) as err:
        y.grad = Tensor([3, 6], ms.float32)
    assert "The grad dtype and shape should be same" in str(err.value)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard')
def test_backward_twice_exception():
    """
    Feature: Test backward api
    Description: Test backward api set grad none
    Expectation: Success.
    """
    x = Tensor([3], ms.float32)
    x.requires_grad_()
    y = Tensor([5], ms.float32)
    y.requires_grad_()
    z = x * y
    z.backward()
    with pytest.raises(RuntimeError) as err:
        z.backward()
    assert "Try to backward the graph twice" in str(err.value)


class FunctionNet(_Function):
    @staticmethod
    def forward(ctx, x):
        t = x * x
        z = t + t
        return z

    @staticmethod
    def backward(ctx, z):
        return z * 3


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard')
def test_backward_twice_custom_function_exception():
    """
    Feature: Test backward api
    Description: Test backward api set grad none
    Expectation: Success.
    """
    x = Tensor([3], ms.float32)
    x.requires_grad_()
    y = Tensor([5], ms.float32)
    y.requires_grad_()
    z = x * y
    z = FunctionNet.apply(z)
    z.backward()
    with pytest.raises(RuntimeError) as err:
        z.backward()
    assert "Try to backward the graph twice" in str(err.value)


def test_backward_twice():
    """
    Feature: Test backward api
    Description: Test backward api set grad none
    Expectation: Success.
    """
    x = Tensor([3], ms.float32)
    x.requires_grad_()
    y = Tensor([5], ms.float32)
    y.requires_grad_()
    z = x * y
    z.backward(retain_graph=True)
    z.backward()
    assert np.allclose(x.grad.asnumpy(), np.array([10], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(y.grad.asnumpy(), np.array([6], dtype=np.float32), 0.00001, 0.00001)


def test_backward_high_order():
    """
    Feature: Test backward api
    Description: Test backward api high order
    Expectation: Success.
    """
    x = Tensor([3], ms.float32)
    x.requires_grad_()
    z = x * x
    z.backward(create_graph=True)
    x.grad.backward()
    assert np.allclose(x.grad.asnumpy(), np.array([8.], dtype=np.float32), 0.00001, 0.00001)


def test_backward_register_hook():
    """
    Feature: Test backward api
    Description: Test backward api register hook
    Expectation: Success.
    """
    x = Tensor([3], ms.float32)
    x.requires_grad_()
    x.register_hook(lambda grad_out: grad_out * 2)
    z = x * x
    z.backward()
    assert np.allclose(x.grad.asnumpy(), np.array([12.], dtype=np.float32), 0.00001, 0.00001)


def test_backward_set_requires_grad_register_hook():
    """
    Feature: Test backward api
    Description: Test backward api set requires grad false and register hook
    Expectation: Success.
    """
    x = Tensor([3], ms.float32)
    x.requires_grad_()
    x.register_hook(lambda grad_out: grad_out * 2)
    x.requires_grad = False
    x.requires_grad = True
    z = x * x
    z.backward()
    assert np.allclose(x.grad.asnumpy(), np.array([12.], dtype=np.float32), 0.00001, 0.00001)


def test_backward_is_leaf():
    """
    Feature: Test backward api
    Description: Test backward api is leaf
    Expectation: Success.
    """
    x = Tensor([3], ms.float32)
    assert x.is_leaf is True
    x.requires_grad_()
    z = x * x
    assert z.is_leaf is False


def test_backward_retain_grad():
    """
    Feature: Test backward api
    Description: Test backward api retain_grad
    Expectation: Success.
    """
    x = Tensor([2.0, 1.0], ms.float32)
    x.requires_grad = True
    assert not x.retains_grad
    y = x * x
    y.retain_grad()
    y.sum().backward(retain_graph=True)
    assert y.retains_grad
    assert np.allclose(y.grad.asnumpy(), np.array([1.0, 1.0], dtype=np.float32), 0.00001, 0.00001)

    z = y * y
    z.sum().backward()
    assert np.allclose(y.grad.asnumpy(), np.array([9.0, 3.0], dtype=np.float32), 0.00001, 0.00001)


def test_backward_retain_grad_inplace():
    """
    Feature: Test backward api
    Description: Test backward api retain_grad inplace
    Expectation: Success.
    """
    x = Tensor([2.0, 1.0], ms.float32)
    x.requires_grad = True
    y = x + 1.0
    y.retain_grad()
    assert y.retains_grad
    y *= 2.0
    assert y.retains_grad
    z = y * y
    z.sum().backward()
    assert np.allclose(y.grad.asnumpy(), np.array([12.0, 8.0], dtype=np.float32), 0.00001, 0.00001)


def test_backward_retain_grad_view_inplace():
    """
    Feature: Test backward api
    Description: Test backward api retain_grad view inplace
    Expectation: Success.
    """
    x = Tensor([2.0, 1.0], ms.float32)
    x.requires_grad = True
    y = x * x
    y.retain_grad()
    assert y.retains_grad
    y[0] *= 2.0
    assert y.retains_grad
    z = y * 2.0
    z.sum().backward()
    assert np.allclose(y.grad.asnumpy(), np.array([2.0, 2.0], dtype=np.float32), 0.00001, 0.00001)


def test_backward_version():
    """
    Feature: Test backward api
    Description: Test backward api version
    Expectation: Success.
    """
    x = Tensor([2.0, 1.0], ms.float32)
    assert not x._version
    x_view = x[0]
    x += 2.0
    assert x._version == 1
    assert x_view._version == 1
    x_view += 1.0
    assert x._version == 2


def test_backward_output_nr():
    """
    Feature: Test backward api
    Description: Test backward api output_nr
    Expectation: Success.
    """
    x = Tensor([2.0, 1.0], ms.float32)
    assert not x.output_nr
    output = ms.mint.split(x, 1, 0)
    assert not output[1].output_nr

    x.requires_grad = True
    output = ms.mint.split(x, 1, 0)
    assert output[1].output_nr == 1


def test_backward_with_recompute_non_use_reentrant():
    """
    Feature: Test backward api
    Description: Test backward api with non-reentrant recompute
    Expectation: Success.
    """
    def fn(x, y):
        z1 = x + 2.0
        z2 = y + 1.0
        return z1 * z2

    x = Tensor([2.0, 1.0], ms.float32)
    y = Tensor([1.0, 3.0], ms.float32)
    x.requires_grad = True
    y.requires_grad = True
    res = recompute(fn, x, y, use_reentrant=False)
    res.sum().backward()
    assert np.allclose(x.grad.asnumpy(), np.array([2.0, 4.0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(y.grad.asnumpy(), np.array([4.0, 3.0], dtype=np.float32), 0.00001, 0.00001)


def test_backward_recompute_auto_session_id():
    """
    Feature: Test backward api with non-reentrant recompute session context
    Description: Test auto-generated recompute session id can retain and then consume recomputed tensors.
    Expectation: The second backward reuses retained recomputed tensors without rerunning recompute.
    """
    class Net:
        def __init__(self):
            self.call_count = 0

        def __call__(self, x):
            self.call_count += 1
            y = ms.ops.sin(x)
            return y * y

    x_np = np.array([[0.2, -0.4], [0.7, -0.9]], np.float32)
    x = Tensor(x_np)
    x.requires_grad = True
    net = Net()
    out = recompute(net, x, use_reentrant=False, early_stop=False)
    sens = ms.ops.ones_like(out)

    with _recompute_session_ctx(retain_on_unpack=True) as session_id:
        assert session_id is not None
        first_grad = run_backward((out,), (sens,), True, False, (x,),
                                  allow_unreachable=True, accumulate_grad=False)

    assert net.call_count == 2

    try:
        with _recompute_session_ctx(session_id, retain_on_unpack=False):
            second_grad = run_backward((out,), (sens,), False, False, (x,),
                                       allow_unreachable=True, accumulate_grad=False)
    finally:
        _clear_recompute_session(session_id)

    expected = 2 * np.sin(x_np) * np.cos(x_np)
    assert net.call_count == 2
    assert np.allclose(first_grad[0].asnumpy(), expected, 0.00001, 0.00001)
    assert np.allclose(second_grad[0].asnumpy(), expected, 0.00001, 0.00001)


def test_backward_recompute_handle_collector_ctx():
    """
    Feature: Test backward api with non-reentrant recompute handle collector context
    Description: Test users can collect recompute handles from a forward range and run recompute manually.
    Expectation: The collected handle can recompute directly; later recompute calls are not appended to that list.
    """
    class Net:
        def __init__(self):
            self.call_count = 0

        def __call__(self, x):
            self.call_count += 1
            return ms.ops.sin(x) * ms.ops.cos(x)

    x_np = np.array([0.2, -0.4], np.float32)
    x = Tensor(x_np)
    x.requires_grad = True
    net = Net()

    with _recompute_handle_collector_ctx() as handles:
        out = recompute(net, x, use_reentrant=False, early_stop=False)
        assert len(handles) == 1

    assert len(handles) == 1
    assert hasattr(handles[0], "recompute")

    session_id = None
    try:
        with _recompute_session_ctx(retain_on_unpack=False) as session_id:
            handles[0].recompute(session_id)
            sens = ms.ops.ones_like(out)
            manual_grad = run_backward((out,), (sens,), False, False, (x,),
                                       allow_unreachable=True, accumulate_grad=False)
    finally:
        _clear_recompute_session(session_id)

    expected = np.cos(x_np) * np.cos(x_np) - np.sin(x_np) * np.sin(x_np)
    assert net.call_count == 2
    assert np.allclose(manual_grad[0].asnumpy(), expected, 0.00001, 0.00001)

    _ = recompute(net, x, use_reentrant=False, early_stop=False)
    assert len(handles) == 1


class ReturnSelfOps(_Function):
    @staticmethod
    def forward(ctx, *args):
        return args

    @staticmethod
    def backward(ctx, *grads):
        return grads


def test_backward_grad_fn():
    """
    Feature: Test backward api
    Description: Test backward api grad_fn
    Expectation: Success.
    """
    x = Tensor([2.0, 1.0], ms.float32)
    y = Tensor([1.0, 3.0], ms.float32)
    assert x.grad_fn is None

    x.requires_grad = True
    y.requires_grad = True
    assert x.grad_fn is not None
    assert y.grad_fn is not None

    x1, y1 = ReturnSelfOps.apply(x, y)
    assert x1.grad_fn is not None
    assert x1.grad_fn.name() == "ReturnSelfOps"

    z = y1 * y
    assert z.grad_fn is not None

    z_next_edges = z.grad_fn.next_functions
    assert z_next_edges[0][0] is not None
    assert z_next_edges[0][1] == 1
    assert z_next_edges[1][0] is not None
    assert z_next_edges[1][1] == 0


def test_backward_grad_fn_view_inplace():
    """
    Feature: Test backward api
    Description: Test backward api grad_fn, when view + inplace
    Expectation: Success.
    """
    x = Tensor([2.0, 1.0], ms.float32)
    y = Tensor([1.0, 3.0], ms.float32)
    y.requires_grad = True
    x[0] = y[1]
    assert x.grad_fn is not None
    assert x.grad_fn.name() == "CopySlice"

    x = Tensor([2.0, 1.0], ms.float32)
    y = Tensor([1.0, 3.0], ms.float32)
    y.requires_grad = True
    x_view = x[1]
    x[0] = y[1]
    assert x_view.grad_fn is not None
    assert x_view.grad_fn.name() == "AsStrided"


def test_backward_grad_fn_register_pre_hook():
    """
    Feature: Test backward api
    Description: Test backward api grad_fn register_pre_hook
    Expectation: Success.
    """
    counter = [0]

    def pre_hook1(grads):
        counter[0] += 1

    def pre_hook2(grads):
        counter[0] += 1
        return tuple(grad * 2 for grad in grads)

    def pre_hook3(grads):
        counter[0] += 1
        return tuple(grad + 1.0 for grad in grads)

    x = Tensor([2.0, 1.0], ms.float32)
    y = Tensor([1.0, 3.0], ms.float32)
    x.requires_grad = True
    y.requires_grad = True

    z = x * y
    z_grad_fn = z.grad_fn
    handle1 = z_grad_fn.register_prehook(pre_hook1)
    handle2 = z_grad_fn.register_prehook(pre_hook2)
    z_grad_fn.register_prehook(pre_hook3)
    z.sum().backward(retain_graph=True)
    assert counter[0] == 3
    assert np.allclose(x.grad.asnumpy(), np.array([3.0, 9.0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(y.grad.asnumpy(), np.array([6.0, 3.0], dtype=np.float32), 0.00001, 0.00001)

    handle1.remove()
    handle2.remove()
    z.sum().backward()
    assert counter[0] == 4
    assert np.allclose(x.grad.asnumpy(), np.array([5.0, 15.0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(y.grad.asnumpy(), np.array([10.0, 5.0], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard')
def test_backward_grad_fn_register_pre_hook_return_error():
    """
    Feature: Test backward api
    Description: Test backward api grad_fn register_pre_hook, hook return incorrect format
    Expectation: Raise Runtime Error.
    """

    def pre_hook1(grads):
        return grads[0]

    def pre_hook2(grads):
        return grads * 2

    x = Tensor([3.0, 2.0], ms.float32)
    y = Tensor([1.0, 2.0], ms.float32)
    x.requires_grad = True
    y.requires_grad = True

    z = x * y
    z.grad_fn.register_prehook(pre_hook1)
    with pytest.raises(RuntimeError) as err:
        z.backward()
    assert "hook pre_hook1 should return a tuple of grad." in str(err.value)

    z = x + y
    z.grad_fn.register_prehook(pre_hook2)
    with pytest.raises(RuntimeError) as err:
        z.backward()
    assert "hook pre_hook2 returned incorrect length 2, expected 1." in str(err.value)


def test_backward_grad_fn_register_post_hook():
    """
    Feature: Test backward api
    Description: Test backward api grad_fn register_post_hook
    Expectation: Success.
    """

    counter = [0]

    def post_hook1(grad_inputs, grad_outputs):
        counter[0] += 1
        assert len(grad_outputs) == 1

    def post_hook2(grad_inputs, grad_outputs):
        counter[0] += 1
        return tuple(grad + 1.0 for grad in grad_inputs)

    def post_hook3(grad_inputs, grad_outputs):
        counter[0] += 1
        return tuple(grad * 2.0 for grad in grad_inputs)

    x = Tensor([2.0, 1.0], ms.float32)
    y = Tensor([1.0, 3.0], ms.float32)
    x.requires_grad = True
    y.requires_grad = True

    z = x + y
    out = z * x
    out_grad_fn = out.grad_fn
    out_grad_fn.register_hook(post_hook1)
    handle2 = out_grad_fn.register_hook(post_hook2)
    out_grad_fn.register_hook(post_hook3)

    out.sum().backward(retain_graph=True)
    assert counter[0] == 3
    assert np.allclose(x.grad.asnumpy(), np.array([14.0, 14.0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(y.grad.asnumpy(), np.array([6.0, 4.0], dtype=np.float32), 0.00001, 0.00001)

    handle2.remove()
    out.sum().backward()
    assert counter[0] == 5
    assert np.allclose(x.grad.asnumpy(), np.array([24.0, 24.0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(y.grad.asnumpy(), np.array([10.0, 6.0], dtype=np.float32), 0.00001, 0.00001)


def test_backward_grad_fn_register_post_hook_leaf():
    """
    Feature: Test backward api
    Description: Test backward api grad_fn register post hook on leaf node
    Expectation: Success.
    """

    x = Tensor([2.0, 1.0], ms.float32)
    y = Tensor([1.0, 3.0], ms.float32)
    x.requires_grad = True
    y.requires_grad = True

    def make_param_post_hook(param):
        def post_hook(_, grad_outputs):
            param.grad += grad_outputs[0]

        return post_hook

    x.grad_fn.register_hook(make_param_post_hook(x))
    y.grad_fn.register_hook(make_param_post_hook(y))

    out = x * y
    out.sum().backward()
    assert np.allclose(x.grad.asnumpy(), np.array([2.0, 6.0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(y.grad.asnumpy(), np.array([4.0, 2.0], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard')
def test_backward_cpu_cases_suit():
    """
    Feature: Aggregate backward precision validations.
    Description: Execute all cpu_linux normal accuracy cases in sequence.
    Expectation: Success.
    """
    test_backward_normal()
    test_backward_normal2()
    test_backward_with_jit()
    test_backward_accumulate()
    test_backward_get_grad()
    test_backward_set_grad_none()
    test_backward_twice()
    test_backward_high_order()
    test_backward_register_hook()
    test_backward_set_requires_grad_register_hook()
    test_backward_is_leaf()
    test_backward_output_nr()
    test_backward_with_recompute_non_use_reentrant()
    test_backward_grad_fn_register_pre_hook()
    test_backward_grad_fn_register_post_hook_leaf()
    test_backward_with_gradient_edge_outputs()
    test_backward_with_gradient_edge_inputs()
    test_backward_with_gradient_edge_both()
    test_backward_grad_fn_input_metadata()
    test_backward_grad_fn_input_metadata_complex()
    test_backward_grad_fn_input_metadata_leaf()


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          mem_peak=2.00)
def test_backward_ascend_cases_suit():
    """
    Feature: Aggregate backward precision validations for platform_ascend.
    Description: Execute all platform_ascend normal accuracy cases in sequence.
    Expectation: Success.
    """
    test_backward_retain_grad()
    test_backward_retain_grad_inplace()
    test_backward_retain_grad_view_inplace()
    test_backward_version()
    test_backward_grad_fn()
    test_backward_grad_fn_view_inplace()
    test_backward_grad_fn_register_post_hook()
    test_backward_with_inputs()
    test_backward_detach()


def test_backward_with_inputs():
    """
    Feature: Test backward api
    Description: Test backward api with inputs
    Expectation: Success.
    """
    x = Tensor([2.0, 1.0], ms.float32)
    x.requires_grad = True
    y = x * x
    y *= 2.0
    z = y * 2.0
    z.sum().backward(inputs=[y])
    assert x.grad is None
    assert np.allclose(y.grad.asnumpy(), np.array([2.0, 2.0], dtype=np.float32), 0.00001, 0.00001)
    with pytest.raises(RuntimeError) as err:
        y.sum().backward()
    assert "Try to backward the graph twice" in str(err.value)


def test_backward_detach():
    """
    Feature: Test backward api
    Description: Test backward api detach
    Expectation: Success.
    """
    x = Tensor([2.0, 1.0], ms.float32).to("Ascend")
    x.requires_grad = True
    x_d = x.detach()

    assert not x_d.requires_grad
    x_d += 1.0
    # .to version is not same as pt.
    assert x._version == 2

    y = x + 2.0
    z = y.detach() + x * x
    z.sum().backward()
    assert np.allclose(x.grad.asnumpy(), np.array([6.0, 4.0], dtype=np.float32), 0.00001, 0.00001)


def test_backward_with_gradient_edge_outputs():
    """
    Feature: Test backward api with _GradientEdge as outputs
    Description: Test backward api using _GradientEdge instead of direct tensor, compare with direct tensor
    Expectation: Success.
    """
    x1 = Tensor([3.0], ms.float32)
    x1.requires_grad = True
    y1 = Tensor([5.0], ms.float32)
    y1.requires_grad = True
    z1 = x1 * y1
    grads = run_grad((z1,), (x1, y1))
    x_grad_direct = grads[0].asnumpy()
    y_grad_direct = grads[1].asnumpy()

    x2 = Tensor([3.0], ms.float32)
    x2.requires_grad = True
    y2 = Tensor([5.0], ms.float32)
    y2.requires_grad = True
    z2 = x2 * y2
    grad_edge = _GradientEdge(grad_node=z2.grad_fn, output_index=z2.output_nr, keep_alive_token=None)
    new_grads = run_grad((grad_edge,), (x2, y2))
    x_grad_edge = new_grads[0].asnumpy()
    y_grad_edge = new_grads[1].asnumpy()

    assert np.allclose(x_grad_direct, x_grad_edge, 0.00001, 0.00001)
    assert np.allclose(y_grad_direct, y_grad_edge, 0.00001, 0.00001)


def test_backward_with_gradient_edge_inputs():
    """
    Feature: Test backward api with _GradientEdge as inputs
    Description: Test backward api using _GradientEdge for inputs parameter
    Expectation: Success.
    """
    x1 = Tensor([2.0, 1.0], ms.float32)
    x1.requires_grad = True
    y1 = x1 * x1
    y1 *= 2.0
    z1 = y1 * 2.0

    grads = run_grad((z1,), (y1,))
    y_grad_direct = grads[0].asnumpy()

    x2 = Tensor([2.0, 1.0], ms.float32)
    x2.requires_grad = True
    y2 = x2 * x2
    y2 *= 2.0
    z2 = y2 * 2.0

    grad_edge_input = _GradientEdge(grad_node=y2.grad_fn, output_index=y2.output_nr, keep_alive_token=None)
    new_grads = run_grad((z2,), (grad_edge_input,))
    y_grad_edge = new_grads[0].asnumpy()

    assert np.allclose(y_grad_direct, y_grad_edge, 0.00001, 0.00001)


def test_backward_with_gradient_edge_both():
    """
    Feature: Test backward api with _GradientEdge as both outputs and inputs
    Description: Test backward api using _GradientEdge for both outputs and inputs
    Expectation: Success.
    """
    x1 = Tensor([3.0, 2.0], ms.float32)
    x1.requires_grad = True
    y1 = Tensor([1.0, 2.0], ms.float32)
    y1.requires_grad = True
    z1 = x1 * y1
    w1 = z1 * z1

    grads = run_grad((w1,), (x1, y1))
    x_grad_direct = grads[0].asnumpy()
    y_grad_direct = grads[1].asnumpy()

    x2 = Tensor([3.0, 2.0], ms.float32)
    x2.requires_grad = True
    y2 = Tensor([1.0, 2.0], ms.float32)
    y2.requires_grad = True
    z2 = x2 * y2
    w2 = z2 * z2

    grad_edge_output = _GradientEdge(grad_node=w2.grad_fn, output_index=w2.output_nr, keep_alive_token=None)
    grad_edge_input_x = _GradientEdge(grad_node=x2.grad_fn, output_index=x2.output_nr, keep_alive_token=None)
    grad_edge_input_y = _GradientEdge(grad_node=y2.grad_fn, output_index=y2.output_nr, keep_alive_token=None)
    new_grads = run_grad((grad_edge_output,), (grad_edge_input_x, grad_edge_input_y))
    x_grad_edge = new_grads[0].asnumpy()
    y_grad_edge = new_grads[1].asnumpy()

    assert np.allclose(x_grad_direct, x_grad_edge, 0.00001, 0.00001)
    assert np.allclose(y_grad_direct, y_grad_edge, 0.00001, 0.00001)


def test_backward_grad_fn_input_metadata():
    """
    Feature: Test grad_fn _input_metadata attribute
    Description: Test that _input_metadata can be read correctly from grad_fn
    Expectation: Success.
    """
    x = Tensor([2.0, 1.0], ms.float32)
    x.requires_grad = True
    y = Tensor([1.0, 3.0], ms.float32)
    y.requires_grad = True

    z = x * y
    z_grad_fn = z.grad_fn

    assert hasattr(z_grad_fn, '_input_metadata')
    input_metadata = z_grad_fn._input_metadata

    assert isinstance(input_metadata, tuple)
    assert len(input_metadata) == 1

    for meta in input_metadata:
        assert meta is not None
        assert hasattr(meta, 'shape')
        assert hasattr(meta, 'dtype')
        assert hasattr(meta, 'strides')
        assert hasattr(meta, 'storage_offset')

    assert len(input_metadata[0].shape) == 1
    assert input_metadata[0].shape[0] == 2


def test_backward_grad_fn_input_metadata_complex():
    """
    Feature: Test grad_fn _input_metadata attribute with complex operations
    Description: Test _input_metadata with multiple outputs using mint.split
    Expectation: Success.
    """
    x = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], ms.float32)
    x.requires_grad = True

    z = x * x
    outputs = ms.mint.split(z, 1, 0)

    assert len(outputs) == 3
    assert outputs[0].grad_fn is not None
    assert outputs[1].grad_fn is not None
    assert outputs[2].grad_fn is not None

    out0_grad_fn = outputs[0].grad_fn
    assert hasattr(out0_grad_fn, '_input_metadata')
    input_metadata = out0_grad_fn._input_metadata

    assert isinstance(input_metadata, tuple)
    assert len(input_metadata) == 3

    for meta in input_metadata:
        assert meta is not None
        assert hasattr(meta, 'shape')
        assert hasattr(meta, 'dtype')
        assert hasattr(meta, 'strides')
        assert hasattr(meta, 'storage_offset')
        assert len(meta.shape) == 2
        assert meta.shape[0] == 1
        assert meta.shape[1] == 2

    out1_grad_fn = outputs[1].grad_fn
    assert hasattr(out1_grad_fn, '_input_metadata')
    out1_input_metadata = out1_grad_fn._input_metadata
    assert isinstance(out1_input_metadata, tuple)
    assert len(out1_input_metadata) == 3

    out2_grad_fn = outputs[2].grad_fn
    assert hasattr(out2_grad_fn, '_input_metadata')
    out2_input_metadata = out2_grad_fn._input_metadata
    assert isinstance(out2_input_metadata, tuple)
    assert len(out2_input_metadata) == 3


def test_backward_grad_fn_input_metadata_leaf():
    """
    Feature: Test grad_fn _input_metadata attribute for leaf nodes
    Description: Test _input_metadata for leaf nodes (tensors with requires_grad=True)
    Expectation: Success.
    """
    x = Tensor([2.0, 1.0], ms.float32)
    x.requires_grad = True

    assert x.grad_fn is not None
    x_grad_fn = x.grad_fn

    assert hasattr(x_grad_fn, '_input_metadata')
    input_metadata = x_grad_fn._input_metadata

    assert isinstance(input_metadata, tuple)

    y = x * x
    y_grad_fn = y.grad_fn
    assert hasattr(y_grad_fn, '_input_metadata')
    y_input_metadata = y_grad_fn._input_metadata

    assert isinstance(y_input_metadata, tuple)
    assert len(y_input_metadata) == 1

    for meta in y_input_metadata:
        assert meta is not None
        assert hasattr(meta, 'shape')
        assert hasattr(meta, 'dtype')


def test_nested_backward_with_unpaired_cell_hook():
    """
    Feature: Test nested backward with unpaired cell backward hook isolation by engine_id
    Description: The outer network registers a backward hook. The input x does not require grad,
                 so the outer network's input-side hook never fires, making the hook unpaired.
                 A sub-cell with custom bprop triggers inner GradOperation during outer backward.
                 When the inner backward ends and calls ProcessUnPairedCellHook, it must NOT
                 clear the outer engine's unpaired hooks. The outer hook should still fire when
                 the outer backward ends.
    Expectation: outer_hook fires during outer backward, not prematurely during inner backward.
    """
    from mindspore import nn, Parameter

    hook_call_log = []

    class InnerHookCell(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor(np.array([1.0]).astype(np.float32)), name='ihw')

        def construct(self, b):
            return b * self.w

    inner_hook_cell = InnerHookCell()

    def inner_hook_fn(_cell, _grad_input, _grad_output):
        hook_call_log.append("inner_hook")

    inner_hook_cell.register_backward_hook(inner_hook_fn)

    class CustomBpropFunction(_Function):
        @staticmethod
        def forward(ctx, x):
            return x * x

        @staticmethod
        def backward(ctx, dout):
            b = Tensor(np.array([1.0]).astype(np.float32))
            b.requires_grad = True
            _pynative_executor.set_enable_grad(True)
            inner_out = inner_hook_cell(b)
            inner_out.backward()
            _pynative_executor.set_enable_grad(False)
            hook_call_log.append("inner_backward_done")
            return (dout * 2,)

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor(np.array([3.0]).astype(np.float32)), name='net_w')

        def construct(self, x):
            z = CustomBpropFunction.apply(x)
            return z * self.w

    net = Net()

    def outer_hook_fn(_cell, _grad_input, _grad_output):
        hook_call_log.append("outer_hook")

    net.register_backward_hook(outer_hook_fn)

    x = Tensor(np.array([3.0]).astype(np.float32))
    x.requires_grad = True
    out = net(x)
    out.backward()

    assert "inner_backward_done" in hook_call_log, f"Inner backward didn't run, log: {hook_call_log}"
    assert "outer_hook" in hook_call_log, f"Outer hook not called at all, log: {hook_call_log}"
    inner_done_idx = hook_call_log.index("inner_backward_done")
    outer_hook_idx = hook_call_log.index("outer_hook")
    assert outer_hook_idx > inner_done_idx, (
        f"outer_hook fired before inner backward completed, log: {hook_call_log}"
    )
