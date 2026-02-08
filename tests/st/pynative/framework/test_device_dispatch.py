# Copyright 2025 Huawei Technologies Co., Ltd
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

"""Test Pynative dispatch"""

import os
import pytest
import numpy as np
from tests.mark_utils import arg_mark
from mindspore import Tensor, Parameter, context, mint, nn, ops
from mindspore import dtype as mstype
from mindspore.ops import composite as C

grad_all = C.GradOperation(get_all=True)

# Ensure pynative mode
context.set_context(mode=context.PYNATIVE_MODE)


########################################
# Basic: device dispatch forward & backward
########################################

def test_pynative_device_dispatch_cpu_forward():
    """
    Feature: Pynative dispatch
    Description: Forward op dispatch on CPU device
    Expectation: result on CPU
    """
    x = Tensor(1.0, mstype.float32)
    y = x.sin()
    assert "CPU" in y.device


def test_pynative_device_dispatch_ascend_forward():
    """
    Feature: Pynative dispatch
    Description: Forward op dispatch on Ascend device
    Expectation: result on Ascend
    """
    x = Tensor(1.0, mstype.float32).to("Ascend")
    y = x.sin()
    assert "Ascend" in y.device


def test_pynative_device_dispatch_cpu_to_ascend_chain_ops():
    """
    Feature: Pynative dispatch
    Description: Tensor created on CPU, then moved to Ascend and used in a chain of ops
    Expectation: final result on Ascend
    """
    x = Tensor(2.0, mstype.float32)
    assert "CPU" in x.device

    x = x.to("Ascend")
    y = x.sin() + x.cos() * 2.0
    assert "Ascend" in y.device


class SimpleNet(nn.Cell):
    def construct(self, x):
        # A simple combination of operators for backward graph testing
        return ops.sin(x) * 2.0 + ops.cos(x)


def test_pynative_device_dispatch_cpu_backward():
    """
    Feature: Pynative dispatch
    Description: Backward (grad) dispatch on CPU device
    Expectation: grad runs successfully and result on CPU
    """
    net = SimpleNet()
    x = Tensor(1.0, mstype.float32)

    grads = grad_all(net)(x)
    (dx,) = grads

    assert "CPU" in dx.device
    expected = 2.0 * np.cos(1.0) - np.sin(1.0)
    np.testing.assert_allclose(dx.asnumpy(), expected, rtol=1e-3, atol=1e-3)


def test_pynative_device_dispatch_ascend_backward():
    """
    Feature: Pynative dispatch
    Description: Backward (grad) dispatch on Ascend device
    Expectation: grad runs successfully and result on Ascend
    """
    net = SimpleNet()
    x = Tensor(1.0, mstype.float32).to("Ascend")

    grads = grad_all(net)(x)
    (dx,) = grads

    assert "Ascend" in dx.device
    expected = 2.0 * np.cos(1.0) - np.sin(1.0)
    np.testing.assert_allclose(dx.asnumpy(), expected, rtol=1e-3, atol=1e-3)


def test_pynative_device_dispatch_parameter_on_ascend():
    """
    Feature: Pynative dispatch
    Description: Cell Parameter on Ascend, forward & backward use the same device
    Expectation: output and grad both on Ascend
    """
    class ParamNet(nn.Cell):
        def __init__(self):
            super().__init__()
            w_init = Tensor(2.0, mstype.float32).to("Ascend")
            self.w = Parameter(w_init, name="w")

        def construct(self, x):
            return self.w * x + ops.sin(x)

    net = ParamNet()
    x = Tensor(3.0, mstype.float32).to("Ascend")

    y = net(x)
    assert "Ascend" in y.device

    grads = grad_all(net)(x)
    dx, = grads
    assert "Ascend" in dx.device


def test_pynative_device_dispatch_broadcast_cpu():
    """
    Feature: Pynative dispatch
    Description: Broadcast op on CPU device
    Expectation: broadcast result on CPU and value is correct
    """
    x = Tensor(np.ones((2, 3), np.float32))
    y = x + Tensor(2.0, mstype.float32)
    assert "CPU" in y.device
    np.testing.assert_allclose(y.asnumpy(), np.ones((2, 3), np.float32) + 2.0)


def test_pynative_device_dispatch_broadcast_ascend():
    """
    Feature: Pynative dispatch
    Description: Broadcast op on Ascend device
    Expectation: broadcast result on Ascend and value is correct
    """
    x = Tensor(np.ones((2, 3), np.float32)).to("Ascend")
    y = x + Tensor(2.0, mstype.float32).to("Ascend")
    assert "Ascend" in y.device
    np.testing.assert_allclose(y.asnumpy(), np.ones((2, 3), np.float32) + 2.0)


def test_pynative_device_dispatch_mixed_device_error():
    """
    Feature: Pynative dispatch
    Description: Mixed device inputs should raise or be forbidden in pynative mode
    Expectation: raise RuntimeError or ValueError
    """
    x_cpu = Tensor(1.0, mstype.float32)
    x_ascend = Tensor(1.0, mstype.float32).to("Ascend")

    with pytest.raises((RuntimeError, ValueError)):
        _ = x_cpu + x_ascend


def test_pynative_device_dispatch_grad_scalar_output_ascend():
    """
    Feature: Pynative dispatch
    Description: Scalar output grad on Ascend device
    Expectation: grad runs successfully and grad device is Ascend
    """
    class ScalarNet(nn.Cell):
        def construct(self, x):
            # sum produces scalar output
            return mint.sum(ops.sin(x))

    net = ScalarNet()
    x = Tensor(np.ones((4,), np.float32)).to("Ascend")

    grads = grad_all(net)(x)
    (dx,) = grads
    assert "Ascend" in dx.device
    expected = np.full((4,), np.cos(1.0), dtype=np.float32)
    np.testing.assert_allclose(dx.asnumpy(), expected, rtol=1e-3, atol=1e-3)


def test_pynative_device_dispatch_dtype_cast_between_devices():
    """
    Feature: Pynative dispatch
    Description: Device + dtype cast flow works in pynative mode
    Expectation: result on target device and dtype correct
    """
    x = Tensor(1.0, mstype.float32)
    x = x.to("Ascend")
    y = ops.cast(x, mstype.float16)
    assert "Ascend" in y.device
    assert y.dtype == mstype.float16


########################################
# Assign op related tests
########################################

def test_pynative_device_dispatch_assign_cpu():
    """
    Feature: Pynative dispatch + Assign
    Description: Use ops.Assign on CPU
    Expectation: parameter and output stay on CPU, value updated correctly
    """
    param = Parameter(Tensor(0.0, mstype.float32), name="param_cpu")
    value = Tensor(3.0, mstype.float32)
    assert "CPU" in param.device
    assert "CPU" in value.device

    assign = ops.Assign()
    out = assign(param, value)

    assert "CPU" in out.device
    assert "CPU" in param.device
    np.testing.assert_allclose(param.asnumpy(), 3.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out.asnumpy(), 3.0, rtol=1e-6, atol=1e-6)


def test_pynative_device_dispatch_assign_ascend():
    """
    Feature: Pynative dispatch + Assign
    Description: Use ops.Assign on Ascend
    Expectation: parameter and output stay on Ascend, value updated correctly
    """
    param = Parameter(Tensor(0.0, mstype.float32).to("Ascend"), name="param_ascend")
    value = Tensor(5.0, mstype.float32).to("Ascend")

    assert "Ascend" in param.device
    assert "Ascend" in value.device

    assign = ops.Assign()
    out = assign(param, value)

    assert "Ascend" in out.device
    assert "Ascend" in param.device
    np.testing.assert_allclose(param.asnumpy(), 5.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out.asnumpy(), 5.0, rtol=1e-6, atol=1e-6)


def test_pynative_device_dispatch_assign_in_cell_ascend_forward_backward():
    """
    Feature: Pynative dispatch + Assign
    Description: Assign used inside Cell on Ascend, verify forward and backward
    Expectation: forward & grad run, parameter/device correct
    """
    class AssignNet(nn.Cell):
        def __init__(self):
            super().__init__()
            w_init = Tensor(1.0, mstype.float32).to("Ascend")
            self.w = Parameter(w_init, name="w")
            self.assign = ops.Assign()

        def construct(self, x):
            # update w to mean(x), then use w in computation
            new_w = ops.reduce_mean(x)
            self.assign(self.w, new_w)
            out = self.w * x
            return out

    net = AssignNet()
    x = Tensor(np.ones((2, 2), np.float32)).to("Ascend")

    # forward
    y = net(x)
    assert "Ascend" in y.device
    np.testing.assert_allclose(y.asnumpy(), np.ones((2, 2), np.float32), rtol=1e-6, atol=1e-6)
    assert "Ascend" in net.w.device
    np.testing.assert_allclose(net.w.asnumpy(), 1.0, rtol=1e-6, atol=1e-6)

    # backward
    grads = grad_all(net)(x)
    (dx,) = grads
    assert "Ascend" in dx.device
    np.testing.assert_allclose(dx.asnumpy(), np.ones((2, 2), np.float32), rtol=1e-6, atol=1e-6)


def test_pynative_device_dispatch_assign_mixed_device_error():
    """
    Feature: Pynative dispatch + Assign
    Description: Assign between different devices should fail
    Expectation: raise RuntimeError or ValueError
    """
    param_cpu = Parameter(Tensor(0.0, mstype.float32), name="param_cpu")
    value_ascend = Tensor(1.0, mstype.float32).to("Ascend")

    assign = ops.Assign()
    with pytest.raises((RuntimeError, ValueError)):
        _ = assign(param_cpu, value_ascend)


########################################
# View-like ops: reshape / slice / expand_dims / flatten / transpose
########################################

def test_pynative_device_dispatch_view_reshape_cpu():
    """
    Feature: Pynative dispatch + reshape
    Description: reshape on CPU
    Expectation: result stays on CPU
    """
    x = Tensor(np.arange(6).astype(np.float32))
    y = ops.reshape(x, (2, 3))

    assert "CPU" in y.device
    np.testing.assert_allclose(y.asnumpy(), [[0, 1, 2], [3, 4, 5]])


def test_pynative_device_dispatch_view_reshape_ascend():
    """
    Feature: Pynative dispatch + reshape
    Description: reshape on Ascend
    Expectation: result stays on Ascend
    """
    x = Tensor(np.arange(6).astype(np.float32)).to("Ascend")
    y = ops.reshape(x, (2, 3))

    assert "Ascend" in y.device
    np.testing.assert_allclose(y.asnumpy(), [[0, 1, 2], [3, 4, 5]])


def test_pynative_device_dispatch_view_slice_cpu():
    """
    Feature: Pynative dispatch + slice
    Description: slice on CPU
    Expectation: result stays on CPU
    """
    x = Tensor(np.arange(6).reshape(2, 3).astype(np.float32))
    y = ops.slice(x, (0, 0), (1, 2))

    assert "CPU" in y.device
    np.testing.assert_allclose(y.asnumpy(), [[0, 1]])


def test_pynative_device_dispatch_view_slice_ascend():
    """
    Feature: Pynative dispatch + slice
    Description: slice on Ascend
    Expectation: result stays on Ascend
    """
    x = Tensor(np.arange(6).reshape(2, 3).astype(np.float32)).to("Ascend")
    y = ops.slice(x, (0, 0), (1, 2))

    assert "Ascend" in y.device
    np.testing.assert_allclose(y.asnumpy(), [[0, 1]])


def test_pynative_device_dispatch_view_expand_dims_cpu():
    """
    Feature: Pynative dispatch + expand_dims
    Description: expand_dims on CPU
    Expectation: result stays on CPU
    """
    x = Tensor(np.array([1, 2, 3], np.float32))
    y = ops.expand_dims(x, 0)

    assert "CPU" in y.device
    np.testing.assert_allclose(y.asnumpy(), [[1, 2, 3]])


def test_pynative_device_dispatch_view_expand_dims_ascend():
    """
    Feature: Pynative dispatch + expand_dims
    Description: expand_dims on Ascend
    Expectation: result stays on Ascend
    """
    x = Tensor(np.array([1, 2, 3], np.float32)).to("Ascend")
    y = ops.expand_dims(x, 0)

    assert "Ascend" in y.device
    np.testing.assert_allclose(y.asnumpy(), [[1, 2, 3]])


def test_pynative_device_dispatch_view_flatten_cpu():
    """
    Feature: Pynative dispatch + flatten
    Description: flatten on CPU
    Expectation: result stays on CPU
    """
    x = Tensor(np.arange(6).reshape(2, 3).astype(np.float32))
    y = mint.flatten(x)

    assert "CPU" in y.device
    np.testing.assert_allclose(y.asnumpy(), np.arange(6))


def test_pynative_device_dispatch_view_flatten_ascend():
    """
    Feature: Pynative dispatch + flatten
    Description: flatten on Ascend
    Expectation: result stays on Ascend
    """
    x = Tensor(np.arange(6).reshape(2, 3).astype(np.float32)).to("Ascend")
    y = mint.flatten(x)

    assert "Ascend" in y.device
    np.testing.assert_allclose(y.asnumpy(), np.arange(6))


def test_pynative_device_dispatch_view_transpose_cpu():
    """
    Feature: Pynative dispatch + transpose
    Description: transpose on CPU
    Expectation: result stays on CPU
    """
    x = Tensor(np.arange(6).reshape(2, 3).astype(np.float32))
    y = mint.transpose(x, 1, 0)

    assert "CPU" in y.device
    np.testing.assert_allclose(y.asnumpy(), [[0, 3], [1, 4], [2, 5]])


def test_pynative_device_dispatch_view_transpose_ascend():
    """
    Feature: Pynative dispatch + transpose
    Description: transpose on Ascend
    Expectation: result stays on Ascend
    """
    x = Tensor(np.arange(6).reshape(2, 3).astype(np.float32)).to("Ascend")
    y = mint.transpose(x, 1, 0)

    assert "Ascend" in y.device
    np.testing.assert_allclose(y.asnumpy(), [[0, 3], [1, 4], [2, 5]])


########################################
# View-like ops + backward
########################################

def test_pynative_device_dispatch_view_reshape_backward_ascend():
    """
    Feature: Pynative dispatch + reshape + backward
    Description: reshape backward on Ascend.(Disabled now)
    Expectation: grad computed on Ascend
    """
    class ReshapeNet(nn.Cell):
        def construct(self, x):
            y = ops.reshape(x, (2, 3))
            return mint.sum(y)

    net = ReshapeNet()
    x = Tensor(np.arange(6).astype(np.float32)).to("Ascend")

    (dx,) = grad_all(net)(x)
    assert "Ascend" in dx.device
    np.testing.assert_allclose(dx.asnumpy(), np.ones(6))


def test_pynative_device_dispatch_view_flatten_backward_ascend():
    """
    Feature: Pynative dispatch + flatten + backward
    Description: flatten backward on Ascend
    Expectation: grad computed on Ascend
    """
    class FlatNet(nn.Cell):
        def construct(self, x):
            y = mint.flatten(x)
            return mint.sum(y)

    net = FlatNet()
    x = Tensor(np.ones((2, 3), np.float32)).to("Ascend")

    (dx,) = grad_all(net)(x)
    assert "Ascend" in dx.device
    np.testing.assert_allclose(dx.asnumpy(), np.ones((2, 3)))


########################################
# View + mixed device error
########################################

def test_pynative_device_dispatch_view_mixed_device_error():
    """
    Feature: Pynative dispatch + view ops
    Description: view ops combined with tensors on different devices should fail
    Expectation: raise RuntimeError or ValueError
    """
    x_cpu = Tensor(np.arange(6).reshape(2, 3), mstype.float32)
    x_ascend = Tensor(np.arange(6).reshape(2, 3), mstype.float32).to("Ascend")

    with pytest.raises((RuntimeError, ValueError)):
        _ = mint.reshape(x_cpu, (3, 2)) + x_ascend


########################################
# Tensor getitem / setitem (CPU / Ascend)
########################################

def test_pynative_device_dispatch_getitem_cpu():
    """
    Feature: Pynative dispatch + Tensor getitem
    Description: Basic tensor indexing on CPU
    Expectation: slice result on CPU, value is correct
    """
    x = Tensor(np.arange(6).reshape(2, 3).astype(np.float32))
    y = x[0]
    z = x[0, 1]

    assert "CPU" in y.device
    assert "CPU" in z.device
    np.testing.assert_allclose(y.asnumpy(), np.array([0, 1, 2], np.float32))
    np.testing.assert_allclose(z.asnumpy(), np.array(1.0, np.float32))


def test_pynative_device_dispatch_getitem_ascend():
    """
    Feature: Pynative dispatch + Tensor getitem
    Description: Basic tensor indexing on Ascend
    Expectation: slice result on Ascend, value is correct
    """
    x = Tensor(np.arange(6).reshape(2, 3).astype(np.float32)).to("Ascend")
    y = x[1]
    z = x[1, 2]

    assert "Ascend" in y.device
    assert "Ascend" in z.device
    np.testing.assert_allclose(y.asnumpy(), np.array([3, 4, 5], np.float32))
    np.testing.assert_allclose(z.asnumpy(), np.array(5.0, np.float32))


def test_pynative_device_dispatch_setitem_cpu():
    """
    Feature: Pynative dispatch + Tensor setitem
    Description: In-place indexing assignment on CPU
    Expectation: tensor stays on CPU and value updated correctly
    """
    context.set_context(device_target="CPU")
    x = Tensor(np.zeros((2, 3), np.float32))
    assert "CPU" in x.device

    x[0, 1] = 5.0
    x[1] = Tensor(np.array([7, 8, 9], np.float32))

    assert "CPU" in x.device
    expected = np.array([[0, 5, 0],
                         [7, 8, 9]], dtype=np.float32)
    np.testing.assert_allclose(x.asnumpy(), expected)
    context.set_context(device_target="Ascend")


def test_pynative_device_dispatch_setitem_ascend():
    """
    Feature: Pynative dispatch + Tensor setitem
    Description: In-place indexing assignment on Ascend
    Expectation: tensor stays on Ascend and value updated correctly
    """
    x = Tensor(np.zeros((2, 3), np.float32)).to("Ascend")
    assert "Ascend" in x.device

    x[0, 2] = 3.0
    x[1, :] = Tensor(np.array([1, 2, 3], np.float32)).to("Ascend")

    assert "Ascend" in x.device
    expected = np.array([[0, 0, 3],
                         [1, 2, 3]], dtype=np.float32)
    np.testing.assert_allclose(x.asnumpy(), expected)


def test_pynative_device_dispatch_setitem_broadcast_cpu():
    """
    Feature: Pynative dispatch + Tensor setitem broadcast
    Description: Broadcasting value in setitem on CPU
    Expectation: tensor stays on CPU and broadcast assignment is correct
    """
    context.set_context(device_target="CPU")
    x = Tensor(np.zeros((2, 3), np.float32))
    x[:, 1] = 2.0

    assert "CPU" in x.device
    expected = np.array([[0, 2, 0],
                         [0, 2, 0]], dtype=np.float32)
    np.testing.assert_allclose(x.asnumpy(), expected)
    context.set_context(device_target="Ascend")


def test_pynative_device_dispatch_setitem_broadcast_ascend():
    """
    Feature: Pynative dispatch + Tensor setitem broadcast
    Description: Broadcasting value in setitem on Ascend
    Expectation: tensor stays on Ascend and broadcast assignment is correct
    """
    x = Tensor(np.zeros((2, 3), np.float32)).to("Ascend")
    x[:, 0] = 1.0

    assert "Ascend" in x.device
    expected = np.array([[1, 0, 0],
                         [1, 0, 0]], dtype=np.float32)
    np.testing.assert_allclose(x.asnumpy(), expected)


def test_pynative_device_dispatch_getitem_view_backward_ascend():
    """
    Feature: Pynative dispatch + getitem + backward
    Description: slice view participates in backward on Ascend
    Expectation: grad of original tensor on Ascend, value correct
    """
    class SliceNet(nn.Cell):
        def construct(self, x):
            y = x[0]
            return mint.sum(y)

    net = SliceNet()
    x = Tensor(np.arange(6).reshape(2, 3).astype(np.float32)).to("Ascend")

    (dx,) = grad_all(net)(x)
    assert "Ascend" in dx.device

    expected = np.array([[1, 1, 1],
                         [0, 0, 0]], dtype=np.float32)
    np.testing.assert_allclose(dx.asnumpy(), expected)


def test_pynative_device_dispatch_setitem_mixed_device_error():
    """
    Feature: Pynative dispatch + Tensor setitem
    Description: setitem with different-device value should fail
    Expectation: raise RuntimeError or ValueError
    """
    x_cpu = Tensor(np.zeros((2, 3), np.float32))
    v_ascend = Tensor(np.ones((3,), np.float32)).to("Ascend")

    with pytest.raises((RuntimeError, ValueError)):
        x_cpu[0] = v_ascend


def test_pynative_device_dispatch_getitem_mixed_device_usage_error():
    """
    Feature: Pynative dispatch + Tensor getitem
    Description: Using slice from Ascend tensor together with CPU tensor should fail
    Expectation: raise RuntimeError or ValueError
    """
    x_cpu = Tensor(np.ones((3,), np.float32))
    x_ascend = Tensor(np.ones((3,), np.float32)).to("Ascend")
    y_ascend_slice = x_ascend[:]

    assert "Ascend" in y_ascend_slice.device

    with pytest.raises((RuntimeError, ValueError)):
        _ = x_cpu + y_ascend_slice


########################################
# Dynamic graph hooks: Tensor hook / Cell hook
########################################

def test_pynative_tensor_hook_on_cpu():
    """
    Feature: Pynative Tensor hook
    Description: Register grad hook on Tensor on CPU device
    Expectation: hook is called, sees grad on CPU, and can modify grad
    """
    hook_called = {"flag": False}
    captured_grad = {"value": None}

    class HookNet(nn.Cell):
        def construct(self, x):
            y = x * 2.0
            z = mint.sum(y)
            return z

    net = HookNet()
    x = Tensor(3.0, mstype.float32)

    def grad_hook(grad):
        hook_called["flag"] = True
        captured_grad["value"] = grad
        assert "CPU" in grad.device
        # Grad of L = sum(2x) is 2; multiply by 2 again
        return grad * 2.0

    x.register_hook(grad_hook)

    (dx,) = grad_all(net)(x)

    assert hook_called["flag"] is True
    assert captured_grad["value"] is not None
    assert "CPU" in dx.device

    np.testing.assert_allclose(captured_grad["value"].asnumpy(), np.array(2.0, np.float32))
    np.testing.assert_allclose(dx.asnumpy(), np.array(4.0, np.float32))


def test_pynative_tensor_hook_on_ascend():
    """
    Feature: Pynative Tensor hook
    Description: Register grad hook on Tensor on Ascend device
    Expectation: hook is called, sees grad on Ascend, and grad modification works
    """
    hook_called = {"flag": False}
    captured_grad = {"value": None}

    class HookNet(nn.Cell):
        def construct(self, x):
            y = x * 3.0
            z = mint.sum(y)
            return z

    net = HookNet()
    x = Tensor(2.0, mstype.float32).to("Ascend")

    def grad_hook(grad):
        hook_called["flag"] = True
        captured_grad["value"] = grad
        assert "Ascend" in grad.device
        # Do not modify grad
        return grad

    x.register_hook(grad_hook)

    (dx,) = grad_all(net)(x)

    assert hook_called["flag"] is True
    assert captured_grad["value"] is not None
    assert "Ascend" in dx.device

    np.testing.assert_allclose(dx.asnumpy(), np.array(3.0, np.float32))
    np.testing.assert_allclose(captured_grad["value"].asnumpy(), np.array(3.0, np.float32))


def test_pynative_tensor_hook_mixed_device_error():
    """
    Feature: Pynative Tensor hook
    Description: Grad hook returns tensor on different device should fail
    Expectation: raise RuntimeError or ValueError
    """
    class HookNet(nn.Cell):
        def construct(self, x):
            return mint.sum(x * 2.0)

    net = HookNet()
    x = Tensor(1.0, mstype.float32).to("Ascend")

    def bad_grad_hook(grad):
        grad_cpu = grad.asnumpy()
        return Tensor(grad_cpu, mstype.float32)  # default device is CPU

    x.register_hook(bad_grad_hook)

    with pytest.raises((RuntimeError, ValueError)):
        _ = grad_all(net)(x)


def test_pynative_cell_forward_hook_like_cpu():
    """
    Feature: Pynative Cell forward hook-like behavior
    Description: Simulate forward hook by wrapping Cell and calling hook in construct (CPU)
    Expectation: hook sees input/output on CPU, and can modify output
    """
    records = {"inputs": None, "outputs": None, "called": False}

    class InnerNet(nn.Cell):
        def construct(self, x):
            return x * 2.0

    class WrapperNet(nn.Cell):
        def __init__(self, net, hook_fn):
            super().__init__()
            self.net = net
            self.hook_fn = hook_fn

        def construct(self, x):
            out = self.net(x)
            new_out = self.hook_fn(self.net, (x,), out)
            return new_out

    def forward_hook(cell, inputs, output):
        records["called"] = True
        records["inputs"] = inputs
        records["outputs"] = output
        assert "CPU" in inputs[0].device
        assert "CPU" in output.device
        # Modify output: add 1
        return output + Tensor(1.0)

    net = WrapperNet(InnerNet(), forward_hook)

    x = Tensor(2.0, mstype.float32)
    y = net(x)

    assert records["called"] is True
    assert records["inputs"] is not None
    assert records["outputs"] is not None
    assert "CPU" in y.device

    np.testing.assert_allclose(y.asnumpy(), np.array(5.0, np.float32))


def test_pynative_cell_forward_hook_like_ascend():
    """
    Feature: Pynative Cell forward hook-like behavior on Ascend
    Description: Wrapper cell calls hook in construct to simulate forward hook
    Expectation: hook sees input/output on Ascend
    """
    records = {"inputs": None, "outputs": None, "called": False}

    class InnerNet(nn.Cell):
        def construct(self, x):
            return x ** 2

    class WrapperNet(nn.Cell):
        def __init__(self, net, hook_fn):
            super().__init__()
            self.net = net
            self.hook_fn = hook_fn

        def construct(self, x):
            out = self.net(x)
            new_out = self.hook_fn(self.net, (x,), out)
            return new_out

    def forward_hook(cell, inputs, output):
        records["called"] = True
        records["inputs"] = inputs
        records["outputs"] = output
        assert "Ascend" in inputs[0].device
        assert "Ascend" in output.device
        # Do not modify output
        return output

    net = WrapperNet(InnerNet(), forward_hook)

    x = Tensor(3.0, mstype.float32).to("Ascend")
    y = net(x)

    assert records["called"] is True
    assert "Ascend" in y.device
    np.testing.assert_allclose(y.asnumpy(), np.array(9.0, np.float32))


def test_pynative_cell_bprop_hook_cpu():
    """
    Feature: Pynative Cell backward hook (bprop)
    Description: Implement bprop in Cell to inspect and modify grad on CPU
    Expectation: bprop is called, grad can be modified
    """
    records = {"called": False, "out_grad": None, "x_grad": None}

    class BpropNet(nn.Cell):
        def construct(self, x):
            return x * 2.0

        def bprop(self, x, out, dout):
            records["called"] = True
            records["out_grad"] = dout
            # Original chain: dL/dx = dout * 2; multiply by 3 more
            dx = dout * 2.0 * 3.0
            records["x_grad"] = dx
            return (dx,)

    net = BpropNet()
    x = Tensor(1.0, mstype.float32)

    (dx,) = grad_all(net)(x)

    assert records["called"] is True
    assert records["out_grad"] is not None
    assert records["x_grad"] is not None
    assert "CPU" in dx.device

    np.testing.assert_allclose(dx.asnumpy(), np.array(6.0, np.float32))


def test_pynative_cell_bprop_hook_ascend():
    """
    Feature: Pynative Cell backward hook (bprop) on Ascend
    Description: Implement bprop in Cell to inspect grad on Ascend
    Expectation: bprop is called, grad device is Ascend
    """
    records = {"called": False, "out_grad": None, "x_grad": None}

    class BpropNet(nn.Cell):
        def construct(self, x):
            # L = sum(x^2)
            return mint.sum(x * x)

        def bprop(self, x, out, dout):
            records["called"] = True
            records["out_grad"] = dout
            dx = 2 * x * dout
            records["x_grad"] = dx
            return (dx,)

    net = BpropNet()
    x = Tensor(np.array([1., 2., 3.], np.float32)).to("Ascend")

    (dx,) = grad_all(net)(x)

    assert records["called"] is True
    assert records["out_grad"] is not None
    assert records["x_grad"] is not None
    assert "Ascend" in dx.device

    expected = np.array([2., 4., 6.], np.float32)
    np.testing.assert_allclose(dx.asnumpy(), expected)


def test_pynative_cell_bprop_mixed_device_error():
    """
    Feature: Pynative Cell backward hook (bprop)
    Description: bprop returns grad on different device should fail
    Expectation: raise RuntimeError or ValueError
    """
    class BadBpropNet(nn.Cell):
        def construct(self, x):
            return mint.sum(x * 2.0)

        def bprop(self, x, out, dout):
            # Convert Ascend dout to CPU grad and return CPU Tensor
            grad_np = (2.0 * dout.asnumpy()).astype(np.float32)
            dx_cpu = Tensor(grad_np, mstype.float32)  # default device is CPU
            return (dx_cpu,)

    net = BadBpropNet()
    x = Tensor(1.0, mstype.float32).to("Ascend")

    with pytest.raises((RuntimeError, ValueError)):
        _ = grad_all(net)(x)


def test_pynative_device_dispatch_all():
    """
    Feature: Pynative dispatch
    Description: Test pynative dispatch
    Expectation: run success
    """
    test_pynative_device_dispatch_cpu_forward()
    test_pynative_device_dispatch_ascend_forward()
    test_pynative_device_dispatch_cpu_to_ascend_chain_ops()
    test_pynative_device_dispatch_cpu_backward()
    test_pynative_device_dispatch_ascend_backward()
    test_pynative_device_dispatch_parameter_on_ascend()
    test_pynative_device_dispatch_broadcast_cpu()
    test_pynative_device_dispatch_broadcast_ascend()
    #test_pynative_device_dispatch_mixed_device_error()
    test_pynative_device_dispatch_grad_scalar_output_ascend()
    test_pynative_device_dispatch_dtype_cast_between_devices()
    test_pynative_device_dispatch_assign_cpu()
    #test_pynative_device_dispatch_assign_ascend()
    test_pynative_device_dispatch_assign_in_cell_ascend_forward_backward()
    #test_pynative_device_dispatch_assign_mixed_device_error()
    test_pynative_device_dispatch_view_reshape_cpu()
    test_pynative_device_dispatch_view_reshape_ascend()
    test_pynative_device_dispatch_view_slice_cpu()
    test_pynative_device_dispatch_view_slice_ascend()
    test_pynative_device_dispatch_view_expand_dims_cpu()
    test_pynative_device_dispatch_view_expand_dims_ascend()
    test_pynative_device_dispatch_view_flatten_cpu()
    test_pynative_device_dispatch_view_flatten_ascend()
    test_pynative_device_dispatch_view_transpose_cpu()
    test_pynative_device_dispatch_view_transpose_ascend()
    test_pynative_device_dispatch_view_reshape_backward_ascend()
    test_pynative_device_dispatch_view_flatten_backward_ascend()
    #test_pynative_device_dispatch_view_mixed_device_error()
    test_pynative_device_dispatch_getitem_cpu()
    test_pynative_device_dispatch_getitem_ascend()
    test_pynative_device_dispatch_setitem_cpu()
    test_pynative_device_dispatch_setitem_ascend()
    test_pynative_device_dispatch_setitem_broadcast_cpu()
    test_pynative_device_dispatch_setitem_broadcast_ascend()
    test_pynative_device_dispatch_getitem_view_backward_ascend()
    #test_pynative_device_dispatch_setitem_mixed_device_error()
    #test_pynative_device_dispatch_getitem_mixed_device_usage_error()
    test_pynative_tensor_hook_on_cpu()
    test_pynative_tensor_hook_on_ascend()
    #test_pynative_tensor_hook_mixed_device_error()
    test_pynative_cell_forward_hook_like_cpu()
    test_pynative_cell_forward_hook_like_ascend()
    test_pynative_cell_bprop_hook_cpu()
    test_pynative_cell_bprop_hook_ascend()
    #test_pynative_cell_bprop_mixed_device_error()

@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=0.01)
def test_pynative_device_dispatch():
    """
    Feature: Pynative dispatch
    Description: Test pynative dispatch
    Expectation: run success
    """
    os.environ["MS_DEV_DISABLE_AUTO_H2D"] = "3"
    return_code = os.system(
        "pytest -sv test_device_dispatch.py::test_pynative_device_dispatch_all"
    )
    assert return_code == 0
