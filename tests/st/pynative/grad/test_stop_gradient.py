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

import numpy as np
import mindspore as ms
from mindspore import Tensor, Parameter
from mindspore import nn
from mindspore import ops
from mindspore.graph.api import _pynative_executor
import pytest
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_stop_gradient_single_input():
    """
    Feature: Test stop gradient.
    Description: Test stop gradient single input.
    Expectation: Success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor([2.0], dtype=ms.float32), name='w')

        def construct(self, x):
            intermediate = x * self.w
            stopped = ops.stop_gradient(intermediate)
            return stopped * self.w

    net = Net()
    x = Tensor([3.0], dtype=ms.float32)

    grad_fn = ms.grad(net, grad_position=0, weights=net.trainable_params())
    input_grad, weight_grad = grad_fn(x)

    assert np.allclose(input_grad.asnumpy(), 0)
    assert np.allclose(weight_grad[0].asnumpy(), 6.0)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_stop_gradient_multiple_inputs():
    """
    Feature: Test stop gradient.
    Description: Test stop gradient multi input.
    Expectation: Success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor([2.0], dtype=ms.float32), name='w')

        def construct(self, x, y):
            stopped_x, stopped_y = ops.stop_gradient((x, y))
            return stopped_x * stopped_y + stopped_x * self.w

    net = Net()
    x = Tensor([3.0], dtype=ms.float32)
    y = Tensor([4.0], dtype=ms.float32)

    grad_fn = ms.grad(net, grad_position=(0, 1), weights=net.trainable_params())
    input_grads, weight_grad = grad_fn(x, y)

    assert np.allclose(input_grads[0].asnumpy(), 0)
    assert np.allclose(input_grads[1].asnumpy(), 0)
    assert np.allclose(weight_grad[0].asnumpy(), 3.0)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_stop_gradient_with_python_objects():
    """
    Feature: Test stop gradient.
    Description: Test stop gradient python objects.
    Expectation: Success.
    """
    class Data:
        def __init__(self, value):
            self.value = value

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor([2.0], dtype=ms.float32), name='w')

        def construct(self, x):
            intermediate = x * self.w
            data_obj = Data(intermediate)
            stopped_data = ops.stop_gradient(data_obj)
            result = stopped_data.value + x
            return result

    net = Net()
    x = Tensor([3.0], dtype=ms.float32)
    grad_fn = ms.grad(net, grad_position=0, weights=net.trainable_params())
    input_grad, weight_grad = grad_fn(x)
    assert np.allclose(input_grad.asnumpy(), 3.0)
    assert np.allclose(weight_grad[0].asnumpy(), 3.0)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_stop_gradient_complex_computation():
    """
    Feature: Test stop gradient.
    Description: Test stop gradient complex computation.
    Expectation: Success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w1 = Parameter(Tensor([2.0], dtype=ms.float32), name='w1')
            self.w2 = Parameter(Tensor([3.0], dtype=ms.float32), name='w2')

        def construct(self, x):
            intermediate1 = x * self.w1
            stopped1 = ops.stop_gradient(intermediate1)
            intermediate2 = stopped1 * self.w2
            stopped2 = ops.stop_gradient(intermediate2)
            intermediate3 = stopped1 + stopped2
            stopped3 = ops.stop_gradient(intermediate3)
            return stopped3 * x

    net = Net()
    x = Tensor([4.0], dtype=ms.float32)

    grad_fn = ms.grad(net, grad_position=0, weights=net.trainable_params())
    input_grad, weight_grads = grad_fn(x)

    assert np.allclose(input_grad.asnumpy(), 32.0)
    assert np.allclose(weight_grads[0].asnumpy(), 0.0)
    assert np.allclose(weight_grads[1].asnumpy(), 0.0)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_stop_gradient_nested():
    """
    Feature: Test stop gradient.
    Description: Test stop gradient nested.
    Expectation: Success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor([2.0], dtype=ms.float32), name='w')

        def construct(self, x):
            intermediate = x * self.w
            stopped1 = ops.stop_gradient(intermediate)
            stopped2 = ops.stop_gradient(stopped1)
            return stopped2 * x

    net = Net()
    x = Tensor([3.0], dtype=ms.float32)

    grad_fn = ms.grad(net, grad_position=0, weights=net.trainable_params())
    input_grad, weight_grad = grad_fn(x)

    assert np.allclose(input_grad.asnumpy(), 6.0)
    assert np.allclose(weight_grad[0].asnumpy(), 0.0)


class StopGradientInplaceNet(nn.Cell):
    def construct(self, x):
        y = x * x
        ops.stop_gradient_(y)
        z = y * x
        return z


class StopGradientInplaceViewNet(nn.Cell):
    def construct(self, x):
        y = x * x
        y = y[0]
        ops.stop_gradient_(y)
        z = y * x
        return z


class StopGradientInplaceParameterNet(nn.Cell):
    def __init__(self):
        super(StopGradientInplaceParameterNet, self).__init__()
        self.p1 = Parameter(Tensor([2.0, 3.0], dtype=ms.float32))

    def construct(self, x):
        ops.stop_gradient_(self.p1)
        return x * self.p1


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=4.00)
def test_auto_grad_stop_gradient_inplace():
    """
    Feature: Test stop gradient inplace.
    Description: Test stop gradient inplace.
    Expectation: Success.
    """
    x = Tensor([2.0, 3.0], ms.float32)
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)

    net = StopGradientInplaceNet()
    grads = grad_op(net)(x)
    assert np.allclose(grads[0][0].asnumpy(), np.array([4.0, 9.0], dtype=np.float32), 0.00001, 0.00001)

    net = StopGradientInplaceParameterNet()
    grads = grad_op(net, net.trainable_params())(x)
    assert np.allclose(grads[0][0].asnumpy(), np.array([2.0, 3.0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1][0].asnumpy(), np.array([0.0, 0.0], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=1024.01)
def test_auto_grad_stop_gradient_inplace_input():
    """
    Feature: Test stop gradient inplace.
    Description: The input is applied by inplace stop_gradient.
    Expectation: Success.
    """
    x = Tensor([2.0], ms.float32)
    y = Tensor([3.0], ms.float32)

    def fn(x, y):
        ops.stop_gradient_(x)
        return x * y
    grad_op = ops.GradOperation(get_all=True)
    grads = grad_op(fn)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.array([0.0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.array([2.0], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=1024.01)
def test_no_grad_stop_gradient_inplace_view():
    """
    Feature: Test stop gradient inplace view in no grad mode.
    Description: Test stop gradient inplace view in no grad mode.
    Expectation: Raise Runtime Error.
    """
    x = Tensor([2.0, 3.0], ms.float32)
    net = StopGradientInplaceViewNet()
    with pytest.raises(RuntimeError, match="Cannot stop_gradient view inplace"):
        net(x)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=1024.00)
def test_auto_grad_stop_gradient_inplace_view_error():
    """
    Feature: Test stop gradient inplace view exception.
    Description: The operation will raise error.
    Expectation: Success.
    """
    x = Tensor([2.0, 3.0], ms.float32)
    net = StopGradientInplaceViewNet()
    grad_op = ops.GradOperation(get_all=True)
    with pytest.raises(RuntimeError, match="Cannot stop_gradient view inplace"):
        grad_op(net)(x)
