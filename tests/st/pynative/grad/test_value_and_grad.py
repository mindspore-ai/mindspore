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
""" test_auto_grad """

import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, Parameter
from mindspore import nn
from mindspore import ops
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_value_and_grad_basic():
    """
    Feature: Test auto grad by position.
    Description: Test auto grad by position.
    Expectation: Success.
    """
    def fn(x, y):
        return x * x + y * y

    x = ms.Tensor([3.0], dtype=ms.float32)
    y = ms.Tensor([4.0], dtype=ms.float32)

    value, grad = ms.value_and_grad(fn, grad_position=0)(x, y)

    assert np.allclose(value.asnumpy(), 25.0)
    assert np.allclose(grad.asnumpy(), 6.0)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_value_and_grad_weights():
    """
    Feature: Test auto grad by weights.
    Description: Test auto grad by weights.
    Expectation: Success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor([2.0], dtype=ms.float32), name='w')

        def construct(self, x):
            return x * self.w

    net = Net()
    x = Tensor([3.0], dtype=ms.float32)

    value, grad = ms.value_and_grad(net, weights=net.trainable_params())(x)

    assert np.allclose(value.asnumpy(), 6.0)
    assert np.allclose(grad[0].asnumpy(), 2.0)
    assert np.allclose(grad[1][0].asnumpy(), 3.0)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_value_and_grad_both():
    """
    Feature: Test auto grad by both input and weights.
    Description: Test auto grad by both input and weights.
    Expectation: Success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor([2.0], dtype=ms.float32), name='w')

        def construct(self, x):
            return x * self.w

    net = Net()
    x = Tensor([3.0], dtype=ms.float32)

    value, (input_grad, weight_grad) = ms.value_and_grad(
        net, grad_position=0, weights=net.trainable_params()
    )(x)

    assert np.allclose(value.asnumpy(), 6.0)
    assert np.allclose(input_grad.asnumpy(), 2.0)
    assert np.allclose(weight_grad[0].asnumpy(), 3.0)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_value_and_grad_multiple_positions():
    """
    Feature: Test auto grad by multiple positions.
    Description: Test auto grad by multiple positions.
    Expectation: Success.
    """
    def fn(x, y, z):
        return x * y + z

    x = Tensor([2.0], dtype=ms.float32)
    y = Tensor([3.0], dtype=ms.float32)
    z = Tensor([4.0], dtype=ms.float32)

    value, grads = ms.value_and_grad(fn, grad_position=(0, 1))(x, y, z)

    assert np.allclose(value.asnumpy(), 10.0)
    assert np.allclose(grads[0].asnumpy(), 3.0)
    assert np.allclose(grads[1].asnumpy(), 2.0)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_value_and_grad_has_aux():
    """
    Feature: Test auto grad with auxiliary output.
    Description: Test auto grad with auxiliary output.
    Expectation: Success.
    """
    def fn(x):
        return x * x, x + 1

    x = Tensor([3.0], dtype=ms.float32)

    (value, aux), grad = ms.value_and_grad(fn, has_aux=True)(x)

    assert np.allclose(value.asnumpy(), 9.0)
    assert np.allclose(aux.asnumpy(), 4.0)
    assert np.allclose(grad.asnumpy(), 6.0)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_value_and_grad_return_ids():
    """
    Feature: Test auto grad with return IDs.
    Description: Test auto grad with return IDs.
    Expectation: Success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor([2.0], dtype=ms.float32), name='w')

        def construct(self, x):
            return x * self.w

    net = Net()
    x = Tensor([3.0], dtype=ms.float32)

    value, grad_info = ms.value_and_grad(
        net, grad_position=0, weights=net.trainable_params(), return_ids=True
    )(x)

    assert np.allclose(value.asnumpy(), 6.0)
    assert isinstance(grad_info, tuple)
    assert len(grad_info) == 2
    assert isinstance(grad_info[0], tuple) and len(grad_info[0]) == 2
    assert isinstance(grad_info[1], tuple) and len(grad_info[1][0]) == 2


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_value_and_grad_multiple_weights():
    """
    Feature: Test auto grad with multiple weights.
    Description: Test auto grad with multiple weights.
    Expectation: Success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w1 = Parameter(Tensor([2.0], dtype=ms.float32), name='w1')
            self.w2 = Parameter(Tensor([3.0], dtype=ms.float32), name='w2')

        def construct(self, x):
            return x * self.w1 * self.w2

    net = Net()
    x = Tensor([4.0], dtype=ms.float32)

    value, grads = ms.value_and_grad(net, weights=net.trainable_params())(x)

    assert np.allclose(value.asnumpy(), 24.0)
    assert np.allclose(grads[1][0].asnumpy(), 12.0)
    assert np.allclose(grads[1][1].asnumpy(), 8.0)


class NormalNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.p1 = Parameter(Tensor([1], dtype=ms.float32))
        self.p2 = Parameter(Tensor([2], dtype=ms.float32))

    def construct(self, x):
        y = x + self.p1
        z = y * self.p2
        return z


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_auto_grad_by_position():
    """
    Feature: Test auto grad by position.
    Description: Test auto grad by position.
    Expectation: Success.
    """
    x = Tensor([1], ms.float32)
    net = NormalNet()
    _, grad = ops.value_and_grad(net)(x)
    assert np.allclose(grad.asnumpy(), np.array([2], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_auto_grad_by_position_list():
    """
    Feature: Test auto grad by position.
    Description: Test auto grad by position.
    Expectation: Success.
    """
    x = Tensor([1], ms.float32)
    net = NormalNet()
    _, grad = ops.value_and_grad(net)(x)
    assert np.allclose(grad.asnumpy(), np.array([2], dtype=np.float32), 0.00001, 0.00001)


class AuxNet(nn.Cell):
    def construct(self, x):
        y = x * x
        z = y + y
        h = x * x
        return y, z, h


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_value_and_grad_has_aux_2():
    """
    Feature: Test hax aux.
    Description: Test value_and_grad has aux.
    Expectation: Success.
    """
    x = Tensor([2.0], ms.float32)
    net = AuxNet()
    grad_op = ops.value_and_grad(net, 0, None, True)
    _, grads = grad_op(x)
    assert np.allclose(grads.asnumpy(), np.array([4.0], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_value_and_grad_none_arguments():
    """
    Feature: Test auto grad with none arguments.
    Description: Test auto grad with none arguments.
    Expectation: Raise ValueError.
    """
    def fn(x):
        return x * x

    x = Tensor([3.0], dtype=ms.float32)

    with pytest.raises(ValueError):
        ms.value_and_grad(fn, grad_position=None, weights=None)(x)
        ms.runtime.synchronize()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_value_and_grad_invalid_grad_position_type():
    """
    Feature: Test auto grad with invalid grad_position type.
    Description: Test auto grad with invalid grad_position type.
    Expectation: Raise TypeError.
    """
    def fn(x):
        return x * x

    x = Tensor([3.0], dtype=ms.float32)

    with pytest.raises(TypeError):
        ms.value_and_grad(fn, grad_position="invalid")(x)
        ms.runtime.synchronize()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_value_and_grad_invalid_weights_type():
    """
    Feature: Test auto grad with invalid weights type.
    Description: Test auto grad with invalid weights type.
    Expectation: Raise TypeError.
    """
    def fn(x):
        return x * x

    x = Tensor([3.0], dtype=ms.float32)

    with pytest.raises(TypeError):
        ms.value_and_grad(fn, weights="invalid")(x)
        ms.runtime.synchronize()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_value_and_grad_invalid_has_aux_type():
    """
    Feature: Test auto grad with invalid has_aux type.
    Description: Test auto grad with invalid has_aux type.
    Expectation: Raise TypeError.
    """
    def fn(x):
        return x * x

    x = Tensor([3.0], dtype=ms.float32)

    with pytest.raises(TypeError):
        ms.value_and_grad(fn, has_aux="invalid")(x)
        ms.runtime.synchronize()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_value_and_grad_invalid_return_ids_type():
    """
    Feature: Test auto grad with invalid return_ids type.
    Description: Test auto grad with invalid return_ids type.
    Expectation: Raise TypeError.
    """
    def fn(x):
        return x * x

    x = Tensor([3.0], dtype=ms.float32)

    with pytest.raises(TypeError):
        ms.value_and_grad(fn, return_ids="invalid")(x)
        ms.runtime.synchronize()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_value_and_grad_output_as_input():
    """
    Feature: Value and grad.
    Description: Test feeding the forward output as input into another value_and_grad call.
    Expectation: Success.
    """
    def forward_fn(x):
        return x * x

    x = ops.rand(5, 5, dtype=ms.float32)
    output, _ = ms.value_and_grad(forward_fn)(x)

    _, grad = ms.value_and_grad(forward_fn)(output)
    assert np.allclose(grad.asnumpy(), (output * 2).asnumpy())
