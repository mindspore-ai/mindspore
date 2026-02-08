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
import mindspore as ms
from mindspore import Tensor, Parameter
from mindspore import nn
import pytest
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_grad_basic():
    """
    Feature: Test grad by position.
    Description: Test grad by position.
    Expectation: Success.
    """
    def fn(x, y):
        return x * x + y * y

    x = ms.Tensor([3.0], dtype=ms.float32)
    y = ms.Tensor([4.0], dtype=ms.float32)

    grad_fn = ms.grad(fn, grad_position=0)
    grad = grad_fn(x, y)

    assert np.allclose(grad.asnumpy(), 6.0)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_grad_weights():
    """
    Feature: Test grad by weights.
    Description: Test grad by weights.
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

    grad_fn = ms.grad(net, weights=net.trainable_params())
    grad = grad_fn(x)
    assert np.allclose(grad[1][0].asnumpy(), 3.0)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_grad_both():
    """
    Feature: Test grad by both input and weights.
    Description: Test grad by both input and weights.
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

    grad_fn = ms.grad(net, grad_position=0, weights=net.trainable_params())
    input_grad, weight_grad = grad_fn(x)

    assert np.allclose(input_grad.asnumpy(), 2.0)
    assert np.allclose(weight_grad[0].asnumpy(), 3.0)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_grad_multiple_positions():
    """
    Feature: Test grad by multiple positions.
    Description: Test grad by multiple positions.
    Expectation: Success.
    """
    def fn(x, y, z):
        return x * y + z

    x = Tensor([2.0], dtype=ms.float32)
    y = Tensor([3.0], dtype=ms.float32)
    z = Tensor([4.0], dtype=ms.float32)

    grad_fn = ms.grad(fn, grad_position=(0, 1))
    grads = grad_fn(x, y, z)

    assert np.allclose(grads[0].asnumpy(), 3.0)
    assert np.allclose(grads[1].asnumpy(), 2.0)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_grad_has_aux():
    """
    Feature: Test grad with auxiliary output.
    Description: Test grad with auxiliary output.
    Expectation: Success.
    """
    def fn(x):
        return x * x, x + 1

    x = Tensor([3.0], dtype=ms.float32)

    grad_fn = ms.grad(fn, has_aux=True)
    grad, aux = grad_fn(x)

    assert np.allclose(grad.asnumpy(), 6.0)
    assert np.allclose(aux[0].asnumpy(), 4.0)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_grad_return_ids():
    """
    Feature: Test grad with return IDs.
    Description: Test grad with return IDs.
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

    grad_fn = ms.grad(
        net, grad_position=0, weights=net.trainable_params(), return_ids=True
    )
    grad_info = grad_fn(x)

    assert isinstance(grad_info, tuple)
    assert len(grad_info) == 2
    assert isinstance(grad_info[0], tuple) and len(grad_info[0]) == 2
    assert isinstance(grad_info[1], tuple) and len(grad_info[1][0]) == 2


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_grad_multiple_weights():
    """
    Feature: Test grad with multiple weights.
    Description: Test grad with multiple weights.
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
    grad_fn = ms.grad(net, weights=net.trainable_params())
    grads = grad_fn(x)
    assert np.allclose(grads[1][0].asnumpy(), 12.0)
    assert np.allclose(grads[1][1].asnumpy(), 8.0)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_grad_none_arguments():
    """
    Feature: Test grad with none arguments.
    Description: Test grad with none arguments.
    Expectation: Raise ValueError.
    """
    def fn(x):
        return x * x

    x = Tensor([3.0], dtype=ms.float32)

    with pytest.raises(ValueError):
        ms.grad(fn, grad_position=None, weights=None)(x)
        ms.runtime.synchronize()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_grad_invalid_grad_position_type():
    """
    Feature: Test grad with invalid grad_position type.
    Description: Test grad with invalid grad_position type.
    Expectation: Raise TypeError.
    """
    def fn(x):
        return x * x

    x = Tensor([3.0], dtype=ms.float32)

    with pytest.raises(TypeError):
        ms.grad(fn, grad_position="invalid")(x)
        ms.runtime.synchronize()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_grad_invalid_weights_type():
    """
    Feature: Test grad with invalid weights type.
    Description: Test grad with invalid weights type.
    Expectation: Raise TypeError.
    """
    def fn(x):
        return x * x

    x = Tensor([3.0], dtype=ms.float32)

    with pytest.raises(TypeError):
        ms.grad(fn, weights="invalid")(x)
        ms.runtime.synchronize()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_grad_invalid_has_aux_type():
    """
    Feature: Test grad with invalid has_aux type.
    Description: Test grad with invalid has_aux type.
    Expectation: Raise TypeError.
    """
    def fn(x):
        return x * x

    x = Tensor([3.0], dtype=ms.float32)

    with pytest.raises(TypeError):
        ms.grad(fn, has_aux="invalid")(x)
        ms.runtime.synchronize()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_grad_invalid_return_ids_type():
    """
    Feature: Test grad with invalid return_ids type.
    Description: Test grad with invalid return_ids type.
    Expectation: Raise TypeError.
    """
    def fn(x):
        return x * x

    x = Tensor([3.0], dtype=ms.float32)

    with pytest.raises(TypeError):
        ms.grad(fn, return_ids="invalid")(x)
        ms.runtime.synchronize()
