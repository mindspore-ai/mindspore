# Copyright 2021 Huawei Technologies Co., Ltd
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
"""test function grad in pynative mode"""
import numpy as np
import pytest
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore import ms_function
from mindspore.ops.functional import grad

context.set_context(mode=context.PYNATIVE_MODE)


class SingleInputSingleOutputNet(nn.Cell):
    def construct(self, x):
        return x**3

class SingleInputMultipleOutputsNet(nn.Cell):
    def construct(self, x):
        return x**3, 2*x

class MultipleInputsSingleOutputNet(nn.Cell):
    def construct(self, x, y, z):
        return x*y*z

class MultipleInputsMultipleOutputsNet(nn.Cell):
    def construct(self, x, y, z):
        return x**2 + y**2 + z**2, x*y*z


def function(x, y, z):
    return x**2 + y**2 + z**2, x*y*z

def iteration_grad_function(x, y, z):
    return x**2*y*z

@ms_function
def grad_warp_with_msfunction(x, y, z):
    output = grad(function)(x, y, z)
    return output


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_single_input_single_output_cell_pynative():
    """
    Features: Function grad.
    Description: Test F.grad with single input and single output net in pynative mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    net = SingleInputSingleOutputNet()
    expect_grad = Tensor(np.array([[3, 12], [27, 48]]).astype(np.float32))
    real_grad = grad(net)(x)
    assert np.allclose(real_grad.asnumpy(), expect_grad.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_single_input_multiple_outputs_cell_pynative():
    """
    Features: Function grad.
    Description: Test F.grad with single input and multiple outputs net in pynative mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    net = SingleInputMultipleOutputsNet()
    expect_grad = Tensor(np.array([[5, 14], [29, 50]]).astype(np.float32))
    real_grad = grad(net)(x)
    assert np.allclose(real_grad.asnumpy(), expect_grad.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_multiple_inputs_single_output_cell_pynative():
    """
    Features: Function grad.
    Description: Test F.grad with multiple inputs and single output net in pynative mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[-2, 3], [-1, 2]]).astype(np.float32))
    z = Tensor(np.array([[0, 3], [5, -1]]).astype(np.float32))
    net = MultipleInputsSingleOutputNet()
    expect_grad1 = Tensor(np.array([[0, 6], [15, -4]]).astype(np.float32))
    expect_grad2 = Tensor(np.array([[-2, 6], [-3, 8]]).astype(np.float32))
    real_grad = grad(net, grad_position=(1, 2))(x, y, z)
    assert isinstance(real_grad, tuple)
    assert len(real_grad) == 2
    assert np.allclose(real_grad[0].asnumpy(), expect_grad1.asnumpy())
    assert np.allclose(real_grad[1].asnumpy(), expect_grad2.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_multiple_inputs_multiple_outputs_cell_pynative():
    """
    Features: Function grad.
    Description: Test F.grad with multiple inputs and multiple outputs net in pynative mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[-2, 3], [-1, 2]]).astype(np.float32))
    z = Tensor(np.array([[0, 3], [5, -1]]).astype(np.float32))
    net = MultipleInputsMultipleOutputsNet()
    expect_grad1 = Tensor(np.array([[-4, 12], [13, 0]]).astype(np.float32))
    expect_grad2 = Tensor(np.array([[-2, 12], [7, 6]]).astype(np.float32))
    real_grad = grad(net, grad_position=(1, 2))(x, y, z)
    assert isinstance(real_grad, tuple)
    assert len(real_grad) == 2
    assert np.allclose(real_grad[0].asnumpy(), expect_grad1.asnumpy())
    assert np.allclose(real_grad[1].asnumpy(), expect_grad2.asnumpy())

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_function_with_sens_pynative():
    """
    Features: Function grad.
    Description: Test F.grad with function setting sens_param in pynative mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[-2, 3], [-1, 2]]).astype(np.float32))
    z = Tensor(np.array([[0, 3], [5, -1]]).astype(np.float32))
    v = Tensor(np.array([[-1, 3], [2, 1]]).astype(np.float32))
    expect_grad1 = Tensor(np.array([[4, 36], [26, 0]]).astype(np.float32))
    expect_grad2 = Tensor(np.array([[2, 36], [14, 6]]).astype(np.float32))
    real_grad = grad(function, grad_position=(1, 2), sens_param=True)(x, y, z, (v, v))
    assert isinstance(real_grad, tuple)
    assert len(real_grad) == 2
    assert np.allclose(real_grad[0].asnumpy(), expect_grad1.asnumpy())
    assert np.allclose(real_grad[1].asnumpy(), expect_grad2.asnumpy())

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_iteration_function_pynative():
    """
    Features: Function grad.
    Description: Test calling F.grad iterative with function in pynative mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[-2, 3], [-1, 2]]).astype(np.float32))
    z = Tensor(np.array([[0, 3], [5, -1]]).astype(np.float32))
    expect_grad1 = Tensor(np.array([[0, 12], [30, -8]]).astype(np.float32))
    expect_grad2 = Tensor(np.array([[-4, 12], [-6, 16]]).astype(np.float32))
    real_grad = grad(grad(iteration_grad_function), grad_position=(1, 2))(x, y, z)
    assert isinstance(real_grad, tuple)
    assert len(real_grad) == 2
    assert np.allclose(real_grad[0].asnumpy(), expect_grad1.asnumpy())
    assert np.allclose(real_grad[1].asnumpy(), expect_grad2.asnumpy())

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_warp_with_msfunction_pynative():
    """
    Features: Function grad.
    Description: Test F.grad warpped with ms_function in pynative mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[-2, 3], [-1, 2]]).astype(np.float32))
    z = Tensor(np.array([[0, 3], [5, -1]]).astype(np.float32))
    expect_grad = Tensor(np.array([[2, 13], [1, 6]]).astype(np.float32))
    real_grad = grad_warp_with_msfunction(x, y, z)
    assert np.allclose(real_grad.asnumpy(), expect_grad.asnumpy())

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_with_grad_position_twice_pynative():
    """
    Features: Function grad.
    Description: Test F.grad with function setting grad_position twice in pynative mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    z = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    net = MultipleInputsSingleOutputNet()
    out1 = grad(net, grad_position=0)(x, y, z)
    out2 = grad(net, grad_position=(0, 1))(x, y, z)
    assert isinstance(out1, Tensor)
    assert isinstance(out2, tuple)
