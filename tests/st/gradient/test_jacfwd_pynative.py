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
"""test function jacfwd in pynative mode"""
import numpy as np
import pytest
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore import ms_function
from mindspore.ops import jacfwd

context.set_context(mode=context.PYNATIVE_MODE)


class SingleInputSingleOutputNet(nn.Cell):
    def construct(self, x):
        return x ** 3


class SingleInputMultipleOutputsNet(nn.Cell):
    def construct(self, x):
        return x ** 3, 2 * x


class MultipleInputsSingleOutputNet(nn.Cell):
    def construct(self, x, y, z):
        return x * y * z


class MultipleInputsMultipleOutputsNet(nn.Cell):
    def construct(self, x, y, z):
        return x ** 2 + y ** 2 + z ** 2, x * y * z


def function(x, y, z):
    return x ** 2 + y ** 2 + z ** 2, x * y * z


def iteration_jac_function(x, y, z):
    return x ** 2 * y * z


@ms_function
def jac_wrap_with_ms_function(x, y, z):
    output = jacfwd(function, has_aux=True)(x, y, z)
    return output


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_jac_single_input_single_output_cell_pynative():
    """
    Features: Function jacfwd.
    Description: Test ops.jacfwd with single input and single output net in pynative mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    net = SingleInputSingleOutputNet()
    expect_jac = np.array([[[[3, 0], [0, 0]], [[0, 12], [0, 0]]],
                           [[[0, 0], [27, 0]], [[0, 0], [0, 48]]]]).astype(np.float32)
    jac = jacfwd(net)(x)
    assert np.allclose(jac.asnumpy(), expect_jac)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_jac_single_input_multiple_outputs_cell_pynative():
    """
    Features: Function jacfwd.
    Description: Test ops.jacfwd with single input and multiple outputs net in pynative mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    net = SingleInputMultipleOutputsNet()
    expect_jac_0 = np.array([[[[3, 0], [0, 0]], [[0, 12], [0, 0]]],
                             [[[0, 0], [27, 0]], [[0, 0], [0, 48]]]]).astype(np.float32)
    expect_jac_1 = np.array([[[[2, 0], [0, 0]], [[0, 2], [0, 0]]],
                             [[[0, 0], [2, 0]], [[0, 0], [0, 2]]]]).astype(np.float32)
    jac = jacfwd(net)(x)
    assert np.allclose(jac[0].asnumpy(), expect_jac_0)
    assert np.allclose(jac[1].asnumpy(), expect_jac_1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_jac_multiple_inputs_single_output_cell_pynative():
    """
    Features: Function jacfwd.
    Description: Test ops.jacfwd with multiple inputs and single output net in pynative mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[-2, 3], [-1, 2]]).astype(np.float32))
    z = Tensor(np.array([[0, 3], [5, -1]]).astype(np.float32))
    net = MultipleInputsSingleOutputNet()
    expect_jac_0 = np.array([[[[0, 0], [0, 0]], [[0, 6], [0, 0]]],
                             [[[0, 0], [15, 0]], [[0, 0], [0, -4]]]]).astype(np.float32)
    expect_jac_1 = np.array([[[[-2, 0], [0, 0]], [[0, 6], [0, 0]]],
                             [[[0, 0], [-3, 0]], [[0, 0], [0, 8]]]]).astype(np.float32)
    jac = jacfwd(net, grad_position=(1, 2))(x, y, z)
    assert isinstance(jac, tuple)
    assert len(jac) == 2
    assert np.allclose(jac[0].asnumpy(), expect_jac_0)
    assert np.allclose(jac[1].asnumpy(), expect_jac_1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_jac_multiple_inputs_multiple_outputs_cell_pynative():
    """
    Features: Function jacfwd.
    Description: Test ops.jacfwd with multiple inputs and multiple outputs net in pynative mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[-2, 3], [-1, 2]]).astype(np.float32))
    z = Tensor(np.array([[0, 3], [5, -1]]).astype(np.float32))
    net = MultipleInputsMultipleOutputsNet()
    expect_jac_0 = np.array([[[[-4, 0], [0, 0]], [[0, 6], [0, 0]]],
                             [[[0, 0], [-2, 0]], [[0, 0], [0, 4]]]]).astype(np.float32)
    expect_jac_1 = np.array([[[[0, 0], [0, 0]], [[0, 6], [0, 0]]],
                             [[[0, 0], [10, 0]], [[0, 0], [0, -2]]]]).astype(np.float32)
    expect_jac_2 = np.array([[[[0, 0], [0, 0]], [[0, 6], [0, 0]]],
                             [[[0, 0], [15, 0]], [[0, 0], [0, -4]]]]).astype(np.float32)
    expect_jac_3 = np.array([[[[-2, 0], [0, 0]], [[0, 6], [0, 0]]],
                             [[[0, 0], [-3, 0]], [[0, 0], [0, 8]]]]).astype(np.float32)
    jac = jacfwd(net, grad_position=(1, 2))(x, y, z)
    assert isinstance(jac, tuple)
    assert len(jac) == 2
    assert np.allclose(jac[0][0].asnumpy(), expect_jac_0)
    assert np.allclose(jac[0][1].asnumpy(), expect_jac_1)
    assert np.allclose(jac[1][0].asnumpy(), expect_jac_2)
    assert np.allclose(jac[1][1].asnumpy(), expect_jac_3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_jac_wrap_with_ms_function_pynative():
    """
    Features: Function jacfwd.
    Description: Test ops.jacfwd warpped with ms_function in pynative mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[-2, 3], [-1, 2]]).astype(np.float32))
    z = Tensor(np.array([[0, 3], [5, -1]]).astype(np.float32))
    expect_jac = np.array([[[[2, 0], [0, 0]], [[0, 4], [0, 0]]],
                           [[[0, 0], [6, 0]], [[0, 0], [0, 8]]]]).astype(np.float32)
    expect_aux = np.array([[0, 18], [-15, -8]]).astype(np.float32)
    jac, aux = jac_wrap_with_ms_function(x, y, z)
    assert np.allclose(jac.asnumpy(), expect_jac)
    assert np.allclose(aux.asnumpy(), expect_aux)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_jac_with_grad_position_twice_pynative():
    """
    Features: Function jacfwd.
    Description: Test ops.jacfwd with function setting grad_position twice in pynative mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[1, 3], [5, 7]]).astype(np.float32))
    z = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    expect_jac_0 = np.array([[[[1, 0], [0, 0]], [[0, 3], [0, 0]]],
                             [[[0, 0], [5, 0]], [[0, 0], [0, 7]]]]).astype(np.float32)
    expect_jac_1 = np.array([[[[1, 0], [0, 0]], [[0, 2], [0, 0]]],
                             [[[0, 0], [3, 0]], [[0, 0], [0, 4]]]]).astype(np.float32)
    net = MultipleInputsSingleOutputNet()
    jac1 = jacfwd(net, grad_position=0)(x, y, z)
    jac2 = jacfwd(net, grad_position=(0, 1))(x, y, z)

    assert np.allclose(jac1.asnumpy(), expect_jac_0)
    assert np.allclose(jac2[1].asnumpy(), expect_jac_1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_jac_with_has_aux_pynative():
    """
    Features: Function jacfwd.
    Description: Test ops.jacfwd with Cell setting grad_position in pynative mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    z = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    expect_jac = np.array([[[[2, 0], [0, 0]], [[0, 4], [0, 0]]],
                           [[[0, 0], [6, 0]], [[0, 0], [0, 8]]]]).astype(np.float32)
    expect_aux = np.array([[1, 4], [9, 16]]).astype(np.float32)
    net = MultipleInputsMultipleOutputsNet()
    jac, aux = jacfwd(net, grad_position=0, has_aux=True)(x, y, z)
    assert np.allclose(jac.asnumpy(), expect_jac)
    assert np.allclose(aux.asnumpy(), expect_aux)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_jac_with_function_has_aux_pynative():
    """
    Features: Function jacfwd.
    Description: Test ops.jacfwd with function setting grad_position in pynative mode.
    Expectation: No exception.
    """

    def fn(x, y, z):
        return x ** 2 + y ** 2 + z ** 2, x * y * z

    def fn2(*args):
        x = args[0]
        y = args[1]
        z = args[2]
        return fn(x, y, z)

    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    z = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    expect_jac = np.array([[[[2, 0], [0, 0]], [[0, 4], [0, 0]]],
                           [[[0, 0], [6, 0]], [[0, 0], [0, 8]]]]).astype(np.float32)
    expect_aux = np.array([[1, 4], [9, 16]]).astype(np.float32)
    jac, aux = jacfwd(fn2, grad_position=0, has_aux=True)(x, y, z)
    assert np.allclose(jac.asnumpy(), expect_jac)
    assert np.allclose(aux.asnumpy(), expect_aux)
