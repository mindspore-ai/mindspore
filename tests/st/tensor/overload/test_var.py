# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Test the overload functional method"""
import pytest
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP

def generate_deprecated_expect_forward_output(input_x, axis=None, ddof=0, keepdims=False):
    if isinstance(input_x, ms.Tensor):
        input_x = input_x.asnumpy()
    return np.var(input_x, axis=axis, ddof=ddof, keepdims=keepdims)

def generate_expect_forward_output(input_x, dim=None, correction=1, keepdim=False):
    if isinstance(input_x, ms.Tensor):
        input_x = input_x.asnumpy()
    return np.var(input_x, axis=dim, ddof=correction, keepdims=keepdim)

class VarNet(nn.Cell):
    def construct(self, input_x):
        return input_x.var()


class VarArgsNet(nn.Cell):
    def construct(self, input_x, *args):
        return input_x.var(*args)


class VarPythonNet(nn.Cell):
    def construct(self, input_x, axis=None, ddof=0, keepdims=False):
        return input_x.var(axis=axis, ddof=ddof, keepdims=keepdims)


class VarPyBoostNet(nn.Cell):
    def construct(self, input_x, dim=None, correction=1, keepdim=False):
        return input_x.var(dim=dim, correction=correction, keepdim=keepdim)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend',
                      'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=4.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_deprecated_tensor_var(mode):
    """
    Feature: tensor.var
    Description: verify the result of the deprecated tensor.var
    Expectation: success
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    input_x = ms.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], ms.float32)

    net = VarNet()
    output = net(input_x)
    expect_output = generate_deprecated_expect_forward_output(input_x)
    np.testing.assert_allclose(output.asnumpy(), expect_output, rtol=1e-4)

    net = VarArgsNet()
    axis = 1
    output = net(input_x, axis)
    expect_output = generate_deprecated_expect_forward_output(input_x, axis)
    np.testing.assert_allclose(output.asnumpy(), expect_output, rtol=1e-4)

    net = VarArgsNet()
    axis, ddof = (0, 1), 1
    output = net(input_x, axis, ddof)
    expect_output = generate_deprecated_expect_forward_output(input_x, axis, ddof)
    np.testing.assert_allclose(output.asnumpy(), expect_output, rtol=1e-4)

    net = VarArgsNet()
    axis, ddof, keepdims = None, 1, True
    output = net(input_x, axis, ddof, keepdims)
    expect_output = generate_deprecated_expect_forward_output(input_x, axis, ddof, keepdims)
    np.testing.assert_allclose(output.asnumpy(), expect_output, rtol=1e-4)

    net = VarPythonNet()
    axis, ddof, keepdims = -1, -1, True
    output = net(input_x, axis=axis, ddof=ddof, keepdims=keepdims)
    expect_output = generate_deprecated_expect_forward_output(input_x, axis=axis, ddof=ddof, keepdims=keepdims)
    np.testing.assert_allclose(output.asnumpy(), expect_output, rtol=1e-4)

    net = VarPythonNet()
    axis, ddof, keepdims = 0, False, True
    output = net(input_x, axis=axis, ddof=ddof, keepdims=keepdims)
    expect_output = generate_deprecated_expect_forward_output(input_x, axis=axis, ddof=ddof, keepdims=keepdims)
    np.testing.assert_allclose(output.asnumpy(), expect_output, rtol=1e-4)

@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_tensor_var(mode):
    """
    Feature: tensor.var
    Description: verify the result of tensor.var
    Expectation: success
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    input_x = ms.Tensor([[[1, 3], [2, 4]], [[5, 7], [6, 8]]], ms.float32)
    net = VarPyBoostNet()
    input_list = []
    input_list.append([input_x, None, 0, False])
    input_list.append([input_x, 0, 1, True])
    input_list.append([input_x, (-1,), 3, True])
    input_list.append([input_x, (1, 2), 3, False])

    for i in range(len(input_list)):
        args = input_list[i]
        output = net(*args)
        expect_output = generate_expect_forward_output(*args)
        np.testing.assert_allclose(output.asnumpy(), expect_output, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=4.0)
def test_tensor_var_dynamic():
    """
    Feature: test tensor.var
    Description: test tensor.var dynamic shape
    Expectation: success
    """

    input1 = ms.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], ms.float32)
    input2 = ms.Tensor([[[1, 3], [2, 4]], [[5, 7], [6, 8]]], ms.float32)

    net = VarPythonNet()
    TEST_OP(
        net,
        [[input1, 0, 1, False], [input2, 1, 3, True]],
        disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'],
        disable_case=['ScalarTensor'],
        case_config={'disable_input_check': True},
    )

    net = VarPyBoostNet()
    TEST_OP(
        net,
        [[input1, 0, 1, False], [input2, 1, 2, True]],
        disable_mode=['GRAPH_MODE_GE'],
        disable_case=['ScalarTensor'],
        case_config={'disable_input_check': True},
    )
