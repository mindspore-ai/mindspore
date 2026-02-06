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
import numpy as np
import pytest
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.utils import test_utils

import mindspore as ms
import mindspore.nn as nn


class SubPythonNet(nn.Cell):
    def construct(self, x, y):
        return x.sub(y)


class SubPythonNet1(nn.Cell):
    def construct(self, x, y):
        return x.__sub__(y)


class SubPythonNet2(nn.Cell):
    def construct(self, x, y):
        return x.__isub__(y)


class SubPyboostNet(nn.Cell):
    def construct(self, x, other, *, alpha=1):
        return x.sub(other, alpha=alpha)


class SubPyboostNet1(nn.Cell):
    def construct(self, x, other, *, alpha=1):
        return x.__sub__(other, alpha=alpha)


class SubPyboostNet2(nn.Cell):
    def construct(self, x, other, *, alpha=1):
        return x.__isub__(other, alpha=alpha)


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


@test_utils.run_with_cell
def sub_forward_func(x, y):
    return x.sub(y)


@test_utils.run_with_cell
def sub_ext_forward_func(x, other, *, alpha=1):
    return x.sub(other, alpha=alpha)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=4.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_method_sub_python(mode):
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.sub.
    Expectation: Run success
    """

    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})

    net = SubPythonNet()
    net1 = SubPythonNet1()
    net2 = SubPythonNet2()
    x = ms.Tensor(np.array([1, 2, 3]), dtype=ms.int32)
    y = ms.Tensor(np.array([4, 5, 6]), dtype=ms.int32)
    output = net(x, y)
    expect_output = np.array([-3, -3, -3], dtype=np.int32)
    assert np.allclose(output.asnumpy(), expect_output)
    output1 = net1(x, y)
    assert np.allclose(output1.asnumpy(), expect_output)
    output2 = net2(x, y)
    assert np.allclose(output2.asnumpy(), expect_output)



@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=1024.0)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_method_sub_pyboost(mode):
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.sub.
    Expectation: Run success
    """

    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})

    net = SubPyboostNet()
    net1 = SubPyboostNet1()
    net2 = SubPyboostNet2()
    x = ms.Tensor(np.array([4, 5, 6]), dtype=ms.float32)
    y = ms.Tensor(1, ms.int32)
    alpha = 2
    output = net(x, y, alpha=alpha)
    expect_output = np.array([2, 3, 4], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)
    output1 = net1(x, y, alpha=alpha)
    assert np.allclose(output1.asnumpy(), expect_output)
    output2 = net2(x, y, alpha=alpha)
    assert np.allclose(output2.asnumpy(), expect_output)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.0)
def test_tensor_sub_dynamic():
    """
    Feature: Test sub op.
    Description: Test sub dynamic shape.
    Expectation: the result match with expected result.
    """
    ms_data1 = ms.Tensor(generate_random_input((4, 6), np.float32))
    y1 = ms.Tensor(generate_random_input((4, 6), np.float32))
    ms_data2 = ms.Tensor(generate_random_input((5, 2, 7, 3), np.float32))
    y2 = ms.Tensor(generate_random_input((5, 2, 7, 3), np.float32))
    TEST_OP(sub_forward_func, [[ms_data1, y1], [ms_data2, y2]],
            disable_mode=["GRAPH_MODE_GE"],
            case_config={'all_dim_zero': True})
    TEST_OP(sub_ext_forward_func, [[ms_data1, y1], [ms_data2, y2]],
            disable_mode=['GRAPH_MODE_GE'],
            case_config={'all_dim_zero': True})
