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
from mindspore import nn
from mindspore.graph.api import _pynative_executor


class SelectPythonNet(nn.Cell):
    def construct(self, x, condition, y):
        return x.select(condition, y)


class SelectPyboostNet(nn.Cell):
    def construct(self, x, dim, index):
        return x.select(dim, index)


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


@test_utils.run_with_cell
def select_ext_forward_func(x, dim, index):
    return x.select(dim, index)


@test_utils.run_with_cell
def select_forward_func(x, condition, y):
    return x.select(condition, y)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_method_select_python(mode):
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.select.
    Expectation: Run success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})

    net = SelectPythonNet()
    cond = ms.Tensor([True, False])
    x = ms.Tensor([2, 3], ms.float32)
    y = ms.Tensor([1, 2], ms.float32)
    output = net(x, cond, y)
    expect_output = np.array([2., 2.], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)

    with pytest.raises((ValueError, TypeError)) as error_info:
        net(x, 1, y)
        _pynative_executor.sync()

    with pytest.raises(TypeError) as error_info:
        temp_y = int(1)
        net(x, cond, temp_y)
        _pynative_executor.sync()
    assert "For 'Tensor.select', if the argument 'y' is int, then the tensor type should be int32 but got " in str(
        error_info.value)

    with pytest.raises(TypeError) as error_info:
        temp_y = float(1)
        x = ms.Tensor([2, 3], ms.int32)
        net(x, cond, temp_y)
        _pynative_executor.sync()
    assert "For 'Tensor.select', if the argument 'y' is float, then the tensor type should be float32 but got " in str(
        error_info.value)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=1024.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_method_select_pyboost(mode):
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.select.
    Expectation: Run success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})

    net = SelectPyboostNet()
    x = ms.Tensor([[2, 3, 4, 5], [3, 2, 4, 5]], ms.float32)
    output = net(x, 0, 0)
    expect_output = np.array([2, 3, 4, 5], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)
    output = net(x, 1, 0)
    expect_output = np.array([2, 3], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)
    output = net(x, 1, 3)
    expect_output = np.array([5, 5], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=1024.0)
def test_tensor_select_ext_dynamic():
    """
    Feature: Test select op.
    Description: Test select dynamic shape.
    Expectation: the result match with expected result.
    """
    ms_data1 = ms.Tensor(generate_random_input((4, 3, 6), np.float32))
    dim1 = 1
    index1 = 1
    ms_data2 = ms.Tensor(generate_random_input((5, 2, 7, 3), np.float32))
    dim2 = 2
    index2 = 3
    TEST_OP(select_ext_forward_func, [[ms_data1, dim1, index1], [ms_data2, dim2, index2]],
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'],
            disable_case=['EmptyTensor', 'ScalarTensor'],
            case_config={'all_dim_zero': True})


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.01)
def test_tensor_select_dynamic():
    """
    Feature: Test select op.
    Description: Test select dynamic shape.
    Expectation: the result match with expected result.
    """
    ms_data1 = ms.Tensor(generate_random_input((4, 3, 6), np.float32))
    condition1 = ms.Tensor(generate_random_input((4, 3, 6), np.bool_))
    y1 = ms.Tensor(generate_random_input((4, 3, 6), np.float32))
    ms_data2 = ms.Tensor(generate_random_input((5, 2, 7, 3), np.float32))
    condition2 = ms.Tensor(generate_random_input((5, 2, 7, 3), np.bool_))
    y2 = ms.Tensor(generate_random_input((5, 2, 7, 3), np.float32))
    TEST_OP(select_forward_func, [[ms_data1, condition1, y1], [ms_data2, condition2, y2]],
            disable_mode=['GRAPH_MODE_GE'], case_config={'all_dim_zero': True})
