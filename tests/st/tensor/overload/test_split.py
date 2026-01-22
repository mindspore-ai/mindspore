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


class SplitPythonNet(nn.Cell):
    def construct(self, x, split_size_or_sections, axis=0):
        return x.split(split_size_or_sections=split_size_or_sections, axis=axis)


class SplitPyboostNet(nn.Cell):
    def construct(self, x, split_size, dim=0):
        return x.split(split_size=split_size, dim=dim)


class SplitPyboostNet1(nn.Cell):
    def construct(self, x, split_size, dim=0):
        return x.split(split_size, dim)


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


@test_utils.run_with_cell
def split_python_forward_func(x, split_size_or_sections, axis=0):
    return x.split(split_size_or_sections=split_size_or_sections, axis=axis)


@test_utils.run_with_cell
def split_pyboost_forward_func(x, split_size, dim=0):
    return x.split(split_size=split_size, dim=dim)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_method_split_python(mode):
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.split.
    Expectation: Run success
    """

    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = SplitPythonNet()

    a = np.array(np.arange(20).reshape((10, 2)), dtype=np.float32)
    x = ms.Tensor(a, dtype=ms.float32)
    split_size_or_sections = 5
    out = net(x, split_size_or_sections, axis=0)
    expect = [np.array(np.arange(10).reshape((5, 2)), dtype=np.float32),
              np.array(np.arange(10, 20).reshape((5, 2)), dtype=np.float32)]
    for res, exp in zip(out, expect):
        assert np.allclose(res.asnumpy(), exp)
    split_size_or_sections = 1
    out = net(x, split_size_or_sections, axis=1)
    expect = [np.array(np.arange(0, 20, 2).reshape((10, 1)), dtype=np.float32),
              np.array(np.arange(1, 20, 2).reshape((10, 1)), dtype=np.float32)]
    for res, exp in zip(out, expect):
        assert np.allclose(res.asnumpy(), exp)

    a = np.array(np.arange(20).reshape((10, 2)), dtype=np.float32)
    x = ms.Tensor(a, dtype=ms.float32)
    split_size_or_sections = [2, 3, 5]
    out = net(x, split_size_or_sections, axis=0)
    expect = [np.array([[0, 1], [2, 3]], dtype=np.float32),
              np.array([[4, 5], [6, 7], [8, 9]], dtype=np.float32),
              np.array([[10, 11], [12, 13], [14, 15], [16, 17], [18, 19]], dtype=np.float32)]
    for res, exp in zip(out, expect):
        assert np.allclose(res.asnumpy(), exp)

    a = np.array(np.arange(20).reshape((2, 10)), dtype=np.float32)
    x = ms.Tensor(a, dtype=ms.float32)
    out = net(x, split_size_or_sections, axis=1)
    expect = [np.array([[0, 1], [10, 11]], dtype=np.float32),
              np.array([[2, 3, 4], [12, 13, 14]], dtype=np.float32),
              np.array([[5, 6, 7, 8, 9], [15, 16, 17, 18, 19]], dtype=np.float32)]
    for res, exp in zip(out, expect):
        assert np.allclose(res.asnumpy(), exp)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_method_split_pyboost(mode):
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.split.
    Expectation: Run success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = SplitPyboostNet()
    net1 = SplitPyboostNet1()

    a = np.array(np.arange(20).reshape((10, 2)), dtype=np.float32)
    x = ms.Tensor(a, dtype=ms.float32)
    split_size = 5

    if mode == 0 and ms.get_context('device_target') != 'Ascend':
        with pytest.raises(RuntimeError) as error_info:
            net(x, split_size, dim=0)
            _pynative_executor.sync()
        assert "Unsupported op [SplitTensor] on" in str(error_info.value)

    out = net1(x, split_size, dim=0)
    expect = [np.array(np.arange(10).reshape((5, 2)), dtype=np.float32),
              np.array(np.arange(10, 20).reshape((5, 2)), dtype=np.float32)]
    for res, exp in zip(out, expect):
        assert np.allclose(res.asnumpy(), exp)

    a = np.array(np.arange(20).reshape((10, 2)), dtype=np.float32)
    x = ms.Tensor(a, dtype=ms.float32)
    split_size = (2, 3, 5)

    out = net1(x, split_size, dim=0)
    expect = [np.array([[0, 1], [2, 3]], dtype=np.float32),
              np.array([[4, 5], [6, 7], [8, 9]], dtype=np.float32),
              np.array([[10, 11], [12, 13], [14, 15], [16, 17], [18, 19]], dtype=np.float32)]
    for res, exp in zip(out, expect):
        assert np.allclose(res.asnumpy(), exp)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_tensor_split_pyboost_dynamic():
    """
    Feature: Test split op.
    Description: Test split dynamic shape.
    Expectation: the result match with expected result.
    """
    ms_data1 = ms.Tensor(generate_random_input((4, 6), np.float32))
    split_size1 = 2
    dim1 = 0
    ms_data2 = ms.Tensor(generate_random_input((5, 2, 7, 3), np.float32))
    split_size2 = 3
    dim2 = 2
    TEST_OP(split_pyboost_forward_func, [[ms_data1, split_size1, dim1], [ms_data2, split_size2, dim2]],
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'],
            disable_case=['EmptyTensor', 'ScalarTensor'],
            case_config={'disable_nontensor_dynamic_type': 'BOTH'})

    split_size1 = (2, 2)
    dim1 = 0
    split_size2 = (3, 2, 2)
    dim2 = 2
    TEST_OP(split_pyboost_forward_func, [[ms_data1, split_size1, dim1], [ms_data2, split_size2, dim2]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor', 'ScalarTensor'],
            case_config={'disable_nontensor_dynamic_type': 'BOTH'})


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_tensor_split_python_dynamic():
    """
    Feature: Test split op.
    Description: Test split dynamic shape.
    Expectation: the result match with expected result.
    """
    ms_data1 = ms.Tensor(generate_random_input((4, 6), np.float32))
    split_size_or_sections1 = (2, 2)
    axis1 = 0
    ms_data2 = ms.Tensor(generate_random_input((5, 2, 7, 3), np.float32))
    split_size_or_sections2 = (4, 1, 2)
    axis2 = 2
    TEST_OP(split_python_forward_func,
            [[ms_data1, split_size_or_sections1, axis1], [ms_data2, split_size_or_sections2, axis2]],
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'],
            disable_case=['EmptyTensor', 'ScalarTensor'],
            case_config={'disable_nontensor_dynamic_type': 'BOTH'})
