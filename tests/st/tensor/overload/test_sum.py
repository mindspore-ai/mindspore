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
import mindspore as ms
import mindspore.nn as nn
import numpy as np
import pytest

from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.utils import test_utils


class SumPythonNet(nn.Cell):
    def construct(self, x, axis=None, dtype=None, keepdims=False, initial=None):
        return x.sum(axis, dtype, keepdims, initial)


class SumPythonKVNet(nn.Cell):
    def construct(self, x, axis=None, dtype=None, keepdims=False, initial=None):
        return x.sum(axis=axis, dtype=dtype, keepdims=keepdims, initial=initial)


class SumPyboostNet(nn.Cell):
    def construct(self, x, dim=None, keepdim=False, *, dtype=None):
        return x.sum(dim, keepdim, dtype=dtype)


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


@test_utils.run_with_cell
def sum_ext_forward_func(x, dim=None, keepdim=False, *, dtype=None):
    return x.sum(dim, keepdim, dtype=dtype)


@test_utils.run_with_cell
def sum_forward_func(x, axis=None, dtype=None, keepdims=False, initial=None):
    return x.sum(axis, dtype, keepdims, initial)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_method_sum_python(mode):
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.sum.
    Expectation: Run success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = SumPythonNet()
    net1 = SumPythonKVNet()

    x = ms.Tensor([[[1, 2, 3], [2, 3, 4]]], ms.float32)
    output = net(x)
    expect_output = 15.0
    assert np.allclose(output.asnumpy(), expect_output)

    output = net1(x, axis=[0, 1], keepdims=True)
    expect_output = np.array([[[3., 5., 7.]]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)

    output = net1(x, axis=(2,), keepdims=False)
    expect_output = np.array([[6., 9.]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)

    output = net1(x, axis=2, keepdims=True)
    expect_output = np.array([[[6.], [9.]]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)

    output = net1(x, dtype=ms.bool_, initial=12)
    expect_output = True
    assert np.allclose(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=4.00)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_method_sum_pyboost(mode):
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.sum.
    Expectation: Run success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = SumPyboostNet()

    x = ms.Tensor(np.array([[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
                            [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
                            [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]]]), ms.float32)
    output = net(x, dtype=None)
    expect_output = 270.0
    assert np.allclose(output.asnumpy(), expect_output)

    output = net(x, dim=2, dtype=None)
    expect_output = np.array([[6., 12., 18.],
                              [24., 30., 36.],
                              [42., 48., 54.]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)

    output = net(x, dim=2, keepdim=True, dtype=None)
    expect_output = np.array([[[6.],
                               [12.],
                               [18.]],
                              [[24.],
                               [30.],
                               [36.]],
                              [[42.],
                               [48.],
                               [54.]]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=4.00)
def test_tensor_sum_dynamic():
    """
    Feature: Test sum op.
    Description: Test sum dynamic shape.
    Expectation: the result match with expected result.
    """
    ms_data1 = ms.Tensor(generate_random_input((4, 6), np.float32))
    dim1 = 0
    keepdim1 = False
    ms_data2 = ms.Tensor(generate_random_input((5, 2, 7, 3), np.float32))
    dim2 = 2
    keepdim2 = True
    TEST_OP(sum_ext_forward_func,
            [[ms_data1, dim1, keepdim1], [ms_data2, dim2, keepdim2]],
            disable_mode=['GRAPH_MODE_GE'])

    ms_data1 = ms.Tensor(generate_random_input((2, 6), np.float32))
    axis1 = 0
    dtype1 = ms.float32
    keepdims1 = True
    initial1 = 3
    ms_data2 = ms.Tensor(generate_random_input((3, 2, 7, 3), np.float32))
    axis2 = 2
    dtype2 = ms.float16
    keepdims2 = False
    initial2 = 2
    TEST_OP(sum_forward_func,
            [[ms_data1, axis1, dtype1, keepdims1, initial1], [ms_data2, axis2, dtype2, keepdims2, initial2]],
            disable_mode=['GRAPH_MODE_GE'],
            case_config={'disable_resize': True})
