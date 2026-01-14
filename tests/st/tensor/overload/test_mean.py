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
from mindspore.graph.api import _pynative_executor


class MeanNet(nn.Cell):
    def construct(self, x, dim=None, keepdim=False, *, dtype=None):
        return x.mean(dim, keepdim, dtype=dtype)


class MeanKVNet(nn.Cell):
    def construct(self, x, dim=None, keepdim=False, *, dtype=None):
        return x.mean(dim=dim, keepdim=keepdim, dtype=dtype)


class MeanKVDisruptNet(nn.Cell):
    def construct(self, x, dim=None, keepdim=False, *, dtype=None):
        return x.mean(keepdim=keepdim, dim=dim, dtype=dtype)


class MeanNetpython(nn.Cell):
    def construct(self, x, axis=None, keep_dims=False):
        return x.mean(keep_dims=keep_dims, axis=axis)


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


@test_utils.run_with_cell
def mean_ext_forward_func(x, dim=None, keepdim=False, *, dtype=None):
    return x.mean(dim, keepdim, dtype=dtype)


@test_utils.run_with_cell
def mean_forward_func(x, axis=None, keep_dims=False):
    return x.mean(axis, keep_dims)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_method_mean_python(mode):
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.mean.
    Expectation: Run success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})

    # test 1: using positional args
    net = MeanNetpython()
    x = ms.Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
    output = net(x, 1, True)
    result = output.shape
    expected = np.array([3, 1, 5, 6], dtype=np.float32)
    assert np.allclose(result, expected)

    # test 2: using default args.
    x = ms.Tensor(np.array([[[2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2]],
                            [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
                            [[6, 6, 6, 6, 6, 6], [8, 8, 8, 8, 8, 8], [10, 10, 10, 10, 10, 10]]]), ms.float32)
    output = net(x)
    expected = 5.0
    assert np.allclose(output.asnumpy(), expected)

    # test 3: using k-v args.
    output = net(x, axis=0, keep_dims=True)
    expected = np.array([[[4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
                          [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                          [6.0, 6.0, 6.0, 6.0, 6.0, 6.0]]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expected)

    # test 4: using k-v out of order args.
    output = net(x, axis=1, keep_dims=True)
    expected = np.array([[[2.0, 2.0, 2.0, 2.0, 2.0, 2.0]],
                         [[5.0, 5.0, 5.0, 5.0, 5.0, 5.0]],
                         [[8.0, 8.0, 8.0, 8.0, 8.0, 8.0]]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=1024.0)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_method_mean_pyboost(mode):
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.mean.
    Expectation: Run success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})

    # test 1: using positional args
    net = MeanNet()
    x = ms.Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
    output = net(x, 1, True, dtype=None)
    result = output.shape
    expected = np.array([3, 1, 5, 6], dtype=np.float32)
    assert np.allclose(result, expected)

    # test 2: using default args.
    x = ms.Tensor(np.array([[[2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2]],
                            [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
                            [[6, 6, 6, 6, 6, 6], [8, 8, 8, 8, 8, 8], [10, 10, 10, 10, 10, 10]]]), ms.float32)
    output = net(x, dtype=None)
    expected = 5.0
    assert np.allclose(output.asnumpy(), expected)

    # test 3: using k-v args.
    net = MeanKVNet()
    output = net(x, dim=0, keepdim=True, dtype=None)
    expected = np.array([[[4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
                          [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                          [6.0, 6.0, 6.0, 6.0, 6.0, 6.0]]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expected)

    # test 4: using k-v out of order args.
    net = MeanKVDisruptNet()
    output = net(x, keepdim=True, dim=1, dtype=None)
    expected = np.array([[[2.0, 2.0, 2.0, 2.0, 2.0, 2.0]],
                         [[5.0, 5.0, 5.0, 5.0, 5.0, 5.0]],
                         [[8.0, 8.0, 8.0, 8.0, 8.0, 8.0]]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expected)

    # test 5: error input
    net = MeanNet()
    with pytest.raises(TypeError):
        net(x, dim=float(2.0), keepdim=True, dtype=None)
        _pynative_executor.sync()

    with pytest.raises(TypeError):
        net(x, dim=1, keepdim=1, dtype=None)
        _pynative_executor.sync()

    with pytest.raises(TypeError):
        net(x, dim=5, keepdim=1, dtype=None)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential',
          mem_peak=2.00)
def test_tensor_mean_dynamic():
    """
    Feature: Test mean op.
    Description: Test mean dynamic shape.
    Expectation: the result match with expected result.
    """
    ms_data1 = ms.Tensor(generate_random_input((4, 3, 6), np.float32))
    axis1 = 0
    keep_dims1 = False
    ms_data2 = ms.Tensor(generate_random_input((5, 2, 7, 3), np.float32))
    axis2 = 2
    keep_dims2 = True
    TEST_OP(mean_ext_forward_func,
            [[ms_data1, axis1, keep_dims1], [ms_data2, axis2, keep_dims2]],
            disable_mode=['GRAPH_MODE_GE'],
            case_config={'disable_input_check': True})
    TEST_OP(mean_forward_func,
            [[ms_data1, axis1, keep_dims1], [ms_data2, axis2, keep_dims2]],
            disable_mode=['GRAPH_MODE_GE'],
            case_config={'disable_grad': True})
