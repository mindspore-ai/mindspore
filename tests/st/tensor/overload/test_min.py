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


class MinPythonNet(nn.Cell):
    def construct(self, x, axis=None, keepdims=False, *, initial=None, where=True, return_indices=False):
        return x.min(axis, keepdims, initial=initial, where=where, return_indices=return_indices)


class MinDimNet(nn.Cell):
    def construct(self, x, dim, keepdim):
        return x.min(dim, keepdim)


class MinPyboostNet(nn.Cell):
    def construct(self, x):
        return x.min()


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


@test_utils.run_with_cell
def min_forward_func1(x):
    return x.min()


@test_utils.run_with_cell
def min_forward_func2(x, axis=None, keepdims=False, *, initial=None, where=True, return_indices=False):
    return x.min(axis, keepdims, initial=initial, where=where, return_indices=return_indices)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_method_min_python(mode):
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.min.
    Expectation: Run success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})

    # test 1: using positional args
    net = MinPythonNet()
    x = ms.Tensor(np.arange(4).reshape((2, 2)).astype(np.float32))
    output = net(x, 0, False, initial=9, where=ms.Tensor(
        [False, True]), return_indices=False)
    expect_output = np.array([9., 1.], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)

    output = net(x, (0, 1), False, initial=9, where=ms.Tensor(
        [False, True]), return_indices=False)
    expect_output = 1.0
    assert np.allclose(output.asnumpy(), expect_output)

    # test 2: using default args.
    output = net(x)
    assert np.allclose(output.asnumpy(), 0.0)

    # test 3: using k-v args.
    output = net(x, axis=0, keepdims=False, initial=9,
                 where=ms.Tensor([False, True]), return_indices=False)
    expect_output = np.array([9., 1.], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)

    # test 4: error input.
    with pytest.raises(TypeError):
        net(x, axis=0, keepdims=False, initial=9,
            where=None, return_indices=False)
        _pynative_executor.sync()

    with pytest.raises((TypeError, ValueError)):
        net(x, axis=0, keepdims=False, initial=ms.Tensor([False, True]), where=ms.Tensor([False, True]),
            return_indices=False)
        _pynative_executor.sync()

    with pytest.raises(TypeError):
        net(x, axis=0, keepdims=False, initial=9,
            where=ms.Tensor([False, True]), return_indices=1)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_method_min_dim(mode):
    """
    Feature: Functional.
    Description: Test tensor method overload Tensor.min(dim, keepdim=False)
    Expectation: Run success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = MinDimNet()

    x = ms.Tensor(np.arange(4).reshape((2, 2)).astype(np.float32))
    dim = 0
    keepdim = False
    if mode == 0:
        output = net(x, dim, keepdim)
        assert np.allclose(output.asnumpy(), np.array([0.0, 1.0]))
    else:
        output, index = net(x, dim, keepdim)
        assert np.allclose(output.asnumpy(), np.array([0.0, 1.0]))
        assert np.allclose(index.asnumpy(), np.array([0, 0]))


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_method_min_pyboost(mode):
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.min.
    Expectation: Run success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = MinPyboostNet()
    x = ms.Tensor(np.arange(4).reshape((2, 2)).astype(np.float32))
    output = net(x)
    assert np.allclose(output.asnumpy(), 0.0)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
def test_tensor_min_dynamic():
    """
    Feature: Test min op.
    Description: Test min dynamic shape.
    Expectation: the result match with expected result.
    """
    ms_data1 = ms.Tensor(generate_random_input((4, 3, 6), np.float32))
    axis1 = 1
    keepdims1 = False
    ms_data2 = ms.Tensor(generate_random_input((5, 2, 7, 3), np.float32))
    axis2 = 2
    keepdims2 = True
    TEST_OP(min_forward_func1, [[ms_data1], [ms_data2]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor'],
            case_config={'disable_tensor_dynamic_type': 'DYNAMIC_RANK',
                         'disable_resize': True})
    TEST_OP(min_forward_func2, [[ms_data1, axis1, keepdims1], [ms_data2, axis2, keepdims2]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor', 'ScalarTensor'],
            case_config={'disable_tensor_dynamic_type': 'DYNAMIC_RANK',
                         'disable_grad': True,
                         'disable_resize': True})
