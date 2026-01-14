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
import mindspore.common.dtype as mstype


class IsCloseNetPython(nn.Cell):
    def construct(self, x, x2, rtol=1e-05, atol=1e-08, equal_nan=False):
        return x.isclose(x2, rtol, atol, equal_nan)


class IsCloseNet(nn.Cell):
    def construct(self, x, other, rtol=1e-05, atol=1e-08, equal_nan=False):
        return x.isclose(other, rtol, atol, equal_nan)


class IsCloseKVNet(nn.Cell):
    def construct(self, x, other, rtol=1e-05, atol=1e-08, equal_nan=False):
        return x.isclose(other, rtol=rtol, atol=atol, equal_nan=equal_nan)


class IsCloseKVDisruptNet(nn.Cell):
    def construct(self, x, other, rtol=1e-05, atol=1e-08, equal_nan=False):
        return x.isclose(other, equal_nan=equal_nan, atol=atol, rtol=rtol)


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


@test_utils.run_with_cell
def isclose_forward_func1(x, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    return x.isclose(other, rtol, atol, equal_nan)


@test_utils.run_with_cell
def isclose_forward_func2(x, x2, rtol=1e-05, atol=1e-08, equal_nan=False):
    return x.isclose(x2, rtol, atol, equal_nan)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_method_isclose(mode):
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.isclose.
    Expectation: Run success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})

    net = IsCloseNetPython()
    x = ms.Tensor([1.3, 2.1, 3.2, 4.1, 5.1, 6.1], dtype=mstype.float16)
    x2 = ms.Tensor([1.3, 3.3, 2.3, 3.1, 5.1, 5.6], dtype=mstype.float16)
    output = net(x, x2, 1e-05, 1e-08, True)
    expected = np.array([True, False, False, False, True, False], dtype=np.bool_)
    assert np.allclose(output.asnumpy(), expected)

    # test 1: using positional args
    net = IsCloseNet()
    x = ms.Tensor([1.3, 2.1, 3.2, 4.1, 5.1, 6.1], dtype=mstype.float16)
    y = ms.Tensor([1.3, 3.3, 2.3, 3.1, 5.1, 5.6], dtype=mstype.float16)
    output = net(x, y, 1e-05, 1e-08, True)
    expected = np.array([True, False, False, False, True, False], dtype=np.bool_)
    assert np.allclose(output.asnumpy(), expected)

    # test 2: using default args
    net = IsCloseNet()
    x = ms.Tensor([1.3, 2.1, 3.2, 4.1, 5.1, 6.1], dtype=mstype.float16)
    y = ms.Tensor([1.3, 3.3, 2.3, 3.1, 5.1, 5.6], dtype=mstype.float16)
    output = net(x, y)
    expected = np.array([True, False, False, False, True, False], dtype=np.bool_)
    assert np.allclose(output.asnumpy(), expected)

    # test 3: using k-v args
    net = IsCloseKVNet()
    x = ms.Tensor([1.3, 2.1, 3.2, 4.1, 5.1, 6.1], dtype=mstype.float16)
    y = ms.Tensor([1.3, 3.3, 2.3, 3.1, 5.1, 5.6], dtype=mstype.float16)
    output = net(x, y, rtol=1e-03, atol=1e-04, equal_nan=False)
    expected = np.array([True, False, False, False, True, False], dtype=np.bool_)
    assert np.allclose(output.asnumpy(), expected)

    # test 3: using k-v args
    net = IsCloseKVDisruptNet()
    x = ms.Tensor([1.3, 2.1, 3.2, 4.1, 5.1, 6.1], dtype=mstype.float16)
    y = ms.Tensor([1.3, 3.3, 2.3, 3.1, 5.1, 5.6], dtype=mstype.float16)
    output = net(x, y, rtol=1e-03, atol=1e-04, equal_nan=False)
    expected = np.array([True, False, False, False, True, False], dtype=np.bool_)
    assert np.allclose(output.asnumpy(), expected)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential',
          mem_peak=2.00)
def test_tensor_isclose_dynamic():
    """
    Feature: Test isclose op.
    Description: Test isclose dynamic shape.
    Expectation: the result match with expected result.
    """
    ms_data1 = ms.Tensor(generate_random_input((2, 3, 4, 5), np.float32))
    other1 = ms.Tensor(generate_random_input((2, 3, 4, 5), np.float32))
    x2_1 = ms.Tensor(generate_random_input((2, 3, 4, 5), np.float32))
    rtol1 = 1e-04
    atol1 = 1e-07
    equal_nan1 = True
    ms_data2 = ms.Tensor(generate_random_input((6, 2, 5), np.float32))
    other2 = ms.Tensor(generate_random_input((6, 2, 5), np.float32))
    x2_2 = ms.Tensor(generate_random_input((6, 2, 5), np.float32))
    rtol2 = 1e-05
    atol2 = 1e-08
    equal_nan2 = False
    TEST_OP(isclose_forward_func1,
            [[ms_data1, other1, rtol1, atol1, equal_nan1], [ms_data2, other2, rtol2, atol2, equal_nan2]],
            disable_mode=["GRAPH_MODE_GE"],
            case_config={'disable_grad': True, 'all_dim_zero': True})
    TEST_OP(isclose_forward_func2,
            [[ms_data1, x2_1, rtol1, atol1, equal_nan1], [ms_data2, x2_2, rtol2, atol2, equal_nan2]],
            disable_mode=["GRAPH_MODE_GE"],
            case_config={'disable_grad': True, 'all_dim_zero': True})
