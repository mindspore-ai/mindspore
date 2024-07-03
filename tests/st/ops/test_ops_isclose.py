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
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, Tensor, context
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


@test_utils.run_with_cell
def isclose_forward_func(x, y, rtol, atol, equal_nan):
    return ops.isclose(x, y, rtol, atol, equal_nan)


def generate_random_input(*shape):
    """return an random integer array with parameter shape"""
    res = np.random.randint(low=1, high=5, size=shape)
    if isinstance(res, np.ndarray):
        return res.astype(np.float32)
    return float(res)


def compare_with_numpy(x, y, rtol=1e-05, atol=1e-08, equal_nan=False):
    ms_result = isclose_forward_func(Tensor(x), Tensor(y), rtol, atol, equal_nan)
    np_result = np.isclose(x, y, rtol, atol, equal_nan)
    print(ms_result)
    print(np_result)
    return np.array_equal(ms_result, np_result)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_isclose_dtype(context_mode):
    """
    Feature: ops.isclose.
    Description: Test cases for IsClose operator of different dtypes.
    Expectation: The result match numpy isclose.
    """
    context.set_context(mode=context_mode)
    x = np.array([[1., -1., 2.], [3.1, 2, 1.]], dtype=np.float16)
    y = np.array([[1.2, -1., 2.1], [3., 2., 1.1]], dtype=np.float16)
    assert compare_with_numpy(x, y)
    x = np.array([[1., -1., 2.], [3.1, 2, 1.]], dtype=np.float32)
    y = np.array([[1.2, -1., 2.1], [3., 2., 1.1]], dtype=np.float32)
    assert compare_with_numpy(x, y)
    x = np.array([[1., -1., 2.], [3.1, 2, 1.]], dtype=np.float64)
    y = np.array([[1.2, -1., 2.1], [3., 2., 1.1]], dtype=np.float64)
    assert compare_with_numpy(x, y)
    x = np.array([[1, -1, 2], [3, 2, 1]], dtype=np.int8)
    y = np.array([[6, -1., 2], [3, 3, 2]], dtype=np.int8)
    assert compare_with_numpy(x, y)
    x = np.array([[1, -1, 2], [3, 2, 1]], dtype=np.int16)
    y = np.array([[6, -1., 2], [3, 3, 2]], dtype=np.int16)
    assert compare_with_numpy(x, y)
    x = np.array([[1, -1, 2], [3, 2, 1]], dtype=np.int32)
    y = np.array([[6, -1., 2], [3, 3, 2]], dtype=np.int32)
    assert compare_with_numpy(x, y)
    x = np.array([[1, -1, 2], [3, 2, 1]], dtype=np.int64)
    y = np.array([[6, -1., 2], [3, 3, 2]], dtype=np.int64)
    assert compare_with_numpy(x, y)
    x = np.array([[1, -1, 2], [3, 2, 1]], dtype=np.uint8)
    y = np.array([[6, -1., 2], [3, 3, 2]], dtype=np.uint8)
    assert compare_with_numpy(x, y)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_isclose_attr(context_mode):
    """
    Feature: ops.isclose
    Description: Test cases for IsClose operator of different attributes.
    Expectation: The result match numpy isclose.
    """
    context.set_context(mode=context_mode)

    x = generate_random_input(2, 3, 4, 5)
    diff = (np.random.random((2, 3, 4, 5)).astype("float32") - 0.5) / 1000
    y = x + diff
    assert compare_with_numpy(x, y, atol=1e-3)
    assert compare_with_numpy(x, y, atol=1e-3, rtol=1e-4)
    assert compare_with_numpy(x, y, atol=1e-2, rtol=1e-6)
    assert compare_with_numpy(x, y, atol=2, rtol=1)

    x = generate_random_input(2, 3, 4, 5)
    y = generate_random_input(4, 5)
    assert compare_with_numpy(x, y)

    x = np.array(1.0).astype("float32")
    y = np.array(1.0 + 1e-8).astype("float32")
    assert compare_with_numpy(x, y, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_isclose_dynamic_shape_testop():
    """
    Feature: Test isclose with dynamic shape in graph mode using TEST_OP.
    Description: call ops.isclose with valid input and index.
    Expectation: return the correct value.
    """
    x1 = generate_random_input(3, 4, 5)
    y1 = generate_random_input(3, 4, 5)
    x2 = generate_random_input(3, 7, 8, 3)
    y2 = generate_random_input(3, 7, 8, 3)
    rtol = 1e-05
    atol = 1e-08
    equal_nan = False
    TEST_OP(isclose_forward_func,
            [[Tensor(x1), Tensor(y1), rtol, atol, equal_nan], [Tensor(x2), Tensor(y2), rtol, atol, equal_nan]],
            '', disable_input_check=True, disable_yaml_check=True, disable_grad=True)
