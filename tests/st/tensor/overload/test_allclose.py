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
from mindspore import mint, Tensor, context
from tests.mark_utils import arg_mark


def allclose_forward_func(x, y, rtol, atol, equal_nan):
    return mint.allclose(x, y, rtol, atol, equal_nan)


def allclose_forward_func_tensor(x, y, rtol, atol, equal_nan):
    return x.allclose(y, rtol, atol, equal_nan)


def generate_random_input(*shape):
    """return an random integer array with parameter shape"""
    res = np.random.randint(low=1, high=5, size=shape)
    if isinstance(res, np.ndarray):
        return res.astype(np.float32)
    return float(res)


def compare_with_numpy_tensor(x, y, rtol=1e-05, atol=1e-08, equal_nan=False):
    ms_result = allclose_forward_func_tensor(
        Tensor(x), Tensor(y), rtol, atol, equal_nan)
    np_result = np.allclose(x, y, rtol, atol, equal_nan)
    return ms_result == np_result


def compare_with_numpy(x, y, rtol=1e-05, atol=1e-08, equal_nan=False):
    ms_result = allclose_forward_func(
        Tensor(x), Tensor(y), rtol, atol, equal_nan)
    np_result = np.allclose(x, y, rtol, atol, equal_nan)
    return ms_result == np_result


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_allclose_tensor(context_mode):
    """
    Feature: mint.allclose
    Description: Test cases for AllClose operator of different attributes.
    Expectation: The result match numpy allclose.
    """
    context.set_context(mode=context_mode)

    x = generate_random_input(2, 3, 4, 5)
    diff = (np.random.random((2, 3, 4, 5)).astype("float32") - 0.5) / 1000
    y = x + diff
    assert compare_with_numpy_tensor(x, y, atol=1e-3)
    assert compare_with_numpy_tensor(x, y, atol=1e-3, rtol=1e-4)
    assert compare_with_numpy_tensor(x, y, atol=1e-2, rtol=1e-6)
    assert compare_with_numpy_tensor(x, y, atol=2, rtol=1)

    x = generate_random_input(2, 3, 4, 5)
    y = generate_random_input(4, 5)
    assert compare_with_numpy_tensor(x, y)

    x = np.array(1.0).astype("float32")
    y = np.array(1.0 + 1e-8).astype("float32")
    assert compare_with_numpy_tensor(x, y, equal_nan=True)
