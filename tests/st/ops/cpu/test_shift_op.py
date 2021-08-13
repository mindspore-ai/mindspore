# Copyright 2021 Huawei Technologies Co., Ltd
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
import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import PrimitiveWithInfer, prim_attr_register
from mindspore._checkparam import Validator as validator
from mindspore.common import dtype as mstype

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class Shift(PrimitiveWithInfer):
    """
        Shift op frontend implementation
    """

    @prim_attr_register
    def __init__(self, periods=1, axis=-1, fill_value=np.nan):
        """Initialize Sort"""
        self.periods = validator.check_value_type("periods", periods, [int], self.name)
        self.axis = validator.check_value_type("axis", axis, [int], self.name)
        self.fill_value = validator.check_value_type("fill_value", fill_value, [float], self.name)

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid("x_dtype", x_dtype,
                                           [mstype.float32, mstype.float64, mstype.int32, mstype.int64, mstype.bool_],
                                           self.name)
        return x_dtype


class ShiftNet(nn.Cell):
    def __init__(self, periods=1, axis=-1, fill_value=np.nan):
        super(ShiftNet, self).__init__()
        self.shift = Shift(periods, axis, fill_value)

    def construct(self, x):
        return self.shift(x)


def numpy_shift(array: np.ndarray, periods: int, axis: int, fill_value=np.nan) -> np.ndarray:
    """
    numpy implementation for validation
    """
    size = array.shape[axis]
    assert periods in range(-size, size)
    assert axis in range(-array.ndim, array.ndim)

    copy_src_indices = [slice(None)] * array.ndim
    copy_dst_indices = [slice(None)] * array.ndim
    fill_indices = [slice(None)] * array.ndim

    if periods > 0:
        fill_indices[axis] = slice(None, periods)
        copy_src_indices[axis] = slice(None, -periods)
        copy_dst_indices[axis] = slice(periods, None)
    elif periods < 0:
        fill_indices[axis] = slice(periods, None)
        copy_src_indices[axis] = slice(-periods, None)
        copy_dst_indices[axis] = slice(None, periods)
    else:
        return array.copy()

    result = np.empty_like(array)
    result[tuple(fill_indices)] = fill_value
    result[tuple(copy_dst_indices)] = array[tuple(copy_src_indices)]

    return result


def compare(arr: np.ndarray, periods: int, axis: int, fill_value=np.nan):
    numpy_result = numpy_shift(arr, periods=periods, axis=axis, fill_value=fill_value)
    shift = ShiftNet(periods=periods, axis=axis, fill_value=fill_value)
    mindspore_result = shift(Tensor(arr)).asnumpy()

    print('numpy:\n')
    print(numpy_result)
    print('mindspore:\n')
    print(mindspore_result)
    assert np.allclose(numpy_result, mindspore_result, equal_nan=True)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('fill_value, dtype',
                         [(0.0, np.float32),
                          (0.0, np.float64),
                          (0.0, np.int32),
                          (0.0, np.int64),
                          (0.0, np.bool_),
                          (1.0, np.float32),
                          (1.0, np.float64),
                          (1.0, np.int32),
                          (1.0, np.int64),
                          (1.0, np.bool_),
                          (np.nan, np.float32),
                          (np.nan, np.float64),
                          (np.nan, np.bool_)]
                         )
def test_no_shift(fill_value, dtype):
    arr = np.random.random((40, 60, 50, 30)).astype(dtype)

    compare(arr, axis=0, periods=0, fill_value=fill_value)
    compare(arr, axis=1, periods=0, fill_value=fill_value)
    compare(arr, axis=2, periods=0, fill_value=fill_value)
    compare(arr, axis=3, periods=0, fill_value=fill_value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('fill_value, dtype',
                         [(0.0, np.float32),
                          (0.0, np.float64),
                          (0.0, np.int32),
                          (0.0, np.int64),
                          (0.0, np.bool_),
                          (1.0, np.float32),
                          (1.0, np.float64),
                          (1.0, np.int32),
                          (1.0, np.int64),
                          (1.0, np.bool_),
                          (np.nan, np.float32),
                          (np.nan, np.float64),
                          (np.nan, np.bool_)]
                         )
def test_fancy_1d(fill_value, dtype):
    arr = np.random.random((1, 1, 50, 1)).astype(dtype)

    axis = 2
    compare(arr, axis=axis, periods=-35, fill_value=fill_value)
    compare(arr, axis=axis, periods=28, fill_value=fill_value)

    arr = np.random.random((70, 1, 1, 1)).astype(dtype)

    axis = 0
    compare(arr, axis=axis, periods=-35, fill_value=fill_value)
    compare(arr, axis=axis, periods=28, fill_value=fill_value)

    arr = np.random.random((1, 1, 1, 80)).astype(dtype)

    axis = 3
    compare(arr, axis=axis, periods=-35, fill_value=fill_value)
    compare(arr, axis=axis, periods=28, fill_value=fill_value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('fill_value, dtype',
                         [(0.0, np.float32),
                          (0.0, np.float64),
                          (0.0, np.int32),
                          (0.0, np.int64),
                          (0.0, np.bool_),
                          (1.0, np.float32),
                          (1.0, np.float64),
                          (1.0, np.int32),
                          (1.0, np.int64),
                          (1.0, np.bool_),
                          (np.nan, np.float32),
                          (np.nan, np.float64),
                          (np.nan, np.bool_)]
                         )
def test_2d(fill_value, dtype):
    arr = np.random.random((30, 40)).astype(dtype)
    axis = 0
    compare(arr, axis=axis, periods=-24, fill_value=fill_value)
    compare(arr, axis=axis, periods=27, fill_value=fill_value)

    axis = 1
    compare(arr, axis=axis, periods=-35, fill_value=fill_value)
    compare(arr, axis=axis, periods=28, fill_value=fill_value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('fill_value, dtype',
                         [(0.0, np.float32),
                          (0.0, np.float64),
                          (0.0, np.int32),
                          (0.0, np.int64),
                          (0.0, np.bool_),
                          (1.0, np.float32),
                          (1.0, np.float64),
                          (1.0, np.int32),
                          (1.0, np.int64),
                          (1.0, np.bool_),
                          (np.nan, np.float32),
                          (np.nan, np.float64),
                          (np.nan, np.bool_)]
                         )
def test_4d(fill_value, dtype):
    arr = np.random.random((30, 40, 50, 60)).astype(dtype)

    axis = 0
    compare(arr, axis=axis, periods=-24, fill_value=fill_value)
    compare(arr, axis=axis, periods=28, fill_value=fill_value)

    axis = 1
    compare(arr, axis=axis, periods=-24, fill_value=fill_value)
    compare(arr, axis=axis, periods=34, fill_value=fill_value)

    axis = 2
    compare(arr, axis=axis, periods=-24, fill_value=fill_value)
    compare(arr, axis=axis, periods=48, fill_value=fill_value)

    axis = 3
    compare(arr, axis=axis, periods=-48, fill_value=fill_value)
    compare(arr, axis=axis, periods=52, fill_value=fill_value)
