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

from functools import partial
from typing import Tuple, List
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import PrimitiveWithInfer, prim_attr_register
from mindspore import _checkparam as validator
from mindspore.common import dtype as mstype
import numpy as np
import pytest

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class Rolling(PrimitiveWithInfer):
    """
        Shift op frontend implementation
    """

    @prim_attr_register
    def __init__(self, window: int, min_periods: int, center: bool, axis: int, closed: str,
                 method: str):
        """Initialize Sort"""
        self.window = validator.check_value_type("window", window, [int], self.name)
        self.min_periods = validator.check_value_type("min_periods", min_periods, [int], self.name)
        self.center = validator.check_value_type("center", center, [bool], self.name)
        self.axis = validator.check_value_type("axis", axis, [int], self.name)
        self.closed = validator.check_value_type("closed", closed, [str], self.name)
        self.method = validator.check_value_type("method", method, [str], self.name)

        self.init_prim_io_names(inputs=['x'], outputs=['output'])

    def __infer__(self, x):
        out_shapes = x['shape']
        return {
            'shape': tuple(out_shapes),
            'dtype': x['dtype'],
            'value': None
        }

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid(x_dtype, [mstype.float32, mstype.float64, mstype.int32, mstype.int64],
                                           self.name, True)
        return x_dtype


class RollingNet(nn.Cell):
    def __init__(self, window: int, min_periods: int, center: bool, axis: int, closed: str,
                 method: str):
        super(RollingNet, self).__init__()
        self.rolling = Rolling(window, min_periods, center, axis, closed, method)

    def construct(self, x):
        return self.rolling(x)


def get_window_bounds(num_values: int, window_size: int, center: bool, closed: str = 'right') -> Tuple[List, List]:
    assert closed in {'left', 'both', 'right', 'neither'}
    offset = (window_size - 1) // 2 if center else 0

    end = np.arange(offset + 1, num_values + 1 + offset, dtype=np.int64)
    start = end - window_size
    if closed in {'left', 'both'}:
        start -= 1
    if closed in {'left', 'neither'}:
        end -= 1

    end = np.clip(end, 0, num_values)
    start = np.clip(start, 0, num_values)

    return list(start), list(end)


def numpy_rolling(array: np.ndarray, window: int, min_periods: int, center: bool, axis: int, closed: str,
                  method: str) -> np.ndarray:
    assert window > 0
    assert 0 < min_periods <= window
    assert axis in range(-array.ndim, array.ndim)
    reduce_map = {'max': np.max, 'min': np.min, 'mean': np.mean, 'sum': np.sum, 'std': partial(np.std, ddof=1),
                  'var': partial(np.var, ddof=1)}
    assert method in reduce_map

    size = array.shape[axis]
    start, end = get_window_bounds(size, window, center, closed)

    rolling_indices = [[slice(None)] * array.ndim for _ in range(len(start))]
    for i, j, indice in zip(start, end, rolling_indices):
        indice[axis] = None if j - i < min_periods else slice(i, j)
        # print(f'i={i}, j={j}, index={index}, indice={rolling_indices[index][axis]}')

    shape = list(array.shape)
    shape[axis] = 1
    nan_array = np.empty(shape)
    if array.dtype == np.float32 or array.dtype == np.float64:
        nan_array[:] = np.nan
    elif array.dtype == np.int32 or array.dtype == np.int64:
        nan_array[:] = 0

    arrays = [
        nan_array.copy() if not indice[axis]
        else reduce_map[method](array[tuple(indice)], axis=axis, keepdims=True).reshape(shape)
        for indice in rolling_indices]

    return np.stack(arrays, axis=axis).reshape(array.shape).astype(array.dtype)


@pytest.mark.parametrize('shape', [(10, 8, 15, 7), (5, 3, 8, 10)])
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize('window, min_periods', [(3, 3), (5, 3)])
@pytest.mark.parametrize('center', [True, False])
@pytest.mark.parametrize('axis', [2, 3, -1])
@pytest.mark.parametrize('closed', ['left', 'both', 'right', 'neither'])
@pytest.mark.parametrize('method', ['max', 'min', 'mean', 'sum', 'std', 'var'])
def test_two_way(shape: List[int], dtype, window: int, min_periods: int, center: bool, axis: int, closed: str,
                 method: str) -> np.ndarray:
    if dtype in (np.int32, np.int64):
        arr = np.random.randint(0, 100, size=shape)
    else:
        arr = np.random.random(shape).astype(dtype)
    expect_result = numpy_rolling(arr, window=window, min_periods=min_periods, center=center, axis=axis, closed=closed,
                                  method=method)
    rolling = RollingNet(window=window, min_periods=min_periods, center=center, axis=axis, closed=closed,
                         method=method)
    actual_result = rolling(Tensor(arr)).asnumpy()
    print('arr: \n', arr, arr.dtype, arr.shape)
    print('np: \n', expect_result, expect_result.dtype, expect_result.shape)
    print('mine: \n', actual_result, actual_result.dtype, actual_result.shape)
    print(f'center: {center}, axis: {axis}, method: {method}')
    assert np.allclose(expect_result, actual_result, equal_nan=True)
