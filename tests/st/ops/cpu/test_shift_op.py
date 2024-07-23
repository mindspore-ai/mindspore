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
from tests.mark_utils import arg_mark
import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import PrimitiveWithInfer, prim_attr_register
from mindspore import _checkparam as validator
from mindspore.common import dtype as mstype

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class Shift(PrimitiveWithInfer):
    """
        Shift op frontend implementation
    """

    @prim_attr_register
    def __init__(self, periods=1, axis=-1):
        """Initialize Sort"""
        self.periods = validator.check_value_type("periods", periods, [int], self.name)
        self.axis = validator.check_value_type("axis", axis, [int], self.name)
        self.init_prim_io_names(inputs=['x', 'fill_value'], outputs=['output'])

    def __infer__(self, x, fill_value):
        out_shapes = x['shape']
        return {
            'shape': tuple(out_shapes),
            'dtype': x['dtype'],
            'value': None
        }

    def infer_dtype(self, x_dtype, fill_value_type):
        validator.check_scalar_or_tensor_types_same({"x_dtype": x_dtype, "fill_value": fill_value_type},
                                                    [mstype.float32, mstype.float64, mstype.int32, mstype.int64,
                                                     mstype.bool_],
                                                    self.name, True)
        return x_dtype


class ShiftNet(nn.Cell):
    def __init__(self, periods=1, axis=-1):
        super(ShiftNet, self).__init__()
        self.shift = Shift(periods, axis)

    def construct(self, x, fill_value):
        return self.shift(x, fill_value)


def numpy_shift(array: np.ndarray, periods: int, axis: int, fill_value=np.nan) -> np.ndarray:
    """
    numpy implementation for validation
    """
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
    shift = ShiftNet(periods=periods, axis=axis)
    mindspore_result = shift(Tensor(arr), fill_value=fill_value).asnumpy()

    print('numpy:\n')
    print(numpy_result)
    print('mindspore:\n')
    print(mindspore_result)
    assert np.allclose(numpy_result, mindspore_result, equal_nan=True)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('dtype, fill_value',
                         [(np.float32, 0.0), (np.float32, 5.3), (np.float32, -5.5), (np.float32, np.nan),
                          (np.float64, 0.0), (np.float64, 5.3), (np.float64, -5.5), (np.float64, np.nan),
                          (np.int32, 0), (np.int32, 1), (np.int32, 5), (np.int32, -4),
                          (np.int64, 0), (np.int64, 1), (np.int64, 5), (np.int64, -4),
                          (np.bool_, True), (np.bool_, False)])
@pytest.mark.parametrize('axis', [0, 1, 2, 3])
def test_no_shift(fill_value, dtype, axis):
    arr = np.random.random((4, 6, 5, 3)).astype(dtype)
    compare(arr, axis=axis, periods=0, fill_value=fill_value)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('dtype, fill_value',
                         [(np.float32, 0.0), (np.float32, 5.3), (np.float32, -5.5), (np.float32, np.nan),
                          (np.float64, 0.0), (np.float64, 5.3), (np.float64, -5.5), (np.float64, np.nan),
                          (np.int32, 0), (np.int32, 1), (np.int32, 5), (np.int32, -4),
                          (np.int64, 0), (np.int64, 1), (np.int64, 5), (np.int64, -4),
                          (np.bool_, True), (np.bool_, False)])
@pytest.mark.parametrize('periods', [-35, 18, 25])
def test_fancy_1d(fill_value, dtype, periods):
    arr = np.random.random((1, 1, 20, 1)).astype(dtype)
    compare(arr, axis=2, periods=periods, fill_value=fill_value)

    arr = np.random.random((30, 1, 1, 1)).astype(dtype)
    compare(arr, axis=0, periods=periods, fill_value=fill_value)

    arr = np.random.random((1, 1, 1, 30)).astype(dtype)
    compare(arr, axis=3, periods=periods, fill_value=fill_value)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('dtype, fill_value',
                         [(np.float32, 0.0), (np.float32, 5.3), (np.float32, -5.5), (np.float32, np.nan),
                          (np.float64, 0.0), (np.float64, 5.3), (np.float64, -5.5), (np.float64, np.nan),
                          (np.int32, 0), (np.int32, 1), (np.int32, 5), (np.int32, -4),
                          (np.int64, 0), (np.int64, 1), (np.int64, 5), (np.int64, -4),
                          (np.bool_, True), (np.bool_, False)])
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('periods', [-3, 7, -5, 8, 9])
def test_2d(fill_value, dtype, axis, periods):
    arr = np.random.random((10, 10)).astype(dtype)
    compare(arr, axis=axis, periods=periods, fill_value=fill_value)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('dtype, fill_value',
                         [(np.float32, 0.0), (np.float32, 5.3), (np.float32, -5.5), (np.float32, np.nan),
                          (np.float64, 0.0), (np.float64, 5.3), (np.float64, -5.5), (np.float64, np.nan),
                          (np.int32, 0), (np.int32, 1), (np.int32, 5), (np.int32, -4),
                          (np.int64, 0), (np.int64, 1), (np.int64, 5), (np.int64, -4),
                          (np.bool_, True), (np.bool_, False)])
@pytest.mark.parametrize('axis', [0, 1, 2, 3])
@pytest.mark.parametrize('periods', [-30, 30, -45, 55])
def test_4d(fill_value, dtype, axis, periods):
    arr = np.random.random((30, 40, 10, 20)).astype(dtype)
    compare(arr, axis=axis, periods=periods, fill_value=fill_value)
