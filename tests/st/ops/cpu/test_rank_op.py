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

from typing import List
from random import sample
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import PrimitiveWithInfer, prim_attr_register
from mindspore._checkparam import Validator as validator
from mindspore.common import dtype as mstype
import numpy as np
import pandas as pd
import pytest

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class Rank(PrimitiveWithInfer):
    """
        Shift op frontend implementation
    """

    # size_t axis_{0};
    # rank::Method method_{rank::MethodNotDefined};
    # rank::NaOption option_{rank::OptionNotDefined};
    # bool ascending_{true};
    # bool pct_{false};
    @prim_attr_register
    def __init__(self, axis: int, method: str, na_option: str, ascending: bool, pct: bool):
        """Initialize Sort"""
        self.axis = validator.check_value_type("axis", axis, [int], self.name)
        self.method = validator.check_value_type("method", method, [str], self.name)
        self.na_option = validator.check_value_type("na_option", na_option, [str], self.name)
        self.ascending = validator.check_value_type("ascending", ascending, [bool], self.name)
        self.pct = validator.check_value_type("pct", pct, [bool], self.name)

        self.init_prim_io_names(inputs=['x'], outputs=['output'])

    def __infer__(self, x):
        out_shapes = x['shape']
        return {
            'shape': tuple(out_shapes),
            'dtype': mstype.float32,
            'value': None
        }


class RankNet(nn.Cell):
    def __init__(self, axis: int, method: str, na_option: str, ascending: bool, pct: bool):
        super(RankNet, self).__init__()
        self.rank = Rank(axis, method, na_option, ascending, pct)

    def construct(self, x):
        return self.rank(x)


def pandas_rank(arr, **kwargs):
    ser = pd.DataFrame(arr)
    result = ser.rank(**kwargs)
    return result.to_numpy()


@pytest.mark.parametrize('shape', [(10,)])
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize('method', ['dense', 'first', 'max', 'min', 'average'])
@pytest.mark.parametrize('na_option', ["keep", "top", "bottom"])
@pytest.mark.parametrize('ascending', [True, False])
@pytest.mark.parametrize('pct', [False, True])
def test_rank_1d(shape: List[int], dtype, method: str, ascending: bool, pct: bool, na_option: str):
    np.random.seed(0)

    if dtype in (np.int32, np.int64):
        arr = np.random.randint(0, 100, size=shape).astype(dtype)
    else:
        arr = np.random.random(size=shape).astype(dtype)
        arr.flat[sample(range(arr.size), int(arr.size / 10))] = np.nan

    pd_result = pandas_rank(arr, method=method, ascending=ascending, pct=pct, na_option=na_option).flatten()
    rank = RankNet(0, method=method, ascending=ascending, pct=pct, na_option=na_option)
    mind_result = rank(Tensor(arr)).asnumpy()

    print('arr: \n', arr, arr.dtype, arr.shape)
    print('pandas: \n', pd_result, pd_result.dtype, pd_result.shape)
    print('mind: \n', mind_result, mind_result.dtype, mind_result.shape)
    print(f'method: {method}, ascending: {ascending}, pct: {pct} na_option: {na_option}')
    assert np.allclose(pd_result, mind_result, equal_nan=True)


@pytest.mark.parametrize('shape', [(5, 6)])
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize('method', ['dense', 'first', 'max', 'min', 'average'])
@pytest.mark.parametrize('na_option', ["keep", "top", "bottom"])
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('ascending', [True, False])
@pytest.mark.parametrize('pct', [False, True])
def test_rank_2d(shape: List[int], dtype, method: str, ascending: bool, pct: bool, axis: int, na_option: str):
    np.random.seed(0)

    if dtype in (np.int32, np.int64):
        arr = np.random.randint(0, 100, size=shape).astype(dtype)
    else:
        arr = np.random.random(size=shape).astype(dtype)
        arr.flat[sample(range(arr.size), int(arr.size / 10))] = np.nan

    pd_result = pandas_rank(arr, method=method, ascending=ascending, pct=pct, na_option=na_option, axis=axis)
    rank = RankNet(axis=axis, method=method, ascending=ascending, pct=pct, na_option=na_option)
    mind_result = rank(Tensor(arr)).asnumpy()

    print('arr: \n', arr, arr.dtype, arr.shape)
    print('pandas: \n', pd_result, pd_result.dtype, pd_result.shape)
    print('mind: \n', mind_result, mind_result.dtype, mind_result.shape)
    print(f'axis: {axis}, method: {method}, ascending: {ascending}, pct: {pct} na_option: {na_option}')
    assert np.allclose(pd_result, mind_result, equal_nan=True)
