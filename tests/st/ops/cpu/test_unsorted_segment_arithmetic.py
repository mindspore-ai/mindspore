# Copyright 2022 Huawei Technologies Co., Ltd
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
import mindspore.ops as ops
from mindspore.common import dtype as mstype
from mindspore.common import Tensor


context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


UnsortedSegmentArith_func_map = {
    "max": ops.UnsortedSegmentMax,
    "min": ops.UnsortedSegmentMin,
    "sum": ops.UnsortedSegmentSum,
    "prod": ops.UnsortedSegmentProd,
}


arith_np_func_map = {
    "prod": lambda a, b: a * b,
    "sum": lambda a, b: a + b,
    "max": np.maximum,
    "min": np.minimum,
}


def init_result(func, shape, dtype):
    result = np.ones(shape, dtype)
    if func == 'sum':
        result = np.zeros(shape, dtype)
    if func == 'min':
        if dtype == np.int32:
            result = result * np.iinfo(dtype).max
        if dtype == np.float32:
            result = result * np.finfo(dtype).max
    if func == 'max':
        if dtype == np.int32:
            result = result * np.iinfo(dtype).min
        if dtype == np.float32:
            result = result * np.finfo(dtype).min
    return result


class TestUnsortedSegmentArithmeticNet(nn.Cell):
    def __init__(self, func, num_segments):
        super(TestUnsortedSegmentArithmeticNet, self).__init__()
        self.arith_func = UnsortedSegmentArith_func_map.get(func)()
        self.num_segments = num_segments

    def construct(self, data, ids):
        return self.arith_func(data, ids, self.num_segments)


def unsorted_segment_arith_expected(func, x, segment_ids, num_segments):
    np_inp = x.asnumpy().copy()
    np_ids = segment_ids.asnumpy().copy()

    f = arith_np_func_map.get(func)

    inp_shape = np_inp.shape
    ids_shape = np_ids.shape
    cal_shape = inp_shape[len(ids_shape):]

    out_shape = np.concatenate(([num_segments], cal_shape), axis=0).astype(np.int32)
    result = init_result(func, out_shape, np_inp.dtype)

    inp_size = np_inp.size
    ids_size = np_ids.size
    cal_size = np.int32(result.size / num_segments)

    trans_inp_batch = np.int32(inp_size / cal_size)
    trans_inp_shape = np.concatenate(([trans_inp_batch], cal_shape), axis=0).astype(np.int32)

    trans_inp = np_inp.reshape(trans_inp_shape)
    trans_ids = np_ids.reshape(ids_size)


    for i in range(ids_size):
        out_index = trans_ids[i]
        if out_index < 0:
            continue
        if out_index >= num_segments:
            continue
        result[out_index] = f(result[out_index], trans_inp[i])

    return result


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', ['sum', 'min', 'max'])
@pytest.mark.parametrize('data_type', [mstype.float32, mstype.int32])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_unsorted_segment_arithmetic_one_d(func, data_type, index_type):
    """
    Feature: UnsortedSegment* operators.
    Description: test cases for UnsortedSegment* operator
    Expectation: the result match numpy implementation.
    """
    x = Tensor(np.array([1, 2, 3, 4]), data_type)
    segment_ids = Tensor(np.array([0, 0, 1, 2]), index_type)
    num_segments = 5

    net = TestUnsortedSegmentArithmeticNet(func, num_segments)
    graph_output = net(x, segment_ids)
    expected = unsorted_segment_arith_expected(func, x, segment_ids, num_segments)
    np.testing.assert_array_almost_equal(graph_output.asnumpy(), expected)



@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', ['sum'])
@pytest.mark.parametrize('data_type', [mstype.float32])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_unsorted_segment_arithmetic_mul_d(func, data_type, index_type):
    """
    Feature: UnsortedSegment* operators.
    Description: test cases for UnsortedSegment* operator
    Expectation: the result match numpy implementation.
    """
    x = Tensor(np.random.randint(0, 100, size=[2, 3, 4, 3, 2]), data_type)
    segment_ids = Tensor(np.random.randint(0, 5, size=[2, 3]), index_type)
    num_segments = 5

    net = TestUnsortedSegmentArithmeticNet(func, num_segments)
    graph_output = net(x, segment_ids)
    expected = unsorted_segment_arith_expected(func, x, segment_ids, num_segments)
    np.testing.assert_array_almost_equal(graph_output.asnumpy(), expected)
