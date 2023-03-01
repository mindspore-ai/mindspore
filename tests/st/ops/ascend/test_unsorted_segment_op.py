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

import pytest
import numpy as np
import mindspore.nn as nn
import mindspore.context as context
from mindspore.common import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P


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
        if dtype in [np.int32, np.uint8, np.int16, np.int8, np.int64, np.uint16, np.uint32, np.uint64]:
            result = result * np.iinfo(dtype).max
        if dtype in [np.float32, np.float64]:
            result = result * np.finfo(dtype).max
    if func == 'max':
        if dtype in [np.int32, np.uint8, np.int16, np.int8, np.int64, np.uint16, np.uint32, np.uint64]:
            result = result * np.iinfo(dtype).min
        if dtype in [np.float32, np.float64]:
            result = result * np.finfo(dtype).min
    return result


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
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', ['min', 'max', 'sum'])
def test_unsorted_segment_op(func):
    """
    Feature: test_unsorted_segment_op* operators.
    Description: test cases for test_unsorted_segment_op* operator
    Expectation: the result match numpy implementation.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    x = Tensor(np.array([1, 2, 3, 4]).astype(np.float32))
    segment_ids = Tensor(np.array([0, 0, 1, 2]).astype(np.int32))
    num_segments = 4

    if func == 'min':
        graph_output = P.UnsortedSegmentMin()(x, segment_ids, num_segments)
    if func == 'max':
        graph_output = P.UnsortedSegmentMax()(x, segment_ids, num_segments)
    if func == 'sum':
        graph_output = P.UnsortedSegmentSum()(x, segment_ids, num_segments)

    expected = unsorted_segment_arith_expected(func, x, segment_ids, num_segments)
    np.testing.assert_array_almost_equal(graph_output.asnumpy(), expected)



class TestUnsortedSegmentArithmeticNet(nn.Cell):
    def __init__(self, func, num_segments):
        super(TestUnsortedSegmentArithmeticNet, self).__init__()
        if func == 'min':
            self.func = P.UnsortedSegmentMin()
        if func == 'max':
            self.func = P.UnsortedSegmentMax()
        if func == 'sum':
            self.func = P.UnsortedSegmentSum()
        self.num_segments = num_segments

    def construct(self, x, segment_ids):
        return self.func(x, segment_ids, self.num_segments)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', ['min', 'max', 'sum'])
def test_unsorted_segment_op_dynamic_shape(func):
    """
    Feature: test_unsorted_segment_op_dynamic_shape.
    Description: test cases for tensor.segment_op* api
    Expectation: the result match numpy implementation.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    x = Tensor(np.random.randint(0, 100, size=[4, 3, 2]), mstype.float32)
    segment_ids = Tensor(np.random.randint(0, 5, size=[4]), mstype.int32)
    num_segments = 5

    net = TestUnsortedSegmentArithmeticNet(func, num_segments)
    x_dy = Tensor(shape=(4, 3, None), dtype=mstype.float32)
    net.set_inputs(x_dy, segment_ids)

    output = net(x, segment_ids)
    expected = unsorted_segment_arith_expected(func, x, segment_ids, num_segments)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)
