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

import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common import dtype as mstype
from mindspore.common import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F

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
        if dtype in [
                np.int32, np.uint8, np.int16, np.int8, np.int64, np.uint16,
                np.uint32, np.uint64
        ]:
            result = result * np.iinfo(dtype).max
        if dtype in [np.float32, np.float64]:
            result = result * np.finfo(dtype).max
    if func == 'max':
        if dtype in [
                np.int32, np.uint8, np.int16, np.int8, np.int64, np.uint16,
                np.uint32, np.uint64
        ]:
            result = result * np.iinfo(dtype).min
        if dtype in [np.float32, np.float64]:
            result = result * np.finfo(dtype).min
    return result


class TestUnsortedSegmentArithmeticNet(nn.Cell):

    def __init__(self, func, num_segments):
        super(TestUnsortedSegmentArithmeticNet, self).__init__()
        self.arith_func = UnsortedSegmentArith_func_map.get(func)()
        self.num_segments = num_segments

    def construct(self, data, ids):
        return self.arith_func(data, ids, self.num_segments)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_unsorted_segment_max_dynamic_shape():
    """
    Feature: test UnsortedSegmentMax op in cpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    net = TestUnsortedSegmentArithmeticNet('max', num_segments=2)

    input_x_dyn = Tensor(shape=[3, None], dtype=ms.float32)
    segment_ids = Tensor([0, 1, 1], dtype=ms.int32)
    net.set_inputs(input_x_dyn, segment_ids)

    input_x = Tensor(
        np.array([[1, 2, 3], [4, 5, 6], [4, 2, 1]]).astype(np.float32))
    output = net(input_x, segment_ids)
    expect_shape = (2, 3)
    assert output.asnumpy().shape == expect_shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_unsorted_segment_min_dynamic_shape():
    """
    Feature: test UnsortedSegmentMin op in cpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    net = TestUnsortedSegmentArithmeticNet('min', num_segments=2)

    input_x_dyn = Tensor(shape=[3, None], dtype=ms.float32)
    segment_ids = Tensor([0, 1, 1], dtype=ms.int32)

    net.set_inputs(input_x_dyn, segment_ids)

    input_x = Tensor(
        np.array([[1, 2, 3], [4, 5, 6], [4, 2, 1]]).astype(np.float32))
    output = net(input_x, segment_ids)
    expect_shape = (2, 3)
    assert output.asnumpy().shape == expect_shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_unsorted_segment_prod_dynamic_shape():
    """
    Feature: test UnsortedSegmentProd op in cpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    net = TestUnsortedSegmentArithmeticNet('prod', num_segments=2)

    input_x_dyn = Tensor(shape=[3, None], dtype=ms.float32)
    segment_ids = Tensor([0, 1, 1], dtype=ms.int32)

    net.set_inputs(input_x_dyn, segment_ids)

    input_x = Tensor(
        np.array([[1, 2, 3], [4, 5, 6], [4, 2, 1]]).astype(np.float32))
    output = net(input_x, segment_ids)
    expect_shape = (2, 3)
    assert output.asnumpy().shape == expect_shape


def unsorted_segment_arith_expected(func, x, segment_ids, num_segments):
    np_inp = x.asnumpy().copy()
    np_ids = segment_ids.asnumpy().copy()

    f = arith_np_func_map.get(func)

    inp_shape = np_inp.shape
    ids_shape = np_ids.shape
    cal_shape = inp_shape[len(ids_shape):]

    out_shape = np.concatenate(([num_segments], cal_shape),
                               axis=0).astype(np.int32)
    result = init_result(func, out_shape, np_inp.dtype)

    inp_size = np_inp.size
    ids_size = np_ids.size
    cal_size = np.int32(result.size / num_segments)

    trans_inp_batch = np.int32(inp_size / cal_size)
    trans_inp_shape = np.concatenate(([trans_inp_batch], cal_shape),
                                     axis=0).astype(np.int32)

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
@pytest.mark.parametrize('func', ['min', 'max', 'sum'])
@pytest.mark.parametrize('data_type', [mstype.float32, mstype.int32])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_unsorted_segment_arithmetic_one_d(func, data_type, index_type):
    """
    Feature: UnsortedSegment* operators.
    Description: test cases for UnsortedSegment* operator
    Expectation: the result match numpy implementation.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    x = Tensor(np.array([1, 2, 3, 4]), data_type)
    segment_ids = Tensor(np.array([0, 0, 1, 2]), index_type)
    num_segments = 5

    net = TestUnsortedSegmentArithmeticNet(func, num_segments)
    graph_output = net(x, segment_ids)
    expected = unsorted_segment_arith_expected(func, x, segment_ids,
                                               num_segments)
    np.testing.assert_array_almost_equal(graph_output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', ['min', 'max'])
def test_unsorted_segment_arithmetic_error(func):
    """
    Feature: UnsortedSegment* operators.
    Description: test cases for UnsortedSegment* operator
    Expectation: the result match numpy implementation.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    x = Tensor(np.array([1, 2, 3, 4]), mstype.float32)
    segment_ids = Tensor(np.array([0, 0, 1, 2]), mstype.int32)
    num_segments = 2

    net = TestUnsortedSegmentArithmeticNet(func, num_segments)

    with pytest.raises(RuntimeError):
        net(x, segment_ids)


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
    context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
    x = Tensor(np.random.randint(0, 100, size=[2, 3, 4, 3, 2]), data_type)
    segment_ids = Tensor(np.random.randint(0, 5, size=[2, 3]), index_type)
    num_segments = 5

    net = TestUnsortedSegmentArithmeticNet(func, num_segments)
    graph_output = net(x, segment_ids)
    expected = unsorted_segment_arith_expected(func, x, segment_ids,
                                               num_segments)
    np.testing.assert_array_almost_equal(graph_output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', ['min', 'max'])
def test_tensor_check(func):
    """
    Feature: test_tensor_check.
    Description: test cases for tensor func
    Expectation: raise TypeError.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    x = Tensor(np.array([1, 2, 3, 4]), mstype.float32)
    segment_ids = Tensor(np.array([0, 0, 1, 2]), mstype.int32)
    num_segments = 5

    if func == 'min':
        output_ms = x.unsorted_segment_min(segment_ids, num_segments)
    if func == 'max':
        output_ms = x.unsorted_segment_max(segment_ids, num_segments)
    if func == 'sum':
        output_ms = x.unsorted_segment_sum(segment_ids, num_segments)

    expected = unsorted_segment_arith_expected(func, x, segment_ids,
                                               num_segments)
    np.testing.assert_array_almost_equal(output_ms.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', ['min', 'max', 'sum'])
def test_functional_check(func):
    """
    Feature: test_functional_check.
    Description: test cases for functional func.
    Expectation: raise TypeError.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    x = Tensor(np.array([1, 2, 3, 4]), mstype.float32)
    segment_ids = Tensor(np.array([0, 0, 1, 2]), mstype.int32)
    num_segments = 5

    if func == 'min':
        output_ms = F.unsorted_segment_min(x, segment_ids, num_segments)
    if func == 'max':
        output_ms = F.unsorted_segment_max(x, segment_ids, num_segments)
    if func == 'sum':
        output_ms = F.unsorted_segment_sum(x, segment_ids, num_segments)

    expected = unsorted_segment_arith_expected(func, x, segment_ids,
                                               num_segments)
    np.testing.assert_array_almost_equal(output_ms.asnumpy(), expected)


def min_vmap_graph(x, segment_ids, num_segments):
    return P.UnsortedSegmentMin()(x, segment_ids, num_segments)


def max_vmap_graph(x, segment_ids, num_segments):
    return P.UnsortedSegmentMax()(x, segment_ids, num_segments)


def sum_vmap_graph(x, segment_ids, num_segments):
    return P.UnsortedSegmentSum()(x, segment_ids, num_segments)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', ['min', 'max', 'sum'])
def test_vmap(func):
    """
    Feature: test_vmap.
    Description: in_axes : 0, None, None
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

    x = Tensor(np.random.rand(2, 3, 4, 5).astype(np.float32))
    segment_ids = Tensor(np.random.randint(0, 5, size=[4]), mstype.int32)
    num_segments = 5
    in_axes = (0, None, None)
    out_axes = 0

    if func == 'min':
        vmap_graph = min_vmap_graph
    if func == 'max':
        vmap_graph = max_vmap_graph
    if func == 'sum':
        vmap_graph = sum_vmap_graph

    vmap_round_net = ops.vmap(ops.vmap(vmap_graph, in_axes, out_axes), in_axes,
                              out_axes)
    output = vmap_round_net(x, segment_ids, num_segments)

    expected = []
    for i in range(0, 2):
        for j in range(0, 3):
            x_s = x[i, j]
            out_s = unsorted_segment_arith_expected(func, x_s, segment_ids,
                                                    num_segments)
            expected.append(out_s)

    output_shape = (2, 3, 5, 5)
    expected = np.array(expected).reshape(output_shape)
    np.testing.assert_allclose(output.asnumpy(), expected, rtol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', ['min', 'max', 'sum'])
def test_vmap2(func):
    """
    Feature: test_vmap.
    Description: in_axes : 0, 0, None
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

    x = Tensor(np.random.rand(2, 3, 4, 5).astype(np.float32))
    segment_ids = Tensor(np.random.randint(0, 5, size=[2, 3, 4]), mstype.int32)
    num_segments = 5
    in_axes = (0, 0, None)
    out_axes = 0

    if func == 'min':
        vmap_graph = min_vmap_graph
    if func == 'max':
        vmap_graph = max_vmap_graph
    if func == 'sum':
        vmap_graph = sum_vmap_graph

    vmap_round_net = ops.vmap(ops.vmap(vmap_graph, in_axes, out_axes), in_axes,
                              out_axes)
    output = vmap_round_net(x, segment_ids, num_segments)

    expected = []
    for i in range(0, 2):
        for j in range(0, 3):
            x_s = x[i, j]
            ids_s = segment_ids[i, j]
            out_s = unsorted_segment_arith_expected(func, x_s, ids_s,
                                                    num_segments)
            expected.append(out_s)

    output_shape = (2, 3, 5, 5)
    expected = np.array(expected).reshape(output_shape)
    np.testing.assert_allclose(output.asnumpy(), expected, rtol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', ['sum'])
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('data_type', [
    mstype.uint8, mstype.uint16, mstype.uint32, mstype.uint64, mstype.int8,
    mstype.int16, mstype.int32, mstype.int64, mstype.float16, mstype.float32,
    mstype.float64
])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_unsorted_segment_arithmetic_dytpe(mode, func, data_type, index_type):
    """
    Feature: UnsortedSegmentSum operators dtype test.
    Description: test cases for UnsortedSegmentSum operator
    Expectation: the result match numpy implementation.
    """
    context.set_context(mode=mode, device_target='CPU')
    x = Tensor(np.random.randint(0, 100, size=[2, 3, 4, 3, 2]), data_type)
    segment_ids = Tensor(np.random.randint(0, 5, size=[2, 3]), index_type)
    num_segments = 5

    net = TestUnsortedSegmentArithmeticNet(func, num_segments)
    graph_output = net(x, segment_ids)
    expected = unsorted_segment_arith_expected(func, x, segment_ids,
                                               num_segments)
    np.testing.assert_array_almost_equal(graph_output.asnumpy(), expected)
