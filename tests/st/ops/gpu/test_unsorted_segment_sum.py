# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
from mindspore.common import dtype as mstype
from mindspore.ops.operations import _inner_ops as inner
from mindspore.ops import operations as P


class UnsortedSegmentSumNet(nn.Cell):
    def __init__(self, num_segments):
        super(UnsortedSegmentSumNet, self).__init__()
        self.unsorted_segment_sum = P.UnsortedSegmentSum()
        self.num_segments = num_segments

    def construct(self, data, ids):
        return self.unsorted_segment_sum(data, ids, self.num_segments)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_1D():
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    input_x = Tensor([1, 2, 3, 4], mstype.float32)
    segment_ids = Tensor([0, 0, 1, 2], mstype.int32)
    num_segments = 4

    net = UnsortedSegmentSumNet(num_segments)
    output = net(input_x, segment_ids)
    expect = [3, 3, 4, 0]
    assert (output.asnumpy() == expect).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_2D():
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    input_x = Tensor([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]], mstype.float32)
    segment_ids = Tensor([2, 1, 1], mstype.int32)
    num_segments = 4

    net = UnsortedSegmentSumNet(num_segments)
    output = net(input_x, segment_ids)
    expect = [[0, 0, 0, 0],
              [14, 16, 18, 20],
              [1, 2, 3, 4],
              [0, 0, 0, 0]]
    assert (output.asnumpy() == expect).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_3D():
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    input_x = Tensor(np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3))
    segment_ids = Tensor([2, 1, 1, -1], mstype.int32)
    num_segments = 5

    net = UnsortedSegmentSumNet(num_segments)
    output = net(input_x, segment_ids)
    expect = [[[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]],
              [[45., 47., 49.],
               [51., 53., 55.],
               [57., 59., 61.],
               [63., 65., 67.],
               [69., 71., 73.]],
              [[0., 1., 2.],
               [3., 4., 5.],
               [6., 7., 8.],
               [9., 10., 11.],
               [12., 13., 14.]],
              [[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]],
              [[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]]]
    assert (output.asnumpy() == expect).all()


# Testing Dynamic Shape
class UnsortedSegmentSumDynNet(nn.Cell):
    def __init__(self, num_segments, dyn_a=True, dyn_b=True):
        super(UnsortedSegmentSumDynNet, self).__init__()
        self.unsorted_segment_sum = P.UnsortedSegmentSum()
        self.gpu_convert_to_dynamic_shape = inner.GpuConvertToDynamicShape()
        self.num_segments = num_segments
        self.to_dyn_1 = dyn_a
        self.to_dyn_2 = dyn_b

    def construct(self, data, ids):
        # testing selective inputs being dynamic
        if self.to_dyn_1:
            data = self.gpu_convert_to_dynamic_shape(data)
        if self.to_dyn_2:
            ids = self.gpu_convert_to_dynamic_shape(ids)
        return self.unsorted_segment_sum(data, ids, self.num_segments)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dyn_ab():
    """
    Tests for Dynamic shape with both inputs dynamic
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    num_segments = 4
    net = UnsortedSegmentSumDynNet(num_segments)
    # test 1
    input_x = Tensor([1, 2, 3, 4], mstype.float32)
    segment_ids = Tensor([0, 0, 1, 2], mstype.int32)
    output = net(input_x, segment_ids)
    expect = [3, 3, 4, 0]
    assert (output.asnumpy() == expect).all()
    # test 2
    input_x = Tensor([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]], mstype.float32)
    segment_ids = Tensor([2, 1, 1], mstype.int32)
    output = net(input_x, segment_ids)
    expect = [[0, 0, 0, 0],
              [14, 16, 18, 20],
              [1, 2, 3, 4],
              [0, 0, 0, 0]]
    assert (output.asnumpy() == expect).all()
    # test 3
    input_x = Tensor(np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3))
    segment_ids = Tensor([2, 1, 1, -1], mstype.int32)
    output = net(input_x, segment_ids)
    expect = [[[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]],
              [[45., 47., 49.],
               [51., 53., 55.],
               [57., 59., 61.],
               [63., 65., 67.],
               [69., 71., 73.]],
              [[0., 1., 2.],
               [3., 4., 5.],
               [6., 7., 8.],
               [9., 10., 11.],
               [12., 13., 14.]],
              [[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]]]
    assert (output.asnumpy() == expect).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dyn_a():
    """
    Tests for Dynamic shape with first input dynamic
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    num_segments = 6
    net = UnsortedSegmentSumDynNet(num_segments, True, False)
    # test 1
    input_x = Tensor([1, 2, 3, 4], mstype.float32)
    segment_ids = Tensor([0, 0, 1, 2], mstype.int32)
    output = net(input_x, segment_ids)
    expect = [3, 3, 4, 0, 0, 0]
    assert (output.asnumpy() == expect).all()
    # test 2
    input_x = Tensor([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]], mstype.float32)
    segment_ids = Tensor([2, 1, 1], mstype.int32)
    output = net(input_x, segment_ids)
    expect = [[0, 0, 0, 0],
              [14, 16, 18, 20],
              [1, 2, 3, 4],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]]
    assert (output.asnumpy() == expect).all()
    # test 3
    input_x = Tensor(np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3))
    segment_ids = Tensor([2, 1, 1, -1], mstype.int32)
    output = net(input_x, segment_ids)
    expect = [[[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]],
              [[45., 47., 49.],
               [51., 53., 55.],
               [57., 59., 61.],
               [63., 65., 67.],
               [69., 71., 73.]],
              [[0., 1., 2.],
               [3., 4., 5.],
               [6., 7., 8.],
               [9., 10., 11.],
               [12., 13., 14.]],
              [[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]],
              [[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]],
              [[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]]]
    assert (output.asnumpy() == expect).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dyn_b():
    """
    Tests for Dynamic shape with second input dynamic
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    num_segments = 6
    net = UnsortedSegmentSumDynNet(num_segments, False, True)
    # test 1
    input_x = Tensor([1, 2, 3, 4], mstype.float32)
    segment_ids = Tensor([0, 0, 1, 2], mstype.int32)
    output = net(input_x, segment_ids)
    expect = [3, 3, 4, 0, 0, 0]
    assert (output.asnumpy() == expect).all()
    # test 2
    input_x = Tensor([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]], mstype.float32)
    segment_ids = Tensor([2, 1, 1], mstype.int32)
    output = net(input_x, segment_ids)
    expect = [[0, 0, 0, 0],
              [14, 16, 18, 20],
              [1, 2, 3, 4],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]]
    assert (output.asnumpy() == expect).all()
    # test 3
    input_x = Tensor(np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3))
    segment_ids = Tensor([2, 1, 1, -1], mstype.int32)
    output = net(input_x, segment_ids)
    expect = [[[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]],
              [[45., 47., 49.],
               [51., 53., 55.],
               [57., 59., 61.],
               [63., 65., 67.],
               [69., 71., 73.]],
              [[0., 1., 2.],
               [3., 4., 5.],
               [6., 7., 8.],
               [9., 10., 11.],
               [12., 13., 14.]],
              [[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]],
              [[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]],
              [[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]]]
    assert (output.asnumpy() == expect).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32,
                                   np.int64, np.float16, np.float32, np.float64])
def test_uss_dtype(mode, dtype):
    """
    Feature: test ops.UnsortedSegmentSum forward.
    Description: inputs with different data type.
    Expectation: the result match with expect
    """
    context.set_context(mode=mode, device_target='GPU')
    x = np.arange(1, 10).reshape(3, 3).astype(dtype)
    input_x = Tensor(x)
    segment_ids = Tensor([0, 1, 0], mstype.int32)
    num_segments = 2
    net = UnsortedSegmentSumNet(num_segments)
    output = net(input_x, segment_ids)
    expect_np = np.array([[8, 10, 12], [4, 5, 6]], dtype=dtype)
    assert (output.asnumpy() == expect_np).all()
