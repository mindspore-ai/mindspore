# Copyright 2020 Huawei Technologies Co., Ltd
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

import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import _inner_ops as inner
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P


class UnsortedSegmentMinNet(nn.Cell):
    def __init__(self, num_segments):
        super(UnsortedSegmentMinNet, self).__init__()
        self.unsorted_segment_min = P.UnsortedSegmentMin()
        self.num_segments = num_segments

    def construct(self, data, ids):
        return self.unsorted_segment_min(data, ids, self.num_segments)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_1d_int32():
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    input_x = Tensor([1, 2, 3, 4], mstype.int32)
    segment_ids = Tensor([0, 0, 1, 2], mstype.int32)
    num_segments = 4
    net = UnsortedSegmentMinNet(num_segments)
    output = net(input_x, segment_ids).asnumpy()
    expect = np.array([1, 3, 4, 2147483647]).astype(np.int32)
    np.testing.assert_array_almost_equal(output, expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_2d_int32():
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    input_x = Tensor([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]], mstype.int32)
    segment_ids = Tensor([2, 1, 1], mstype.int32)
    num_segments = 4
    net = UnsortedSegmentMinNet(num_segments)
    output = net(input_x, segment_ids).asnumpy()
    expect = np.array([[2147483647, 2147483647, 2147483647, 2147483647],
                       [5, 6, 7, 8],
                       [1, 2, 3, 4],
                       [2147483647, 2147483647, 2147483647, 2147483647]]).astype(np.int32)
    np.testing.assert_array_almost_equal(output, expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_3d_float16():
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    input_x = Tensor(np.arange(
        4 * 5 * 3, dtype=np.float16).reshape(4, 5, 3), dtype=mindspore.float16)
    segment_ids = Tensor([2, 1, 1, -1], mstype.int32)
    num_segments = 5
    net = UnsortedSegmentMinNet(num_segments)
    output = net(input_x, segment_ids).asnumpy()
    expect = np.array([[[6.55e+04, 6.55e+04, 6.55e+04],
                        [6.55e+04, 6.55e+04, 6.55e+04],
                        [6.55e+04, 6.55e+04, 6.55e+04],
                        [6.55e+04, 6.55e+04, 6.55e+04],
                        [6.55e+04, 6.55e+04, 6.55e+04]],
                       [[1.50e+01, 1.60e+01, 1.70e+01],
                        [1.80e+01, 1.90e+01, 2.00e+01],
                        [2.10e+01, 2.20e+01, 2.30e+01],
                        [2.40e+01, 2.50e+01, 2.60e+01],
                        [2.70e+01, 2.80e+01, 2.90e+01]],
                       [[0.00e+00, 1.00e+00, 2.00e+00],
                        [3.00e+00, 4.00e+00, 5.00e+00],
                        [6.00e+00, 7.00e+00, 8.00e+00],
                        [9.00e+00, 1.00e+01, 1.10e+01],
                        [1.20e+01, 1.30e+01, 1.40e+01]],
                       [[6.55e+04, 6.55e+04, 6.55e+04],
                        [6.55e+04, 6.55e+04, 6.55e+04],
                        [6.55e+04, 6.55e+04, 6.55e+04],
                        [6.55e+04, 6.55e+04, 6.55e+04],
                        [6.55e+04, 6.55e+04, 6.55e+04]],
                       [[6.55e+04, 6.55e+04, 6.55e+04],
                        [6.55e+04, 6.55e+04, 6.55e+04],
                        [6.55e+04, 6.55e+04, 6.55e+04],
                        [6.55e+04, 6.55e+04, 6.55e+04],
                        [6.55e+04, 6.55e+04, 6.55e+04]]]).astype(np.float16)
    np.testing.assert_array_almost_equal(output, expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_3d_float32():
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    input_x = Tensor(np.arange(
        4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3), dtype=mindspore.float32)
    segment_ids = Tensor([2, 1, 1, -1], mstype.int32)
    num_segments = 3
    net = UnsortedSegmentMinNet(num_segments)
    output = net(input_x, segment_ids).asnumpy()
    expect = np.array([[[3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38, 3.4028235e+38]],
                       [[1.5000000e+01, 1.6000000e+01, 1.7000000e+01],
                        [1.8000000e+01, 1.9000000e+01, 2.0000000e+01],
                        [2.1000000e+01, 2.2000000e+01, 2.3000000e+01],
                        [2.4000000e+01, 2.5000000e+01, 2.6000000e+01],
                        [2.7000000e+01, 2.8000000e+01, 2.9000000e+01]],
                       [[0.0000000e+00, 1.0000000e+00, 2.0000000e+00],
                        [3.0000000e+00, 4.0000000e+00, 5.0000000e+00],
                        [6.0000000e+00, 7.0000000e+00, 8.0000000e+00],
                        [9.0000000e+00, 1.0000000e+01, 1.1000000e+01],
                        [1.2000000e+01, 1.3000000e+01, 1.4000000e+01]]]).astype(np.float32)
    np.testing.assert_array_almost_equal(output, expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_3d_single_init():
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    input_x = Tensor(np.arange(
        4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3), dtype=mindspore.float32)
    segment_ids = Tensor([3, 0, 1, -1], mstype.int32)
    net = P.UnsortedSegmentMin()

    num_segments = 4
    output = net(input_x, segment_ids, num_segments).asnumpy()
    expect = np.array([[[1.5000000e+01, 1.6000000e+01, 1.7000000e+01],
                        [1.8000000e+01, 1.9000000e+01, 2.0000000e+01],
                        [2.1000000e+01, 2.2000000e+01, 2.3000000e+01],
                        [2.4000000e+01, 2.5000000e+01, 2.6000000e+01],
                        [2.7000000e+01, 2.8000000e+01, 2.9000000e+01]],
                       [[3.0000000e+01, 3.1000000e+01, 3.2000000e+01],
                        [3.3000000e+01, 3.4000000e+01, 3.5000000e+01],
                        [3.6000000e+01, 3.7000000e+01, 3.8000000e+01],
                        [3.9000000e+01, 4.0000000e+01, 4.1000000e+01],
                        [4.2000000e+01, 4.3000000e+01, 4.4000000e+01]],
                       [[3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38, 3.4028235e+38]],
                       [[0.0000000e+00, 1.0000000e+00, 2.0000000e+00],
                        [3.0000000e+00, 4.0000000e+00, 5.0000000e+00],
                        [6.0000000e+00, 7.0000000e+00, 8.0000000e+00],
                        [9.0000000e+00, 1.0000000e+01, 1.1000000e+01],
                        [1.2000000e+01, 1.3000000e+01, 1.4000000e+01]]]).astype(np.float32)
    np.testing.assert_array_almost_equal(output, expect)

    num_segments = 6
    output = net(input_x, segment_ids, num_segments).asnumpy()
    expect = np.array([[[1.5000000e+01, 1.6000000e+01, 1.7000000e+01],
                        [1.8000000e+01, 1.9000000e+01, 2.0000000e+01],
                        [2.1000000e+01, 2.2000000e+01, 2.3000000e+01],
                        [2.4000000e+01, 2.5000000e+01, 2.6000000e+01],
                        [2.7000000e+01, 2.8000000e+01, 2.9000000e+01]],
                       [[3.0000000e+01, 3.1000000e+01, 3.2000000e+01],
                        [3.3000000e+01, 3.4000000e+01, 3.5000000e+01],
                        [3.6000000e+01, 3.7000000e+01, 3.8000000e+01],
                        [3.9000000e+01, 4.0000000e+01, 4.1000000e+01],
                        [4.2000000e+01, 4.3000000e+01, 4.4000000e+01]],
                       [[3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38, 3.4028235e+38]],
                       [[0.0000000e+00, 1.0000000e+00, 2.0000000e+00],
                        [3.0000000e+00, 4.0000000e+00, 5.0000000e+00],
                        [6.0000000e+00, 7.0000000e+00, 8.0000000e+00],
                        [9.0000000e+00, 1.0000000e+01, 1.1000000e+01],
                        [1.2000000e+01, 1.3000000e+01, 1.4000000e+01]],
                       [[3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38, 3.4028235e+38]],
                       [[3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38, 3.4028235e+38]]]).astype(np.float32)
    np.testing.assert_array_almost_equal(output, expect)


# For testing Dynamic Shape operation
class UnsortedSegmentMinDynNet(nn.Cell):
    def __init__(self, num_segments, dyn_a=True, dyn_b=True):
        super(UnsortedSegmentMinDynNet, self).__init__()
        self.unsorted_segment_min = P.UnsortedSegmentMin()
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
        return self.unsorted_segment_min(data, ids, self.num_segments)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_3d_float32_ab_dyn():
    """
    Test for Dynamic shape with both inputs dynamic
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    input_x = Tensor(np.arange(
        4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3), dtype=mindspore.float32)
    segment_ids = Tensor([2, 1, 1, -1], mstype.int32)
    num_segments = 3
    net = UnsortedSegmentMinDynNet(num_segments)
    output = net(input_x, segment_ids).asnumpy()
    expect = np.array([[[3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38, 3.4028235e+38]],
                       [[1.5000000e+01, 1.6000000e+01, 1.7000000e+01],
                        [1.8000000e+01, 1.9000000e+01, 2.0000000e+01],
                        [2.1000000e+01, 2.2000000e+01, 2.3000000e+01],
                        [2.4000000e+01, 2.5000000e+01, 2.6000000e+01],
                        [2.7000000e+01, 2.8000000e+01, 2.9000000e+01]],
                       [[0.0000000e+00, 1.0000000e+00, 2.0000000e+00],
                        [3.0000000e+00, 4.0000000e+00, 5.0000000e+00],
                        [6.0000000e+00, 7.0000000e+00, 8.0000000e+00],
                        [9.0000000e+00, 1.0000000e+01, 1.1000000e+01],
                        [1.2000000e+01, 1.3000000e+01, 1.4000000e+01]]]).astype(np.float32)
    np.testing.assert_array_almost_equal(output, expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_3d_float32_a_dyn():
    """
    Tests for Dynamic shape with only first input dynamic
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    num_segments = 4
    net = UnsortedSegmentMinDynNet(num_segments, True, False)
    # test 1
    input_x = Tensor(np.arange(
        4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3), dtype=mindspore.float32)
    segment_ids = Tensor([3, 0, 1, -1], mstype.int32)
    output = net(input_x, segment_ids).asnumpy()
    expect = np.array([[[1.5000000e+01, 1.6000000e+01, 1.7000000e+01],
                        [1.8000000e+01, 1.9000000e+01, 2.0000000e+01],
                        [2.1000000e+01, 2.2000000e+01, 2.3000000e+01],
                        [2.4000000e+01, 2.5000000e+01, 2.6000000e+01],
                        [2.7000000e+01, 2.8000000e+01, 2.9000000e+01]],
                       [[3.0000000e+01, 3.1000000e+01, 3.2000000e+01],
                        [3.3000000e+01, 3.4000000e+01, 3.5000000e+01],
                        [3.6000000e+01, 3.7000000e+01, 3.8000000e+01],
                        [3.9000000e+01, 4.0000000e+01, 4.1000000e+01],
                        [4.2000000e+01, 4.3000000e+01, 4.4000000e+01]],
                       [[3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38, 3.4028235e+38]],
                       [[0.0000000e+00, 1.0000000e+00, 2.0000000e+00],
                        [3.0000000e+00, 4.0000000e+00, 5.0000000e+00],
                        [6.0000000e+00, 7.0000000e+00, 8.0000000e+00],
                        [9.0000000e+00, 1.0000000e+01, 1.1000000e+01],
                        [1.2000000e+01, 1.3000000e+01, 1.4000000e+01]]]).astype(np.float32)
    np.testing.assert_array_almost_equal(output, expect)
    # test 2
    input_x = Tensor(np.arange(
        4 * 7 * 2, dtype=np.float32).reshape(4, 7, 2), dtype=mindspore.float32)
    segment_ids = Tensor([3, 0, 1, -1], mstype.int32)
    output = net(input_x, segment_ids).asnumpy()
    expect = np.array([[[1.4000000e+01, 1.5000000e+01],
                        [1.6000000e+01, 1.7000000e+01],
                        [1.8000000e+01, 1.9000000e+01],
                        [2.0000000e+01, 2.1000000e+01],
                        [2.2000000e+01, 2.3000000e+01],
                        [2.4000000e+01, 2.5000000e+01],
                        [2.6000000e+01, 2.7000000e+01]],
                       [[2.8000000e+01, 2.9000000e+01],
                        [3.0000000e+01, 3.1000000e+01],
                        [3.2000000e+01, 3.3000000e+01],
                        [3.4000000e+01, 3.5000000e+01],
                        [3.6000000e+01, 3.7000000e+01],
                        [3.8000000e+01, 3.9000000e+01],
                        [4.0000000e+01, 4.1000000e+01]],
                       [[3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38]],
                       [[0.0000000e+00, 1.0000000e+00],
                        [2.0000000e+00, 3.0000000e+00],
                        [4.0000000e+00, 5.0000000e+00],
                        [6.0000000e+00, 7.0000000e+00],
                        [8.0000000e+00, 9.0000000e+00],
                        [1.0000000e+01, 1.1000000e+01],
                        [1.2000000e+01, 1.3000000e+01]]]).astype(np.float32)
    np.testing.assert_array_almost_equal(output, expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_3d_float32_b_dyn():
    """
    Tests for Dynamic shape with only second input dynamic
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    num_segments = 4
    net = UnsortedSegmentMinDynNet(num_segments, False, True)
    # test 1
    input_x = Tensor(np.arange(
        4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3), dtype=mindspore.float32)
    segment_ids = Tensor([3, 0, 1, -1], mstype.int32)
    output = net(input_x, segment_ids).asnumpy()
    expect = np.array([[[1.5000000e+01, 1.6000000e+01, 1.7000000e+01],
                        [1.8000000e+01, 1.9000000e+01, 2.0000000e+01],
                        [2.1000000e+01, 2.2000000e+01, 2.3000000e+01],
                        [2.4000000e+01, 2.5000000e+01, 2.6000000e+01],
                        [2.7000000e+01, 2.8000000e+01, 2.9000000e+01]],
                       [[3.0000000e+01, 3.1000000e+01, 3.2000000e+01],
                        [3.3000000e+01, 3.4000000e+01, 3.5000000e+01],
                        [3.6000000e+01, 3.7000000e+01, 3.8000000e+01],
                        [3.9000000e+01, 4.0000000e+01, 4.1000000e+01],
                        [4.2000000e+01, 4.3000000e+01, 4.4000000e+01]],
                       [[3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38, 3.4028235e+38]],
                       [[0.0000000e+00, 1.0000000e+00, 2.0000000e+00],
                        [3.0000000e+00, 4.0000000e+00, 5.0000000e+00],
                        [6.0000000e+00, 7.0000000e+00, 8.0000000e+00],
                        [9.0000000e+00, 1.0000000e+01, 1.1000000e+01],
                        [1.2000000e+01, 1.3000000e+01, 1.4000000e+01]]]).astype(np.float32)
    np.testing.assert_array_almost_equal(output, expect)
    # test 2
    input_x = Tensor(np.arange(
        4 * 7 * 2, dtype=np.float32).reshape(4, 7, 2), dtype=mindspore.float32)
    segment_ids = Tensor([3, 0, 1, -1], mstype.int32)
    output = net(input_x, segment_ids).asnumpy()
    expect = np.array([[[1.4000000e+01, 1.5000000e+01],
                        [1.6000000e+01, 1.7000000e+01],
                        [1.8000000e+01, 1.9000000e+01],
                        [2.0000000e+01, 2.1000000e+01],
                        [2.2000000e+01, 2.3000000e+01],
                        [2.4000000e+01, 2.5000000e+01],
                        [2.6000000e+01, 2.7000000e+01]],
                       [[2.8000000e+01, 2.9000000e+01],
                        [3.0000000e+01, 3.1000000e+01],
                        [3.2000000e+01, 3.3000000e+01],
                        [3.4000000e+01, 3.5000000e+01],
                        [3.6000000e+01, 3.7000000e+01],
                        [3.8000000e+01, 3.9000000e+01],
                        [4.0000000e+01, 4.1000000e+01]],
                       [[3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38],
                        [3.4028235e+38, 3.4028235e+38]],
                       [[0.0000000e+00, 1.0000000e+00],
                        [2.0000000e+00, 3.0000000e+00],
                        [4.0000000e+00, 5.0000000e+00],
                        [6.0000000e+00, 7.0000000e+00],
                        [8.0000000e+00, 9.0000000e+00],
                        [1.0000000e+01, 1.1000000e+01],
                        [1.2000000e+01, 1.3000000e+01]]]).astype(np.float32)
    np.testing.assert_array_almost_equal(output, expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_1d_int32_dynamic_shape():
    """
    Feature: UnsortedSegmentMin operation dynamic shape test
    Description: test UnsortedSegmentMin dynamic shape operation
    Expectation: UnsortedSegmentMin output == expect
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    input_x = Tensor([1, 2, 3, 4], mstype.int32)
    segment_ids = Tensor([0, 0, 1, 2], mstype.int32)
    num_segments = 4
    net = UnsortedSegmentMinNet(num_segments)
    output = net(input_x, segment_ids).asnumpy()
    expect = np.array([1, 3, 4, 2147483647]).astype(np.int32)
    np.testing.assert_array_almost_equal(output, expect)
