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
from mindspore import Tensor
from mindspore.ops.operations.array_ops import SegmentMax, SegmentMin, SegmentMean, SegmentSum, SegmentProd
from mindspore.nn import Cell
import mindspore.common.dtype as mstype

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")


class  SegmentMaxNet(Cell):
    def __init__(self):
        super().__init__()
        self.segmentmax = SegmentMax()

    def construct(self, x, segment_ids):
        return self.segmentmax(x, segment_ids)


class  SegmentMinNet(Cell):
    def __init__(self):
        super().__init__()
        self.segmentmin = SegmentMin()

    def construct(self, x, segment_ids):
        return self.segmentmin(x, segment_ids)


class  SegmentMeanNet(Cell):
    def __init__(self):
        super().__init__()
        self.segmentmean = SegmentMean()

    def construct(self, x, segment_ids):
        return self.segmentmean(x, segment_ids)


class  SegmentSumNet(Cell):
    def __init__(self):
        super().__init__()
        self.segmentsum = SegmentSum()

    def construct(self, x, segment_ids):
        return self.segmentsum(x, segment_ids)


class  SegmentProdNet(Cell):
    def __init__(self):
        super().__init__()
        self.segmentprod = SegmentProd()

    def construct(self, x, segment_ids):
        return self.segmentprod(x, segment_ids)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_segment_max_fp():
    """
    Feature: SegmentMax operator.
    Description: test cases for SegmentMax operator.
    Expectation: the result match expectation.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    input_x = Tensor([1, 2, 3], mstype.int32)
    segment_ids = Tensor([0, 6, 6], mstype.int32)
    net = SegmentMaxNet()
    expect = np.array([1, 0, 0, 0, 0, 0, 3]).astype(np.int32)
    output_gr = net(input_x, segment_ids).asnumpy()
    np.testing.assert_array_almost_equal(output_gr, expect)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    output_py = net(input_x, segment_ids).asnumpy()
    np.testing.assert_almost_equal(output_py, expect)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_segment_min_fp():
    """
    Feature: SegmentMin operator.
    Description: test cases for SegmentMin operator.
    Expectation: the result match expectation.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    input_x = Tensor([1, 2, 3, 4], mstype.int32)
    segment_ids = Tensor([0, 0, 1, 5], mstype.int32)
    net = SegmentMinNet()
    expect = np.array([1, 3, 0, 0, 0, 4]).astype(np.int32)
    output_gr = net(input_x, segment_ids).asnumpy()
    np.testing.assert_array_almost_equal(output_gr, expect)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    output_py = net(input_x, segment_ids).asnumpy()
    np.testing.assert_almost_equal(output_py, expect)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_segment_sum_fp():
    """
    Feature: SegmentSum operator.
    Description: test cases for SegmentSum operator.
    Expectation: the result match expectation.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    input_x = Tensor([1 + 2j, 2 + 2j, 3 + 2j], mstype.float32)
    segment_ids = Tensor([0, 0, 2], mstype.int32)
    net = SegmentSumNet()
    expect = np.array([3 + 4j, 0, 3 + 2j]).astype(np.float32)
    output_gr = net(input_x, segment_ids).asnumpy()
    np.testing.assert_array_almost_equal(output_gr, expect)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_segment_mean_fp():
    """
    Feature: SegmentMean operator.
    Description: test cases for SegmentMean operator.
    Expectation: the result match expectation.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    input_x = Tensor([2, 2, 3, 4], mstype.float32)
    segment_ids = Tensor([0, 0, 1, 2], mstype.int32)
    net = SegmentMeanNet()
    expect = np.array([2, 3, 4]).astype(np.float32)
    output_gr = net(input_x, segment_ids).asnumpy()
    np.testing.assert_array_almost_equal(output_gr, expect)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    output_py = net(input_x, segment_ids).asnumpy()
    np.testing.assert_almost_equal(output_py, expect)



@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_segment_prod_fp():
    """
    Feature: SegmentProd operator.
    Description: test cases for SegmentProd operator.
    Expectation: the result match expectation.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    input_x = Tensor([1, 2, 3, 4], mstype.float32)
    segment_ids = Tensor([0, 0, 1, 2], mstype.int32)
    net = SegmentProdNet()
    expect = np.array([2, 3, 4]).astype(np.float32)
    output_gr = net(input_x, segment_ids).asnumpy()
    np.testing.assert_array_almost_equal(output_gr, expect)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    output_py = net(input_x, segment_ids).asnumpy()
    np.testing.assert_almost_equal(output_py, expect)
