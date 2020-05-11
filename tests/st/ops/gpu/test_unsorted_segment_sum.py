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

import pytest
import numpy as np
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common.api import ms_function
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
import mindspore.nn as nn
import mindspore.context as context
from mindspore.common import dtype as mstype

context.set_context(device_target='GPU')

class UnsortedSegmentSumNet(nn.Cell):
    def __init__(self, num_segments):
        super(UnsortedSegmentSumNet, self).__init__()
        self.unsorted_segment_sum = P.UnsortedSegmentSum()
        self.num_segments = num_segments

    def construct(self, data, ids):
        return self.unsorted_segment_sum(data, ids, self.num_segments)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_1D():
      input_x = Tensor([1, 2, 3, 4], mstype.float32)
      segment_ids = Tensor([0, 0, 1, 2], mstype.int32)
      num_segments = 4

      net = UnsortedSegmentSumNet(num_segments)
      output = net(input_x, segment_ids)
      expect = [3, 3, 4, 0]
      assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_2D():
      input_x = Tensor([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12]], mstype.float32)
      segment_ids = Tensor([2, 1, 1], mstype.int32)
      num_segments = 4

      net = UnsortedSegmentSumNet(num_segments)
      output = net(input_x, segment_ids)
      expect = [[ 0,  0,  0,  0],
                [14, 16, 18, 20],
                [ 1,  2,  3,  4],
                [ 0,  0,  0,  0]]
      assert (output.asnumpy() == expect).all()



@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_3D():
    input_x = Tensor(np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3))
    segment_ids = Tensor([2, 1, 1, -1], mstype.int32)
    num_segments = 5

    net = UnsortedSegmentSumNet(num_segments)
    output = net(input_x, segment_ids)
    expect = [[[ 0.,  0.,  0.],
               [ 0.,  0.,  0.],
               [ 0.,  0.,  0.],
               [ 0.,  0.,  0.],
               [ 0.,  0.,  0.]],

              [[45., 47., 49.],
               [51., 53., 55.],
               [57., 59., 61.],
               [63., 65., 67.],
               [69., 71., 73.]],

              [[ 0.,  1.,  2.],
               [ 3.,  4.,  5.],
               [ 6.,  7.,  8.],
               [ 9., 10., 11.],
               [12., 13., 14.]],

              [[ 0.,  0.,  0.],
               [ 0.,  0.,  0.],
               [ 0.,  0.,  0.],
               [ 0.,  0.,  0.],
               [ 0.,  0.,  0.]],

              [[ 0.,  0.,  0.],
               [ 0.,  0.,  0.],
               [ 0.,  0.,  0.],
               [ 0.,  0.,  0.],
               [ 0.,  0.,  0.]]]
    assert (output.asnumpy() == expect).all()
