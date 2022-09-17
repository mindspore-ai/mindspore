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
from mindspore import nn
from mindspore.ops import operations as P
from mindspore import Tensor
from mindspore import context
from .test_grad_of_dynamic import TestDynamicGrad


class UnsortedSegmentMaxNet(nn.Cell):
    def __init__(self, num_segments):
        super(UnsortedSegmentMaxNet, self).__init__()
        self.unsorted_segment_max = P.UnsortedSegmentMax()
        self.num_segments = num_segments

    def construct(self, data, ids):
        return self.unsorted_segment_max(data, ids, self.num_segments)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
def test_dynamic_shape_unsorted_segment_max():
    """
    Feature: UnsortedSegmentMax Grad DynamicShape.
    Description: Test case of dynamic shape for UnsortedSegmentMax grad operator on CPU and GPU.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    num_segments = 2
    test_dynamic = TestDynamicGrad(UnsortedSegmentMaxNet(num_segments))
    input_x = Tensor(
        np.array([[1, 2, 3], [4, 5, 6], [4, 2, 1]]).astype(np.float32))
    segment_ids = Tensor(np.array([0, 1, 1]).astype(np.int32))
    test_dynamic.test_dynamic_grad_net([input_x, segment_ids], False)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
def test_dynamic_rank_unsorted_segment_max():
    """
    Feature: UnsortedSegmentMax Grad DynamicShape.
    Description: Test case of dynamic rank for UnsortedSegmentMax grad operator on CPU and GPU.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    num_segments = 2
    test_dynamic = TestDynamicGrad(UnsortedSegmentMaxNet(num_segments))
    input_x = Tensor(
        np.array([[1, 2, 3], [4, 5, 6], [4, 2, 1]]).astype(np.float32))
    segment_ids = Tensor(np.array([0, 1, 1]).astype(np.int32))
    test_dynamic.test_dynamic_grad_net([input_x, segment_ids], True)
