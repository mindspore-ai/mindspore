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
import mindspore as mp
from mindspore import nn, context, Tensor
from mindspore.ops.operations.array_ops import ResizeNearestNeighborV2
from .test_grad_of_dynamic import TestDynamicGrad


class NetResizeNearestNeighborV2(nn.Cell):
    def __init__(self):
        super(NetResizeNearestNeighborV2, self).__init__()
        self.resize_nearest_neighbor_v2 = ResizeNearestNeighborV2()

    def construct(self, input_tensor, size):
        return self.resize_nearest_neighbor_v2(input_tensor, size)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dynamic_resize_nearest_neighbor_v2_shape():
    """
    Feature: ResizeNearestNeighborV2 Grad DynamicShape.
    Description: Test case of dynamic shape for ResizeNearestNeighborV2 grad operator on CPU, GPU and Ascend.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    test_dynamic = TestDynamicGrad(NetResizeNearestNeighborV2())
    input_tensor = Tensor(np.array([[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]]]).astype(np.float32))
    size = Tensor([2, 2], mp.int32)
    test_dynamic.test_dynamic_grad_net([input_tensor, size])


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dynamic_resize_nearest_neighbor_v2_rank():
    """
    Feature: ResizeNearestNeighborV2 Grad DynamicRank.
    Description: Test case of dynamic rank for ResizeNearestNeighborV2 grad operator on CPU, GPU and Ascend.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    test_dynamic = TestDynamicGrad(NetResizeNearestNeighborV2())
    input_tensor = Tensor(np.array([[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]]]).astype(np.float32))
    size = Tensor([2, 2], mp.int32)
    test_dynamic.test_dynamic_grad_net([input_tensor, size], True)
