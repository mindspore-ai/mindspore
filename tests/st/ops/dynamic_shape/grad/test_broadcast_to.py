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
from mindspore import ops, nn, context, Tensor
from .test_grad_of_dynamic import TestDynamicGrad


class NetBroadcastTo(nn.Cell):
    def __init__(self):
        super(NetBroadcastTo, self).__init__()
        self.op = ops.BroadcastTo((16, 2, 3))

    def construct(self, x):
        return self.op(x)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_shape_broadcast_to():
    """
    Feature: BroadcastTo Grad DynamicShape.
    Description: Test case of dynamic shape for BroadcastTo grad.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    test_dynamic = TestDynamicGrad(NetBroadcastTo())
    x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]).astype(np.float16))
    test_dynamic.test_dynamic_grad_net(x)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_rank_broadcast_to():
    """
    Feature: BroadcastTo Grad DynamicRank.
    Description: Test case of dynamic rank for BroadcastTo grad.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    test_dynamic = TestDynamicGrad(NetBroadcastTo())
    x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]).astype(np.float16))
    test_dynamic.test_dynamic_grad_net(x, True)
