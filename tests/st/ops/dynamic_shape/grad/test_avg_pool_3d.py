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
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore import ops
from .test_grad_of_dynamic import TestDynamicGrad


class NetAvgPool3D(nn.Cell):
    def __init__(self):
        super(NetAvgPool3D, self).__init__()
        self.op = ops.AvgPool3D(kernel_size=2, strides=1, pad_mode="valid")

    def construct(self, x):
        return self.op(x)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_shape_avg_pool_3d():
    """
    Feature: AvgPool3D Grad DynamicShape.
    Description: Test case of dynamic shape for AvgPool3D grad.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    test_dynamic = TestDynamicGrad(NetAvgPool3D())
    x = Tensor(np.arange(1 * 2 * 2 * 2 * 3).reshape((1, 2, 2, 2, 3)).astype(np.float16))
    test_dynamic.test_dynamic_grad_net(x)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_rank_avg_pool_3d():
    """
    Feature: AvgPool3D Grad DynamicRank.
    Description: Test case of dynamic rank for AvgPool3D grad.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    test_dynamic = TestDynamicGrad(NetAvgPool3D())
    x = Tensor(np.arange(1 * 2 * 2 * 2 * 3).reshape((1, 2, 2, 2, 3)).astype(np.float16))
    test_dynamic.test_dynamic_grad_net(x, True)
