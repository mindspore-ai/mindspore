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
from mindspore import Tensor
from mindspore import context
from mindspore.ops.operations.math_ops import Igamma
from .test_grad_of_dynamic import TestDynamicGrad


class IgammaNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.igamma = Igamma()

    def construct(self, a, x):
        return self.igamma(a, x)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
def test_dynamic_shape_igamma():
    """
    Feature: Igamma Grad DynamicShape.
    Description: Test case of dynamic shape for Igamma grad operator on CPU.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    a_np = np.array([[10, 22], [20, 50]]).astype(np.float32)
    x_np = np.array([[10, 22], [20, 50]]).astype(np.float32)
    test_dynamic = TestDynamicGrad(IgammaNet())
    test_dynamic.test_dynamic_grad_net(
        [Tensor(a_np), Tensor(x_np)], False)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
def test_dynamic_rank_igamma():
    """
    Feature: Igamma Grad DynamicShape.
    Description: Test case of dynamic rank for Igamma grad operator on CPU.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    a_np = np.array([[10, 22], [20, 50]]).astype(np.float32)
    x_np = np.array([[10, 22], [20, 50]]).astype(np.float32)
    test_dynamic = TestDynamicGrad(IgammaNet())
    test_dynamic.test_dynamic_grad_net(
        [Tensor(a_np), Tensor(x_np)], True)
