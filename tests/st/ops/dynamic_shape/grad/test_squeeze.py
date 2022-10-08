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
from mindspore import context
from mindspore import Tensor
from .test_grad_of_dynamic import TestDynamicGrad


class NetSqueeze(nn.Cell):
    def __init__(self):
        super(NetSqueeze, self).__init__()
        self.squeeze = P.Squeeze()

    def construct(self, x):
        return self.squeeze(x)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dynamic_squeeze_shape():
    """
    Feature: Squeeze Grad DynamicShape.
    Description: Test case of dynamic shape for Squeeze grad operator on CPU, GPU and Ascend.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    test_dynamic = TestDynamicGrad(NetSqueeze())
    x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]).astype(np.float16))
    test_dynamic.test_dynamic_grad_net(x)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dynamic_squeeze_rank():
    """
    Feature: Squeeze Grad DynamicRank.
    Description: Test case of dynamic rank for Squeeze grad operator on CPU, GPU and Ascend.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    test_dynamic = TestDynamicGrad(NetSqueeze())
    x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]).astype(np.float16))
    test_dynamic.test_dynamic_grad_net(x, True)
