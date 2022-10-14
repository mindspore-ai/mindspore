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


class NetTranspose(nn.Cell):
    def __init__(self):
        super(NetTranspose, self).__init__()
        self.transpose = P.Transpose()

    def construct(self, x, perm):
        return self.transpose(x, perm)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_dynamic_shape_transpose():
    """
    Feature: Transpose Grad DynamicShape.
    Description: Test case of dynamic shape for Transpose grad operator on CPU, GPU and Ascend.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    test_dynamic = TestDynamicGrad(NetTranspose())
    x = Tensor(np.array([[1, 2, 3], [3, 2, 1]]))
    perm = (0, 1)
    test_dynamic.test_dynamic_grad_net([x, perm], False)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_dynamic_rank_transpose():
    """
    Feature: Transpose Grad DynamicShape.
    Description: Test case of dynamic rank for Transpose grad operator on CPU, GPU and Ascend.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    test_dynamic = TestDynamicGrad(NetTranspose())
    x = Tensor(np.array([[1, 2, 3], [3, 2, 1]]))
    perm = (0, 1)
    test_dynamic.test_dynamic_grad_net([x, perm], True)
