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
from mindspore import nn, context, Tensor
from mindspore.ops import operations as P
from .test_grad_of_dynamic import TestDynamicGrad

context.set_context(mode=context.PYNATIVE_MODE)


class StackNet(nn.Cell):
    def __init__(self):
        super(StackNet, self).__init__()
        self.stack = P.Stack(axis=2)

    def construct(self, x1, x2):
        return self.stack((x1, x2))


def stack_test(is_dyn_rank):
    x1_np = np.array([0] * 16).astype(np.float32)
    x1_np = np.reshape(x1_np, (2, 2, 2, 2))
    x1 = Tensor(x1_np)
    x2 = Tensor(np.arange(16).reshape(2, 2, 2, 2).astype(np.float32))
    tester = TestDynamicGrad(StackNet())
    tester.test_dynamic_grad_net([x1, x2], is_dyn_rank)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_stack_dyn_shape():
    """
    Feature: Stack Grad DynamicShape.
    Description: Test case of dynamic shape for Stack grad operator.
    Expectation: success.
    """
    stack_test(False)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_stack_dyn_rank():
    """
    Feature: Stack Grad DynamicShape.
    Description: Test case of dynamic rank for Stack grad operator.
    Expectation: success.
    """
    stack_test(True)
