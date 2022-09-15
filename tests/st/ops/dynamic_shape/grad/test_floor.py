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


class FloorNet(nn.Cell):
    def __init__(self):
        super(FloorNet, self).__init__()
        self.floor = ops.Floor()

    def construct(self, x):
        return self.floor(x)


def floor_test(is_dyn_rank):
    x = Tensor(np.random.randn(32, 8, 32).astype(np.float32))
    tester = TestDynamicGrad(FloorNet())
    tester.test_dynamic_grad_net([x], is_dyn_rank)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
def test_floor_dyn_shape():
    """
    Feature: Floor Grad DynamicShape.
    Description: Test case of dynamic shape for Floor grad operator.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    floor_test(False)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
def test_floor_dyn_rank():
    """
    Feature: Floor Grad DynamicShape.
    Description: Test case of dynamic rank for Floor grad operator.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    floor_test(True)
