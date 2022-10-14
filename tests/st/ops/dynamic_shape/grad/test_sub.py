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


class NetSub(nn.Cell):
    def __init__(self):
        super(NetSub, self).__init__()
        self.sub = ops.Sub()

    def construct(self, x, y):
        return self.sub(x, y)


def sub_test(is_dyn_rank):
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32))
    y = Tensor(np.array([[7, 8, 9]]).astype(np.float32))
    tester = TestDynamicGrad(NetSub())
    tester.test_dynamic_grad_net([x, y], is_dyn_rank)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_sub_dyn_shape():
    """
    Feature: Sub Grad DynamicShape.
    Description: Test case of dynamic shape for Sub grad operator.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    sub_test(False)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_sub_dyn_rank():
    """
    Feature: Sub Grad DynamicShape.
    Description: Test case of dynamic rank for Sub grad operator.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    sub_test(True)
