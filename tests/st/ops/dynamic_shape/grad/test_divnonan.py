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

context.set_context(mode=context.PYNATIVE_MODE)


class NetDivNoNan(nn.Cell):
    def __init__(self):
        super(NetDivNoNan, self).__init__()
        self.div = ops.DivNoNan()

    def construct(self, x, y):
        return self.div(x, y)


def divnonan_test(is_dyn_rank):
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32))
    y = Tensor(np.array([[7, 8, 9]]).astype(np.float32))
    tester = TestDynamicGrad(NetDivNoNan())
    tester.test_dynamic_grad_net([x, y], is_dyn_rank)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
def test_divnonan_dyn_shape():
    """
    Feature: DivNoNan Grad DynamicShape.
    Description: Test case of dynamic shape for DivNoNan grad operator.
    Expectation: success.
    """
    divnonan_test(False)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
def test_divnonan_dyn_rank():
    """
    Feature: DivNoNan Grad DynamicShape.
    Description: Test case of dynamic rank for DivNoNan grad operator.
    Expectation: success.
    """
    divnonan_test(True)
