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
import mindspore as ms
from mindspore import ops, nn, context, Tensor
from .test_grad_of_dynamic import TestDynamicGrad

context.set_context(mode=context.PYNATIVE_MODE)


class NetAddcmul(nn.Cell):
    def __init__(self):
        super(NetAddcmul, self).__init__()
        self.add = ops.Addcmul()

    def construct(self, a, x1, x2, v):
        return self.add(a, x1, x2, v)


def addcmul_test(is_dyn_rank):
    a = Tensor(np.array([1, 1, 1]).astype(np.float32))
    x1 = Tensor(np.array([[1], [2], [3]]).astype(np.float32))
    x2 = Tensor(np.array([[1, 2, 3]]).astype(np.float32))
    v = Tensor([1], ms.float32)
    tester = TestDynamicGrad(NetAddcmul())
    tester.test_dynamic_grad_net([a, x1, x2, v], is_dyn_rank)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_addcmul_dyn_shape():
    """
    Feature: Addcmul Grad DynamicShape.
    Description: Test case of dynamic shape for Addcmul grad operator.
    Expectation: success.
    """
    addcmul_test(False)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_addcmul_dyn_rank():
    """
    Feature: Addcmul Grad DynamicShape.
    Description: Test case of dynamic rank for Addcmul grad operator.
    Expectation: success.
    """
    addcmul_test(True)
