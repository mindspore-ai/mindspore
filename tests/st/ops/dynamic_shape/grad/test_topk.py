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


class NetTopK(nn.Cell):
    def __init__(self, k):
        super(NetTopK, self).__init__()
        self.topk = ops.TopK()
        self.k = k

    def construct(self, x):
        return self.topk(x, self.k)


def topk_test(is_dyn_rank):
    x = Tensor(np.array([[1, 2, 3, 4, 5]]).astype(np.float32))
    k = 3
    tester = TestDynamicGrad(NetTopK(k))
    tester.test_dynamic_grad_net([x], is_dyn_rank)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
def test_topk_dyn_shape():
    """
    Feature: TopK Grad DynamicShape.
    Description: Test case of dynamic shape for TopK grad operator.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    topk_test(False)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
def test_topk_dyn_rank():
    """
    Feature: TopK Grad DynamicShape.
    Description: Test case of dynamic rank for TopK grad operator.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    topk_test(True)
