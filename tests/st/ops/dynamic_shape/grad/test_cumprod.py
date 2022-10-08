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


class CumProdNet(nn.Cell):
    def __init__(self):
        super(CumProdNet, self).__init__()
        self.op = ops.CumProd()

    def construct(self, x, axis):
        return self.op(x, axis)


def dyn_grad_func(dtype=np.float16, is_dynamic_rank=False):
    test_dynamic = TestDynamicGrad(CumProdNet())
    x = Tensor(np.random.rand(2, 3, 4, 4).astype(dtype))
    axis = 0
    inputs = [x, axis]
    test_dynamic.test_dynamic_grad_net(inputs, is_dynamic_rank=is_dynamic_rank)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cumprod_dynamic_shape():
    """
    Feature: Test the bprop process of CumProd in PyNative mode with dynamic shape inputs
    Description: The inputs are dynamic shape and the bprop function invokes the operator itself.
    Expectation: Assert the result is equal to that of static shape inputs
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    dyn_grad_func(is_dynamic_rank=False)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cumprod_dynamic_rank():
    """
    Feature: Test the bprop process of CumProd in PyNative mode with dynamic rank inputs
    Description: The inputs are dynamic rank and the bprop function invokes the operator itself.
    Expectation: Assert the result is equal to that of static shape inputs
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    dyn_grad_func(dtype=np.float32, is_dynamic_rank=True)
