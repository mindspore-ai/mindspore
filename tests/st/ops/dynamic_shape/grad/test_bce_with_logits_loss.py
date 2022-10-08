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


class BCEWithLogitsLossNet(nn.Cell):
    def __init__(self, reduction):
        super(BCEWithLogitsLossNet, self).__init__()
        self.op = ops.BCEWithLogitsLoss(reduction=reduction)

    def construct(self, predict, target, weight, pos_weight):
        return self.op(predict, target, weight, pos_weight)


def dyn_grad_func(dtype=np.float16, is_dynamic_rank=False):
    test_dynamic = TestDynamicGrad(BCEWithLogitsLossNet("mean"))
    predict = Tensor(np.arange(6).reshape(2, 3).astype(dtype))
    target = Tensor(np.arange(34, 40).reshape(2, 3).astype(dtype))
    weight = Tensor(np.array([2, 3, 1]).astype(dtype))
    pos_weight = Tensor(np.array([6, 3, 4]).astype(dtype))
    inputs = [predict, target, weight, pos_weight]
    test_dynamic.test_dynamic_grad_net(inputs, is_dynamic_rank=is_dynamic_rank)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_bcewithlogitsloss_dynamic_shape():
    """
    Feature: Test the bprop process of BCEWithLogitsLoss in PyNative modee with dynamic shape inputs
    Description: The inputs are dynamic shape and the bprop function uses these shapes when reduction mode is mean
    Expectation: Assert the result is equal to that of static shape inputs
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    dyn_grad_func(is_dynamic_rank=False)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bcewithlogitsloss_dynamic_rank():
    """
    Feature: Test the bprop process of BCEWithLogitsLoss in PyNative mode with dynamic rank inputs
    Description: The inputs are dynamic rank and the bprop function uses these shapes when reduction mode is mean
    Expectation: Assert the result is equal to that of static shape inputs
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    dyn_grad_func(dtype=np.float32, is_dynamic_rank=True)
