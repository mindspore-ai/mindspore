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
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.ops import operations as P
from .test_grad_of_dynamic import TestDynamicGrad


class AddcdivNet(nn.Cell):
    def __init__(self):
        super(AddcdivNet, self).__init__()
        self.addcdiv = P.Addcdiv()

    def construct(self, input_data, x1, x2, value):
        return self.addcdiv(input_data, x1, x2, value)


def dynamic_shape():
    type_s = np.float32
    test_dynamic = TestDynamicGrad(AddcdivNet())
    input_data = Tensor(np.array([12]).astype(type_s))
    x1 = Tensor(np.array([7]).astype(type_s))
    x2 = Tensor(np.array([3]).astype(type_s))
    value = Tensor(np.array([37]).astype(type_s))
    test_dynamic.test_dynamic_grad_net((input_data, x1, x2, value))


def dynamic_rank():
    type_s = np.float32
    test_dynamic = TestDynamicGrad(AddcdivNet())
    input_data = Tensor(np.array([12]).astype(type_s))
    x1 = Tensor(np.array([7]).astype(type_s))
    x2 = Tensor(np.array([3]).astype(type_s))
    value = Tensor(np.array([37]).astype(type_s))
    test_dynamic.test_dynamic_grad_net((input_data, x1, x2, value), True)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_addcdiv():
    """
    Feature: Addcdiv Grad DynamicShape.
    Description: Test case of dynamic shape for Addcdiv grad operator.
    Expectation: success.
    """
    # Graph mode
    context.set_context(mode=context.GRAPH_MODE)
    dynamic_shape()
    dynamic_rank()
    # PyNative mode
    context.set_context(mode=context.PYNATIVE_MODE)
    dynamic_shape()
    dynamic_rank()
