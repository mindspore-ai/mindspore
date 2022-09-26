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


class Atan2Net(nn.Cell):
    def __init__(self):
        super(Atan2Net, self).__init__()
        self.atan2 = P.Atan2()

    def construct(self, x, y):
        return self.atan2(x, y)


def dynamic_shape():
    test_dynamic = TestDynamicGrad(Atan2Net())
    x = Tensor(np.array([0, 1]).astype(np.float32))
    y = Tensor(np.array([1, 1]).astype(np.float32))
    test_dynamic.test_dynamic_grad_net((x, y))


def dynamic_rank():
    test_dynamic = TestDynamicGrad(Atan2Net())
    x = Tensor(np.array([0, 1]).astype(np.float32))
    y = Tensor(np.array([1, 1]).astype(np.float32))
    test_dynamic.test_dynamic_grad_net((x, y), True)


def test_dynamic_atan2_cpu():
    """
    Feature: Atan2 Grad DynamicShape.
    Description: Test case of dynamic shape for Atan2 grad operator on CPU.
    Expectation: success.
    """
    # Graph mode
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    dynamic_shape()
    dynamic_rank()
    # PyNative mode
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    dynamic_shape()
    dynamic_rank()


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_dynamic_atan2_gpu():
    """
    Feature: Atan2 Grad DynamicShape.
    Description: Test case of dynamic shape for Atan2 grad operator on GPU.
    Expectation: success.
    """
    # Graph mode
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    dynamic_shape()
    dynamic_rank()
    # PyNative mode
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    dynamic_shape()
    dynamic_rank()


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_dynamic_atan2_ascend():
    """
    Feature: Atan2 Grad DynamicShape.
    Description: Test case of dynamic shape for Atan2 grad operator on Ascend.
    Expectation: success.
    """
    # Graph mode
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    dynamic_shape()
    dynamic_rank()
    # PyNative mode
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    dynamic_shape()
    dynamic_rank()
