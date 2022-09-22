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
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.ops import operations as P
from .test_grad_of_dynamic import TestDynamicGrad


class MaskedFillNet(nn.Cell):
    def __init__(self):
        super(MaskedFillNet, self).__init__()
        self.maskedfill = P.MaskedFill()

    def construct(self, input_0, mask, value):
        return self.maskedfill(input_0, mask, value)


def run_dynamic_shape():
    test_dynamic = TestDynamicGrad(MaskedFillNet())
    input_0 = Tensor(np.array([1., 2., 3., 4.]), ms.float32)
    mask = Tensor(np.array([True, True, False, True]), ms.bool_)
    value = Tensor(0.5, ms.float32)
    test_dynamic.test_dynamic_grad_net([input_0, mask, value])


def run_dynamic_rank():
    test_dynamic = TestDynamicGrad(MaskedFillNet())
    input_0 = Tensor(np.array([1., 2., 3., 4.]), ms.float32)
    mask = Tensor(np.array([True, True, False, True]), ms.bool_)
    value = Tensor(0.5, ms.float32)
    test_dynamic.test_dynamic_grad_net([input_0, mask, value], True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
def test_dynamic_maskedfill_gpu():
    """
    Feature: MaskedFill Grad DynamicShape.
    Description: Test case of dynamic shape for  MaskedFill grad operator on GPU.
    Expectation: success.
    """
    # Graph mode
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    run_dynamic_shape()
    run_dynamic_rank()
    # PyNative mode
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    run_dynamic_shape()
    run_dynamic_rank()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
def test_dynamic_maskedfill_ascend():
    """
    Feature: MaskedFill Grad DynamicShape.
    Description: Test case of dynamic shape for  MaskedFill grad operator on Ascend.
    Expectation: success.
    """
    # Graph mode
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_dynamic_shape()
    run_dynamic_rank()
    # PyNative mode
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    run_dynamic_shape()
    run_dynamic_rank()
