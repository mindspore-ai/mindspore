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
import mindspore
from mindspore import ops, nn, context, Tensor
from .test_grad_of_dynamic import TestDynamicGrad

context.set_context(mode=context.PYNATIVE_MODE)


class TestMul(nn.Cell):
    def __init__(self):
        super(TestMul, self).__init__()
        self.ops = ops.Mul()

    def construct(self, x, y):
        return self.ops(x, y)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_mul_dynamic_shape():
    """
    Feature: Mul Grad DynamicShape.
    Description: Test case of dynamic shape for Mul grad operator on GPU.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

    test_dynamic = TestDynamicGrad(TestMul())
    input_x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
    input_y = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
    x = [input_x, input_y]
    test_dynamic.test_dynamic_grad_net(x)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_mul_dynamic_rank():
    """
    Feature: Mul Grad DynamicShape.
    Description: Test case of dynamic rank for Mul grad operator on GPU.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

    test_dynamic = TestDynamicGrad(TestMul())
    input_x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
    input_y = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
    x = [input_x, input_y]
    test_dynamic.test_dynamic_grad_net(x, True)
