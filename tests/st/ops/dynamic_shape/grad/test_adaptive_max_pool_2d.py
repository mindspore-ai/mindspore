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
import pytest
import numpy as np
import mindspore.ops.operations.nn_ops as P
from mindspore import nn, context, Tensor
from .test_grad_of_dynamic import TestDynamicGrad


class Net(nn.Cell):

    def __init__(self, output_size):
        super(Net, self).__init__()
        self.op = P.AdaptiveMaxPool2D(output_size=output_size, return_indices=True)

    def construct(self, x):
        return self.op(x)


def grad_dyn_case(is_dynamic_rank):
    x = Tensor(np.random.rand(4, 4, 6, 8).astype(np.float32))
    tester = TestDynamicGrad(Net((None, 4)))
    tester.test_dynamic_grad_net([x], is_dynamic_rank)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_dynamic_shape():
    """
    Feature: test AdaptiveMaxPool2D Grad in dynamic shape.
    Description: input is dynamic shape.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    grad_dyn_case(False)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_dynamic_rank():
    """
    Feature: test AdaptiveMaxPool2D Grad in dynamic rank.
    Description: input is dynamic rank.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    grad_dyn_case(True)
