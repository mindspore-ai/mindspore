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
from mindspore import nn
from mindspore.ops import operations as P
from mindspore import Tensor
from mindspore import context
from .test_grad_of_dynamic import TestDynamicGrad


class KLDivLossNet(nn.Cell):
    def __init__(self, reduction="none"):
        super(KLDivLossNet, self).__init__()
        self.kl_div_loss = P.KLDivLoss(reduction)

    def construct(self, x, y):
        return self.kl_div_loss(x, y)


def grad_dyn_case(is_dynamic_rank, reduction="none"):
    prediction = Tensor(np.log(np.array([[0.3, 0.7], [0.5, 0.5]])).astype(np.float32))
    target = Tensor(np.array([[-1, 1], [1, -1]]).astype(np.float32))
    test_dynamic = TestDynamicGrad(KLDivLossNet(reduction))
    test_dynamic.test_dynamic_grad_net([prediction, target], is_dynamic_rank)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_grad_dynamic_shape():
    """
    Feature: test UnsortedSegmentMin dynamic shape on GPU.
    Description: input is dynamic shape.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    grad_dyn_case(False)
    grad_dyn_case(False, "mean")


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_grad_dynamic_rank():
    """
    Feature: test UnsortedSegmentMin dynamic rank on GPU.
    Description: input is dynamic rank.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    grad_dyn_case(True)
    grad_dyn_case(True, "mean")


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cpu_grad_dynamic_shape():
    """
    Feature: test UnsortedSegmentMin dynamic shape on CPU.
    Description: input is dynamic shape.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    grad_dyn_case(False)
    grad_dyn_case(False, "mean")


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cpu_grad_dynamic_rank():
    """
    Feature: test UnsortedSegmentMin dynamic rank on CPU.
    Description: input is dynamic rank.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    grad_dyn_case(True)
    grad_dyn_case(True, "mean")
