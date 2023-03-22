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
from mindspore import nn, context, Tensor
from mindspore.ops.operations.math_ops import BesselJ1
from .test_grad_of_dynamic import TestDynamicGrad


class BesselJ1Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.bessel_j1 = BesselJ1()

    def construct(self, x):
        return self.bessel_j1(x)


def grad_dyn_case(is_dynamic_rank):
    test_dynamic = TestDynamicGrad(BesselJ1Net())
    x_np = np.array([1, 2, 3, 4]).astype(np.float32)
    test_dynamic.test_dynamic_grad_net(Tensor(x_np), is_dynamic_rank)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_grad_dynamic_shape():
    """
    Feature: test BesselJ1 grad dynamic shape.
    Description: input is dynamic shape.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    grad_dyn_case(False)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_grad_dynamic_rank():
    """
    Feature: test BesselJ1 grad dynamic rank.
    Description: input is dynamic rank.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    grad_dyn_case(True)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_grad_dynamic_shape_2():
    """
    Feature: test BesselJ1 grad dynamic shape.
    Description: input is dynamic shape.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.GRAPH_MODE)
    grad_dyn_case(False)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_grad_dynamic_rank_2():
    """
    Feature: test BesselJ1 grad dynamic rank.
    Description: input is dynamic shape.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.GRAPH_MODE)
    grad_dyn_case(True)
