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
from mindspore.ops.operations import Lerp
from .test_grad_of_dynamic import TestDynamicGrad


class NetLerp(nn.Cell):
    def __init__(self):
        super(NetLerp, self).__init__()
        self.lerp = Lerp()

    def construct(self, x, y, z):
        return self.lerp(x, y, z)


def grad_dyn_case(is_dynamic_rank):
    test_dynamic = TestDynamicGrad(NetLerp())
    x = np.array([[1.0, -1.0, 2.0], [3.1, 2, 1.0]], dtype=np.float32)
    y = np.array([[1.2, -1.0, 2.1], [3.0, 2.0, 1.1]], dtype=np.float32)
    z = np.array([[1.0, -1.2, 0.9], [0.1, 2.0, 1.0]], dtype=np.float32)
    test_dynamic.test_dynamic_grad_net([Tensor(x), Tensor(y), Tensor(z)], is_dynamic_rank)


def grad_partial_dyn_case(is_dynamic_rank):
    test_dynamic = TestDynamicGrad(NetLerp())
    test_dynamic.skip_convert_in_ids = [2]
    x = np.array([[1.0, -1.0, 2.0], [3.1, 2, 1.0]], dtype=np.float32)
    y = np.array([[1.2, -1.0, 2.1], [3.0, 2.0, 1.1]], dtype=np.float32)
    z = np.array([[1.0, -1.2, 0.9], [0.1, 2.0, 1.0]], dtype=np.float32)
    test_dynamic.test_dynamic_grad_net([Tensor(x), Tensor(y), Tensor(z)], is_dynamic_rank)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_grad_dynamic_shape():
    """
    Feature: test Lerp grad dynamic shape.
    Description: input is dynamic shape.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    grad_dyn_case(False)
    grad_partial_dyn_case(False)
    context.set_context(mode=context.GRAPH_MODE)
    grad_dyn_case(False)
    grad_partial_dyn_case(False)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_grad_dynamic_rank():
    """
    Feature: test Lerp grad dynamic rank.
    Description: input is dynamic rank.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    grad_dyn_case(True)
    grad_partial_dyn_case(True)
    context.set_context(mode=context.GRAPH_MODE)
    grad_dyn_case(True)
    grad_partial_dyn_case(True)


@pytest.mark.skip(reason="Ascend does not support dynamic shape")
@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ascend_grad_dynamic_shape():
    """
    Feature: test Lerp grad dynamic rank on Ascend.
    Description: input is dynamic rank.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    grad_dyn_case(False)
    grad_partial_dyn_case(False)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    grad_dyn_case(False)
    grad_partial_dyn_case(False)


@pytest.mark.skip(reason="Ascend does not support dynamic shape")
@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ascend_grad_dynamic_rank():
    """
    Feature: test Lerp grad dynamic rank on Ascend.
    Description: input is dynamic rank.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    grad_dyn_case(True)
    grad_partial_dyn_case(True)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    grad_dyn_case(True)
    grad_partial_dyn_case(True)
