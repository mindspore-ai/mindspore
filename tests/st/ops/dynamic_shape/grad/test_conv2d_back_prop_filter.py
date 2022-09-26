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
from mindspore.ops.operations import _grad_ops as G
from .test_grad_of_dynamic import TestDynamicGrad


class NetConv2DBackpropFilter(nn.Cell):
    def __init__(self):
        super(NetConv2DBackpropFilter, self).__init__()
        self.conv_filter = G.Conv2DBackpropFilter(1, 3, pad_mode="valid", pad=0, mode=1, stride=(1, 1),
                                                  dilation=(1, 1, 1, 1), group=1)
        self.w_shape = (1, 1, 3, 3)

    def construct(self, out, x):
        return self.conv_filter(out, x, self.w_shape)


def grad_dyn_case(is_dynamic_rank):
    test_dynamic = TestDynamicGrad(NetConv2DBackpropFilter())
    np.random.seed(1)
    out = Tensor(np.random.normal(0, 1, (1, 1, 4, 4)).astype(np.float32))
    x = Tensor(np.random.normal(0, 2, (1, 1, 6, 6)).astype(np.float32))
    test_dynamic.test_dynamic_grad_net([out, x], is_dynamic_rank)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_gpu_grad_dynamic_shape():
    """
    Feature: test Conv2DBackpropFilter dynamic shape on GPU.
    Description: input is dynamic shape.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    grad_dyn_case(False)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_gpu_grad_dynamic_rank():
    """
    Feature: test Conv2DBackpropFilter dynamic rank on GPU.
    Description: input is dynamic rank.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    grad_dyn_case(True)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
def test_cpu_grad_dynamic_shape():
    """
    Feature: test Conv2DBackpropFilter dynamic shape on CPU.
    Description: input is dynamic shape.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    grad_dyn_case(False)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
def test_cpu_grad_dynamic_rank():
    """
    Feature: test Conv2DBackpropFilter dynamic rank on CPU.
    Description: input is dynamic rank.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    grad_dyn_case(True)


def test_ascend_grad_dynamic_shape():
    """
    Feature: test Conv2DBackpropFilter dynamic shape on Ascend.
    Description: input is dynamic shape.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    grad_dyn_case(False)


def test_ascend_grad_dynamic_rank():
    """
    Feature: test Conv2DBackpropFilter dynamic rank on Ascend.
    Description: input is dynamic rank.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    grad_dyn_case(True)
