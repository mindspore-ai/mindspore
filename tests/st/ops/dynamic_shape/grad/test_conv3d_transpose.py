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
from mindspore.ops.operations import Conv3DTranspose
from .test_grad_of_dynamic import TestDynamicGrad


class NetConv3dTranspose(nn.Cell):
    def __init__(self):
        super(NetConv3dTranspose, self).__init__()
        in_channel = 2
        out_channel = 2
        kernel_size = 2
        self.conv_trans = Conv3DTranspose(
            in_channel, out_channel, kernel_size, pad_mode="pad", pad=1, stride=1, dilation=1, group=1
        )

    def construct(self, x, w):
        return self.conv_trans(x, w)


def grad_dyn_case(is_dynamic_rank):
    x = Tensor(np.arange(1 * 2 * 3 * 3 * 3).reshape(1, 2, 3, 3, 3).astype(np.float32))
    w = Tensor(np.ones((2, 2, 2, 2, 2)).astype(np.float32))
    test_dynamic = TestDynamicGrad(NetConv3dTranspose())
    test_dynamic.test_dynamic_grad_net([x, w], is_dynamic_rank)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_grad_dynamic_shape():
    """
    Feature: test Conv3DTranspose grad dynamic shape on GPU.
    Description: input is dynamic shape.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    grad_dyn_case(False)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    grad_dyn_case(False)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_grad_dynamic_rank():
    """
    Feature: test Conv3DTranspose grad dynamic rank on GPU.
    Description: input is dynamic rank.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    grad_dyn_case(True)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    grad_dyn_case(True)


@pytest.mark.skip(reason="CPU无Conv3DBackpropFilter, Conv3DBackpropInput, kernel实现")
@pytest.mark.level2
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cpu_grad_dynamic_shape():
    """
    Feature: test Conv3DTranspose grad dynamic shape on CPU.
    Description: input is dynamic shape.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    grad_dyn_case(False)


@pytest.mark.skip(reason="CPU无Conv3DBackpropFilter, Conv3DBackpropInput, kernel实现")
@pytest.mark.level2
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cpu_grad_dynamic_rank():
    """
    Feature: test Conv3DTranspose grad dynamic rank on CPU.
    Description: input is dynamic rank.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    grad_dyn_case(True)
