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
from tests.mark_utils import arg_mark

import numpy as np
import pytest
from mindspore import nn, context, Tensor
from mindspore.ops.operations import Conv3D
from .test_grad_of_dynamic import TestDynamicGrad


class NetConv3d(nn.Cell):
    def __init__(self):
        super(NetConv3d, self).__init__()
        out_channel = 4
        kernel_size = 2
        self.conv = Conv3D(out_channel, kernel_size, mode=1, pad_mode="valid", pad=0, stride=1, dilation=1, group=1)

    def construct(self, x, w):
        return self.conv(x, w)


def grad_dyn_case(is_dynamic_rank):
    test_dynamic = TestDynamicGrad(NetConv3d())
    input_np = np.arange(1 * 3 * 3 * 3 * 3).reshape(1, 3, 3, 3, 3).astype(np.float32)
    weight_np = np.arange(4 * 3 * 2 * 2 * 2).reshape(4, 3, 2, 2, 2).astype(np.float32)
    test_dynamic.test_dynamic_grad_net([Tensor(input_np), Tensor(weight_np)], is_dynamic_rank)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_grad_dynamic_shape_1():
    """
    Feature: test Conv3D grad dynamic shape.
    Description: input is dynamic shape.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    grad_dyn_case(False)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_grad_dynamic_rank_1():
    """
    Feature: test Conv3D grad dynamic rank.
    Description: input is dynamic rank.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    grad_dyn_case(True)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_gpu_grad_dynamic_shape_2():
    """
    Feature: test Conv3D grad dynamic shape on GPU.
    Description: input is dynamic shape.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    grad_dyn_case(False)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_gpu_grad_dynamic_rank_2():
    """
    Feature: test Conv3D grad dynamic shape on GPU.
    Description: input is dynamic shape.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    grad_dyn_case(True)
