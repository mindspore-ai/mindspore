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


class Conv2dTransposeNet(nn.Cell):
    def __init__(self):
        super(Conv2dTransposeNet, self).__init__()
        out_channel = 1
        kernel_size = 3
        self.conv_input = P.Conv2DTranspose(out_channel,
                                            kernel_size,
                                            pad_mode="valid",
                                            pad=0,
                                            mode=1,
                                            stride=1,
                                            dilation=1,
                                            group=1)

    def construct(self, out, w, shape):
        return self.conv_input(out, w, shape)


def dynamic_shape():
    test_dynamic = TestDynamicGrad(Conv2dTransposeNet(), skip_convert_out_ids=[2])
    w = Tensor(np.array([[[[1, 0, -1], [1, 0, -1], [1, 0, -1]]]]).astype(np.float32))
    x = Tensor(np.array([[[
        [3, 0, 1, 2, 7, 4],
        [1, 5, 8, 9, 3, 1],
        [2, 7, 2, 5, 1, 3],
        [0, 1, 3, 1, 7, 8],
        [4, 2, 1, 6, 2, 8],
        [2, 4, 5, 2, 3, 9]]]]).astype(np.float32))
    out = Tensor(np.array([[[
        [-5, -4, 0, 8],
        [-10, -2, 2, 3],
        [0, -2, -4, -7],
        [-3, -2, -3, -16]]]]).astype(np.float32))
    test_dynamic.test_dynamic_grad_net((out, w, x.shape))


def dynamic_rank():
    test_dynamic = TestDynamicGrad(Conv2dTransposeNet(), skip_convert_out_ids=[2])
    w = Tensor(np.array([[[[1, 0, -1], [1, 0, -1], [1, 0, -1]]]]).astype(np.float32))
    x = Tensor(np.array([[[
        [3, 0, 1, 2, 7, 4],
        [1, 5, 8, 9, 3, 1],
        [2, 7, 2, 5, 1, 3],
        [0, 1, 3, 1, 7, 8],
        [4, 2, 1, 6, 2, 8],
        [2, 4, 5, 2, 3, 9]]]]).astype(np.float32))
    out = Tensor(np.array([[[
        [-5, -4, 0, 8],
        [-10, -2, 2, 3],
        [0, -2, -4, -7],
        [-3, -2, -3, -16]]]]).astype(np.float32))
    test_dynamic.test_dynamic_grad_net((out, w, x.shape), True)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dynamic_conv2dtranspose_cpu():
    """
    Feature: Conv2dTranspose Grad DynamicShape.
    Description: Test case of dynamic shape for Conv2dTranspose grad operator on CPU.
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
def test_dynamic_conv2dtranspose_gpu():
    """
    Feature: Conv2dTranspose Grad DynamicShape.
    Description: Test case of dynamic shape for Conv2dTranspose grad operator on GPU.
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


def test_dynamic_conv2dtranspose_ascend():
    """
    Feature: Conv2dTranspose Grad DynamicShape.
    Description: Test case of dynamic shape for Conv2dTranspose grad operator on Ascend.
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
