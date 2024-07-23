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
import mindspore.context as context
from mindspore import Tensor
import mindspore.nn as nn
from mindspore.ops.operations._grad_ops import ScaleAndTranslateGrad
import pytest


class NetScaleAndTranslateGrad(nn.Cell):
    def __init__(self, kernel_type, antialias):
        super(NetScaleAndTranslateGrad, self).__init__()
        self.sclae_and_translate_grad_fun = ScaleAndTranslateGrad(kernel_type=kernel_type, antialias=antialias)

    def construct(self, grads, original_image, scale, translation):
        return self.sclae_and_translate_grad_fun(grads, original_image, scale, translation)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_scale_and_translate_grad_graph():
    """
    Feature: test operations in result and output type
    Description: test in graph mode on GPU
    Expectation: success or throw pytest error
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    grad = np.array([1, 2, 3, 4])
    grad = grad.reshape([1, 2, 2, 1]).astype(np.float32)
    gradients = Tensor(grad)
    ori = np.array([1, 2, 3, 4])
    ori = ori.reshape([1, 1, 4, 1]).astype(np.float32)
    origin = Tensor(ori)
    scale = np.array([1, 1]).astype(np.float32)
    scale_ms = Tensor(scale)
    translation = np.array([0.5, 0.5]).astype(np.float32)
    translation_ms = Tensor(translation)
    kernel_type = "mitchellcubic"
    antialias = True
    scale_and_translate_grad = NetScaleAndTranslateGrad(kernel_type, antialias)
    output = scale_and_translate_grad(gradients, origin, scale_ms, translation_ms)
    expected_output = np.array([[[[7.3784475], [2.822894], [-0.2013415], [0.0]]]]).astype(np.float32)
    error = np.ones(shape=expected_output.shape) * 1.0e-4
    diff = output.asnumpy() - expected_output
    assert np.all(abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_scale_and_translate_grad_pynative():
    """
    Feature: test operations in result and output type
    Description: test in graph mode on GPU
    Expectation: success or throw pytest error
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    grad = np.array([1, 2, 3, 4])
    grad = grad.reshape([1, 2, 2, 1]).astype(np.float32)
    gradients = Tensor(grad)
    ori = np.array([1, 2, 3, 4])
    ori = ori.reshape([1, 1, 4, 1]).astype(np.float32)
    origin = Tensor(ori)
    scale = np.array([1, 1]).astype(np.float32)
    scale_ms = Tensor(scale)
    translation = np.array([0.5, 0.5]).astype(np.float32)
    translation_ms = Tensor(translation)
    kernel_type = "mitchellcubic"
    antialias = True
    scale_and_translate_grad = NetScaleAndTranslateGrad(kernel_type, antialias)
    output = scale_and_translate_grad(gradients, origin, scale_ms, translation_ms)
    expected_output = np.array([[[[7.3784475], [2.822894], [-0.2013415], [0.0]]]]).astype(np.float32)
    error = np.ones(shape=expected_output.shape) * 1.0e-4
    diff = output.asnumpy() - expected_output
    assert np.all(abs(diff) < error)
