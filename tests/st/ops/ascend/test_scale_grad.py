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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations.inner_ops import ScaleGrad
from mindspore.common import dtype

context.set_context(device_target="Ascend")


class Net(nn.Cell):
    def __init__(self, scale):
        super(Net, self).__init__()
        self.scale_grad = ScaleGrad()
        self.scale = scale

    def construct(self, origin_grads):
        return self.scale_grad(origin_grads, self.scale)


def test_scale_grad_grad_float32_scale_float32():
    """
    Feature: Scale Grad fusion operation
    Description: test the rightness of ScaleGrad kernel, gradient's dtype is float32, scale's dtype is float32
    Expectation: the output is wrong
    """
    scale = Tensor(1024.0, dtype.float32)
    gradients = []
    for _ in range(3):
        gradients.append(Tensor(np.ones([3, 3]).astype(np.float32)))
    gradients_input = tuple(gradients)
    scale_grad = Net(scale)
    scale_grad(gradients_input)


def test_scale_grad_grad_float32_scale_float16():
    """
    Feature: Scale Grad fusion operation
    Description: test the rightness of ScaleGrad kernel, gradient's dtype is float32, scale's dtype is float16
    Expectation: the output is wrong
    """
    scale = Tensor(1024.0, dtype.float32)
    gradients = []
    for _ in range(3):
        gradients.append(Tensor(np.ones([3, 3]).astype(np.float32)))
    gradients_input = tuple(gradients)
    scale_grad = Net(scale)
    scale_grad(gradients_input)


def test_scale_grad_grad_float16_scale_float32():
    """
    Feature: Scale Grad fusion operation
    Description: test the rightness of ScaleGrad kernel, gradient's dtype is float16, scale's dtype is float32
    Expectation: the output is wrong
    """
    scale = Tensor(1024.0, dtype.float32)
    gradients = []
    for _ in range(3):
        gradients.append(Tensor(np.ones([3, 3]).astype(np.float16)))
    gradients_input = tuple(gradients)
    scale_grad = Net(scale)
    scale_grad(gradients_input)


def test_scale_grad_grad_float16_scale_float16():
    """
    Feature: Scale Grad fusion operation
    Description: test the rightness of ScaleGrad kernel, gradient's dtype is float16, scale's dtype is float16
    Expectation: the output is wrong
    """
    scale = Tensor(1024.0, dtype.float16)
    gradients = []
    for _ in range(3):
        gradients.append(Tensor(np.ones([3, 3]).astype(np.float16)))
    gradients_input = tuple(gradients)
    scale_grad = Net(scale)
    scale_grad(gradients_input)


def test_scale_grad_grad_mixed_scale_float32():
    """
    Feature: Scale Grad fusion operation
    Description: test the rightness of ScaleGrad kernel, gradient's dtype is mixed, scale's dtype is float32
    Expectation: the output is wrong
    """
    scale = Tensor(1024.0, dtype.float32)
    gradients = []
    for i in range(3):
        if (i % 2) == 0:
            gradients.append(Tensor(np.ones([3, 3]).astype(np.float32)))
        else:
            gradients.append(Tensor(np.ones([3, 3]).astype(np.float16)))
    gradients_input = tuple(gradients)
    scale_grad = Net(scale)
    scale_grad(gradients_input)


def test_scale_grad_grad_mixed_scale_float16():
    """
    Feature: Scale Grad fusion operation
    Description: test the rightness of ScaleGrad kernel, gradient's dtype is mixed, scale's dtype is float16
    Expectation: the output is wrong
    """
    scale = Tensor(1024.0, dtype.float16)
    gradients = []
    for i in range(3):
        if (i % 2) == 0:
            gradients.append(Tensor(np.ones([3, 3]).astype(np.float32)))
        else:
            gradients.append(Tensor(np.ones([3, 3]).astype(np.float16)))
    gradients_input = tuple(gradients)
    scale_grad = Net(scale)
    scale_grad(gradients_input)
