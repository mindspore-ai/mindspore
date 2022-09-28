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
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import _grad_ops as G


context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class NetMaxPoolGradGrad(nn.Cell):
    def __init__(self, mode, kernel, stride):
        super(NetMaxPoolGradGrad, self).__init__()
        self.maxpool_grad_grad_fun = G.MaxPoolGradGrad(pad_mode=mode,
                                                       kernel_size=kernel,
                                                       strides=stride)

    def construct(self, x, out, grad):
        return self.maxpool_grad_grad_fun(x, out, grad)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_maxpool2d_grad_grad_fp16():
    """
    Feature: MaxPool2dGradGrad gpu kernel
    Description: test the rightness of MaxPool2dGradGrad gpu kernel, pad_mode: VALID, dtype: float16
    Expectation: the output is same as expect output
    """
    data = (np.arange(1 * 1 * 6 * 6).astype(np.float16)).reshape(1, 1, 6, 6)
    x = Tensor(data)
    d = Tensor(data / 10)
    out = Tensor(np.array([[[
        [7, 9, 11],
        [19, 21, 23],
        [31, 33, 35]
    ]]]).astype(np.float16))
    expect_result = (np.array([[[
        [0.7, 0.9, 1.1],
        [1.9, 2.1, 2.3],
        [3.1, 3.3, 3.5]
    ]]])).astype(np.float16)

    maxpool2d_grad_grad = NetMaxPoolGradGrad("VALID", 2, 2)
    output = maxpool2d_grad_grad(x, out, d)
    assert np.allclose(output.asnumpy(), expect_result)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_maxpool2d_grad_grad_fp32():
    """
    Feature: MaxPool2dGradGrad gpu kernel
    Description: test the rightness of MaxPool2dGradGrad gpu kernel, pad_mode: SAME, dtype: float
    Expectation: the output is same as expect output
    """
    data = (np.arange(2 * 1 * 3 * 3).astype(np.float32)).reshape(2, 1, 3, 3)
    x = Tensor(data)
    d = Tensor(data / 10 * (-1))
    out = Tensor(np.array([[[[4, 5, 5],
                             [7, 8, 8],
                             [7, 8, 8]]],
                           [[[13, 14, 14],
                             [16, 17, 17],
                             [16, 17, 17]]]]).astype(np.float32))
    expect_result = (np.array([[[[-0.4, -0.5, -0.5],
                                 [-0.7, -0.8, -0.8],
                                 [-0.7, -0.8, -0.8]]],
                               [[[-1.3, -1.4, -1.4],
                                 [-1.6, -1.7, -1.7],
                                 [-1.6, -1.7, -1.7]]]]).astype(np.float32))

    maxpool2d_grad_grad = NetMaxPoolGradGrad("SAME", 3, 1)
    output = maxpool2d_grad_grad(x, out, d)
    assert np.allclose(output.asnumpy(), expect_result)
