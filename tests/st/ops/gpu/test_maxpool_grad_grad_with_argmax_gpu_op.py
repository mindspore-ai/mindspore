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


class NetMaxPoolGradGradWithArgmax(nn.Cell):
    def __init__(self, mode, kernel, stride):
        super(NetMaxPoolGradGradWithArgmax, self).__init__()
        self.maxpool_grad_grad_argmax_fun = G.MaxPoolGradGradWithArgmax(pad_mode=mode,
                                                                        kernel_size=kernel,
                                                                        strides=stride)

    def construct(self, x, grad, argmax):
        return self.maxpool_grad_grad_argmax_fun(x, grad, argmax)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("argmax_type", [np.int32, np.int64])
def test_maxpool_grad_grad_argmax_fp16(argmax_type):
    """
    Feature: MaxPoolGradGradWithArgmax gpu kernel
    Description: test the rightness of MaxPoolGradGradWithArgmax gpu kernel, pad_mode: VALID, dtype: float16
    Expectation: the output is same as expect output
    """
    data = (np.arange(1 * 2 * 4 * 4).astype(np.float16)).reshape(1, 2, 4, 4)
    x = Tensor(data)
    grad = Tensor(data / 10)
    argmax = Tensor(np.array([[[[5, 7],
                                [13, 15]],
                               [[21, 23],
                                [29, 31]]]]).astype(argmax_type))
    expect_result = (np.array([[[[0.5, 0.7],
                                 [1.3, 1.5]],
                                [[2.1, 2.3],
                                 [2.9, 3.1]]]])).astype(np.float16)

    maxpool_grad_grad_argmax = NetMaxPoolGradGradWithArgmax("VALID", 2, 2)
    output = maxpool_grad_grad_argmax(x, grad, argmax)
    assert np.allclose(output.asnumpy(), expect_result)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("argmax_type", [np.int32, np.int64])
def test_maxpool_grad_grad_argmax_fp32(argmax_type):
    """
    Feature: MaxPoolGradGradWithArgmax gpu kernel
    Description: test the rightness of MaxPoolGradGradWithArgmax gpu kernel, pad_mode: SAME, dtype: float
    Expectation: the output is same as expect output
    """
    data = (np.arange(2 * 1 * 3 * 3).astype(np.float32)
            ).reshape(2, 1, 3, 3) * (-1)
    x = Tensor(data)
    grad = Tensor(data / 10)
    argmax = Tensor(np.array([[[[0, 0, 1],
                                [0, 0, 1],
                                [3, 3, 4]]],
                              [[[0, 0, 1],
                                [0, 0, 1],
                                [3, 3, 4]]]]).astype(argmax_type))
    expect_result = (np.array([[[[0, 0, -0.1],
                                 [0, 0, -0.1],
                                 [-0.3, -0.3, -0.4]]],
                               [[[-0.9, -0.9, -1.0],
                                 [-0.9, -0.9, -1.0],
                                 [-1.2, -1.2, -1.3]]]]).astype(np.float32))

    maxpool_grad_grad_argmax = NetMaxPoolGradGradWithArgmax("SAME", 3, 1)
    output = maxpool_grad_grad_argmax(x, grad, argmax)
    assert np.allclose(output.asnumpy(), expect_result)
