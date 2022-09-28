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


class NetMaxPool3DGradGrad(nn.Cell):
    def __init__(self, mode, kernel, stride):
        super(NetMaxPool3DGradGrad, self).__init__()
        self.maxpool_grad_grad_fun = G.MaxPool3DGradGrad(pad_mode=mode,
                                                         kernel_size=kernel,
                                                         strides=stride)

    def construct(self, x, out, grad):
        return self.maxpool_grad_grad_fun(x, out, grad)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu
@pytest.mark.env_onecard
def test_maxpool3d_grad_grad_fp16():
    """
    Feature: MaxPool3DGradGrad gpu kernel
    Description: test the rightness of MaxPool3DGradGrad gpu kernel, pad_mode: VALID, dtype: float16
    Expectation: the output is same as expect output
    """
    data = (np.arange(1 * 2 * 2 * 6 * 6).astype(np.float16)
            ).reshape(1, 2, 2, 6, 6)
    x = Tensor(data)
    d = Tensor(data / 10)
    out = Tensor(np.array([[[[[43, 45, 47],
                              [55, 57, 59],
                              [67, 69, 71]]],
                            [[[115, 117, 119],
                              [127, 129, 131],
                              [139, 141, 143]]]]]).astype(np.float16))
    expect_result = np.array([[[[[4.3, 4.5, 4.7],
                                 [5.5, 5.7, 5.9],
                                 [6.7, 6.9, 7.1]]],
                               [[[11.5, 11.7, 11.9],
                                 [12.7, 12.9, 13.1],
                                 [13.9, 14.1, 14.3]]]]]).astype(np.float16)

    maxpool3d_grad_grad = NetMaxPool3DGradGrad("VALID", 2, 2)
    output = maxpool3d_grad_grad(x, out, d)
    assert np.allclose(output.asnumpy(), expect_result)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu
@pytest.mark.env_onecard
def test_maxpool3d_grad_grad_fp32():
    """
    Feature: MaxPool3DGradGrad gpu kernel
    Description: test the rightness of MaxPool3DGradGrad gpu kernel, pad_mode: SAME, dtype: float
    Expectation: the output is same as expect output
    """
    data = (np.arange(2 * 1 * 3 * 3 * 3).astype(np.float32)
            ).reshape(2, 1, 3, 3, 3)
    x = Tensor(data)
    d = Tensor(data / 10)
    out = Tensor(np.array([[[[[13, 14, 14],
                              [16, 17, 17],
                              [16, 17, 17]],
                             [[22, 23, 23],
                              [25, 26, 26],
                              [25, 26, 26]],
                             [[22, 23, 23],
                              [25, 26, 26],
                              [25, 26, 26]]]],

                           [[[[40, 41, 41],
                              [43, 44, 44],
                              [43, 44, 44]],
                             [[49, 50, 50],
                              [52, 53, 53],
                              [52, 53, 53]],
                             [[49, 50, 50],
                              [52, 53, 53],
                              [52, 53, 53]]]]]).astype(np.float32))

    expect_result = (np.array([[[[[1.3, 1.4, 1.4],
                                  [1.6, 1.7, 1.7],
                                  [1.6, 1.7, 1.7]],
                                 [[2.2, 2.3, 2.3],
                                  [2.5, 2.6, 2.6],
                                  [2.5, 2.6, 2.6]],
                                 [[2.2, 2.3, 2.3],
                                  [2.5, 2.6, 2.6],
                                  [2.5, 2.6, 2.6]]]],

                               [[[[4.0, 4.1, 4.1],
                                  [4.3, 4.4, 4.4],
                                  [4.3, 4.4, 4.4]],
                                 [[4.9, 5.0, 5.0],
                                  [5.2, 5.3, 5.3],
                                  [5.2, 5.3, 5.3]],
                                 [[4.9, 5.0, 5.0],
                                  [5.2, 5.3, 5.3],
                                  [5.2, 5.3, 5.3]]]]])).astype(np.float32)

    maxpool3d_grad_grad = NetMaxPool3DGradGrad("SAME", 3, 1)
    output = maxpool3d_grad_grad(x, out, d)
    assert np.allclose(output.asnumpy(), expect_result)
