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
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops.functional import vmap


context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class NetMaxPool3DGradGrad(nn.Cell):
    def __init__(self, mode, kernel, stride):
        super(NetMaxPool3DGradGrad, self).__init__()
        self.maxpool_grad_grad_fun = G.MaxPool3DGradGrad(pad_mode=mode,
                                                         kernel_size=kernel,
                                                         strides=stride)

    def construct(self, x, out, grad):
        return self.maxpool_grad_grad_fun(x, out, grad)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_maxpool3d_grad_grad_fp16():
    """
    Feature: MaxPool3dGradGrad cpu kernel
    Description: test the rightness of MaxPool3dGradGrad cpu kernel, pad_mode: VALID, dtype: float16
    Expectation: the output is same as expect output
    """
    data = (np.arange(1 * 3 * 2 * 4 * 4).astype(np.float16)
            ).reshape(1, 3, 2, 4, 4)
    x = Tensor(data)
    d = Tensor(data / 10)
    out = Tensor(np.array([[[[[21, 23],
                              [29, 31]]],
                            [[[53, 55],
                              [61, 63]]],
                            [[[85, 87],
                              [93, 95]]]]]).astype(np.float16))
    expect_result = (np.array([[[[[2.1, 2.3],
                                  [2.9, 3.1]]],
                                [[[5.3, 5.5],
                                  [6.1, 6.3]]],
                                [[[8.5, 8.7],
                                  [9.3, 9.5]]]]])).astype(np.float16)

    maxpool3d_grad_grad = NetMaxPool3DGradGrad("VALID", 2, 2)
    output = maxpool3d_grad_grad(x, out, d)
    assert np.allclose(output.asnumpy(), expect_result)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_maxpool3d_grad_grad_fp32():
    """
    Feature: MaxPool3dGradGrad cpu kernel
    Description: test the rightness of MaxPool3dGradGrad cpu kernel, pad_mode: SAME, dtype: float
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


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('axis', [2])
def test_maxpool3d_grad_grad_vmap(axis):
    """
    Feature: MaxPool3DGradGrad cpu kernel
    Description: test the rightness of MaxPool3DGradGrad cpu kernel vmap feature.
    Expectation: Success.
    """
    maxpool3d_grad_grad = NetMaxPool3DGradGrad("SAME", 3, 1)

    x = np.random.random((2, 3, 5, 5, 5, axis)).astype(np.float32)
    y = np.random.random((2, 3, 5, 5, 5, axis)).astype(np.float32)
    grad = np.random.random((2, 3, 5, 5, 5, axis)).astype(np.float32)

    output_vmap = vmap(maxpool3d_grad_grad, in_axes=(-1, -1, -1)
                       )(Tensor(x), Tensor(y), Tensor(grad))

    def manually_batched(x, y, grad):
        output = []
        for i in range(x.shape[-1]):
            output.append(maxpool3d_grad_grad(Tensor(x[:, :, :, :, :, i]), Tensor(
                y[:, :, :, :, :, i]), Tensor(grad[:, :, :, :, :, i])).asnumpy())
        return np.stack(output)

    expect = manually_batched(x, y, grad)
    assert np.array_equal(output_vmap.asnumpy(), expect)
