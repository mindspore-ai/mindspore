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


class NetMaxPoolGradGradWithArgmax(nn.Cell):
    def __init__(self, mode, kernel, stride):
        super(NetMaxPoolGradGradWithArgmax, self).__init__()
        self.maxpool_grad_grad_argmax_fun = G.MaxPoolGradGradWithArgmax(pad_mode=mode,
                                                                        kernel_size=kernel,
                                                                        strides=stride)

    def construct(self, x, grad, argmax):
        return self.maxpool_grad_grad_argmax_fun(x, grad, argmax)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("argmax_type", [np.int32, np.int64])
def test_maxpool_grad_grad_argmax_fp16(argmax_type):
    """
    Feature: MaxPoolGradGradWithArgmax cpu kernel
    Description: test the rightness of MaxPoolGradGradWithArgmax cpu kernel, pad_mode: VALID, dtype: float16
    Expectation: the output is same as expect output
    """
    data = (np.arange(1 * 2 * 6 * 6).astype(np.float16)).reshape(1, 2, 6, 6)
    x = Tensor(data)
    grad = Tensor(data / 10)
    argmax = Tensor(np.array([[[[7, 9, 11],
                                [19, 21, 23],
                                [31, 33, 35]],
                               [[43, 45, 47],
                                [55, 57, 59],
                                [67, 69, 71]]]]).astype(argmax_type))
    expect_result = (np.array([[[[0.7, 0.9, 1.1],
                                 [1.9, 2.1, 2.3],
                                 [3.1, 3.3, 3.5]],
                                [[4.3, 4.5, 4.7],
                                 [5.5, 5.7, 5.9],
                                 [6.7, 6.9, 7.1]]]])).astype(np.float16)

    maxpool_grad_grad_argmax = NetMaxPoolGradGradWithArgmax("VALID", 2, 2)
    output = maxpool_grad_grad_argmax(x, grad, argmax)
    assert np.allclose(output.asnumpy(), expect_result)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("argmax_type", [np.int32, np.int64])
def test_maxpool_grad_grad_argmax_fp32(argmax_type):
    """
    Feature: MaxPoolGradGradWithArgmax cpu kernel
    Description: test the rightness of MaxPoolGradGradWithArgmax cpu kernel, pad_mode: SAME, dtype: float
    Expectation: the output is same as expect output
    """
    data = (np.arange(2 * 1 * 5 * 5).astype(np.float32)).reshape(2, 1, 5, 5)
    x = Tensor(-1 * data)
    grad = Tensor(data / 10)
    argmax = Tensor(np.array([[[[0, 0, 1, 2, 3],
                                [0, 0, 1, 2, 3],
                                [5, 5, 6, 7, 8],
                                [10, 10, 11, 12, 13],
                                [15, 15, 16, 17, 18]]],
                              [[[0, 0, 1, 2, 3],
                                [0, 0, 1, 2, 3],
                                [5, 5, 6, 7, 8],
                                [10, 10, 11, 12, 13],
                                [15, 15, 16, 17, 18]]]]
                             ).astype(argmax_type))
    expect_result = (np.array(
        [[[[0, 0, 0.1, 0.2, 0.3],
           [0, 0, 0.1, 0.2, 0.3],
           [0.5, 0.5, 0.6, 0.7, 0.8],
           [1.0, 1.0, 1.1, 1.2, 1.3],
           [1.5, 1.5, 1.6, 1.7, 1.8]]],
         [[[2.5, 2.5, 2.6, 2.7, 2.8],
           [2.5, 2.5, 2.6, 2.7, 2.8],
           [3.0, 3.0, 3.1, 3.2, 3.3],
           [3.5, 3.5, 3.6, 3.7, 3.8],
           [4.0, 4.0, 4.1, 4.2, 4.3]]]])).astype(np.float32)

    maxpool_grad_grad_argmax = NetMaxPoolGradGradWithArgmax("SAME", 3, 1)
    output = maxpool_grad_grad_argmax(x, grad, argmax)
    assert np.allclose(output.asnumpy(), expect_result)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('axis', [2])
def test_maxpool_grad_grad_argmax_vmap(axis):
    """
    Feature: MaxPoolGradGradWithArgmax cpu kernel
    Description: test the rightness of MaxPoolGradGradWithArgmax cpu kernel vmap feature.
    Expectation: Success.
    """
    maxpool_grad_grad_argmax = NetMaxPoolGradGradWithArgmax("SAME", 3, 1)

    x = np.random.random((2, 3, 5, 5, axis)).astype(np.float32)
    grad = np.random.random((2, 3, 5, 5, axis)).astype(np.float32)
    argmax = np.random.random((2, 3, 5, 5, axis)).astype(np.int32)

    output_vmap = vmap(maxpool_grad_grad_argmax, in_axes=(-1, -1, -1)
                       )(Tensor(x), Tensor(grad), Tensor(argmax))

    def manually_batched(x, grad, argmax):
        output = []
        for i in range(x.shape[-1]):
            output.append(maxpool_grad_grad_argmax(Tensor(x[:, :, :, :, i]), Tensor(
                grad[:, :, :, :, i]), Tensor(argmax[:, :, :, :, i])).asnumpy())
        return np.stack(output)

    expect = manually_batched(x, grad, argmax)
    assert np.array_equal(output_vmap.asnumpy(), expect)
