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
import mindspore as ms
from mindspore import Tensor
import mindspore.context as context
import mindspore.nn as nn
from mindspore import dtype as mstype
from mindspore.ops.functional import vmap
from mindspore.ops.operations import _grad_ops as G

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class MaxPoolGradWithArgmaxV2(nn.Cell):
    def __init__(self, kernel_size, strides, pads, dilation, ceil_mode, argmax_type=mstype.int64):
        super(MaxPoolGradWithArgmaxV2, self).__init__()
        self.grad = G.MaxPoolGradWithArgmaxV2(kernel_size, strides, pads, dilation, ceil_mode, argmax_type)

    def construct(self, x, grad, argmax):
        return self.grad(x, grad, argmax)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_maxpool_grad_with_argmax_v2():
    """
    Feature: test maxPoolGradWithArgmaxV2 cpu version.
    Description: comparing to gpu version
    Expectation: expect correct result.
    """
    x = Tensor(np.array([[[
        [0, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11],
        [12, 13, 14, 15, 16, 17],
        [18, 19, 20, 21, 22, 23],
        [24, 25, 26, 27, 28, 29],
        [30, 31, 32, 33, 34, 35]
    ]]]).astype(np.float32))
    dy = Tensor(np.array([[[
        [0.7, 0.9, 0.11],
        [0.19, 0.21, 0.23],
        [0.31, 0.33, 0.35]
    ]]]).astype(np.float32))
    index = Tensor(np.array([[[
        [7, 9, 11],
        [19, 21, 23],
        [31, 33, 35]
    ]]]).astype(np.int32))
    expect_result = (np.array([[[
        [0., 0., 0., 0., 0., 0.],
        [0., 0.7, 0., 0.9, 0., 0.11],
        [0., 0., 0., 0., 0., 0.],
        [0., 0.19, 0., 0.21, 0., 0.23],
        [0., 0., 0., 0., 0., 0.],
        [0., 0.31, 0., 0.33, 0., 0.35]]]]).astype(np.float32))
    grad_max_pool = MaxPoolGradWithArgmaxV2(kernel_size=2, strides=2, pads=0, dilation=1, ceil_mode=False)
    actual_output = grad_max_pool(x, dy, index)
    assert (actual_output.asnumpy() == expect_result).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_maxpool_grad_with_argmax_v2_fp16():
    """
    Feature: test MaxPoolGradWithArgmaxV2 cpu version.
    Description: comparing to gpu version
    Expectation: expect correct result.
    """
    x = Tensor(np.array([[[
        [0, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11],
        [12, 13, 14, 15, 16, 17],
        [18, 19, 20, 21, 22, 23],
        [24, 25, 26, 27, 28, 29],
        [30, 31, 32, 33, 34, 35]
    ]]]).astype(np.float16))
    dy = Tensor(np.array([[[
        [0.7, 0.9, 0.11],
        [0.19, 0.21, 0.23],
        [0.31, 0.33, 0.35]
    ]]]).astype(np.float16))
    index = Tensor(np.array([[[
        [7, 9, 11],
        [19, 21, 23],
        [31, 33, 35]
    ]]]).astype(np.int32))
    expect_result = np.array([[[
        [0., 0., 0., 0., 0., 0.],
        [0., 0.7, 0., 0.9, 0., 0.11],
        [0., 0., 0., 0., 0., 0.],
        [0., 0.19, 0., 0.21, 0., 0.23],
        [0., 0., 0., 0., 0., 0.],
        [0., 0.31, 0., 0.33, 0., 0.35]
    ]]]).astype(np.float16)
    grad_max_pool = MaxPoolGradWithArgmaxV2(kernel_size=2, strides=2, pads=0, dilation=1, ceil_mode=False)
    actual_output = grad_max_pool(x, dy, index)
    assert (actual_output.asnumpy() == expect_result).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_maxpool_grad_with_argmax_v2_vmap():
    """
    Feature: test maxPoolGradWithArgmaxV2 gpu vmap version.
    Description: comparing to gpu version
    Expectation: expect correct result.
    """
    x = Tensor(np.arange(6 * 6).reshape(1, 1, 6, 6, 1).astype(np.float32))
    dy = Tensor(np.array([[[
        [[0.7], [0.9], [0.11]],
        [[0.19], [0.21], [0.23]],
        [[0.31], [0.33], [0.35]]
    ]]]).astype(np.float32))
    index = Tensor(np.array([[[
        [[7], [9], [11]],
        [[19], [21], [23]],
        [[31], [33], [35]]
    ]]]).astype(np.int32))
    grad_max_pool = MaxPoolGradWithArgmaxV2(kernel_size=2, strides=2, pads=0, dilation=1, ceil_mode=False)
    nest_vmap = vmap(grad_max_pool, in_axes=(-1, -1, -1), out_axes=0)
    actual_output = nest_vmap(x, dy, index)
    assert actual_output.shape == (1, 1, 1, 6, 6)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_maxpool_grad_with_argmax_v2_x_dynamic_shape():
    """
    Feature: test MaxPoolGradWithArgmaxV2 cpu version.
    Description: comparing to gpu version
    Expectation: expect correct result.
    """
    x = Tensor(np.array([[[
        [0, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11],
        [12, 13, 14, 15, 16, 17],
        [18, 19, 20, 21, 22, 23],
        [24, 25, 26, 27, 28, 29],
        [30, 31, 32, 33, 34, 35]
    ]]]), dtype=ms.float16)
    dy = Tensor(np.array([[[
        [0.7, 0.9, 0.11],
        [0.19, 0.21, 0.23],
        [0.31, 0.33, 0.35]
    ]]]), dtype=ms.float16)
    index = Tensor(np.array([[[
        [7, 9, 11],
        [19, 21, 23],
        [31, 33, 35]
    ]]]), dtype=ms.int32)
    expect_result = np.array([[[
        [0., 0., 0., 0., 0., 0.],
        [0., 0.7, 0., 0.9, 0., 0.11],
        [0., 0., 0., 0., 0., 0.],
        [0., 0.19, 0., 0.21, 0., 0.23],
        [0., 0., 0., 0., 0., 0.],
        [0., 0.31, 0., 0.33, 0., 0.35]
    ]]]).astype(np.float16)
    grad_max_pool = MaxPoolGradWithArgmaxV2(kernel_size=2, strides=2, pads=0, dilation=1, ceil_mode=False)
    x_dyn = Tensor(shape=[x.shape[0], None, x.shape[2], x.shape[3]], dtype=ms.float16)
    dy_dyn = Tensor(shape=[dy.shape[0], None, None, dy.shape[3]], dtype=ms.float16)
    index_dyn = Tensor(shape=[index.shape[0], index.shape[1], index.shape[2], None], dtype=ms.int32)
    grad_max_pool.set_inputs(x_dyn, dy_dyn, index_dyn)
    actual_output = grad_max_pool(x, dy, index)
    assert (actual_output.asnumpy() == expect_result).all()
