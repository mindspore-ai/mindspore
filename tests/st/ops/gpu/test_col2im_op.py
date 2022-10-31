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
from mindspore import dtype as mstype
from mindspore import Tensor
from mindspore.ops.operations.array_ops import Col2Im
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype

np.random.seed(1)


class Col2ImTest(nn.Cell):

    def __init__(self, kernel_size, dilation, padding, stride):
        super(Col2ImTest, self).__init__()
        self.c2i = Col2Im(kernel_size, dilation, padding, stride)

    def construct(self, x, output_size):
        return self.c2i(x, output_size)


def dyn_case():
    kernel_size = [2, 2]
    dilation = [2, 2]
    padding = [2, 2]
    stride = [2, 2]
    col2im = Col2ImTest(kernel_size=kernel_size,
                        dilation=dilation,
                        padding=padding,
                        stride=stride)

    x_dyn = Tensor(shape=[None, None, None, None], dtype=mstype.float32)
    output_size_dyn = Tensor(shape=[None], dtype=mstype.int32)
    col2im.set_inputs(x_dyn, output_size_dyn)

    x = Tensor(np.random.rand(16, 16, 4, 25).astype(np.float32))
    output_size = Tensor([8, 8], dtype=mstype.int32)
    output = col2im(x, output_size)

    expect_shape = (16, 16, 8, 8)
    assert output.shape == expect_shape


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_col2im_dyn():
    """
    Feature: Col2Im function.
    Description:  test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    dyn_case()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    dyn_case()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode, input_type",
                         [(context.GRAPH_MODE, np.float32),
                          (context.PYNATIVE_MODE, np.float32),
                          (context.GRAPH_MODE, np.float16),
                          (context.PYNATIVE_MODE, np.float16),
                          (context.GRAPH_MODE, np.float64),
                          (context.PYNATIVE_MODE, np.float64),
                          (context.GRAPH_MODE, np.complex64),
                          (context.PYNATIVE_MODE, np.complex64),
                          (context.GRAPH_MODE, np.complex128),
                          (context.PYNATIVE_MODE, np.complex128)])
def test_col2im_op(mode, input_type):
    """
    Feature: Celu cpu kernel
    Description: test the celu alpha = 1.0.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=mode, device_target='GPU')
    x = Tensor(np.random.rand(16, 16, 4, 25).astype(input_type))
    output_size = Tensor([8, 8], dtype=mstype.int32)
    kernel_size = [2, 2]
    dilation = [2, 2]
    padding = [2, 2]
    stride = [2, 2]
    expect_shape = (16, 16, 8, 8)
    col2im = Col2ImTest(kernel_size=kernel_size,
                        dilation=dilation,
                        padding=padding,
                        stride=stride)
    output = col2im(x, output_size)
    assert output.shape == expect_shape

    output_func = F.col2im(x, output_size, kernel_size, dilation, padding,
                           stride)
    assert output_func.shape == expect_shape

    assert x.col2im(output_size, kernel_size, dilation, padding,
                    stride).shape == expect_shape
