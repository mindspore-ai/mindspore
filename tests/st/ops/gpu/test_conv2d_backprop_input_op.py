# Copyright 2019 Huawei Technologies Co., Ltd
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
from mindspore.common.api import ms_function
from mindspore.ops import operations as P

context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')


class Conv2dInput(nn.Cell):
    def __init__(self):
        super(Conv2dInput, self).__init__()
        out_channel = 1
        kernel_size = 3
        self.conv_input = P.Conv2DBackpropInput(out_channel,
                                                kernel_size,
                                                pad_mode="valid",
                                                pad=0,
                                                mode=1,
                                                stride=1,
                                                dilation=1,
                                                group=1)

        self.get_shape = P.Shape()

    @ms_function
    def construct(self, out, w, x):
        return self.conv_input(out, w, self.get_shape(x))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_conv2d_backprop_input():
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
    conv2d_input = Conv2dInput()
    output = conv2d_input(out, w, x)
    expect = np.array([[[[-5, -4, 5, 12, 0, -8],
                         [-15, -6, 17, 17, -2, -11],
                         [-15, -8, 13, 12, 2, -4],
                         [-13, -6, 8, -14, 5, 20],
                         [-3, -4, -4, -19, 7, 23],
                         [-3, -2, 0, -14, 3, 16]]]]).astype(np.float32)

    assert (abs(output.asnumpy() - expect) < np.ones(shape=[1, 1, 6, 6]) * 1.0e-4).all()
