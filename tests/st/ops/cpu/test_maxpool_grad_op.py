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
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.ops.operations import _grad_ops as G

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class Net_Pool_Grad(nn.Cell):
    def __init__(self):
        super(Net_Pool_Grad, self).__init__()
        self.maxpool_grad_fun = G.MaxPoolGrad(pad_mode="VALID",
                                              kernel_size=2,
                                              strides=2)

        self.x = Parameter(initializer(
            Tensor(np.array([[[
                [0, 1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10, 11],
                [12, 13, 14, 15, 16, 17],
                [18, 19, 20, 21, 22, 23],
                [24, 25, 26, 27, 28, 29],
                [30, 31, 32, 33, 34, 35]
            ]]]).astype(np.float32)), [1, 1, 6, 6]), name='x')

        self.a = Parameter(initializer(
            Tensor(np.array([[[
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3]
            ]]]).astype(np.float32)), [1, 1, 3, 3]), name='a')

        self.d = Parameter(initializer(
            Tensor(np.array([[[
                [7, 9, 11],
                [19, 21, 23],
                [31, 33, 35]
            ]]]).astype(np.float32)), [1, 1, 3, 3]), name='d')

    def construct(self):
        return self.maxpool_grad_fun(self.x, self.a, self.d)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_maxpool2d_grad():
    maxpool2d_grad = Net_Pool_Grad()
    output = maxpool2d_grad()
    print(output)

    expect_result = (np.array([[[
        [0, 0, 0, 0, 0, 0],
        [0, 7, 0, 9, 0, 11],
        [0, 0, 0, 0, 0, 0],
        [0, 19, 0, 21, 0, 23],
        [0, 0, 0, 0, 0, 0],
        [0, 31, 0, 33, 0, 35]
    ]]]))
    assert (output.asnumpy() == expect_result).all()
