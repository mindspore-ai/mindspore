# Copyright 2021 Huawei Technologies Co., Ltd
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


class NetReluGrad(nn.Cell):
    def __init__(self):
        super(NetReluGrad, self).__init__()
        self.relu6_grad = G.ReLU6Grad()
        self.x = Parameter(initializer(Tensor(np.array([[[[1, 0, 6],
                                                          [-2, 3, 6],
                                                          [-3, 1, 8]]]]).astype(np.float32)), [1, 1, 3, 3]), name='x')
        self.dy = Parameter(initializer(Tensor(np.array([[[[1, 2, 3],
                                                           [4, 5, 6],
                                                           [7, 8, 9]]]]).astype(np.float32)), [1, 1, 3, 3]), name='dy')

    def construct(self):
        return self.relu6_grad(self.dy, self.x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_relu_grad():
    relu_grad = NetReluGrad()
    output = relu_grad()
    expect = np.array([[[[1, 0, 3], [0, 5, 6], [0, 8, 0]]]]).astype(np.float32)
    error = np.ones(shape=[3, 3]) * 1.0e-6
    diff = np.abs(output.asnumpy() - expect)
    assert np.all(diff < error)
