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
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.ops.operations import _grad_ops as G

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class NetReLUGradV2(nn.Cell):
    def __init__(self):
        super(NetReLUGradV2, self).__init__()
        self.relu_grad_v2 = G.ReluGradV2()
        self.dy = Parameter(initializer(Tensor(np.array([[[[1, 0, 1],
                                                           [0, 1, 0],
                                                           [1, 1, 1]]]]).astype(np.float32)), [1, 1, 3, 3]), name='dy')
        self.mask = Parameter(initializer(Tensor(np.array([[[[0, 1, 1],
                                                             [1, 0, 1],
                                                             [1, 1, 0]]]])
                                                 .astype(np.uint8)), [1, 1, 3, 3]), name='mask')

    def construct(self):
        return self.relu_grad_v2(self.dy, self.mask)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_relu_grad_v2():
    """
    Feature: ReLUGradV2 cpu kernel.
    Description: test the rightness of ReLUGradV2 cpu kernel.
    Expectation: the output is almost same as numpy output.
    """
    relu_grad_v2 = NetReLUGradV2()
    output = relu_grad_v2()
    expect = np.array([[[[0, 0, 1], [0, 0, 0], [1, 1, 0]]]]).astype(np.float32)
    error = np.ones(shape=[3, 3]) * 1.0e-6
    diff = np.abs(output.asnumpy() - expect)
    assert np.all(diff < error)
