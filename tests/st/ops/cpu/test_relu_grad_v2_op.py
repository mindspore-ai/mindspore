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
from mindspore.common.parameter import Parameter
from mindspore.ops.operations import _grad_ops as G

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64, np.int32])
def test_relu_grad_v2(dtype):
    """
    Feature: ReLUGradV2 cpu kernel.
    Description: test the rightness of ReLUGradV2 cpu kernel.
    Expectation: the output is almost same as numpy output.
    """
    class NetReLUGradV2(nn.Cell):
        def __init__(self):
            super(NetReLUGradV2, self).__init__()
            self.relu_grad_v2 = G.ReluGradV2()
            self.dy = Parameter(Tensor(np.array([[[[1, 0, 1],
                                                   [0, 1, 0],
                                                   [1, 1, 1]]]], dtype=dtype)), name='dy')
            self.mask = Parameter(Tensor(np.array([[[[0, 1, 1],
                                                     [1, 0, 1],
                                                     [1, 1, 0]]]], dtype=np.uint8)), name='mask')

        def construct(self):
            return self.relu_grad_v2(self.dy, self.mask)

    relu_grad_v2 = NetReLUGradV2()
    output = relu_grad_v2()
    expect = np.array([[[[0, 0, 1],
                         [0, 0, 0],
                         [1, 1, 0]]]], dtype=dtype)
    assert np.allclose(output.asnumpy(), expect)
