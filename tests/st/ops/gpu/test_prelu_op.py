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
from mindspore.ops import operations as P

class NetPReLU(nn.Cell):
    def __init__(self):
        super(NetPReLU, self).__init__()
        self.prelu = P.PReLU()

    def construct(self, x, weight):
        return self.prelu(x, weight)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_prelu_float16():
    weight = Tensor(np.array([0.25]).astype(np.float16))
    x = Tensor(np.array([[[[-1, 1, 10],
                           [1, -1, 1],
                           [10, 1, -1]]]]).astype(np.float16))
    expect = np.array([[[[-0.25, 1, 10,],
                         [1, -0.25, 1,],
                         [10, 1, -0.25]]]]).astype(np.float16)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    prelu = NetPReLU()
    output = prelu(x, weight)
    assert (output.asnumpy() == expect).all()

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    prelu = NetPReLU()
    output = prelu(x, weight)
    assert (output.asnumpy() == expect).all()

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_prelu_float32():
    weight = Tensor(np.array([0.25]).astype(np.float32))
    x = Tensor(np.array([[[[-1, 1, 10],
                           [1, -1, 1],
                           [10, 1, -1]]]]).astype(np.float32))
    expect = np.array([[[[-0.25, 1, 10,],
                         [1, -0.25, 1,],
                         [10, 1, -0.25]]]]).astype(np.float32)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    prelu = NetPReLU()
    output = prelu(x, weight)
    assert (output.asnumpy() == expect).all()

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    prelu = NetPReLU()
    output = prelu(x, weight)
    assert (output.asnumpy() == expect).all()
