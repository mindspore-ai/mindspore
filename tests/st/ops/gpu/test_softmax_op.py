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

import pytest
import numpy as np
from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.nn as nn
import mindspore.context as context

class NetSoftmax(nn.Cell):
    def __init__(self):
        super(NetSoftmax, self).__init__()
        axis = -2
        self.softmax1 = P.Softmax()
        self.softmax2 = P.Softmax(axis)

    def construct(self, x):
        return self.softmax1(x), self.softmax2(x)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_softmax():
    x = Tensor(np.array([[0.1, 0.3, 0.6, -0.3],
                         [0.2, -0.6, 0.8, 0.6],
                         [0.6, -1.2, 0.4, 0.6]]).astype(np.float32))
    expect1 = np.ones(3)
    expect2 = np.ones(4)
    error1 = expect1 * 1.0e-6
    error2 = expect2 * 1.0e-6

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    Softmax = NetSoftmax()
    output = Softmax(x)
    outputSum1 = output[0].asnumpy().sum(axis=1)
    outputSum2 = output[1].asnumpy().sum(axis=0)
    diff1 = np.abs(outputSum1 - expect1)
    diff2 = np.abs(outputSum2 - expect2)
    assert np.all(diff1 < error1)
    assert np.all(diff2 < error2)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    Softmax = NetSoftmax()
    output = Softmax(x)
    outputSum1 = output[0].asnumpy().sum(axis=1)
    outputSum2 = output[1].asnumpy().sum(axis=0)
    diff1 = np.abs(outputSum1 - expect1)
    diff2 = np.abs(outputSum2 - expect2)
    assert np.all(diff1 < error1)
    assert np.all(diff2 < error2)
