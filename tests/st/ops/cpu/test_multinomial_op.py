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
from mindspore.ops import composite as C

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class Net(nn.Cell):
    def __init__(self, sample, replacement, seed=0):
        super(Net, self).__init__()
        self.sample = sample
        self.replacement = replacement
        self.seed = seed

    def construct(self, x):
        return C.multinomial(x, self.sample, self.replacement, self.seed)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_multinomial_net():
    x0 = Tensor(np.array([0.9, 0.2]).astype(np.float32))
    x1 = Tensor(np.array([[0.9, 0.2], [0.9, 0.2]]).astype(np.float32))
    net0 = Net(1, True, 20)
    net1 = Net(2, True, 20)
    net2 = Net(6, True, 20)
    out0 = net0(x0)
    out1 = net1(x0)
    out2 = net2(x1)
    assert out0.asnumpy().shape == (1,)
    assert out1.asnumpy().shape == (2,)
    assert out2.asnumpy().shape == (2, 6)
