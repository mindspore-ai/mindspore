# Copyright 2020 Huawei Technologies Co., Ltd
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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self, shape, seed=0, seed2=0):
        super(Net, self).__init__()
        self.gamma = P.Gamma(seed=seed, seed2=seed2)
        self.shape = shape

    def construct(self, alpha, beta):
        return self.gamma(self.shape, alpha, beta)


def test_net_1D():
    seed = 10
    shape = (3, 2, 4)
    alpha = 1.0
    beta = 1.0
    net = Net(shape=shape, seed=seed)
    talpha, tbeta = Tensor(alpha, mstype.float32), Tensor(beta, mstype.float32)
    output = net(talpha, tbeta)
    assert output.shape == (3, 2, 4)


def test_net_ND():
    seed = 10
    shape = (3, 1, 2)
    alpha = np.array([[[1], [2]], [[3], [4]], [[5], [6]]]).astype(np.float32)
    beta = np.array([1.0]).astype(np.float32)
    net = Net(shape=shape, seed=seed)
    talpha, tbeta = Tensor(alpha), Tensor(beta)
    output = net(talpha, tbeta)
    assert output.shape == (3, 2, 2)
