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

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self, num_true=1, num_sampled=5, unique=True, range_max=5, seed=0):
        super(Net, self).__init__()
        self.sampler = P.LogUniformCandidateSampler(num_true, num_sampled, unique, range_max, seed)

    def construct(self, x):
        return self.sampler(x)


def test_net_true():
    x = np.array([[1, 7], [0, 4], [3, 3]])
    net = Net(2, 5, True, 5)
    output = net(Tensor(x))
    print(output)


def test_net_false():
    x = np.array([[1, 7], [0, 4], [3, 3]])
    net = Net(2, 5, False, 10)
    output = net(Tensor(x))
    print(output)
