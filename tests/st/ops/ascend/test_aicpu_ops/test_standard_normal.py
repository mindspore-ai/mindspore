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

import mindspore.context as context
import mindspore.nn as nn
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self, shape, seed=0, seed2=0):
        super(Net, self).__init__()
        self.shape = shape
        self.seed = seed
        self.seed2 = seed2
        self.stdnormal = P.StandardNormal(seed, seed2)

    def construct(self):
        return self.stdnormal(self.shape)


def test_net():
    seed = 10
    seed2 = 10
    shape = (3, 2, 4)
    net = Net(shape, seed, seed2)
    output = net()
    assert output.shape == (3, 2, 4)
