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

import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class Net(nn.Cell):
    def __init__(self, shape, seed=0, seed2=0):
        super(Net, self).__init__()
        self.shape = shape
        self.seed = seed
        self.seed2 = seed2
        self.stdnormal = P.StandardNormal(seed, seed2)

    def construct(self):
        return self.stdnormal(self.shape)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_net():
    seed = 10
    seed2 = 10
    shape = (5, 6, 8)
    net = Net(shape, seed, seed2)
    output = net()
    assert output.shape == (5, 6, 8)
    outnumpyflatten_1 = output.asnumpy().flatten()

    seed = 0
    seed2 = 10
    shape = (5, 6, 8)
    net = Net(shape, seed, seed2)
    output = net()
    assert output.shape == (5, 6, 8)
    outnumpyflatten_2 = output.asnumpy().flatten()
    # same seed should generate same random number
    assert (outnumpyflatten_1 == outnumpyflatten_2).all()

    seed = 0
    seed2 = 0
    shape = (130, 120, 141)
    net = Net(shape, seed, seed2)
    output = net()
    assert output.shape == (130, 120, 141)
