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

from tests.mark_utils import arg_mark
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.softmax = P.Softmax(axis=1)
        self.cast = P.Cast()
        self.relu = P.ReLU()
        self.biasadd = P.BiasAdd()

    def construct(self, x, y):
        x = self.relu(x)
        x = self.relu(x)
        x = self.relu(x)
        x = self.biasadd(x, y)
        x = self.relu(x)
        x = self.relu(x)
        x = self.relu(x)
        return x


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_net():
    x = np.random.randn(32, 10).astype(np.float32)
    y = np.random.randn(10).astype(np.float32)
    net = Net()
    output = net(Tensor(x), Tensor(y))
    print(output.asnumpy())
