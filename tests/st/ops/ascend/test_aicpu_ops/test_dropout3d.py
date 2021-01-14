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

import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self, keep_prob, inplace):
        super(Net, self).__init__()
        self.drop = P.Dropout3d(keep_prob=keep_prob, inplace=inplace)

    def construct(self, x):
        return self.drop(x)


class NetInplace(nn.Cell):
    def __init__(self, keep_prob, inplace, x):
        super(NetInplace, self).__init__()
        self.drop = P.Dropout3d(keep_prob=keep_prob, inplace=inplace)
        self.x = x

    def construct(self):
        return self.drop(self.x)


def test_net_float32():
    x = Tensor(np.random.randn(3, 4, 3, 3, 3), mindspore.float32)
    net = Net(0.7, False)
    output = net(x)
    print(x)
    print(output)

    y = (output.asnumpy() == x.asnumpy()/0.7).reshape(3*4, 3*3*3)
    for i in range(3*4):
        if not y[i].all():
            assert y[i].sum() == 0


def test_net_float32_inplace():
    x = mindspore.Parameter(Tensor(np.random.randn(3, 4, 3, 3, 3), mindspore.float32))
    net = NetInplace(0.7, True, x)
    output = net()
    print(Tensor(x))
    print(output)
    assert np.array_equal(x.asnumpy(), output.asnumpy())
