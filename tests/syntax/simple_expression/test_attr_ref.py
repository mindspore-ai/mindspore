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
""" test syntax for logic expression """

import mindspore.nn as nn
import mindspore
from mindspore import context
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor

context.set_context(mode=context.GRAPH_MODE)


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.weight = Parameter(Tensor(3, mindspore.float32), name="w")
        self.m = 2

    def construct(self, x, y):
        self.weight = x
        self.m = 3
        #self.l = 1
        #y.weight = x
        print(self.weight)
        return x

def test_attr_ref():
    x = Tensor(4, mindspore.float32)
    net_y = Net()
    net = Net()
    ret = net(x, net_y)
    print(ret)
