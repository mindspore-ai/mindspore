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

import logging
import numpy as np
import mindspore.context as context
import mindspore.ops.composite as C
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.nn.composite_ops import ReLU

log = logging.getLogger("ME")
log.setLevel(level=logging.DEBUG)
context.set_context(mode=context.GRAPH_MODE, save_graphs=True, device_target="Ascend")

class NetBackwordFuse1(Cell):
    def __init__(self):
        super(NetBackwordFuse1, self).__init__()
        self.relu = ReLU()
        self.reduce_sum = P.ReduceSum(keep_dims=True)

    def construct(self, x):
        relu = self.relu(x)
        mul = P.Mul()(relu, 2.0)
        add = relu + mul
        out = self.reduce_sum(add, (0, ))
        return out

class NetBackwordFuse2(Cell):
    def __init__(self):
        super(NetBackwordFuse2, self).__init__()
        self.relu = ReLU()
        self.reduce_sum = P.ReduceSum(keep_dims=True)

    def construct(self, x):
        relu = self.relu(x)
        mul = P.Mul()(relu, 2.0)
        reduce = self.reduce_sum(relu, (0, ))
        div = 1.0 / reduce
        add1 = reduce + div
        out = relu + add1
        return out

def test_composite_fuse1():
    x = np.random.normal(0, 1, [2, 3, 1, 3]).astype(np.float32)
    net = NetBackwordFuse1()
    result = net(Tensor(x))
    print("================relu result=======================")
    print("x: {}".format(x))
    print("result: {}".format(result))
    print("=======================================")

def test_composite_fuse2():
    x = np.random.normal(0, 1, [2, 3, 1, 3]).astype(np.float32)
    net = NetBackwordFuse2()
    result = net(Tensor(x))
    print("================relu result=======================")
    print("x: {}".format(x))
    print("result: {}".format(result))
    print("=======================================")

test_composite_fuse1()
