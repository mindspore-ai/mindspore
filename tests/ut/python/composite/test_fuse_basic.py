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

class NetBasicFuse1(Cell):
    def __init__(self):
        super(NetBasicFuse1, self).__init__()

    def construct(self, x):
        mul = P.Mul()(x, 2.0)
        add = mul + 1.0
        reduce = P.ReduceSum()(add, (0, ))
        return reduce

def test_basic_fuse1():
    x = np.random.normal(0, 1, [2, 3]).astype(np.float32)
    net = NetBasicFuse1()
    result = net(Tensor(x))
    print("================result=======================")
    print("x: {}".format(x))
    print("result: {}".format(result))
    print("=======================================")


test_basic_fuse1()
