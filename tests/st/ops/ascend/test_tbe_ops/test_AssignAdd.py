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
from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.nn as nn
from mindspore.common.api import ms_function
import numpy as np
import mindspore.context as context
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    """Net definition"""

    def __init__(self):
        super(Net, self).__init__()
        self.AssignAdd = P.AssignAdd()
        self.inputdata = Parameter(initializer('normal', [1]), name="global_step")
        print("inputdata: ", self.inputdata)

    def construct(self, x):
        out = self.AssignAdd(self.inputdata, x)
        return out


def test_net():
    """test AssignAdd"""
    net = Net()
    x = Tensor(np.ones([1]).astype(np.float32) * 100)

    print("MyPrintResult dataX:", x)
    result = net(x)
    print("MyPrintResult data::", result.asnumpy())
