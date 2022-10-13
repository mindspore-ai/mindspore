# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    """Net definition"""

    def __init__(self):
        super(Net, self).__init__()
        self.assign_sub = P.AssignSub()
        self.inputdata = Parameter(initializer('normal', [1]), name="global_step")
        print("inputdata: ", self.inputdata)

    def construct(self, x):
        self.assign_sub(self.inputdata, x)
        return self.inputdata


def test_net():
    """
    Feature: test AssignSub.
    Description: test AssignSub.
    Expectation: No exception.
    """
    net = Net()
    x = Tensor(np.ones([1]).astype(np.int32) * 100)

    print("MyPrintResult dataX:", x)
    result = net(x)
    print("MyPrintResult data::", result.asnumpy())
