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
"""
test assign sub
"""
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import Tensor
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter

context.set_context(mode=context.GRAPH_MODE)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.b = Parameter(initializer('ones', [5]), name='b')
        self.sub = P.AssignSub()

    def construct(self, value):
        self.sub(self.b, value)
        return self.b


def test_net():
    net = Net()
    input_data = Tensor(np.ones([5]).astype(np.float32))
    output = net(input_data)
    print(output.asnumpy().shape)
    print(output.asnumpy())
