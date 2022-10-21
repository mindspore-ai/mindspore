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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import jit
from mindspore.ops import operations as P
from mindspore.ops.composite import GradOperation

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True)
        self.network = network

    @jit
    def construct(self, input_):
        return self.grad(self.network)(input_)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.relu_v2 = P.ReLUV2()

    def construct(self, x):
        return self.relu_v2(x)


def test_net():
    x = Tensor(np.ones((2, 3, 3, 4)).astype(np.float32))
    relu_net = Net()
    relu_output = relu_net(x)
    net = Grad(Net())
    output_grad = net(x)
    print(relu_output[0].asnumpy())
    print(relu_output[1].asnumpy())
    print(len(output_grad))
    print(output_grad[0].asnumpy())
