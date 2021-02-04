# Copyright 2019-2021 Huawei Technologies Co., Ltd
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
from mindspore.common.api import ms_function
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P
from mindspore.ops.composite import GradOperation

# context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
context.set_context(device_target="Ascend")


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network

    @ms_function
    def construct(self, input_, output_grad):
        return self.grad(self.network)(input_, output_grad)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.bn = P.BatchNorm()
        self.scale = Parameter(initializer('ones', [64]), name='scale')
        self.b = Parameter(initializer('zeros', [64]), name='b')
        self.mean = Parameter(initializer('ones', [64]), name='mean')
        self.variance = Parameter(initializer('zeros', [64]), name='variance')

    def construct(self, x):
        return self.bn(x, self.scale, self.b, self.mean, self.variance)[0]


def test_net():
    x = np.random.randn(1, 64, 112, 112).astype(np.float32)
    sens = np.random.randn(1, 64, 112, 112).astype(np.float32)
    net = Grad(Net())
    output = net(Tensor(x), Tensor(sens))
    print("***********x*********")
    print(x)

    print("***********output y*********")
    print(output.asnumpy())
