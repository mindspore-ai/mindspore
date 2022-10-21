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
""" test nn pad """
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import jit
from mindspore.ops.composite import GradOperation


class Net(nn.Cell):
    def __init__(self, raw_paddings, mode):
        super(Net, self).__init__()
        self.pad = nn.Pad(raw_paddings, mode=mode)

    @jit
    def construct(self, x):
        return self.pad(x)


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network

    @jit
    def construct(self, x, grads):
        return self.grad(self.network)(x, grads)


def test_pad_train():
    mode = 'CONSTANT'
    x = np.random.random(size=(2, 3)).astype(np.float32)
    raw_paddings = ((1, 1), (2, 2))
    grads = np.random.random(size=(4, 7)).astype(np.float32)
    grad = Grad(Net(raw_paddings, mode))
    output = grad(Tensor(x), Tensor(grads))
    print("=================output====================")
    print(output)


def test_pad_infer():
    mode = 'CONSTANT'
    x = np.random.random(size=(2, 3)).astype(np.float32)
    raw_paddings = ((1, 1), (2, 2))
    net = Net(raw_paddings, mode)
    output = net(Tensor(x))
    print("=================output====================")
    print(output)
