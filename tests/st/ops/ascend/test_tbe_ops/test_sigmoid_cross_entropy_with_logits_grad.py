# Copyright 2019 Huawei Technologies Co., Ltd
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

context.set_context(device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.sigmoid_cross_entropy_with_logits = P.SigmoidCrossEntropyWithLogits()

    @jit
    def construct(self, features, labels):
        return self.sigmoid_cross_entropy_with_logits(features, labels)


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network

    @jit
    def construct(self, features, labels, dout):
        return self.grad(self.network)(features, labels, dout)


def test_net():
    features = np.random.randn(2, 3).astype(np.float16)
    labels = np.random.randn(2, 3).astype(np.float16)
    dout = np.random.randn(2, 3).astype(np.float16)

    net = Grad(Net())
    output = net(Tensor(features), Tensor(labels), Tensor(dout))
    print("=================output====================")
    print(output)
