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
from mindspore.ops import operations as P
from mindspore.ops.composite import GradOperation

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self, sigma=1.0):
        super(Net, self).__init__()
        self.SmoothL1Loss = P.SmoothL1Loss(sigma)

    def construct(self, pred, gt):
        return self.SmoothL1Loss(pred, gt)


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, pred, gt, dout):
        return self.grad(self.network)(pred, gt, dout)


def test_net():
    pred = np.random.randn(2, 4).astype(np.float32)
    gt = np.random.randn(2, 4).astype(np.float32)
    dout = np.random.randn(2, 4).astype(np.float32)
    smooth_l1_loss_grad = Grad(Net())
    output = smooth_l1_loss_grad(Tensor(pred), Tensor(gt), Tensor(dout))
    print("------------- input ---------------")
    print("predict:\n", pred)
    print("grount truth:\n", gt)
    print("dout:\n", dout)
    print("------------- output ---------------")
    print("predict grad:\n", output[0].asnumpy())
