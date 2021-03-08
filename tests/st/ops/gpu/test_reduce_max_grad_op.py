# Copyright 2021 Huawei Technologies Co., Ltd
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
from mindspore.ops import composite as C
from mindspore.ops import operations as P

def test_reduce_sum_grad():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.op = P.ReduceMax()

        def construct(self, x1, x2):
            return self.op(x1, x2)


    class GradNet(nn.Cell):
        def __init__(self, network):
            super(GradNet, self).__init__()
            self.grad = C.GradOperation(get_all=True, sens_param=True)
            self.network = network

        def construct(self, x1, x2, dy):
            return self.grad(self.network)(x1, x2, dy)

    net = Net()
    grad_net = GradNet(net)
    x1 = Tensor(np.array([[1, 2], [5, 4], [9, 16]]).astype(np.float32))
    x2 = 1
    dy = Tensor(np.array([2, 10, 1]).astype(np.float32))
    out = grad_net(x1, x2, dy)
    expected = np.array([[0, 2], [10, 0], [0, 1]])
    np.testing.assert_allclose(out[0].asnumpy(), expected, rtol=1e-6)

    x1 = Tensor(np.array([[9, 2], [4, 5], [1, 16]]).astype(np.float32))
    x2 = 0
    dy = Tensor(np.array([10, 11]).astype(np.float32))
    out = grad_net(x1, 0, dy)
    expected = np.array([[10, 0], [0, 0], [0, 11]])
    np.testing.assert_allclose(out[0].asnumpy(), expected, rtol=1e-6)
