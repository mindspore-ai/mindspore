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
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from mindspore.ops.composite import GradOperation


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.loss = P.CTCLoss()
        self.div = P.RealDiv()
        self.mean = P.ReduceMean()

    def construct(self, probs, label, input_length, indices):
        x, _ = self.loss(probs, indices, label, input_length)
        x = self.mean(x)
        return x


class GradData(nn.Cell):
    def __init__(self, network):
        super(GradData, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=False)
        self.network = network

    def construct(self, probs, indices, labels, input_lengths):
        return self.grad(self.network)(probs, indices, labels, input_lengths)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ctcloss():
    probs = Tensor([[[-4.4131, -4.6093, -3.4333, -3.9268, -2.8917, -3.4093, -4.2243, -1.1379, -7.1046, -0.6902],
                     [-2.5109, -3.3397, -4.9384, -1.2723, -1.1443, -2.4683, -2.6768, -4.1282, -2.7062, -3.1906],
                     [-2.5092, -1.6392, -2.0864, -4.0059, -1.5610, -2.3223, -2.4816, -2.9922, -3.1412, -2.3311]],

                    [[-2.1243, -3.5773, -3.1108, -4.4253, -2.7080, -1.9653, -2.0499, -2.4418, -1.8620, -1.5229],
                     [-2.2479, -3.5128, -1.4189, -2.8701, -1.8562, -2.2752, -2.7019, -2.1865, -2.5634, -2.9869],
                     [-3.2144, -1.3986, -3.1083, -3.9634, -3.5131, -3.2317, -2.6200, -1.7938, -1.8159, -1.7255]],

                    [[-3.1301, -2.1649, -0.9286, -2.9452, -2.5992, -2.0263, -2.9201, -3.2155, -2.8302, -3.3636],
                     [-1.4661, -3.6311, -2.4781, -4.6180, -2.7308, -1.7019, -1.5570, -2.6012, -4.0788, -2.3073],
                     [-2.6833, -1.5033, -3.6922, -2.6360, -2.6974, -2.6847, -2.7579, -2.1396, -1.4093, -2.9630]],

                    [[-2.0094, -2.3024, -3.3673, -1.0220, -2.8326, -2.2613, -3.0535, -2.9879, -3.7015, -2.4510],
                     [-1.9071, -3.2603, -2.3229, -2.0572, -4.3450, -2.1284, -2.6306, -1.3824, -2.9815, -2.5061],
                     [-2.7931, -3.7631, -3.2440, -4.3887, -1.0271, -3.8851, -1.2418, -4.5123, -2.2993, -2.4607]],

                    [[-1.5763, -2.7539, -3.6941, -3.8166, -1.2599, -2.6903, -2.5826, -4.8208, -2.9562, -1.6321],
                     [-3.3031, -3.0087, -1.9982, -1.9081, -3.8731, -2.8764, -2.2485, -2.3808, -1.4283, -2.1625],
                     [-2.4516, -3.2394, -4.2053, -4.3541, -2.5229, -4.0717, -1.4894, -2.3151, -1.1098, -2.3465]]],
                   dtype=mstype.float32)
    labels = Tensor([3, 4, 6, 4, 7, 1, 4, 6, 6, 8], dtype=mstype.int32)
    indices = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2], [2, 3]]
    indices = Tensor(indices, dtype=mstype.int64)
    input_lengths = Tensor([5, 5, 5], dtype=mstype.int32)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    net = Net()
    ctc_loss = net(probs, labels, input_lengths, indices)
    expect_loss = [9.083767]
    assert np.allclose(ctc_loss.asnumpy(), expect_loss)

    grad = GradData(net)(probs, labels, input_lengths, indices)
    grad = P.ReduceMean()(grad[0])
    expect_grad = [-5.9604646e-09]
    assert np.allclose(grad.asnumpy(), expect_grad, atol=1e-5)
