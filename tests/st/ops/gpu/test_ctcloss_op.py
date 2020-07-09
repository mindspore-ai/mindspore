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
        self.loss = P.CTCLossV2()
        self.div = P.RealDiv()
        self.cast = P.Cast()
        self.mean = P.ReduceMean()

    def construct(self, probs, label, input_length, label_length):
        x, _ = self.loss(probs, label, input_length, label_length)
        x = self.div(x, self.cast(label_length, mstype.float32))
        x = self.mean(x)
        return x

class GradData(nn.Cell):
    def __init__(self, network):
        super(GradData, self).__init__()
        self.grad = GradOperation(name="get_all", get_all=True, sens_param=False)
        self.network = network

    def construct(self, probs, labels, input_lengths, label_lengths):
        return self.grad(self.network)(probs, labels, input_lengths, label_lengths)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
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
    labels = Tensor([9, 4, 6, 4, 7, 1, 4, 6, 6, 8], dtype=mstype.int32)
    input_lengths = Tensor([5, 5, 5], dtype=mstype.int32)
    label_lengths = Tensor([3, 3, 4], dtype=mstype.int32)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = Net()
    ctc_loss = net(probs, labels, input_lengths, label_lengths)
    expect_loss = [2.4099]
    assert np.allclose(ctc_loss.asnumpy(), expect_loss)

    grad = GradData(net)(probs, labels, input_lengths, label_lengths)
    expect_grad = [[[8.8442e-05, 1.1065e-03, 3.5867e-03, 2.1896e-03, 6.1646e-03,
                     3.6738e-03, 1.6262e-03, 3.5610e-02, 9.1258e-05, -5.4134e-02],
                    [-3.7523e-03, 3.9386e-03, 7.9623e-04, 3.1132e-02, -6.2954e-02,
                     9.4143e-03, 7.6425e-03, 1.7902e-03, 7.4211e-03, 4.5719e-03],
                    [6.7778e-03, 1.6178e-02, 1.0344e-02, 1.5173e-03, -6.5840e-02,
                     8.1707e-03, 6.9674e-03, 4.1814e-03, 3.6026e-03, 8.0991e-03]],

                   [[-1.2581e-02, 3.1057e-03, 4.9517e-03, 1.3301e-03, -2.6320e-02,
                     1.5568e-02, 1.4305e-02, 9.6671e-03, 1.7262e-02, -2.7292e-02],
                    [-1.5566e-02, 3.3126e-03, 2.6887e-02, 6.2993e-03, -3.9716e-02,
                     1.1420e-02, 7.4531e-03, -1.4252e-02, 8.5603e-03, 5.6048e-03],
                    [3.3483e-03, 2.0579e-02, 3.7231e-03, 1.5832e-03, 2.4837e-03,
                     3.2909e-03, -7.7267e-02, 1.3861e-02, 1.3558e-02, 1.4840e-02]],

                   [[-8.0007e-03, 1.2751e-02, 4.3901e-02, 5.8435e-03, -7.2627e-02,
                     1.4647e-02, -8.0584e-03, 4.4595e-03, 6.5557e-03, 5.2891e-04],
                    [-3.6006e-02, 1.5308e-03, 9.3225e-03, 1.0969e-03, -2.5098e-03,
                     2.0260e-02, 2.3419e-02, -3.0053e-02, 1.8809e-03, 1.1059e-02],
                    [-7.7639e-02, 1.8533e-02, 2.0764e-03, 5.9706e-03, 5.6150e-03,
                     5.6868e-03, 5.2854e-03, 9.8085e-03, 2.0360e-02, 4.3053e-03]],

                   [[-2.6776e-02, 1.1113e-02, 3.8314e-03, 3.9986e-02, -1.6020e-02,
                     1.1579e-02, -4.1635e-02, 5.5992e-03, 2.7429e-03, 9.5786e-03],
                    [-6.8619e-03, -6.4066e-03, 1.0888e-02, 1.4201e-02, 1.4413e-03,
                     1.3225e-02, 8.0039e-03, -4.9191e-02, 5.6352e-03, 9.0651e-03],
                    [5.1026e-03, 1.9343e-03, 3.2506e-03, 1.0347e-03, 2.9837e-02,
                     1.7121e-03, -5.9261e-02, 9.1443e-04, 8.3608e-03, 7.1146e-03]],

                   [[-2.0848e-02, 7.0754e-03, 2.7633e-03, 2.4447e-03, 3.1520e-02,
                     7.5401e-03, -5.8895e-02, 8.9559e-04, 5.7796e-03, 2.1724e-02],
                    [-1.3499e-03, -1.0019e-01, 1.5064e-02, 1.6485e-02, 2.3104e-03,
                     6.2597e-03, 1.1729e-02, 1.0275e-02, 2.6635e-02, 1.2782e-02],
                    [7.1796e-03, 3.2656e-03, 1.2430e-03, 1.0712e-03, 6.6856e-03,
                     1.4207e-03, 1.8792e-02, 8.2297e-03, -5.5865e-02, 7.9753e-03]]]
    assert np.allclose(grad[0].asnumpy(), expect_grad, atol=1e-5)
