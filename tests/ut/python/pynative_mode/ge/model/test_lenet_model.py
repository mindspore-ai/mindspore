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
""" test_lenet_model """
import numpy as np

import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.nn.optim import Momentum
from mindspore.nn import WithGradCell, WithLossCell
from ....ut_filter import non_graph_engine


class LeNet5(nn.Cell):
    """ LeNet5 definition """

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, 10)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = P.Flatten()

    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@non_graph_engine
def test_lenet_pynative_train_net():
    """ test_lenet_pynative_train_net """
    data = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([1, 10]).astype(np.float32))
    dout = Tensor(np.ones([1]).astype(np.float32))
    iteration_num = 1
    verification_step = 0

    net = LeNet5()

    for i in range(0, iteration_num):
        # get the gradients
        loss_fn = nn.SoftmaxCrossEntropyWithLogits(is_grad=False)
        grad_fn = nn.SoftmaxCrossEntropyWithLogits()
        grad_net = WithGradCell(net, grad_fn, sens=dout)
        gradients = grad_net(data, label)

        # update parameters
        opt = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
        opt(gradients)

        # verification
        if i == verification_step:
            loss_net = WithLossCell(net, loss_fn)
            loss_output = loss_net(data, label)
            print("The loss of %s-th iteration is %s" % (i, loss_output.asnumpy()))


def test_lenet_pynative_train_model():
    """ test_lenet_pynative_train_model """
    # get loss from model.compute_loss
    return
