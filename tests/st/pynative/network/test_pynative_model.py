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
""" test_pynative_model """
import numpy as np

import mindspore.nn as nn
from mindspore import Parameter, ParameterTuple, Tensor
from mindspore import context
from mindspore.nn.optim import Momentum
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from tests.mark_utils import arg_mark


grad_by_list = C.GradOperation(get_by_list=True)


def setup_module(module):
    context.set_context(mode=context.PYNATIVE_MODE)


class GradWrap(nn.Cell):
    """ GradWrap definition """

    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network
        self.weights = ParameterTuple(network.get_parameters())

    def construct(self, x, label):
        weights = self.weights
        return grad_by_list(self.network, weights)(x, label)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_softmaxloss_grad():
    """ test_softmaxloss_grad """

    class NetWithLossClass(nn.Cell):
        """ NetWithLossClass definition """

        def __init__(self, network):
            super(NetWithLossClass, self).__init__()
            self.loss = nn.SoftmaxCrossEntropyWithLogits()
            self.network = network

        def construct(self, x, label):
            predict = self.network(x)
            return self.loss(predict, label)

    class Net(nn.Cell):
        """ Net definition """

        def __init__(self):
            super(Net, self).__init__()
            self.weight = Parameter(Tensor(np.ones([64, 10]).astype(np.float32)), name="weight")
            self.bias = Parameter(Tensor(np.ones([10]).astype(np.float32)), name="bias")
            self.fc = P.MatMul()
            self.bias_add = P.BiasAdd()

        def construct(self, x):
            x = self.bias_add(self.fc(x, self.weight), self.bias)
            return x

    net = GradWrap(NetWithLossClass(Net()))

    predict = Tensor(np.ones([1, 64]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    print("pynative run")
    out = net.construct(predict, label)
    print("out:", out)
    print(out[0], (out[0]).asnumpy(), ":result")


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_lenet_grad():
    """ test_lenet_grad """

    class NetWithLossClass(nn.Cell):
        """ NetWithLossClass definition """

        def __init__(self, network):
            super(NetWithLossClass, self).__init__()
            self.loss = nn.SoftmaxCrossEntropyWithLogits()
            self.network = network

        def construct(self, x, label):
            predict = self.network(x)
            return self.loss(predict, label)

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

    input_data = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([1, 10]).astype(np.float32))
    iteration_num = 1
    verification_step = 0

    net = LeNet5()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    momen_opti = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = GradWrap(NetWithLossClass(net))
    train_net.set_train()

    for i in range(0, iteration_num):
        # get the gradients
        grads = train_net(input_data, label)
        # update parameters
        success = momen_opti(grads)
        if success is False:
            print("fail to run optimizer")
        # verification
        if i == verification_step:
            fw_output = net(input_data)
            loss_output = loss(fw_output, label)
            print("The loss of %s-th iteration is %s" % (i, loss_output.asnumpy()))
