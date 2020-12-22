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
""" test model train """
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor, Parameter, Model
from mindspore.common.initializer import initializer
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.optim import Momentum
from mindspore.ops import operations as P


# fn is a funcation use i as input
def lr_gen(fn, epoch_size):
    for i in range(epoch_size):
        yield fn(i)


def me_train_tensor(net, input_np, label_np, epoch_size=2):
    """me_train_tensor"""
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr_gen(lambda i: 0.1, epoch_size), 0.9,
                   0.01, 1024)
    Model(net, loss, opt)
    _network = nn.WithLossCell(net, loss)
    _train_net = nn.TrainOneStepCell(_network, opt)
    _train_net.set_train()
    label_np = np.argmax(label_np, axis=-1).astype(np.int32)
    for epoch in range(0, epoch_size):
        print(f"epoch %d" % (epoch))
        _train_net(Tensor(input_np), Tensor(label_np))


def test_bias_add(test_with_simu):
    """test_bias_add"""
    import mindspore.context as context
    is_pynative_mode = (context.get_context("mode") == context.PYNATIVE_MODE)
    # training api is implemented under Graph mode
    if is_pynative_mode:
        context.set_context(mode=context.GRAPH_MODE)
    if test_with_simu:
        return

    class Net(nn.Cell):
        """Net definition"""

        def __init__(self,
                     output_channels,
                     bias_init='zeros',
                     ):
            super(Net, self).__init__()
            self.biasAdd = P.BiasAdd()

            if isinstance(bias_init, Tensor):
                if bias_init.ndim != 1 or bias_init.shape[0] != output_channels:
                    raise ValueError("bias_init shape error")

            self.bias = Parameter(initializer(
                bias_init, [output_channels]), name="bias")

        def construct(self, input_x):
            return self.biasAdd(input_x, self.bias)

    bias_init = Tensor(np.ones([3]).astype(np.float32))
    input_np = np.ones([1, 3, 3, 3], np.float32)
    label_np = np.ones([1, 3, 3, 3], np.int32) * 2
    me_train_tensor(Net(3, bias_init=bias_init), input_np, label_np)


def test_conv(test_with_simu):
    """test_conv"""
    import mindspore.context as context
    is_pynative_mode = (context.get_context("mode") == context.PYNATIVE_MODE)
    # training api is implemented under Graph mode
    if is_pynative_mode:
        context.set_context(mode=context.GRAPH_MODE)
    if test_with_simu:
        return

    class Net(nn.Cell):
        "Net definition"""

        def __init__(self,
                     cin,
                     cout,
                     kernel_size):
            super(Net, self).__init__()
            Tensor(np.ones([6, 3, 3, 3]).astype(np.float32) * 0.01)
            self.conv = nn.Conv2d(cin,
                                  cout,
                                  kernel_size)

        def construct(self, input_x):
            return self.conv(input_x)

    net = Net(3, 6, (3, 3))
    input_np = np.ones([1, 3, 32, 32]).astype(np.float32) * 0.01
    label_np = np.ones([1, 6, 32, 32]).astype(np.int32)
    me_train_tensor(net, input_np, label_np)


def test_net():
    """test_net"""
    import mindspore.context as context
    is_pynative_mode = (context.get_context("mode") == context.PYNATIVE_MODE)
    # training api is implemented under Graph mode
    if is_pynative_mode:
        context.set_context(mode=context.GRAPH_MODE)

    class Net(nn.Cell):
        """Net definition"""

        def __init__(self):
            super(Net, self).__init__()
            Tensor(np.ones([64, 3, 7, 7]).astype(np.float32) * 0.01)
            self.conv = nn.Conv2d(3, 64, (7, 7), pad_mode="same", stride=2)
            self.relu = nn.ReLU()
            self.bn = nn.BatchNorm2d(64)
            self.mean = P.ReduceMean(keep_dims=True)
            self.flatten = nn.Flatten()
            self.dense = nn.Dense(64, 12)

        def construct(self, input_x):
            output = input_x
            output = self.conv(output)
            output = self.bn(output)
            output = self.relu(output)
            output = self.mean(output, (-2, -1))
            output = self.flatten(output)
            output = self.dense(output)
            return output

    net = Net()
    input_np = np.ones([32, 3, 224, 224]).astype(np.float32) * 0.01
    label_np = np.ones([32, 12]).astype(np.int32)
    me_train_tensor(net, input_np, label_np)


def test_bn():
    """test_bn"""
    import mindspore.context as context
    is_pynative_mode = (context.get_context("mode") == context.PYNATIVE_MODE)
    # training api is implemented under Graph mode
    if is_pynative_mode:
        context.set_context(mode=context.GRAPH_MODE)

    class Net(nn.Cell):
        """Net definition"""

        def __init__(self, cin, cout):
            super(Net, self).__init__()
            self.bn = nn.BatchNorm2d(cin)
            self.flatten = nn.Flatten()
            self.dense = nn.Dense(cin, cout)

        def construct(self, input_x):
            output = input_x
            output = self.bn(output)
            output = self.flatten(output)
            output = self.dense(output)
            return output

    net = Net(2048, 16)
    input_np = np.ones([32, 2048, 1, 1]).astype(np.float32) * 0.01
    label_np = np.ones([32, 16]).astype(np.int32)
    me_train_tensor(net, input_np, label_np)
