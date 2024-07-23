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
import os
import numpy as np
from tests.st.compiler.control.cases_register import case_register

import mindspore.nn as nn
import mindspore
from mindspore import context, jit
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import TruncatedNormal
from mindspore.common.parameter import ParameterTuple
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.train.serialization import export, load


def weight_variable():
    return TruncatedNormal(0.02)


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


class LeNet5(nn.Cell):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.batch_size = 32
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, 10)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reshape = P.Reshape()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.reshape(x, (self.batch_size, -1))
        x = self.fc1(x)
        x = self.relu(x)
        print("test print.")
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class WithLossCell(nn.Cell):
    def __init__(self, network):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self.loss = nn.SoftmaxCrossEntropyWithLogits()
        self.network = network

    def construct(self, x, label):
        predict = self.network(x)
        return self.loss(predict, label)


class TrainOneStepCell(nn.Cell):
    def __init__(self, network):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_train()
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = nn.Momentum(self.weights, 0.1, 0.9)
        self.hyper_map = C.HyperMap()
        self.grad = C.GradOperation(get_by_list=True)

    def construct(self, x, label):
        weights = self.weights
        grads = self.grad(self.network, weights)(x, label)
        return self.optimizer(grads)


class SingleIfNet(nn.Cell):
    def construct(self, x, y):
        x += 1
        if x < y:
            y += x
        else:
            y -= x
        y += 5
        return y


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_export_lenet_grad_mindir():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    network = LeNet5()
    network.set_train()
    predict = Tensor(np.ones([32, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.zeros([32, 10]).astype(np.float32))
    net = TrainOneStepCell(WithLossCell(network))
    export(net, predict, label, file_name="lenet_grad", file_format='MINDIR')
    verify_name = "lenet_grad.mindir"
    assert os.path.exists(verify_name)


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_load_mindir_and_run():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    network = LeNet5()
    network.set_train()

    inputs0 = Tensor(np.ones([32, 1, 32, 32]).astype(np.float32) * 0.01)
    outputs0 = network(inputs0)

    inputs = Tensor(np.zeros([32, 1, 32, 32]).astype(np.float32))
    export(network, inputs, file_name="test_lenet_load", file_format='MINDIR')
    mindir_name = "test_lenet_load.mindir"
    assert os.path.exists(mindir_name)

    graph = load(mindir_name)
    loaded_net = nn.GraphCell(graph)
    outputs_after_load = loaded_net(inputs0)
    assert np.allclose(outputs0.asnumpy(), outputs_after_load.asnumpy())


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_single_if():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    network = SingleIfNet()

    x = Tensor(np.array([1]).astype(np.float32))
    y = Tensor(np.array([2]).astype(np.float32))
    origin_out = network(x, y)

    file_name = "if_net"
    export(network, x, y, file_name=file_name, file_format='MINDIR')
    mindir_name = file_name + ".mindir"
    assert os.path.exists(mindir_name)

    graph = load(mindir_name)
    loaded_net = nn.GraphCell(graph)
    outputs_after_load = loaded_net(x, y)
    assert origin_out == outputs_after_load


@case_register.level1
@case_register.target_gpu
def test_jit_net():
    """
    Feature: export jit function.
    Description: export jit function
    Expectation: Correct result and no exception.
    """
    class Net(nn.Cell):
        """
        Net1
        """

        def __init__(self):
            super(Net, self).__init__()
            self.conv2d = P.Conv2D(out_channel=32, kernel_size=3, data_format="NHWC")
            self.weight = Tensor(np.ones([32, 3, 3, 32]), mindspore.float32)
            self.one = Tensor(np.ones([1, 1, 1, 1]), mindspore.float32)

        @jit
        def fun(self, x):
            """
            自定义方法
            """
            x = self.conv2d(x, self.weight)
            return x

        def construct(self, x):
            """
            construct
            """
            x = self.fun(x)
            x += Tensor(np.ones([1, 1, 1, 1]), mindspore.float32)
            return x

    input_x = Tensor(np.ones([10, 32, 32, 32]).astype(np.float32), mindspore.float32)
    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    origin_out = net(input_x)
    file_name = "jit_net"
    export(net, input_x, file_name=file_name, file_format='MINDIR')
    mindir_name = file_name + ".mindir"
    assert os.path.exists(mindir_name)
    context.set_context(mode=context.GRAPH_MODE)
    graph = load(mindir_name)
    loaded_net = nn.GraphCell(graph)
    outputs_after_load = loaded_net(input_x)
    if not np.allclose(origin_out.asnumpy(), outputs_after_load.asnumpy(), 0.0001, 0.0001):
        assert False
    assert True
