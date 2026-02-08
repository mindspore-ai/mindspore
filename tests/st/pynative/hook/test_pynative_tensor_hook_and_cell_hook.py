# Copyright 2025 Huawei Technologies Co., Ltd
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
from mindspore.nn import Momentum, WithLossCell
from mindspore.common.initializer import TruncatedNormal
from mindspore import nn, ops, Tensor, ParameterTuple
from tests.mark_utils import arg_mark


def weight_variable():
    """weight initial"""
    return TruncatedNormal(0.02)


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """weight initial for conv layer"""
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def cell_hook_function_print_grad(cell, grad_input, grad_output):
    assert grad_input[0].asnumpy().shape == (32, 6, 14, 14)
    assert grad_output[0].asnumpy().shape == (32, 16, 10, 10)


def custom_hook_function_print_and_save_grad(grad_out):
    assert grad_out[0].asnumpy().shape == (32, 6, 28, 28)


def fc_with_initialize(input_channels, out_channels):
    """weight initial for fc layer"""
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


class test_custom_hook_function_base():
    def __init__(self):
        pass

    def test_custom_hook_function(self, hook_function, cell_hook_function):
        return hook_function, cell_hook_function


class GradWrap(nn.Cell):
    """ GradWrap definition """

    def __init__(self, network):
        super(GradWrap, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ParameterTuple(filter(lambda x: x.requires_grad, network.get_parameters()))

    def construct(self, x, label):
        weights = self.weights
        return ops.GradOperation(get_by_list=True)(self.network, weights)(x, label)


class LeNet5(nn.Cell):
    def __init__(self, hook_function, cell_hook_function, num_class=10):
        super(LeNet5, self).__init__()
        self.num_class = num_class
        self.batch_size = 32
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.conv2.register_backward_hook(cell_hook_function)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, self.num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reshape = ops.Reshape()
        self.hook = ops.HookBackward(hook_function)

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.hook(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.reshape(x, (self.batch_size, -1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_lenet5_train_hook_function_print_and_save_grad():
    """
    Feature: Test hook backward ops, cell backward hook.
    Description: Test hook backward ops, cell backward hook with LeNet5.
    Expectation: Success
    """
    hook = test_custom_hook_function_base()
    function = hook.test_custom_hook_function(custom_hook_function_print_and_save_grad,
                                              cell_hook_function_print_grad)
    net = LeNet5(hook_function=function[0], cell_hook_function=function[1])
    optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.1, 0.9)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
    net_with_criterion = WithLossCell(net, criterion)
    train_network = GradWrap(net_with_criterion)
    train_network.set_train()

    input_data = Tensor(np.ones([net.batch_size, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([net.batch_size, net.num_class]).astype(np.float32))
    output = net(Tensor(input_data))
    criterion(output, label)
    grads = train_network(input_data, label)
    success = optimizer(grads)
    assert success


CELL_HOOK_DONE = False
VAR_HOOK_DONE = False
CELL_BPROP_DONE = False


def cell_hook_function1(cell, grad_input, grad_output):
    print("cell_id", id(cell))
    global CELL_HOOK_DONE
    CELL_HOOK_DONE = True
    assert grad_input[0].asnumpy().shape == (32, 6, 14, 14)
    assert grad_output[0].asnumpy().shape == (32, 16, 10, 10)


def var_hook_function(grad_out):
    print("grad:", grad_out)
    global VAR_HOOK_DONE
    VAR_HOOK_DONE = True
    assert grad_out[0].asnumpy().shape == (32, 120)


class Block(nn.Cell):
    def __init__(self):
        super(Block, self).__init__()
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(x)
        return x

    def bprop(self, x, out, dout):
        global CELL_BPROP_DONE
        CELL_BPROP_DONE = True
        grad = out.asnumpy() * dout.asnumpy()
        grad = Tensor(grad)
        return (grad,)


class LeNet(nn.Cell):
    """
    Lenet network
    Args:
        num_class (int): Num classes. Default: 10.
    Returns:
        Tensor, output tensor

    Examples:
        >>> LeNet(num_class=10)
    """

    def __init__(self, num_class=10):
        super(LeNet, self).__init__()
        self.num_class = num_class
        self.batch_size = 32
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.conv2.register_backward_hook(cell_hook_function1)
        self.block = Block()
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, self.num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reshape = ops.Reshape()
        self.hook = ops.HookBackward(var_hook_function)

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.block(x)
        x = self.max_pool2d(x)
        x = self.reshape(x, (self.batch_size, -1))
        x = self.fc1(x)
        x = self.hook(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_lenet_with_hook():
    """
    Feature: Test hook backward ops, cell backward hook, custom bprop.
    Description: Test hook backward ops, cell backward hook, custom bprop with LeNet.
    Expectation: Success
    """
    net = LeNet()
    optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.1, 0.9)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
    net_with_criterion = WithLossCell(net, criterion)
    train_network = GradWrap(net_with_criterion)
    train_network.set_train()

    input_data = Tensor(np.ones([net.batch_size, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([net.batch_size, net.num_class]).astype(np.float32))
    output = net(Tensor(input_data))
    loss_output = criterion(output, label)
    grads = train_network(input_data, label)
    optimizer(grads)
    assert CELL_HOOK_DONE
    assert VAR_HOOK_DONE
    assert CELL_BPROP_DONE
    print(loss_output.asnumpy())
