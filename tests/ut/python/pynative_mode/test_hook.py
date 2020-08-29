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

import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import context, Tensor, ParameterTuple
from mindspore.common.initializer import TruncatedNormal
from mindspore.nn import WithLossCell, Momentum
from mindspore.ops import composite as C

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
cell_hook_done = False
var_hook_done = False
cell_bprop_done = False


grad_all = C.GradOperation(get_all=True)


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """weight initial for conv layer"""
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    """weight initial for fc layer"""
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


def weight_variable():
    """weight initial"""
    return TruncatedNormal(0.02)


def cell_hook_function(cell_id, grad_input, grad_output):
    print(cell_id)
    global cell_hook_done
    cell_hook_done = True
    assert (grad_output[0].asnumpy().shape == (32, 6, 14, 14))
    assert (grad_input[0].asnumpy().shape == (32, 16, 10, 10))


def var_hook_function(grad_out):
    print("grad:", grad_out)
    global var_hook_done
    var_hook_done = True
    assert (grad_out[0].asnumpy().shape == (32, 120))


class Block(nn.Cell):
    def __init__(self):
        super(Block, self).__init__()
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(x)
        return x

    def bprop(self, x, out, dout):
        global cell_bprop_done
        cell_bprop_done = True
        grad = out.asnumpy() * dout.asnumpy()
        grad = Tensor(grad)
        return (grad,)

class LeNet5(nn.Cell):
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
        super(LeNet5, self).__init__()
        self.num_class = num_class
        self.batch_size = 32
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.conv2.register_backward_hook(cell_hook_function)
        self.block = Block()
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, self.num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reshape = P.Reshape()
        self.hook = P.HookBackward(var_hook_function)

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


class GradWrap(nn.Cell):
    """ GradWrap definition """
    def __init__(self, network):
        super(GradWrap, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ParameterTuple(filter(lambda x: x.requires_grad, network.get_parameters()))

    def construct(self, x, label):
        weights = self.weights
        return C.GradOperation(get_by_list=True)(self.network, weights)(x, label)


def test_hook():
    net = LeNet5()
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
    success = optimizer(grads)
    assert cell_hook_done
    assert var_hook_done
    assert cell_bprop_done
    print(loss_output.asnumpy())


bprop_debug = False

class MulAdd(nn.Cell):
    def __init__(self):
        super(MulAdd, self).__init__()

    def construct(self, x, y):
        return 2 * x * x + y * y

    def bprop(self, x, y, out, dout):
        global bprop_debug
        bprop_debug = True
        return dout, 2 * y


def test_custom_bprop():
    mul_add = MulAdd()
    mul_add.bprop_debug = True
    x = Tensor(np.array([1, 2, 3]).astype(np.int32))
    y = Tensor(np.array([2, 3, 4]).astype(np.int32))
    grad_all(mul_add)(x, y)
    assert bprop_debug


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()

    def construct(self, x, y):
        return 2 * x * x + y * y

def test_grad_all():
    net = Net()
    x = Tensor(np.array([1, 2, 3]).astype(np.int32))
    y = Tensor(np.array([2, 3, 4]).astype(np.int32))
    res = grad_all(net)(x, y)
    print(res)

def test_check_input():
    net = Net()
    x = np.array([1, 2, 3])
    y = np.array([2, 3, 4])
    with pytest.raises(TypeError):
        net(x, y)
