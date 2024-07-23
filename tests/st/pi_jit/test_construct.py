import pytest
import numpy as np
import mindspore.nn as nn
import mindspore.ops.operations as P

from mindspore import Tensor, jit, context
from mindspore.common.initializer import TruncatedNormal
from tests.mark_utils import arg_mark


# context.set_context(save_graphs = True, save_graphs_path="./grahp_jit")
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
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, self.num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reshape = P.Reshape()

    @jit(mode="PIJit")
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
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def check(output):
    # check output size [32 x 10]
    print("res:", output.size)
    assert output.size == 320


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('input_data', [Tensor(np.ones([32, 1, 32, 32]).astype(np.float32) * 0.01)])
def test_cell_lenet(input_data):
    """
    Feature: LeNet-5 Model Testing
    Description: Test the LeNet-5 model with given input data.
    Expectation: The output size should match the expected size.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    net = LeNet5()
    context.set_context(mode=context.GRAPH_MODE)
    output = net(Tensor(input_data))
    check(output)
