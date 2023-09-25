# Copyright 2023 Huawei Technologies Co., Ltd
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
""" test amp """
import mindspore as ms
from mindspore.train import amp
from mindspore import nn, ops

class NetWithBranch(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 6, 5, pad_mode='valid')
        self.relu = ops.ReLU()
        self.bn = nn.BatchNorm2d(1)

    def construct(self, x):
        x = self.conv(x)
        y1 = self.relu(x)
        y2 = self.bn(x)
        x = y1 + y2
        return x


def test_net_with_branch():
    """
    Feature: Test amp o1.
    Description: Input x has two branch, one need cast, the other don't need to.
    Expectation: Success.
    """
    network = NetWithBranch()
    stree = ms.rewrite.SymbolTree.create(network)
    amp._insert_cast_operator_white_list(stree, amp.AMP_WHITE_LIST, ms.float16) # pylint:disable=protected-access
    amp._remove_duplicated_cast(stree, ms.float16) # pylint:disable=protected-access
    codes = stree.get_code()
    assert codes.count("x = self.outcast_conv(x, mindspore.float32)") == 1
    assert codes.count("x_1 = self.incast_relu0(x, mindspore.float16)") == 1
    assert codes.count("y1 = self.relu(x_1)") == 1
    assert codes.count("y1 = self.outcast_relu(y1, mindspore.float32)") == 1


class NetWithIf(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(1, 6, 5, pad_mode='valid')
        self.conv3 = nn.Conv2d(1, 6, 5, pad_mode='valid')
        self.conv4 = nn.Conv2d(1, 6, 5, pad_mode='valid')
        self.conv5 = nn.Conv2d(1, 6, 5, pad_mode='valid')
        self.relu = ops.ReLU()
        self.bn = nn.BatchNorm2d(1)

    def construct(self, x):
        x = self.conv1(x)
        if self.relu(x):
            x = self.conv2(x)
            x = self.bn(x)
        else:
            x = self.conv3(x)
            x = self.conv4(x)
        x = self.conv5(x)
        return x


def test_net_with_if():
    """
    Feature: Test amp o1.
    Description: Network has if statement, check whether casts are inserted correctly.
    Expectation: Success.
    """
    network = NetWithIf()
    stree = ms.rewrite.SymbolTree.create(network)
    amp._insert_cast_operator_white_list(stree, amp.AMP_WHITE_LIST, ms.float16) # pylint:disable=protected-access
    amp._remove_duplicated_cast(stree, ms.float16) # pylint:disable=protected-access
    codes = stree.get_code()
    assert codes.count("x = self.outcast_conv1(x, mindspore.float32)") == 1
    assert codes.count("x_1 = self.incast_relu0(x, mindspore.float16)") == 1
    assert codes.count("relu_var = self.relu(x_1)") == 1
    assert codes.count("relu_var = self.outcast_relu(relu_var, mindspore.float32)") == 1
    assert codes.count("if relu_var:") == 1
    assert codes.count("x = self.outcast_conv2(x, mindspore.float32)") == 1
    assert codes.count("x = self.outcast_conv3(x, mindspore.float32)") == 0
    assert codes.count("x = self.outcast_conv4(x, mindspore.float32)") == 1
    assert codes.count("x = self.outcast_conv5(x, mindspore.float32)") == 1


class NetWithClassFunction(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(1, 6, 5, pad_mode='valid')
        self.relu = ops.ReLU()
        self.bn = nn.BatchNorm2d(1)

    def construct(self, x):
        x = self.conv1(x)
        x = self.inner_function(x)
        return x

    def inner_function(self, x):
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn(x)
        return x


def test_net_with_class_function():
    """
    Feature: Test amp o1.
    Description: Network has class function, check whether casts are inserted correctly.
    Expectation: Success.
    """
    network = NetWithClassFunction()
    stree = ms.rewrite.SymbolTree.create(network)
    amp._insert_cast_operator_white_list(stree, amp.AMP_WHITE_LIST, ms.float16) # pylint:disable=protected-access
    amp._remove_duplicated_cast(stree, ms.float16) # pylint:disable=protected-access
    codes = stree.get_code()
    assert codes.count("x = self.outcast_conv1(x, mindspore.float32)") == 1
    assert codes.count("x = self.outcast_conv2(x, mindspore.float32)") == 0
    assert codes.count("x = self.outcast_relu(x, mindspore.float32)") == 1
