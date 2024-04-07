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
import numpy as np

class NetWithBranch(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 6, 2, pad_mode='valid')
        self.relu = ops.ReLU()
        self.bn = nn.BatchNorm2d(6)

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
    x = ms.Tensor(np.ones([1, 1, 4, 4]), ms.float32)
    y = network(x)
    # enable parse mindspore cells
    ms.rewrite.common.namespace._ms_cells_to_subtree = True # pylint:disable=protected-access
    stree = ms.rewrite.SymbolTree.create(network)
    amp._insert_cast_for_operators(stree, ms.float16, False, white_list=amp.AMP_WHITE_LIST) # pylint:disable=protected-access
    amp._remove_duplicated_cast(stree, ms.float16) # pylint:disable=protected-access
    codes = stree.get_code()
    assert codes.count("x = amp_cast(x, mindspore.float16)") == 1, codes
    assert codes.count("weight_var = amp_cast(self.weight, mindspore.float16)") == 1, codes
    assert codes.count("output = self.conv2d(x, weight_var)") == 1, codes
    assert codes.count("output = amp_cast(output, mindspore.float32)") == 2, codes
    assert codes.count("output_var = amp_cast(output, mindspore.float16)") == 1, codes
    assert codes.count("bias_var = amp_cast(self.bias, mindspore.float16)") == 1, codes
    assert codes.count("output = self.bias_add(output_var, bias_var)") == 1, codes
    assert codes.count("x_var = amp_cast(x, mindspore.float16)") == 1, codes
    assert codes.count("y1 = self.relu(x_var)") == 1, codes
    assert codes.count("y1 = amp_cast(y1, mindspore.float32)") == 1, codes
    new_net = stree.get_network()
    # disable parse mindspore cells
    ms.rewrite.common.namespace._ms_cells_to_subtree = False # pylint:disable=protected-access
    y1 = new_net(x)
    assert np.allclose(y.asnumpy(), y1.asnumpy(), 0.001, 0.001)


class NetWithIf(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 2, pad_mode='valid')
        self.conv2 = nn.Conv2d(1, 1, 2, pad_mode='valid')
        self.conv3 = nn.Conv2d(1, 1, 2, pad_mode='valid')
        self.conv4 = nn.Conv2d(1, 1, 2, pad_mode='valid')
        self.conv5 = nn.Conv2d(1, 1, 2, pad_mode='valid')
        self.relu = ops.ReLU()
        self.bn = nn.BatchNorm2d(1)

    def construct(self, x):
        x = self.conv1(x)
        if self.relu(x) is not None:
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
    x = ms.Tensor(np.ones([1, 1, 4, 4]), ms.float32)
    y = network(x)
    # enable parse mindspore cells
    ms.rewrite.common.namespace._ms_cells_to_subtree = True # pylint:disable=protected-access
    stree = ms.rewrite.SymbolTree.create(network)
    amp._insert_cast_for_operators(stree, ms.float16, False, white_list=amp.AMP_WHITE_LIST) # pylint:disable=protected-access
    amp._remove_duplicated_cast(stree, ms.float16) # pylint:disable=protected-access
    codes = stree.get_code()
    assert codes.count("x = amp_cast(x, mindspore.float16)") == 5, codes
    assert codes.count("weight_var = amp_cast(self.weight, mindspore.float16)") == 5, codes
    assert codes.count("output = self.conv2d(x, weight_var)") == 5, codes
    assert codes.count("output = amp_cast(output, mindspore.float32)") == 10, codes
    assert codes.count("output_var = amp_cast(output, mindspore.float16)") == 5, codes
    assert codes.count("bias_var = amp_cast(self.bias, mindspore.float16)") == 5, codes
    assert codes.count("output = self.bias_add(output_var, bias_var)") == 5, codes
    assert codes.count("x_var = amp_cast(x, mindspore.float16)") == 1, codes
    assert codes.count("relu_var = self.relu(x_var)") == 1, codes
    assert codes.count("relu_var = amp_cast(relu_var, mindspore.float32)") == 1, codes
    new_net = stree.get_network()
    # disable parse mindspore cells
    ms.rewrite.common.namespace._ms_cells_to_subtree = False # pylint:disable=protected-access
    y1 = new_net(x)
    assert np.allclose(y.asnumpy(), y1.asnumpy(), 0.001, 0.001)


class NetWithClassFunction(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 2, pad_mode='valid')
        self.conv2 = nn.Conv2d(1, 1, 2, pad_mode='valid')
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
    ms.set_context(mode=ms.GRAPH_MODE, save_graphs=True, save_graphs_path="./graphs")
    network = NetWithClassFunction()
    x = ms.Tensor(np.ones([1, 1, 4, 4]), ms.float32)
    y = network(x)
    # enable parse mindspore cells
    ms.rewrite.common.namespace._ms_cells_to_subtree = True # pylint:disable=protected-access
    stree = ms.rewrite.SymbolTree.create(network)
    amp._insert_cast_for_operators(stree, ms.float16, False, white_list=amp.AMP_WHITE_LIST) # pylint:disable=protected-access
    amp._remove_duplicated_cast(stree, ms.float16) # pylint:disable=protected-access
    codes = stree.get_code()
    assert codes.count("x = amp_cast(x, mindspore.float16)") == 3, codes
    assert codes.count("weight_var = amp_cast(self.weight, mindspore.float16)") == 2, codes
    assert codes.count("output = self.conv2d(x, weight_var)") == 2, codes
    assert codes.count("output = amp_cast(output, mindspore.float32)") == 4, codes
    assert codes.count("output_var = amp_cast(output, mindspore.float16)") == 2, codes
    assert codes.count("bias_var = amp_cast(self.bias, mindspore.float16)") == 2, codes
    assert codes.count("output = self.bias_add(output_var, bias_var)") == 2, codes
    assert codes.count("x = self.relu(x)") == 1, codes
    assert codes.count("x = amp_cast(x, mindspore.float32)") == 1, codes
    new_net = stree.get_network()
    # disable parse mindspore cells
    ms.rewrite.common.namespace._ms_cells_to_subtree = False # pylint:disable=protected-access
    y1 = new_net(x)
    assert np.allclose(y.asnumpy(), y1.asnumpy(), 0.001, 0.001)

class CellAndOps(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = nn.Dense(1, 1)
        self.abs = ops.Abs()

    def construct(self, x):
        x = self.relu(x)
        x = self.abs(x)
        return x

def test_cell_and_ops():
    """
    Feature: Test amp o1.
    Description: Network with cells and ops.
    Expectation: Success.
    """
    network = CellAndOps()
    # enable parse mindspore cells
    ms.rewrite.common.namespace._ms_cells_to_subtree = True # pylint:disable=protected-access
    stree = ms.rewrite.SymbolTree.create(network)
    amp._insert_cast_for_operators(stree, ms.float16, False, white_list=amp.AMP_WHITE_LIST) # pylint:disable=protected-access
    amp._remove_duplicated_cast(stree, ms.float16) # pylint:disable=protected-access
    codes = stree.get_code()
    # amp_cast should not exist in `class CellAndOpsOpt(CellAndOps, nn.Cell)`
    assert codes.count("x = amp_cast(x, mindspore.float16)") == 0, codes
    assert codes.count("x = amp_cast(x, mindspore.float32)") == 4, codes
