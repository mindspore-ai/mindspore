# Copyright 2022 Huawei Technologies Co., Ltd
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
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.rewrite import SymbolTree, Node, ScopedValue
from mindspore import Tensor
import numpy as np
from .test_import.net_with_unused_import import NetWithUnusedImport


def external_func(x):
    x = ops.abs(x)
    return x


class OtherClass():
    def other_class_func(self, x):
        x = ops.abs(x)
        return x


class SubSubNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.abs = ops.Abs()

    def construct(self, x):
        x = self.relu(x)
        x = external_func(x)
        x = self.subsubnet_internal_func(x)
        return x

    def subsubnet_internal_func(self, x):
        x = self.abs(x)
        return x


class SubNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.s_cells = nn.SequentialCell([SubSubNet(), SubSubNet()])
        self.subsubnet = SubSubNet()
        self.abs = ops.Abs()
        self.relu = nn.ReLU()
        self.other_class = OtherClass()

    def construct(self, x):
        x = self.relu(x)
        x = self.s_cells(x)
        x = self.subsubnet(x)
        x = external_func(x)
        x = self.subnet_internal_func(x)
        x = self.other_class.other_class_func(x)
        return x

    def subnet_internal_func(self, x):
        x = self.abs(x)
        return x


class MyNet(nn.Cell):
    def __init__(self, sub_net):
        super().__init__()
        self.relu = nn.ReLU()
        self.sub_net = sub_net
        self.s_cells = nn.SequentialCell(nn.ReLU())
        self.s_cells.append(nn.ReLU())
        self.s_cells.append(SubSubNet())
        self.abs = ops.Abs()
        self.sub_net1 = SubNet()
        self.sub_net2 = SubNet()
        self.other_class = OtherClass()

    def construct(self, x):
        x = self.relu(x)
        x = self.sub_net(x)
        x = self.sub_net1(x)
        x = self.sub_net1(x)
        x = self.sub_net2(x)
        x = self.s_cells(x)
        x = external_func(x)
        x = self.internal_func(x)
        x = self.other_class.other_class_func(x)
        return x

    def internal_func(self, x):
        x = self.sub_net(self.abs(x))
        return x


def test_generate_codes_from_symboltree():
    """
    Feature: Python api get_code of Node of Rewrite.
    Description: Test rewrite generate codes from symbol tree.
    Expectation: Success.
    """
    net = MyNet(SubNet())
    stree = SymbolTree.create(net)

    codes = stree.get_code()
    assert codes.count("def external_func") > 0
    assert codes.count("class SubSubNetOpt") == 1
    assert codes.count("def subsubnet_internal_func(self, x):") == 1
    assert codes.count("class SubNetOpt") == 1
    assert codes.count("def subnet_internal_func(self, x):") == 1
    assert codes.count("class MyNetOpt") == 1
    assert codes.count("def internal_func(self, x):") == 1

    subtree = stree.get_node("sub_net1").get_handler().symbol_tree
    subtree.erase_node(subtree.get_node("relu"))
    codes = stree.get_code()
    assert codes.count("class SubNetOpt") == 2

    subtree = stree.get_node("sub_net1_1").get_handler().symbol_tree
    subsubtree = subtree.get_node("subsubnet").symbol_tree
    subsubtree.erase_node(subsubtree.get_node("relu"))
    codes = stree.get_code()
    assert codes.count("class SubNetOpt") == 3
    assert codes.count("class SubSubNetOpt") == 2


class TransformerEncoderLayer(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(x)
        return x


class MOE(nn.Cell):
    def __init__(self):
        super().__init__()
        self.net = TransformerEncoderLayer()

    def construct(self, x):
        x = self.net(x)
        return x


class IfInInitNetSubNet(nn.Cell):
    def __init__(self, use_moe):
        super().__init__()
        self.use_moe = use_moe
        if self.use_moe:
            self.net1 = MOE()
        else:
            self.net2 = TransformerEncoderLayer()

    def construct(self, x):
        if self.use_moe:
            x = self.net1(x)
        else:
            x = self.net2(x)
        return x


class IfInInitNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.subnet1 = IfInInitNetSubNet(True)
        self.subnet2 = IfInInitNetSubNet(False)

    def construct(self, x):
        x = self.subnet1(x)
        x = self.subnet2(x)
        return x

def test_generate_codes_with_if_in_init():
    """
    Feature: Python api get_code of Node of Rewrite.
    Description: Test rewrite generate codes when two subnet in if statement of init func.
    Expectation: Success.
    """
    net = IfInInitNet()
    stree = SymbolTree.create(net)
    codes = stree.get_code()
    assert codes.count("class IfInInitNetOpt(IfInInitNet, nn.Cell):") == 1
    assert codes.count("class IfInInitNetSubNetOpt(IfInInitNetSubNet, nn.Cell):") == 1
    assert codes.count("class IfInInitNetSubNetOpt_1(IfInInitNetSubNet, nn.Cell):") == 1
    assert codes.count("self.net1 = MOEOpt(self.net1)") == 1
    assert codes.count("self.net = TransformerEncoderLayerOpt(self.net)") == 1
    assert codes.count("self.net2 = TransformerEncoderLayerOpt(self.net2)") == 1


class TransformerEncoderLayer2(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(x)
        return x


class MOE2(nn.Cell):
    def __init__(self):
        super().__init__()
        self.net = TransformerEncoderLayer2()

    def construct(self, x):
        x = self.net(x)
        return x


class IfInInitNetSubNet2(nn.Cell):
    def __init__(self, use_moe):
        super().__init__()
        self.use_moe = use_moe
        if self.use_moe:
            self.net = MOE2()
        else:
            self.net = TransformerEncoderLayer2()

    def construct(self, x):
        if self.use_moe:
            x = self.net(x)
        else:
            x = self.net(x)
        return x


class IfInInitNet2(nn.Cell):
    def __init__(self):
        super().__init__()
        self.subnet1 = IfInInitNetSubNet2(True)
        self.subnet2 = IfInInitNetSubNet2(False)

    def construct(self, x):
        x = self.subnet1(x)
        x = self.subnet2(x)
        return x

def test_generate_codes_with_if_in_init_and_construct_same_func_name():
    """
    Feature: Python api get_code of Node of Rewrite.
    Description: Test rewrite generate codes when two subnet of same func name in if statement of
    init func and construct func.
    Expectation: Success.
    """
    net = IfInInitNet2()
    stree = SymbolTree.create(net)
    codes = stree.get_code()
    assert codes.count("class IfInInitNet2Opt(IfInInitNet2, nn.Cell):") == 1
    assert codes.count("class IfInInitNetSubNet2Opt(IfInInitNetSubNet2, nn.Cell):") == 1
    assert codes.count("class IfInInitNetSubNet2Opt_1(IfInInitNetSubNet2, nn.Cell):") == 1
    assert codes.count("self.net = MOE2Opt(self.net)") == 2
    assert codes.count("self.net = TransformerEncoderLayer2Opt(self.net)") == 3


class TransformerEncoderLayer3(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(x)
        return x


class MOE3(nn.Cell):
    def __init__(self):
        super().__init__()
        self.net = TransformerEncoderLayer3()

    def construct(self, x):
        x = self.net(x)
        return x


class IfInInitNetSubNet3(nn.Cell):
    def __init__(self, use_moe):
        super().__init__()
        self.use_moe = use_moe
        if self.use_moe:
            self.net = MOE3()
        else:
            self.net = TransformerEncoderLayer3()

    def construct(self, x):
        x = self.net(x)
        return x


class IfInInitNet3(nn.Cell):
    def __init__(self):
        super().__init__()
        self.subnet1 = IfInInitNetSubNet3(False)
        self.subnet2 = IfInInitNetSubNet3(True)
        self.subnet3 = IfInInitNetSubNet3(False)

    def construct(self, x):
        x = self.subnet1(x)
        x = self.subnet2(x)
        x = self.subnet3(x)
        return x

def test_generate_codes_with_if_in_init_same_func_name():
    """
    Feature: Python api get_code of Node of Rewrite.
    Description: Test rewrite generate codes when two subnet of same func name in if statement of init func.
    Expectation: Success.
    """
    net = IfInInitNet3()
    stree = SymbolTree.create(net)
    codes = stree.get_code()
    assert codes.count("class IfInInitNet3Opt(IfInInitNet3, nn.Cell):") == 1
    assert codes.count("class IfInInitNetSubNet3Opt(IfInInitNetSubNet3, nn.Cell):") == 1
    assert codes.count("class IfInInitNetSubNet3Opt_1(IfInInitNetSubNet3, nn.Cell):") == 1
    assert codes.count("class MOE3Opt(MOE3, nn.Cell):") == 1
    assert codes.count("class TransformerEncoderLayer3Opt(TransformerEncoderLayer3, nn.Cell):") == 1
    assert codes.count("self.subnet1 = IfInInitNetSubNet3Opt(self.subnet1)") == 1
    assert codes.count("self.subnet2 = IfInInitNetSubNet3Opt_1(self.subnet2)") == 1
    assert codes.count("self.subnet3 = IfInInitNetSubNet3Opt(self.subnet3)") == 1
    assert codes.count("self.subnet3 = IfInInitNetSubNet3Opt(self.subnet3)") == 1
    assert codes.count("self.net = MOE3Opt(self.net)") == 1
    assert codes.count("self.net = TransformerEncoderLayer3Opt(self.net)") == 2


class TestAnnotationSubNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(x)
        return x


class TestAnnotationNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.sub_net = self.get_subnet(TestAnnotationSubNet())

    def construct(self, x):
        x = self.relu(x)
        x = self.sub_net(x)
        return x

    def get_subnet(self, subnet: TestAnnotationSubNet):
        return subnet

def test_annotation():
    """
    Feature: Python api get_code of Node of Rewrite.
    Description: Test rewrite generate codes when net has annotation.
    Expectation: Success with annotation being removed.
    """
    net = TestAnnotationNet()
    y0 = net(Tensor(1.0))
    stree = SymbolTree.create(net)
    codes = stree.get_code()
    assert codes.count("def get_subnet(self, subnet):") == 1
    new_net = stree.get_network()
    y = new_net(Tensor(1.0))
    assert np.allclose(y0.asnumpy(), y.asnumpy())


def test_net_with_unused_import():
    """
    Feature: Python api get_code of Node of Rewrite.
    Description: Test rewrite generate codes when import is not used by origin network but used by inserted node.
    Expectation: Success.
    """
    net = NetWithUnusedImport()
    y0 = net(Tensor(1.0))
    stree = SymbolTree.create(net)
    relu_node = stree.get_node("relu")
    relu_target = relu_node.get_targets()[0]
    dst_type = ScopedValue.create_naming_value("float32", "ms")
    new_node = Node.create_call_cell(ops.Cast(), [relu_target], [relu_target, dst_type], name="cast_node")
    stree.insert(stree.after(relu_node), new_node)
    new_net = stree.get_network()
    y = new_net(Tensor(1.0))
    assert np.allclose(y0.asnumpy(), y.asnumpy())


class NetWithStatic(nn.Cell):

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(x)
        x = self.static_func(x)
        return x

    @staticmethod
    def static_func(x):
        return x

class SubNetWithStatic(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(x)
        x = self.static_func(x)
        return x

    @staticmethod
    def static_func(x):
        raise NotImplementedError


class NetWithSubNetStatic(SubNetWithStatic):
    @staticmethod
    def static_func(x):
        return x


def test_net_with_static_method():
    """
    Feature: Rewrite.
    Description: Test rewrite parse network with static method function.
    Expectation: Success.
    """
    net = NetWithStatic()
    y0 = net(Tensor(1.0))
    stree = SymbolTree.create(net)
    new_net = stree.get_network()
    y = new_net(Tensor(1.0))
    assert np.allclose(y0.asnumpy(), y.asnumpy())

    net = NetWithSubNetStatic()
    y0 = net(Tensor(1.0))
    stree = SymbolTree.create(net)
    new_net = stree.get_network()
    y = new_net(Tensor(1.0))
    assert np.allclose(y0.asnumpy(), y.asnumpy())
