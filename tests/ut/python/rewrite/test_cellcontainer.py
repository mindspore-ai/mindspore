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
"""test cell container."""

from mindspore import nn
from mindspore.ops import operations as P

from mindspore.rewrite import SymbolTree, NodeType, TreeNodeHelper, Node, ScopedValue, PatternEngine, Replacement, \
    PatternNode


def _conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                     padding=0, pad_mode='same', weight_init="ones")


def _conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride,
                     padding=0, pad_mode='same', weight_init="ones")


def _bn(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)


class ResidualBlock(nn.Cell):
    expansion = 4

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        channel = out_channel // self.expansion
        self.conv1 = _conv1x1(in_channel, channel, stride=1)
        self.bn1 = _bn(channel)
        self.conv2 = _conv3x3(channel, channel, stride=stride)
        self.bn2 = _bn(channel)
        self.conv3 = _conv1x1(channel, out_channel, stride=1)
        self.bn3 = _bn(out_channel)
        self.relu = nn.ReLU()
        self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel, stride), _bn(out_channel)])

    def construct(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        identity = self.down_sample_layer(identity)
        out = out + identity
        out = self.relu(out)

        return out


class ResNetSimple(nn.Cell):
    def __init__(self):
        super(ResNetSimple, self).__init__(auto_prefix=True)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, pad_mode='pad', weight_init="ones")
        self.bn1 = _bn(16)
        self.relu = P.ReLU()
        self.layer1 = self._make_layer(ResidualBlock, 3, in_channel=63, out_channel=256, stride=1)
        self.layer1.append(self.conv1)
        self.layer1.append(self.bn1)
        self.reshape = P.Reshape()
        self.out_channels = 10

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        return x

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride):
        layers = []
        resnet_block = block(in_channel, out_channel, stride=stride)
        layers.append(resnet_block)
        for _ in range(1, layer_num):
            resnet_block = ResidualBlock(out_channel, out_channel, stride=1)
            layers.append(resnet_block)
        return nn.SequentialCell(layers)


def test_cellcontainer_parse():
    """
    Feature: parse CellContainer Node.
    Description: parse a network with SequentialCell object.
    Expectation: Rewrite can parse a network with SquentialCell object successfully.
    """
    net = ResNetSimple()
    stree = SymbolTree.create(net)
    for node in stree.nodes():
        if node.get_node_type() == NodeType.CellContainer:
            assert len(node.get_handler().node_list) == 5
            for i, n in enumerate(node.get_handler().node_list):
                if i < 3:
                    assert n.get_instance_type() is ResidualBlock
                if i == 3:
                    assert n.get_instance_type() is nn.Conv2d
                if i == 4:
                    assert n.get_instance_type() is nn.BatchNorm2d


def test_cellcontainer_insert():
    """
    Feature: modify CellContainer Node.
    Description: using node in container to set insert location.
    Expectation: raise ValueError.
    """
    net = ResNetSimple()
    stree = SymbolTree.create(net)
    for node in stree.nodes():
        if node.get_node_type() == NodeType.CellContainer:
            assert len(node.get_handler().nodes()) == 5
            for n in node.get_handler().nodes():
                if n.get_instance_type() is nn.Conv2d:
                    position = stree.before(Node(n))
                    new_conv = nn.Conv2d(16, 16, 3)
                    new_conv_node = Node.create_call_cell(new_conv, targets=['x_1'], name='new_conv',
                                                          args=[ScopedValue.create_naming_value('self_max_po')])
                    stree.insert(position, new_conv_node)
                    break
            assert len(node.get_handler().nodes()) == 6
            assert node.get_handler().node_list[3].get_name() == "new_conv"


def test_cellcontainer_insert_ok():
    """
    Feature: modify CellContainer Node.
    Description: Inserts a node within a tree node in CellContainer Node.
    Expectation: Insertion succeeded.
    """
    def _insert_conv(stree: SymbolTree):
        for node in stree.nodes():
            if node.get_instance_type() == nn.BatchNorm2d:
                position = stree.after(node)
                new_conv = nn.Conv2d(16, 16, 3)
                new_conv_node = Node.create_call_cell(new_conv, targets=['x_1'], name='new_conv',
                                                      args=[ScopedValue.create_naming_value('self_max_po')])
                stree.insert(position, new_conv_node)
                break
    net = ResNetSimple()
    stree = SymbolTree.create(net)
    for node in stree.nodes():
        if node.get_node_type() == NodeType.CellContainer:
            for n in node.get_handler().node_list:
                if n.get_node_type() == NodeType.Tree:
                    _insert_conv(TreeNodeHelper.get_sub_tree(Node(n)))
                    break
    new_net = stree.get_network()
    cell_container = getattr(new_net, "layer1")
    assert hasattr(cell_container._cells["0"], "new_conv")


def test_cellcontainer_insert_to_subtree():
    """
    Feature: modify CellContainer Node.
    Description: Inserts a node within a tree node in CellContainer Node.
    Expectation: Insertion succeeded.
    """
    def _insert_conv(stree: SymbolTree):
        for node in stree.nodes():
            if node.get_instance_type() == nn.BatchNorm2d:
                position = stree.after(node)
                new_conv = nn.Conv2d(16, 16, 3)
                new_conv_node = Node.create_call_cell(new_conv, targets=['x_1'], name='new_conv',
                                                      args=[ScopedValue.create_naming_value('self_max_po')])
                stree.insert(position, new_conv_node)
                break
    net = ResNetSimple()
    stree = SymbolTree.create(net)
    for node in stree.nodes():
        if node.get_node_type() == NodeType.CellContainer:
            for n in node.get_handler().node_list:
                if n.get_node_type() == NodeType.Tree:
                    _insert_conv(TreeNodeHelper.get_sub_tree(Node(n)))
                    break
    new_net = stree.get_network()
    cell_container = getattr(new_net, "layer1")
    assert hasattr(cell_container._cells["0"], "new_conv")


def test_cellcontainer_del():
    """
    Feature: modify CellContainer Node.
    Description: delete the CellContainer Node.
    Expectation: success.
    """
    net = ResNetSimple()
    stree = SymbolTree.create(net)
    original_nodes_size = len(stree.get_handler()._nodes)
    for node in stree.nodes():
        if node.get_node_type() == NodeType.CellContainer and node.get_name() == "layer1":
            users = node.get_users()
            for user in users:
                user.set_arg(0, "x")
            stree.erase_node(node)
    assert len(stree.get_handler()._nodes) == original_nodes_size - 1


def test_cellcontainer_del_node():
    """
    Feature: modify CellContainer Node.
    Description: delete the CellContainer Node.
    Expectation: success.
    """
    net = ResNetSimple()
    stree = SymbolTree.create(net)
    for node in stree.nodes():
        if node.get_node_type() == NodeType.CellContainer and node.get_name() == "layer1":
            assert len(node.get_handler().nodes()) == 5
            for n in node.get_handler().nodes():
                users = node.get_users()
                inputs = node.get_inputs()
                for user in users:
                    user.set_arg_by_node(0, inputs[0])
                stree.erase_node(Node(n))
                break
            assert len(node.get_handler().nodes()) == 4


def test_cellcontainer_del_node_in_subtree():
    """
    Feature: modify CellContainer Node.
    Description: delete a node within a tree node in CellContainer Node.
    Expectation: success.
    """
    def _del_node(sub_tree):
        for _node in sub_tree.nodes():
            if _node.get_name() == "conv2":
                users = Node(_node).get_users()
                for user in users:
                    user.set_arg(0, "out")
                sub_tree.erase_node(_node)
    net = ResNetSimple()
    stree = SymbolTree.create(net)
    for node in stree.nodes():
        if node.get_node_type() == NodeType.CellContainer:
            for i, n in enumerate(node.get_handler().node_list):
                if n.get_node_type() == NodeType.Tree and i == 1:
                    sub_tree = n.symbol_tree
                    original_nodes_size = len(sub_tree._nodes)
                    _del_node(sub_tree)
                    assert len(sub_tree._nodes) == original_nodes_size - 1

    new_net = stree.get_network()
    cell_container = getattr(new_net, "layer1")
    assert not hasattr(cell_container._cells["1"], "conv2")


def test_cellcontainer_replace():
    """
    Feature: modify CellContainer Node.
    Description: replace CellContainer Node with another Node.
    Expectation: success.
    """
    def _replace_bn(stree: SymbolTree):
        for node in stree.nodes():
            if node.get_node_type() == NodeType.CellContainer:
                new_conv = nn.Conv2d(16, 16, 3)
                new_conv_node = Node.create_call_cell(new_conv, targets=['x_1'], name='new_conv',
                                                      args=[ScopedValue.create_naming_value('x')])
                stree.replace(node, [new_conv_node])
                break
    net = ResNetSimple()
    stree = SymbolTree.create(net)
    _replace_bn(stree)
    new_net = stree.get_network()
    assert not hasattr(new_net, "layer1")
    assert hasattr(new_net, "new_conv")


def test_cellcontainer_replace_node():
    """
    Feature: modify CellContainer Node.
    Description: replace the CellContainer Node.
    Expectation: success.
    """
    net = ResNetSimple()
    stree = SymbolTree.create(net)
    for node in stree.nodes():
        if node.get_node_type() == NodeType.CellContainer and node.get_name() == "layer1":
            for n in node.get_handler().nodes():
                new_conv = nn.Conv2d(16, 16, 3)
                new_conv_node = Node.create_call_cell(new_conv, targets=['x_1'], name='new_conv',
                                                      args=[ScopedValue.create_naming_value('x')])
                stree.replace(Node(n), [new_conv_node])
                break
            assert node.get_handler().node_list[0].get_name() == "new_conv"
            assert isinstance(node.get_handler().get_instance()._cells["0"], nn.Conv2d)
            break


def test_cellcontainer_replace_in_subtree():
    """
    Feature: modify CellContainer Node.
    Description: replace a node within a tree node in CellContainer Node.
    Expectation: success.
    """
    def _replace_bn(stree: SymbolTree):
        for node in stree.nodes():
            if node.get_name() == "bn1":
                new_conv = nn.Conv2d(16, 16, 3)
                new_conv_node = Node.create_call_cell(new_conv, targets=['x_1'], name='new_conv',
                                                      args=[ScopedValue.create_naming_value('self_max_po')])
                stree.replace(node, [new_conv_node])
                break
    net = ResNetSimple()
    stree = SymbolTree.create(net)
    for node in stree.nodes():
        if node.get_node_type() == NodeType.CellContainer:
            for n in node.get_handler().node_list:
                if n.get_node_type() == NodeType.Tree:
                    _replace_bn(TreeNodeHelper.get_sub_tree(Node(n)))
                    break
    new_net = stree.get_network()
    cell_container = getattr(new_net, "layer1")
    assert not hasattr(cell_container._cells["0"], "bn1")
    assert hasattr(cell_container._cells["0"], "new_conv")


def test_cellcontainer_pattern():
    """
    Feature: modify CellContainer Node.
    Description: apply pattern matching and replacement on the network containing SequentialCell object.
    Expectation: success.
    """
    class ConvBnReplacement(Replacement):
        def build(self, pattern: PatternNode, is_chain_pattern: bool, matched):
            assert is_chain_pattern
            assert pattern.type() == nn.BatchNorm2d
            bn_node: Node = matched.get(pattern.name())
            assert bn_node is not None
            assert len(pattern.get_inputs()) == 1
            add_pattern = pattern.get_inputs()[0]
            assert add_pattern.type() == nn.Conv2d
            add_node: Node = matched.get(add_pattern.name())
            assert add_node is not None
            assert not add_pattern.get_inputs()

            new_maxpool1 = nn.MaxPool2d()
            new_maxpool1_node = Node.create_call_cell(new_maxpool1, ['new_maxpool1'], add_node.get_args())
            new_relu1 = nn.ReLU()
            new_relu1_node = Node.create_call_cell(new_relu1, ['new_relu_1'],
                                                   [ScopedValue.create_naming_value('new_maxpool1')])
            new_relu2 = nn.ReLU()
            new_relu2_node = Node.create_call_cell(new_relu2, ['new_relu_2'],
                                                   [ScopedValue.create_naming_value('new_maxpool1')])
            new_maxpool2 = nn.BiDense(1, 1, 2)
            new_maxpool2_node = Node.create_call_cell(new_maxpool2, ['new_maxpool2'],
                                                      [ScopedValue.create_naming_value('new_relu_1'),
                                                       ScopedValue.create_naming_value('new_relu_2')])
            return [new_maxpool1_node, new_relu1_node, new_relu2_node, new_maxpool2_node]


    class ConvReluPattern(PatternEngine):
        def __init__(self):
            super().__init__([nn.Conv2d, nn.BatchNorm2d], ConvBnReplacement())

    net = ResNetSimple()
    stree = SymbolTree.create(net)
    _pattern = ConvReluPattern()
    _pattern.apply(stree)
    new_net = stree.get_network()
    cell_container = getattr(new_net, "layer1")
    assert not hasattr(cell_container, "conv1")
    assert not hasattr(cell_container, "bn1")
    assert not hasattr(cell_container._cells["0"], "conv1")
    assert not hasattr(cell_container._cells["1"], "conv1")
    assert not hasattr(cell_container._cells["2"], "conv1")
    assert hasattr(cell_container._cells["0"], "new_relu")
    assert hasattr(cell_container._cells["0"], "new_maxpool1")
    assert isinstance(getattr(getattr(cell_container._cells["0"], "down_sample_layer"), "0"), nn.MaxPool2d)
    assert hasattr(cell_container._cells["1"], "new_relu")
    assert hasattr(cell_container._cells["1"], "new_maxpool1")
    assert isinstance(getattr(getattr(cell_container._cells["1"], "down_sample_layer"), "0"), nn.MaxPool2d)
    assert hasattr(cell_container._cells["2"], "new_relu")
    assert hasattr(cell_container._cells["2"], "new_maxpool1")
    assert isinstance(getattr(getattr(cell_container._cells["2"], "down_sample_layer"), "0"), nn.MaxPool2d)
    assert isinstance(getattr(cell_container, "3"), nn.MaxPool2d)
    assert isinstance(getattr(cell_container, "4"), nn.ReLU)
    assert isinstance(getattr(cell_container, "6"), nn.BiDense)


def test_cellcontainer_first_node_inputs():
    """
    Feature: create CellContainer Node.
    Description: nodes in cellcontainer has inputs.
    Expectation: success.
    """
    net = ResNetSimple()
    stree = SymbolTree.create(net)
    for node in stree.nodes():
        if node.get_node_type() == NodeType.CellContainer and node.get_name() == "layer1":
            for n in node.get_handler().nodes():
                inputs = n.get_inputs()
                assert inputs
                assert hasattr(n, "container")
                assert hasattr(n, "valid")
                assert getattr(n, "valid")
