
from collections import OrderedDict

from mindspore import nn
from mindspore.ops import operations as P
from mindspore.rewrite import SymbolTree, PatternEngine, Replacement, PatternNode, Node, ScopedValue
from mindspore.rewrite.api.node_type import NodeType


def make_layer(block, layer_num, in_channel, out_channel, stride, use_se=False, se_block=False):
    """
    Make stage network of ResNet.

    Args:
        block (Cell): Resnet block.
        layer_num (int): Layer number.
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        stride (int): Stride size for the first convolutional layer.
        se_block(bool): Use se block in SE-ResNet50 net. Default: False.
    Returns:
        SequentialCell, the output layer.

    Examples:
        >>> _make_layer(ResidualBlock, 3, 128, 256, 2)
    """
    layers = []

    resnet_block = block(in_channel, out_channel, stride=stride, use_se=use_se)
    layers.append(resnet_block)
    if se_block:
        for _ in range(1, layer_num - 1):
            resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se)
            layers.append(resnet_block)
        resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se, se_block=se_block)
        layers.append(resnet_block)
    else:
        for _ in range(1, layer_num):
            resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se)
            layers.append(resnet_block)
    return nn.SequentialCell(layers)


class ConvBnReplace(Replacement):
    def build(self, pattern: PatternNode, is_chain_pattern: bool, matched: OrderedDict) -> [Node]:
        bn_node: Node = matched.get(pattern.name())
        bn: nn.BatchNorm2d = bn_node.get_instance()
        conv_p = pattern.get_inputs()[0]
        conv_node: Node = matched.get(conv_p.name())
        conv: nn.Conv2d = conv_node.get_instance()
        newconv = nn.Conv2dBnAct(conv.in_channels,
                                 conv.out_channels,
                                 conv.kernel_size,
                                 conv.stride,
                                 conv.pad_mode,
                                 conv.padding,
                                 conv.dilation,
                                 conv.group,
                                 conv.has_bias,
                                 conv.weight_init,
                                 conv.bias_init,
                                 True,
                                 bn.momentum,
                                 bn.eps)
        newconv_node = Node.create_call_cell(newconv, bn_node.get_targets(), conv_node.get_args(),
                                             conv_node.get_kwargs(), "Conv2dBnAct")
        return [newconv_node]


class ConvBnPattern(PatternEngine):
    def __init__(self):
        super().__init__([nn.Conv2d, nn.BatchNorm2d], ConvBnReplace())


class CellBlock(nn.Cell):
    """
    ResNet V1 residual block definition.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        stride (int): Stride size for the first convolutional layer. Default: 1.
        use_se (bool): Enable SE-ResNet50 net. Default: False.
        se_block(bool): Use se block in SE-ResNet50 net. Default: False.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResidualBlock(3, 256, stride=2)
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1,):
        super(CellBlock, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 1, stride=1)
        self.bn1 = nn.BatchNorm2d(6, eps=1e-4, momentum=0.9,
                                  gamma_init=0, beta_init=0, moving_mean_init=0, moving_var_init=1)
        self.relu = nn.ReLU()
        self.down_sample_layer = nn.SequentialCell([nn.Conv2d(in_channel, out_channel, 1)])

    def construct(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        x = self.down_sample_layer(x)
        out = out + x
        return out


class SimpleNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.mul = P.Mul()
        self.dense = nn.Dense(in_channels=32, out_channels=32, weight_init="ones")
        self.mean = P.ReduceMean(keep_dims=False)
        self.split = P.Split(axis=1, output_num=3)
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block = CellBlock(3, 6)

    def construct(self, x):
        y, _, _ = self.split(x)
        y = self.mean(y, (2, 3))
        x = self.mul(x, 1)
        x = self.block(x)
        x = self.conv1(x)
        x = self.max_pool2d(x)
        x = self.dense(x)
        return x, y


class ForNetWithSubTree(nn.Cell):
    def __init__(self):
        super(ForNetWithSubTree, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 1)
        self.conv2 = nn.Conv2d(6, 16, 1)
        self.relu = nn.ReLU()
        self.relu1 = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool2d1 = nn.MaxPool2d(kernel_size=2, stride=2)
        layers1 = [self.conv1, self.conv2, self.max_pool2d, self.relu]
        self.layer1 = nn.SequentialCell(layers1)

        resnet_block1 = CellBlock(3, 6)
        resnet_block2 = CellBlock(6, 16)
        resnet_block3 = CellBlock(16, 32)
        layers = [resnet_block1, resnet_block2, resnet_block3]
        self.layer2 = nn.SequentialCell(layers)
        self.simple_net = SimpleNet()

    def construct(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.simple_net(x)
        return x


def test_erase_subtree_node():
    """
    Feature: parser and erase api.
    Description: erase a node in subtree of `SymbolTree`.
    Expectation: Success.
    """
    net = ForNetWithSubTree()
    stree = SymbolTree.create(net)

    for node in stree.nodes():
        if node.get_name() == "simple_net":
            subtree = node.get_sub_tree()
            orig_node_num = len(subtree.get_handler().nodes())
            for n in subtree.nodes():
                if n.get_instance_type() == nn.MaxPool2d:
                    input_node = n.get_inputs()[0]
                    output_nodes = n.get_users()
                    for out_node in output_nodes:
                        out_node.set_arg_by_node(0, input_node)
                    subtree.erase(n)
                    break
            assert len(subtree.get_handler().nodes()) == orig_node_num - 1
            break


def test_erase_subtree_node_01():
    """
    Feature: parser and erase api.
    Description: erase a node in subtree of `SymbolTree`.
    Expectation: Success.
    """
    net = ForNetWithSubTree()
    stree = SymbolTree.create(net)

    for node in stree.nodes():
        if node.get_name() == "simple_net":
            subtree = node.get_sub_tree()
            orig_node_num = len(subtree.get_handler().nodes())
            for n in subtree.nodes():
                if n.get_name() == "block":
                    input_node = n.get_inputs()[0]
                    output_nodes = n.get_users()
                    for _nn in output_nodes:
                        _nn.set_arg_by_node(0, input_node)
                    subtree.erase(n)
                    assert len(subtree.get_handler().nodes()) == orig_node_num - 1
                    break
            break


def test_erase_subtree_node_02():
    """
    Feature: parser and erase api.
    Description: parser and erase node in subtree of `SymbolTree`.
    Expectation: Success.
    """
    def _remove_bn(subtree):
        for node in subtree.nodes():
            if node.get_name() == "bn1":
                input_node = node.get_inputs()[0]
                output_nodes = node.get_users()
                for n in output_nodes:
                    n.set_arg_by_node(0, input_node)
                subtree.erase(node)
                break

    net = ForNetWithSubTree()
    stree = SymbolTree.create(net)
    for node in stree.nodes():
        if node.get_name() == "simple_net":
            subtree = node.get_sub_tree()
            for n in subtree.nodes():
                if n.get_name() == "block":
                    subtree1 = n.get_sub_tree()
                    _remove_bn(subtree1)
                    assert subtree1.get_node("bn1") is None
                    break


def test_insert_subtree_node():
    """
    Feature: parser and insert api.
    Description: Insert node into subtree in `Symboltree`.
    Expectation: Success.
    """
    def _insert_node(subtree):
        for node in subtree.nodes():
            if node.get_name() == "bn1":
                position = subtree.before(node)
                new_conv = nn.Conv2d(16, 16, 3)
                new_conv_node = Node.create_call_cell(new_conv, targets=['x_1'], name='new_conv',
                                                      args=[ScopedValue.create_naming_value('self_max_po')])
                subtree.insert(position, new_conv_node)

    net = ForNetWithSubTree()
    stree = SymbolTree.create(net)
    for node in stree.nodes():
        if node.get_name() == "simple_net" and  node.get_node_type() == NodeType.Tree:
            subtree = node.get_sub_tree()
            for n in subtree.nodes():
                if n.get_name() == "block":
                    subtree1 = n.get_sub_tree()
                    orig_node_num = len(subtree1.get_handler().nodes())
                    _insert_node(subtree1)
                    assert len(subtree1.get_handler().nodes()) == orig_node_num + 1


def test_resnet_replace_121():
    """
    Feature: parser and replace api.
    Description: Replace one node by one nodes in subtree of `SymbolTree`.
    Expectation: Success.
    """
    net = ForNetWithSubTree()
    stree: SymbolTree = SymbolTree.create(net)
    original_nodes_size = len(stree.get_handler().nodes())
    for node in stree.nodes():
        if node.get_name() == "simple_net" and  node.get_node_type() == NodeType.Tree:
            subtree = node.get_sub_tree()
            for n in subtree.nodes():
                if n.get_instance_type() == nn.Conv2d:
                    conv: nn.Conv2d = n.get_instance()
                    new_conv = Node.create_call_cell(nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size),
                                                     targets=n.get_targets(), args=n.get_args(),
                                                     kwargs=node.get_kwargs(), name="new_conv")
                    subtree.replace(n, [new_conv])
                    break
    assert len(stree.get_handler().nodes()) == original_nodes_size


def test_resnet_replace_12m():
    """
    Feature: parser and replace api.
    Description: Replace one node by multi-nodes in subtree of `SymbolTree`.
    Expectation: Success.
    """
    net = ForNetWithSubTree()
    stree: SymbolTree = SymbolTree.create(net)

    for node in stree.nodes():
        if node.get_name() == "simple_net" and  node.get_node_type() == NodeType.Tree:
            subtree = node.get_sub_tree()
            original_nodes_size = len(subtree.get_handler().nodes())
            for n in subtree.nodes():
                if n.get_instance_type() == nn.Conv2d:
                    conv: nn.Conv2d = n.get_instance()
                    new_conv = Node.create_call_cell(nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size),
                                                     targets=["x"], args=n.get_args(),
                                                     kwargs=node.get_kwargs(), name="new_conv")
                    new_bn = Node.create_call_cell(nn.BatchNorm2d(conv.out_channels),
                                                   targets=n.get_targets(), args=[ScopedValue.create_naming_value("x")],
                                                   kwargs={}, name="new_bn")
                    subtree.replace(n, [new_conv, new_bn])
                    break
            assert len(subtree.get_handler().nodes()) == original_nodes_size + 1


def test_node_fusion_in_subtree():
    """
    Feature: parser and PatternEngine.
    Description: Apply PatternEngine on nodes in `SymbolTree`..
    Expectation: Success.
    """
    net = ForNetWithSubTree()
    stree: SymbolTree = SymbolTree.create(net)
    original_nodes_size = len(stree.get_handler().nodes())
    for node in stree.nodes():
        if node.get_name() == "simple_net" and  node.get_node_type() == NodeType.Tree:
            subtree = node.get_sub_tree()
            original_nodes_size = len(subtree.get_handler().nodes())
            for n in subtree.nodes():
                node_: Node = n
                if node_.get_instance_type() == nn.Conv2d:
                    old_bn = node_.get_users()[0]
                    pos = subtree.after(node_)
                    conv: nn.Conv2d = node_.get_instance()
                    new_bn = Node.create_call_cell(nn.BatchNorm2d(conv.out_channels), targets=["x"],
                                                   args=[node_.get_targets()[0]], kwargs={}, name="new_bn")
                    subtree.insert(pos, new_bn)
                    old_bn.set_arg_by_node(0, new_bn)
                    break
            assert len(subtree.get_handler().nodes()) == original_nodes_size + 1
            ConvBnPattern().apply(subtree)
            assert len(subtree.get_handler().nodes()) == original_nodes_size
            assert not subtree.get_node("conv1")
            assert not subtree.get_node("new_bn")
