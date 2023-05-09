import pytest
from mindspore import Tensor
from mindspore.rewrite import SymbolTree
import mindspore.nn as nn
import mindspore as ms


class BaseNet(nn.Cell):
    def __init__(self, a):
        super().__init__()
        self.relu = nn.ReLU()
        self.a = a

    def construct(self, x):
        return x

    def add_a(self, x):
        x = x + self.a
        return x


class NetA(BaseNet):
    def add_x(self, x):
        x = x + x
        return x


class NetB(NetA):
    def construct(self, x):
        x = self.add_a(x)
        x = self.add_x(x)
        return x


class NetC(nn.Cell):
    def __init__(self, a, b):
        super().__init__()
        self.relu = nn.ReLU()
        self.net_b = NetB(a)
        self.b = b

    def construct(self, x):
        x = self.relu(x)
        x = self.net_b(x)
        x = x + self.b
        return x


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_one_father_class(mode):
    """
    Feature: Test Rewrite.
    Description: Test Rewrite with one father class.
    Expectation: Success.
    """
    ms.set_context(mode=mode)
    net = NetA(Tensor(2))
    stree = SymbolTree.create(net)
    new_net = stree.get_network()
    y = new_net(Tensor(1))
    assert y == 1


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_two_father_classes(mode):
    """
    Feature: Test Rewrite.
    Description: Test Rewrite with two father classes.
    Expectation: Success.
    """
    ms.set_context(mode=mode)
    net = NetB(Tensor(2))
    stree = SymbolTree.create(net)
    new_net = stree.get_network()
    y = new_net(Tensor(1))
    assert y == 6


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_two_father_classes_in_tree(mode):
    """
    Feature: Test Rewrite.
    Description: Test Rewrite with two father classes in tree node.
    Expectation: Success.
    """
    ms.set_context(mode=mode)
    net = NetC(Tensor(2), Tensor(3))
    stree = SymbolTree.create(net)
    new_net = stree.get_network()
    y = new_net(Tensor(1))
    assert y == 9
