import pytest
from mindspore import Tensor
from mindspore.rewrite import SymbolTree
import mindspore.nn as nn
import mindspore as ms
from .models import BaseNet, NoCellNet, NetWithClassVar
from tests.mark_utils import arg_mark


class NetA(BaseNet):
    def add_x(self, x):
        x = x + x
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_one_father_class(mode):
    """
    Feature: Test Rewrite.
    Description: Test Rewrite with one father class.
    Expectation: Success.
    """
    ms.set_context(mode=mode)
    net = NetA(Tensor(2))
    y0 = net(Tensor(1))
    stree = SymbolTree.create(net)
    codes = stree.get_code()
    assert codes.count("class NetAOpt(NetA, BaseNetOpt):") == 1
    assert codes.count("class BaseNetOpt(BaseNet, nn.Cell):") == 1
    new_net = stree.get_network()
    y = new_net(Tensor(1))
    assert y == y0


class NetB(NetA):
    def construct(self, x):
        x = self.add_a(x)
        x = self.add_x(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_two_level_father_classes(mode):
    """
    Feature: Test Rewrite.
    Description: Test Rewrite with two father classes.
    Expectation: Success.
    """
    ms.set_context(mode=mode)
    net = NetB(Tensor(2))
    y0 = net(Tensor(1))
    stree = SymbolTree.create(net)
    codes = stree.get_code()
    assert codes.count("class NetBOpt(NetB, NetAOpt):") == 1, codes
    assert codes.count("class NetAOpt(NetA, BaseNetOpt):") == 1, codes
    assert codes.count("class BaseNetOpt(BaseNet, nn.Cell):") == 1, codes
    new_net = stree.get_network()
    y = new_net(Tensor(1))
    assert y == y0


class NetB1(NetA):
    def construct(self, x):
        x = self.add_a(x)
        x = self.add_x(x)
        return x


class NetC(nn.Cell):
    def __init__(self, a, b):
        super().__init__()
        self.relu = nn.ReLU()
        self.net_b = NetB1(a)
        self.b = b

    def construct(self, x):
        x = self.relu(x)
        x = self.net_b(x)
        x = x + self.b
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_two_level_father_classes_in_tree(mode):
    """
    Feature: Test Rewrite.
    Description: Test Rewrite with two father classes in tree node.
    Expectation: Success.
    """
    ms.set_context(mode=mode)
    net = NetC(Tensor(2), Tensor(3))
    y0 = net(Tensor(1))
    stree = SymbolTree.create(net)
    codes = stree.get_code()
    assert codes.count("class NetCOpt(NetC, nn.Cell):") == 1
    assert codes.count("class NetB1Opt(NetB1, NetAOpt):") == 1
    assert codes.count("class NetAOpt(NetA, BaseNetOpt):") == 1
    assert codes.count("class BaseNetOpt(BaseNet, nn.Cell):") == 1
    new_net = stree.get_network()
    y = new_net(Tensor(1))
    assert y == y0


class NetD(BaseNet, NoCellNet):
    def __init__(self, a, b):
        BaseNet.__init__(self, a)
        NoCellNet.__init__(self, a, b)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(x)
        x = self.no_cell_func(x)
        x = self.add_a(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_two_father_classes_one_not_cell(mode):
    """
    Feature: Test Rewrite.
    Description: Test Rewrite with two father classes, one of them is not subclass of nn.Cell.
    Expectation: Success.
    """
    ms.set_context(mode=mode)
    net = NetD(Tensor(1.0), Tensor(2.0))
    y0 = net(Tensor(1))
    stree = SymbolTree.create(net)
    codes = stree.get_code()
    assert codes.count("class NetDOpt(NetD, BaseNetOpt, NoCellNetOpt):") == 1
    assert codes.count("class NoCellNetOpt(NoCellNet):") == 1
    assert codes.count("class BaseNetOpt(BaseNet, nn.Cell):") == 1
    new_net = stree.get_network()
    y = new_net(Tensor(1))
    assert y == y0


def external_func(x):
    return x


class NetE(nn.Cell):
    var1 = Tensor(1.0)
    var2 = external_func

    def __init__(self, a):
        super().__init__()
        self.a = a

    def construct(self, x):
        x = x + self.a
        x = x + self.var1
        x = NetE.var2(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_net_with_class_var(mode):
    """
    Feature: Test Rewrite.
    Description: Test Rewrite with class variables.
    Expectation: Success.
    """
    ms.set_context(mode=mode)
    net = NetE(Tensor(1))
    y0 = net(Tensor(1))
    stree = SymbolTree.create(net)
    codes = stree.get_code()
    assert codes.count("class NetEOpt(NetE, nn.Cell):") == 1
    assert codes.count("def external_func(x):") == 0
    assert codes.count("var1 = Tensor(1.0)") == 0
    assert codes.count("self.__class__.var1 = obj.__class__.var1") == 1
    assert codes.count("var2 = external_func") == 0
    assert codes.count("self.__class__.var2 = obj.__class__.var2") == 1
    new_net = stree.get_network()
    y = new_net(Tensor(1))
    assert y == y0


class NetF(BaseNet, NoCellNet, NetWithClassVar):
    def __init__(self):
        BaseNet.__init__(self, Tensor(1))
        NoCellNet.__init__(self, Tensor(1), Tensor(2))
        NetWithClassVar.__init__(self, Tensor(1))
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(x)
        x = self.no_cell_func(x)
        x = self.add_a(x)
        x = self.class_var_func(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_father_classes_with_class_var(mode):
    """
    Feature: Test Rewrite.
    Description: Test Rewrite with two father classes, one of them has class variables.
    Expectation: Success.
    """
    ms.set_context(mode=mode)
    net = NetF()
    y0 = net(Tensor(1))
    stree = SymbolTree.create(net)
    codes = stree.get_code()
    assert codes.count("class NetFOpt(NetF, BaseNetOpt, NoCellNetOpt, NetWithClassVarOpt):") == 1
    assert codes.count("class NoCellNetOpt(NoCellNet):") == 1
    assert codes.count("class BaseNetOpt(BaseNet, nn.Cell):") == 1
    assert codes.count("class NetWithClassVarOpt(NetWithClassVar):") == 1
    assert codes.count("var1 = Tensor(1.0)") == 0
    assert codes.count("var2 = external_func") == 0
    assert codes.count("var3 = external_func2") == 0
    assert codes.count("def external_func(x):") == 0
    new_net = stree.get_network()
    y = new_net(Tensor(1))
    assert y == y0


G_DEVICE = 'Ascend'


def g_func(x):
    return x


class BaseNet1(nn.Cell):
    def __init__(self, a):
        super().__init__()
        self.relu = nn.ReLU()
        self.a = a

    def construct(self, x):
        return x

    def add_a(self, x):
        x = x + self.a
        return x


class FatherNet(BaseNet1):
    def add_x(self, x):
        x = x + x
        return x


class MyNet(FatherNet):
    func_var = g_func
    device_var = G_DEVICE
    def __init__(self, a, b):
        super().__init__(a)
        self.relu = nn.ReLU()
        self.b = b

    def construct(self, x):
        x = self.relu(x)
        x = x + self.b
        if MyNet.device_var:
            x = self.add_a(x)
        x = MyNet.func_var(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_two_level_father_classes_with_class_var(mode):
    """
    Feature: Test Rewrite.
    Description: Test Rewrite with two level of father classes with class variables.
    Expectation: Success.
    """
    ms.set_context(mode=mode)
    net = MyNet(Tensor(2), Tensor(3))
    y0 = net(Tensor(1))
    stree = SymbolTree.create(net)
    net = stree.get_network()
    y = net(Tensor(1))
    assert y == y0


class BaseNet2(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(x)
        return x


class FatherNet2(BaseNet2, nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(x)
        return x


class MyNet2(FatherNet2):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_father_classes_has_two_bases(mode):
    """
    Feature: Test Rewrite.
    Description: Test Rewrite when father class has two bases.
    Expectation: Success.
    """
    ms.set_context(mode=mode)
    net = MyNet2()
    y0 = net(Tensor(1))
    stree = SymbolTree.create(net)
    net = stree.get_network()
    y = net(Tensor(1))
    assert y == y0
