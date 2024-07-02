import pytest
from mindspore import Tensor
from mindspore.rewrite import SymbolTree
import mindspore.nn as nn
import mindspore as ms
from .. import BaseNet, NoCellNet, NetWithClassVar


class NetImport(BaseNet, NoCellNet, NetWithClassVar):
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


@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def run_net_with_import(mode):
    """
    Feature: Test Rewrite.
    Description: Test Rewrite with two father classes, one of them has class variables.
    Expectation: Success.
    """
    ms.set_context(mode=mode)
    net = NetImport()
    y0 = net(Tensor(1))
    stree = SymbolTree.create(net)
    codes = stree.get_code()
    assert codes.count("class NetImportOpt(NetImport, BaseNetOpt, NoCellNetOpt, NetWithClassVarOpt):") == 1
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
