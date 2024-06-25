from mindspore.nn import Cell
from mindspore.common.parameter import Parameter
from mindspore.common import dtype
from mindspore.common import Tensor
import mindspore.ops.operations as P
import numpy as np
from .ctrl_factory import CtrlFactory
from tests.mark_utils import arg_mark


class CtrlWhileInWhileBC(Cell):
    def __init__(self, t):
        super().__init__()
        self.add = P.Add()
        self.mul = P.Mul()
        self.assignadd = P.AssignAdd()
        self.para = Parameter(t, name="a")

    def construct(self, x, y):
        out = self.add(y, y)
        while x < 2:
            self.assignadd(self.para, y)
            x += 1
            if x < 4:
                out = self.add(out, out)
                break
            while x + 1 > 1:
                x -= 1
                if x < 7:
                    out = self.mul(out, self.para)
                    continue
                out = self.add(out, y)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_in_while_in_if_break_continue():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with while in while, break out, continue in
        2. run the net
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as psjit
    '''
    input_np = np.random.randn(3, 2).astype(np.float32)
    x = 1
    t = input_np
    y = input_np
    fact = CtrlFactory(x, y)
    ps_net = CtrlWhileInWhileBC(t)
    pi_net = CtrlWhileInWhileBC(t)
    fact.compare(ps_net, pi_net)


class CtrlWhileInWhileCB(Cell):
    def __init__(self, t):
        super().__init__()
        self.add = P.Add()
        self.mul = P.Mul()
        self.assignadd = P.AssignAdd()
        self.para = Parameter(t, name="a")

    def construct(self, x, y):
        out = self.add(y, y)
        while x < 2:
            self.assignadd(self.para, y)
            x += 1
            if x < 4:
                out = self.add(out, out)
                continue
            while x + 1 > 1:
                x -= 1
                if x < 7:
                    out = self.mul(out, self.para)
                    break
                out = self.add(out, y)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_in_if_continue_break():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with while in while, continue out, break in
        2. run the net
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as psjit
    '''
    input_np = np.random.randn(3, 2).astype(np.float32)
    x = 1
    t = input_np
    y = input_np
    fact = CtrlFactory(x, y)
    ps_net = CtrlWhileInWhileCB(t)
    pi_net = CtrlWhileInWhileCB(t)
    fact.compare(ps_net, pi_net)


class CtrlWhileInWhileBR(Cell):
    def __init__(self, t):
        super().__init__()
        self.add = P.Add()
        self.mul = P.Mul()
        self.assignadd = P.AssignAdd()
        self.para = Parameter(t, name="a")

    def construct(self, x, y):
        out = self.add(y, y)
        while x > -4:
            x -= 3
            self.assignadd(self.para, y)
            if x < 0:
                out = self.mul(out, out)
                break
            while x > -4:
                x -= 1
                out = self.add(out, y)
                if x < -1:
                    return out
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_in_while_break_return():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with while in while, break out, return in
        2. run the net
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as psjit
    '''
    input_np = np.random.randn(3, 2).astype(np.float32)
    x = 5
    t = input_np
    y = input_np
    fact = CtrlFactory(x, y)
    ps_net = CtrlWhileInWhileBR(t)
    pi_net = CtrlWhileInWhileBR(t)
    fact.compare(ps_net, pi_net)


class CtrlWhileInWhileRB(Cell):
    def __init__(self, t):
        super().__init__()
        self.add = P.Add()
        self.mul = P.Mul()
        self.assignadd = P.AssignAdd()
        self.para = Parameter(t, name="a")

    def construct(self, x, y):
        out = self.add(y, y)
        while x > -4:
            x -= 3
            self.assignadd(self.para, y)
            if x < 0:
                out = self.mul(out, out)
                return out
            while x > -4:
                x -= 1
                out = self.add(out, y)
                if x < -1:
                    break
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_in_while_return_break():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with while in while, return out, break in
        2. run the net
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as psjit
    '''
    input_np = np.random.randn(3, 2).astype(np.float32)
    x = 5
    t = input_np
    y = input_np
    fact = CtrlFactory(x, y)
    ps_net = CtrlWhileInWhileRB(t)
    pi_net = CtrlWhileInWhileRB(t)
    fact.compare(ps_net, pi_net)


class CtrlWhileInWhileCR(Cell):
    def __init__(self, t):
        super().__init__()
        self.add = P.Add()
        self.mul = P.Mul()
        self.para = Parameter(Tensor(t, dtype.float32), name="a")

    def construct(self, x, y):
        out = self.mul(y, y)
        while x != 3:
            while x > 5:
                x += 1
                if x > 3:
                    x = x - 1
                    return out
                out = self.add(out, self.para)
            x = x + 1
            continue
        out = self.mul(out, y)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_in_while_continue_return():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with while in while, return in, continue out
        2. run the net
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as psjit
    '''
    input_np = np.random.randn(3, 2).astype(np.float32)
    x = 2
    t = 8
    y = input_np
    fact = CtrlFactory(x, y)
    ps_net = CtrlWhileInWhileCR(t)
    pi_net = CtrlWhileInWhileCR(t)
    fact.compare(ps_net, pi_net)
