import mindspore.nn as nn
from mindspore.nn import Cell
from mindspore.common import dtype
from mindspore.common import Tensor
import mindspore.ops.operations as P
import mindspore.ops.functional as F
from mindspore.common import Parameter
import numpy as np
from .ctrl_factory import CtrlFactory
import pytest
from tests.mark_utils import arg_mark


class CtrlWhileIfContinue(Cell):
    def __init__(self):
        super().__init__()
        self.loop = Parameter(Tensor(1, dtype.float32), name="loop")

    def construct(self, x):
        while self.loop < 5:
            self.loop += 1
            if x > 1:
                x += 1
                continue
            x += 1
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_if_continue_not_relevant_gt():
    '''
    Description: test control flow, loop is parameter in init
    if-continue variable is x, different from loop, use cmp operator >
    Expectation: No exception.
    '''
    fact = CtrlFactory(-2)
    ps_net = CtrlWhileIfContinue()
    pi_net = CtrlWhileIfContinue()
    fact.compare(ps_net, pi_net)


class CtrlWhileContinueIn(Cell):
    def __init__(self):
        super().__init__()
        self.addn = P.AddN()

    def construct(self, x):
        s = x
        t = x + 1
        tensor_list = [x, x]
        while len(tensor_list) < 4:
            tensor_list.append(x)
            a = self.addn(tensor_list)
            x += 1
            if t in tensor_list:
                continue
            s += a
        return s


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_in_continue():
    '''
    Description: test control flow while continue, use member operator in
    Expectation: No exception.
    '''
    fact = CtrlFactory(2)
    ps_net = CtrlWhileContinueIn()
    pi_net = CtrlWhileContinueIn()
    fact.compare(ps_net, pi_net)


class CtrlWhileCast(Cell):
    def __init__(self):
        super().__init__()
        self.cast = P.Cast()

    def construct(self, x, loop):
        while loop >= 3:
            loop -= 2
            if self.cast(x, dtype.bool_):
                continue
        return loop


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_continue_cast():
    '''
    Description: test control flow, use op cast
    Expectation: No exception.
    '''
    fact = CtrlFactory(1, 7)
    ps_net = CtrlWhileCast()
    pi_net = CtrlWhileCast()
    fact.compare(ps_net, pi_net)


class CtrlWhileContinueInIf(Cell):
    def __init__(self):
        super().__init__()
        self.mul = P.Mul()

    def construct(self, x):
        while x < 2:
            x += 1
            if x >= 2:
                continue
            elif x == 1:
                x = self.mul(x, x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_continue_in_if():
    '''
    Description: test control flow, while once continue
    Expectation: No exception.
    '''
    fact = CtrlFactory(-3)
    ps_net = CtrlWhileContinueInIf()
    pi_net = CtrlWhileContinueInIf()
    fact.compare(ps_net, pi_net)


class CtrlWhileContinueInElif(Cell):
    def __init__(self):
        super().__init__()
        self.mul = P.Mul()

    def construct(self, x):
        out = self.mul(x, x)
        while x < 2:
            x += 2
            if x <= 0:
                out += x
            elif x != 1:
                continue
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_continue_in_elif():
    """
    Description:
    Test Steps:
        1. create a net which contains while and if, elif in while
        2. run net forward and backward
    Expectation:
        1. the network train return ok
        2. the network forward and backward is the same as psjit
    """
    fact = CtrlFactory(-3)
    ps_net = CtrlWhileContinueInElif()
    pi_net = CtrlWhileContinueInElif()
    fact.compare(ps_net, pi_net)


class CtrlElifTwoContinue(Cell):
    def __init__(self):
        super().__init__()
        self.mul = P.Mul()

    def construct(self, x, t):
        out = t
        while x > 0:
            x -= 1
            if x < 2:
                continue
            elif x < 1:
                continue
            out = self.mul(t, out)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_elif_two_continue():
    '''
    Description: test control flow, if-elif in while, both continue
    Expectation: No exception.
    '''
    fact = CtrlFactory(3, [1, 2, 3])
    ps_net = CtrlElifTwoContinue()
    pi_net = CtrlElifTwoContinue()
    fact.compare(ps_net, pi_net)


class CtrlElifContinueOnce(Cell):
    def __init__(self):
        super().__init__()
        self.mul = P.Mul()

    def construct(self, x, t):
        out = t
        while x < 3:
            x -= 2
            if x > 4:
                x -= 1
            elif x > 6:
                x += 1
            out = self.mul(out, t)
            continue
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_once_elif_continue():
    '''
    Description: test control flow, if-elif in while, continue at last
    Expectation: No exception.
    '''
    fact = CtrlFactory(8, [2, 3, 4])
    ps_net = CtrlElifContinueOnce()
    pi_net = CtrlElifContinueOnce()
    fact.compare(ps_net, pi_net)


class CtrlIfContinueElse(Cell):
    def __init__(self):
        super().__init__()
        self.mul = P.Mul()

    def construct(self, x, y, t):
        out = t
        while x + y > 4:
            if x > 1 and y > 1:
                continue
            elif x > 4 or y > 2:
                out += t
            else:
                out = self.mul(out, t)
            x -= 2
            y += 1
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_else_continue_in_if():
    '''
    Description: test control flow, if-elif-else in while
    Expectation: No exception.
    '''
    x = 9
    y = -2
    t = np.random.rand(3, 4)
    fact = CtrlFactory(x, y, t)
    ps_net = CtrlIfContinueElse()
    pi_net = CtrlIfContinueElse()
    fact.compare(ps_net, pi_net)


class CtrlWhileElseContinueInElif(Cell):
    def __init__(self):
        super().__init__()
        self.mul = P.Mul()

    def construct(self, x, t):
        out = t
        while x < 4:
            x += 1
            if not x > 1:
                out += t
            elif 1 <= x < 2:
                continue
            else:
                out = self.mul(out, x)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_else_elif_continue():
    '''
    Description: test control flow, if-elif-else in while, continue in elif
    use and, not
    Expectation: No exception.
    '''
    x = -1
    t = np.random.rand(3, 4)
    fact = CtrlFactory(x, t)
    ps_net = CtrlWhileElseContinueInElif()
    pi_net = CtrlWhileElseContinueInElif()
    fact.compare(ps_net, pi_net)


class CtrlWhileContinueInIfElse(Cell):
    def __init__(self, a):
        super().__init__()
        self.param = Parameter(Tensor(a, dtype.float32), name="a")
        self.add = P.Add()

    def construct(self, x):
        out = x
        while self.param > -5 and x > -5:
            if self.param > 0:
                continue
            elif self.param > -3:
                out = self.add(out, x)
            else:
                continue
            self.param -= 1
            x -= 1
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_continue_in_if_else():
    '''
    Description: test control flow, if-elif-else in while
    continue in if else, param as condition
    Expectation: No exception.
    '''
    a = -7
    x = -7
    fact = CtrlFactory(x)
    ps_net = CtrlWhileContinueInIfElse(a)
    pi_net = CtrlWhileContinueInIfElse(a)
    fact.compare(ps_net, pi_net)


class CtrlWhileContinueInElifElse(Cell):
    def __init__(self, t):
        super().__init__()
        self.a = Parameter(Tensor(t, dtype.float32), name="t")
        self.mul = P.Mul()

    def construct(self, x):
        while x > 5:
            if x > self.a:
                x -= 2
            elif x == self.a:
                continue
            else:
                continue
            x -= 1
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_continue_in_elif_else():
    '''
    Description: test control flow, if-elif-else in while
    continue in elif and else, compare with param
    Expectation: No exception.
    '''
    t = 3
    fact = CtrlFactory(7)
    ps_net = CtrlWhileContinueInElifElse(t)
    pi_net = CtrlWhileContinueInElifElse(t)
    fact.compare(ps_net, pi_net)


class CtrlWhile2ElifContinueInElif(Cell):
    def __init__(self):
        super().__init__()
        self.reduce = P.ReduceSum()
        self.max = P.ReduceMax()

    def construct(self, x, y):
        while y < 4:
            y += 1
            if self.reduce(x) > 2:
                x[1] -= 2
            elif self.reduce(x) > 1:
                continue
            elif self.max(x) > 2:
                y += 1
            else:
                x[0] += 1
            x = x * y
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_continue_in_elif_else2():
    '''
    Description: test control flow, if-elif-else in while
    continue in elif and else, compare with param
    Expectation: No exception.
    '''
    x = [-2, -3, 4]
    y = 2
    fact = CtrlFactory(x, y)
    ps_net = CtrlWhile2ElifContinueInElif()
    pi_net = CtrlWhile2ElifContinueInElif()
    fact.compare(ps_net, pi_net)


class CtrlWhile2ElifContinueInElse(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, t, x):
        self.add(t, t)
        while t < 20:
            t += 1
            if x.all():
                t += 4
            elif x.any():
                t += 3
            elif not x.all():
                t += 2
            else:
                continue
        return t


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_2elif_continue_in_else():
    '''
    Description: test control flow, if-2elif-else in while
    use tensor.any, tensor.all
    Expectation: No exception.
    '''
    t = 0
    x = [True, False, False]
    fact = CtrlFactory(t)
    fact.ms_input.append(Tensor(x, dtype.bool_))
    ps_net = CtrlWhile2ElifContinueInElse()
    pi_net = CtrlWhile2ElifContinueInElse()
    fact.compare(ps_net, pi_net)


class CtrlWhile2ElifBInIfElif(Cell):
    def __init__(self):
        super().__init__()
        self.cast = P.Cast()

    def construct(self, x):
        while self.cast(x, dtype.bool_):
            x -= 1
            if x < -1:
                continue
            elif x < 3:
                continue
            elif x < 9:
                x -= 1
            else:
                x -= 2
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_2elif_continue_in_ifelif():
    '''
    Description: test control flow, if-2elif-else in while
    continue in if elif, use cast to bool
    Expectation: No exception.
    '''
    x = 12
    fact = CtrlFactory(x)
    ps_net = CtrlWhile2ElifBInIfElif()
    pi_net = CtrlWhile2ElifBInIfElif()
    fact.compare(ps_net, pi_net)


class CtrlWhile2ElifContinueIfElif(Cell):
    def __init__(self):
        super().__init__()
        self.sqrt = F.sqrt
        self.square = F.square

    def construct(self, x):
        while x < 20:
            if self.sqrt(x) > 4:
                x = x + 1
                continue
            elif x > 10:
                x = x + 4
                continue
            elif self.square(x) > 4:
                x += 3
            else:
                x += 2
            x += 1
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_2elif_continue_in_if_elif_usef():
    '''
    Description: test control flow, if-2elif-else in while
    continue in if elif, use F.sqrt, F.square
    Expectation: No exception.
    '''
    x = 1
    fact = CtrlFactory(x)
    ps_net = CtrlWhile2ElifContinueIfElif()
    pi_net = CtrlWhile2ElifContinueIfElif()
    fact.compare(ps_net, pi_net)


class CtrlWhile2ElifContinueInElifElse(Cell):
    def __init__(self):
        super().__init__()
        self.print = P.Print()

    def construct(self, x):
        while x < 20:
            if x > 4:
                self.print(x)
            elif x >= 3:
                x += 1
            elif x * 2 > 4:
                continue
            else:
                continue
            x += 1
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_2elif_continue_in_elif_else():
    '''
    Description: test control flow, if-2elif-else in while
    continue in elif, else, use P.Print
    Expectation: No exception.
    '''
    x = 3
    fact = CtrlFactory(x)
    ps_net = CtrlWhile2ElifContinueInElifElse()
    pi_net = CtrlWhile2ElifContinueInElifElse()
    fact.compare(ps_net, pi_net)


class CtrlWhile2IfContinueTwo(Cell):
    def __init__(self):
        super().__init__()
        self.cell_list = nn.CellList()
        self.cell_list.append(nn.ReLU())
        self.cell_list.append(nn.Tanh())
        self.cell_list.append(nn.Sigmoid())

    def construct(self, t, x):
        out = t
        while x < 3:
            add = self.cell_list[x](t)
            out = out + add
            x += 1
            if add > 1:
                x += 1
            if add < 1:
                continue
        return out


@pytest.mark.skip(reason="ata_expected = array(4, data_me = array(2.6165862), result match error")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_2if_continue_second():
    '''
    Description: test control flow, 2if in while
    continue in second if, use cell list
    Expectation: No exception.
    '''
    x = 0
    t = 1
    fact = CtrlFactory(t)
    fact.ms_input.append(x)
    ps_net = CtrlWhile2IfContinueTwo()
    pi_net = CtrlWhile2IfContinueTwo()
    fact.compare(ps_net, pi_net)
