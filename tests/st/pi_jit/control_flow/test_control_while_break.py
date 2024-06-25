from mindspore.nn import Cell
from mindspore.common import dtype
from mindspore.common import Tensor
import mindspore.ops.operations as P
import mindspore.ops.functional as F
from mindspore.common.parameter import Parameter
from .ctrl_factory import CtrlFactory
import numpy as np
from tests.mark_utils import arg_mark


class CtrlWhileIfBreak(Cell):
    def __init__(self):
        super().__init__()
        self.loop = Parameter(Tensor(1, dtype.float32), name="loop")

    def construct(self, x):
        while self.loop < 5:
            self.loop += 1
            if x > 1:
                x += 1
                break
            x += 1
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_if_break_not_relevant_gt():
    '''
    Description: test control flow, loop is parameter in init
    if-break variable is x, different from loop, use cmp operator >
    Expectation: No exception.
    '''
    fact = CtrlFactory(-2)
    ps_net = CtrlWhileIfBreak()
    pi_net = CtrlWhileIfBreak()
    fact.compare(ps_net, pi_net)


class CtrlWhileBreakIn(Cell):
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
                break
            s += a
        return s


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_in_break():
    '''
    Description: test control flow while break, use member operator in
    Expectation: No exception.
    '''
    fact = CtrlFactory(-2)
    ps_net = CtrlWhileBreakIn()
    pi_net = CtrlWhileBreakIn()
    fact.compare(ps_net, pi_net)


class CtrlWhileCast(Cell):
    def __init__(self):
        super().__init__()
        self.cast = P.Cast()

    def construct(self, x, loop):
        while loop >= 3:
            loop -= 2
            if self.cast(x, dtype.bool_):
                break
        return loop


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_break_cast():
    '''
    Description: test control flow, use op cast
    Expectation: No exception.
    '''
    fact = CtrlFactory(1, 7)
    ps_net = CtrlWhileCast()
    pi_net = CtrlWhileCast()
    fact.compare(ps_net, pi_net)


class CtrlOnceBreak(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        self.add(x, x)
        while x > 2:
            if x > 1:
                pass
            x = x + 1
            break
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_once_break():
    '''
    Description: test control flow, while once break
    Expectation: No exception.
    '''
    fact = CtrlFactory(-2)
    ps_net = CtrlOnceBreak()
    pi_net = CtrlOnceBreak()
    fact.compare(ps_net, pi_net)


class CtrlWhileBreakInIf(Cell):
    def __init__(self):
        super().__init__()
        self.mul = P.Mul()

    def construct(self, x):
        while x < 2:
            x += 1
            if x >= 2:
                break
            elif x == 1:
                x = self.mul(x, x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_break_in_if():
    '''
    Description: test control flow, while, if-elif, break in if
    Expectation: No exception.
    '''
    fact = CtrlFactory(-3)
    ps_net = CtrlWhileBreakInIf()
    pi_net = CtrlWhileBreakInIf()
    fact.compare(ps_net, pi_net)


class CtrlWhileBreakInElif(Cell):
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
                break
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_break_in_elif():
    '''
    Description: test control flow, if-elif in while, break in elif
    Expectation: No exception.
    '''
    fact = CtrlFactory(-3)
    ps_net = CtrlWhileBreakInElif()
    pi_net = CtrlWhileBreakInElif()
    fact.compare(ps_net, pi_net)


class CtrlElifTwoBreak(Cell):
    def __init__(self):
        super().__init__()
        self.mul = P.Mul()

    def construct(self, x, t):
        out = t
        while x > 0:
            x -= 1
            if x < 2:
                break
            elif x < 1:
                break
            out = self.mul(t, out)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_elif_two_break():
    '''
    Description: test control flow, if-elif in while, both break
    Expectation: No exception.
    '''
    fact = CtrlFactory(3, [1, 2, 3])
    ps_net = CtrlElifTwoBreak()
    pi_net = CtrlElifTwoBreak()
    fact.compare(ps_net, pi_net)


class CtrlElifBreakOnce(Cell):
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
            break
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_once_elif_break():
    '''
    Description: test control flow, if-elif in while, both break
    Expectation: No exception.
    '''
    fact = CtrlFactory(8, [2, 3, 4])
    ps_net = CtrlElifBreakOnce()
    pi_net = CtrlElifBreakOnce()
    fact.compare(ps_net, pi_net)


class CtrlIfBreakElse(Cell):
    def __init__(self):
        super().__init__()
        self.mul = P.Mul()

    def construct(self, x, y, t):
        out = t
        while x + y > 4:
            if x > 1 and y > 1:
                break
            elif x > 4 or y > 2:
                out += t
            else:
                out = self.mul(out, t)
            x -= 2
            y += 1
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_else_break_in_if():
    '''
    Description: test control flow, if-elif-else in while
    Expectation: No exception.
    '''
    x = 9
    y = -2
    t = np.random.rand(3, 4)
    fact = CtrlFactory(x, y, t)
    ps_net = CtrlIfBreakElse()
    pi_net = CtrlIfBreakElse()
    fact.compare(ps_net, pi_net)


class CtrlWhileElseBreakInElif(Cell):
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
                break
            else:
                out = self.mul(out, x)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_else_elif_break():
    '''
    Description: test control flow, if-elif-else in while, break in elif
    Expectation: No exception.
    '''
    x = -1
    t = np.random.rand(3, 4)
    fact = CtrlFactory(x, t)
    ps_net = CtrlWhileElseBreakInElif()
    pi_net = CtrlWhileElseBreakInElif()
    fact.compare(ps_net, pi_net)


class CtrlWhileBreakInIfElif(Cell):
    def __init__(self):
        super().__init__()
        self.square = P.Square()
        self.add = P.Add()

    def construct(self, x):
        while x < 5:
            x += 2
            if self.double(x) < 3:
                break
            elif self.sqr(x) < 5:
                break
            else:
                x -= 1
        return x

    def double(self, x):
        return self.add(x, x)

    def sqr(self, x):
        return self.square(x)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_break_func():
    '''
    Description: test control flow, condition func(x), if-elif break
    Expectation: No exception.
    '''
    fact = CtrlFactory(3)
    ps_net = CtrlWhileBreakInIfElif()
    pi_net = CtrlWhileBreakInIfElif()
    fact.compare(ps_net, pi_net)


class CtrlWhile2ElifBreakInElif(Cell):
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
                break
            elif self.max(x) > 2:
                y += 1
            else:
                x[0] += 1
            x = x * y
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_break_func2():
    '''
    Description: test control flow, condition func(x), if-elif break
    Expectation: No exception.
    '''
    x = [-2, -3, 4]
    y = 2
    fact = CtrlFactory(x, y)
    ps_net = CtrlWhile2ElifBreakInElif()
    pi_net = CtrlWhile2ElifBreakInElif()
    fact.compare(ps_net, pi_net)


class CtrlWhile2ElifBreakInElse(Cell):
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
                break
        return t


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_2elif_break_in_else():
    '''
    Description: test control flow, if-elif-elif-else in while
    break in else, use tensor.any(), tensor.all()
    Expectation: No exception.
    '''
    t = 0
    x = [True, False, False]
    fact = CtrlFactory(t)
    fact.ms_input.append(Tensor(x, dtype.bool_))
    ps_net = CtrlWhile2ElifBreakInElse()
    pi_net = CtrlWhile2ElifBreakInElse()
    fact.compare(ps_net, pi_net)


class CtrlWhile2ElifBInIfElif(Cell):
    def __init__(self):
        super().__init__()
        self.cast = P.Cast()

    def construct(self, x):
        while self.cast(x, dtype.bool_):
            x -= 1
            if x < -1:
                break
            elif x < 3:
                break
            elif x < 9:
                x -= 1
            else:
                x -= 2
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_2elif_break_in_ifelif():
    '''
    Description: test control flow, if-elif-elif-else in while
    break in if and elif
    Expectation: No exception.
    '''
    x = 12
    fact = CtrlFactory(x)
    ps_net = CtrlWhile2ElifBInIfElif()
    pi_net = CtrlWhile2ElifBInIfElif()
    fact.compare(ps_net, pi_net)


class CtrlWhile2ElifBreakIfElif(Cell):
    def __init__(self):
        super().__init__()
        self.sqrt = F.sqrt
        self.square = F.square

    def construct(self, x):
        while x < 20:
            if self.sqrt(x) > 4:
                break
            elif x > 10:
                break
            elif self.square(x) > 4:
                x += 3
            else:
                x += 2
            x += 1
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_2elif_break_in_if_elif_usef():
    '''
    Description: test control flow, if-elif-elif-else in while
    break in if and elif, use F.sqrt
    Expectation: No exception.
    '''
    x = 1
    fact = CtrlFactory(x)
    ps_net = CtrlWhile2ElifBreakIfElif()
    pi_net = CtrlWhile2ElifBreakIfElif()
    fact.compare(ps_net, pi_net)


class CtrlWhile2ElifBreakInIfElse(Cell):
    def __init__(self, t):
        super().__init__()
        self.assign = P.Assign()
        self.weight = Parameter(Tensor(t, dtype.float32), name="w")

    def construct(self, x):
        while x < 2:
            x += 1
            if x < -4:
                break
            elif x < -3:
                self.assign(self.weight, x)
            elif x < 0:
                x += 2
            else:
                break
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_2elif_break_in_if_else():
    '''
    Description: test control flow, if-elif-elif-else in while
    break in if and else, assign parameter
    Expectation: No exception.
    '''
    t = 4
    x = -4
    fact = CtrlFactory(x)
    ps_net = CtrlWhile2ElifBreakInIfElse(t)
    pi_net = CtrlWhile2ElifBreakInIfElse(t)
    fact.compare(ps_net, pi_net)


class CtrlWhile2ElifBreakInElifElse(Cell):
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
                break
            else:
                break
            x += 1
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_2elif_break_in_if_else2():
    '''
    Description: test control flow, if-elif-elif-else in while
    break in elif2 and else, print in if
    Expectation: No exception.
    '''
    x = 3
    fact = CtrlFactory(x)
    ps_net = CtrlWhile2ElifBreakInElifElse()
    pi_net = CtrlWhile2ElifBreakInElifElse()
    fact.compare(ps_net, pi_net)
