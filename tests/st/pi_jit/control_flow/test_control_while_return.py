from mindspore.nn import Cell
from mindspore.common.parameter import Parameter
from mindspore.common import dtype
from mindspore.common import Tensor
import mindspore.ops.operations as P
import mindspore.ops.functional as F
import numpy as np
from .ctrl_factory import CtrlFactory
from tests.mark_utils import arg_mark


class CtrlWhileIfReturn(Cell):
    def __init__(self):
        super().__init__()
        self.loop = Parameter(Tensor(1, dtype.float32), name="loop")

    def construct(self, x):
        while self.loop < 5:
            self.loop += 1
            if x > 1:
                x += 1
                return x
            x += 1
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_if_return_not_relevant_gt():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with return in while, condition is parameter
        2. no parameter
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as psjit
    '''
    fact = CtrlFactory(-2)
    ps_net = CtrlWhileIfReturn()
    pi_net = CtrlWhileIfReturn()
    fact.compare(ps_net, pi_net)


class CtrlWhileReturnIn(Cell):
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
                return s
            s += a
        return s


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_in_return():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with return in while, use member op in
        2. no parameter
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as psjit
    '''
    fact = CtrlFactory(2)
    ps_net = CtrlWhileReturnIn()
    pi_net = CtrlWhileReturnIn()
    fact.compare(ps_net, pi_net)


class CtrlWhileCast(Cell):
    def __init__(self):
        super().__init__()
        self.cast = P.Cast()

    def construct(self, x, loop):
        while loop >= 3:
            loop -= 2
            if self.cast(x, dtype.bool_):
                return loop
        return loop


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_return_cast():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with return in while, use op cast
        2. no parameter
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as psjit
    '''
    fact = CtrlFactory(1, 7)
    ps_net = CtrlWhileCast()
    pi_net = CtrlWhileCast()
    fact.compare(ps_net, pi_net)


class CtrlOnceReturn(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        self.add(x, x)
        while x > 2:
            if x > 1:
                pass
            x = x + 1
            return x
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_once_return():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with return in while, once out
        2. no parameter
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as psjit
    '''
    fact = CtrlFactory(-2)
    ps_net = CtrlOnceReturn()
    pi_net = CtrlOnceReturn()
    fact.compare(ps_net, pi_net)


class CtrlWhileReturnInIf(Cell):
    def __init__(self):
        super().__init__()
        self.mul = P.Mul()

    def construct(self, x):
        while x < 2:
            x += 1
            if x >= 2:
                return x
            elif x == 1:
                x = self.mul(x, x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_return_in_if():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with if-elif in while, return in if
        2. no parameter
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as psjit
    '''
    fact = CtrlFactory(-3)
    ps_net = CtrlWhileReturnInIf()
    pi_net = CtrlWhileReturnInIf()
    fact.compare(ps_net, pi_net)


class CtrlWhileReturnInElif(Cell):
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
                return out
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_return_in_elif():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with if-elif in while, return in elif
        2. no parameter
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as psjit
    '''
    fact = CtrlFactory(-3)
    ps_net = CtrlWhileReturnInElif()
    pi_net = CtrlWhileReturnInElif()
    fact.compare(ps_net, pi_net)


class CtrlElifReturnOnce(Cell):
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
            return out
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_once_elif_return():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with if-elif in while, return at last
        2. no parameter
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as psjit
    '''
    fact = CtrlFactory(8, [2, 3, 4])
    ps_net = CtrlElifReturnOnce()
    pi_net = CtrlElifReturnOnce()
    fact.compare(ps_net, pi_net)


class CtrlIfReturnElse(Cell):
    def __init__(self):
        super().__init__()
        self.mul = P.Mul()

    def construct(self, x, y, t):
        out = t
        while x + y > 4:
            if x > 1 and y > 1:
                return out
            elif x > 4 or y > 2:
                out += t
            else:
                out = self.mul(out, t)
            x -= 2
            y += 1
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_else_return_in_if():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with if-elif-else in while, return in if
        2. no parameter
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as psjit
    '''
    x = 9
    y = -2
    t = np.random.rand(3, 4)
    fact = CtrlFactory(x, y, t)
    ps_net = CtrlIfReturnElse()
    pi_net = CtrlIfReturnElse()
    fact.compare(ps_net, pi_net)


class CtrlWhileElseReturnInElif(Cell):
    def __init__(self):
        super().__init__()
        self.mul = P.Mul()

    def construct(self, x, t):
        out = t
        while x < 4:
            x += 1
            if not x > 1:
                out += t
            elif x >= 1 and x < 2:
                return out
            else:
                out = self.mul(out, x)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_else_elif_return():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with if-elif-else in while, return in elif, use and not
        2. no parameter
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as psjit
    use and not
    '''
    x = -1
    t = np.random.rand(3, 4)
    fact = CtrlFactory(x, t)
    ps_net = CtrlWhileElseReturnInElif()
    pi_net = CtrlWhileElseReturnInElif()
    fact.compare(ps_net, pi_net)


class CtrlWhileReturnInIfElif(Cell):
    def __init__(self):
        super().__init__()
        self.square = P.Square()
        self.add = P.Add()

    def construct(self, x):
        while x < 5:
            x += 2
            if self.double(x) < 3:
                return x
            elif self.sqr(x) < 5:
                return x
            else:
                x -= 1
        return x

    def double(self, x):
        return self.add(x, x)

    def sqr(self, x):
        return self.square(x)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_return_func():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with if-elif-else in while, return in if elif, condition of func
        2. no parameter
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as psjit
    '''
    fact = CtrlFactory(3)
    ps_net = CtrlWhileReturnInIfElif()
    pi_net = CtrlWhileReturnInIfElif()
    fact.compare(ps_net, pi_net)


class CtrlWhileReturnInIfElse(Cell):
    def __init__(self, a):
        super().__init__()
        self.param = Parameter(Tensor(a, dtype.float32), name="a")
        self.add = P.Add()

    def construct(self, x):
        out = x
        while self.param > -5 and x > -5:
            if self.param > 0:
                return out
            elif self.param > -3:
                out = self.add(out, x)
            else:
                return out
            self.param -= 1
            x -= 1
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_return_in_if_else():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with if-elif-else in while, return in if else
        2. parameter as condition
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as torch
    '''
    a = -7
    x = -7
    fact = CtrlFactory(x)
    ps_net = CtrlWhileReturnInIfElse(a)
    pi_net = CtrlWhileReturnInIfElse(a)
    fact.compare(ps_net, pi_net)


class CtrlWhileReturnInElifElse(Cell):
    def __init__(self, tensor):
        super().__init__()
        self.a = Parameter(tensor, name="t")
        self.mul = P.Mul()

    def construct(self, x):
        while x > 5:
            if x > self.a:
                x -= 2
            elif x == self.a:
                return x
            else:
                return x
            x -= 1
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_return_in_elif_else():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with if-elif-else in while, return in elif else
        2. parameter as condition
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as torch
    '''
    t = Tensor(3, dtype.float32)
    fact = CtrlFactory(7)
    ps_net = CtrlWhileReturnInElifElse(t)
    pi_net = CtrlWhileReturnInElifElse(t)
    fact.compare(ps_net, pi_net)


class CtrlWhile2ElifReturnInElif(Cell):
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
                return x
            elif self.max(x) > 2:
                y += 1
            else:
                x[0] += 1
            x = x * y
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_2elif_return_in_elif():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with if-2elif-else in while, return in elif
        2. use sum
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as torch
    '''
    x = [-2, -3, 4]
    y = 2
    fact = CtrlFactory(x, y)
    ps_net = CtrlWhile2ElifReturnInElif()
    pi_net = CtrlWhile2ElifReturnInElif()
    fact.compare(ps_net, pi_net)


class CtrlWhile2ElifReturnInElse(Cell):
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
                return t
        return t


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_2elif_return_in_else():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with iwhile 2elif, return in else
        2. use sum
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as torch
    '''
    t = 0
    x = [True, False, False]
    fact = CtrlFactory(t)
    fact.ms_input.append(Tensor(x, dtype.bool_))
    ps_net = CtrlWhile2ElifReturnInElse()
    pi_net = CtrlWhile2ElifReturnInElse()
    fact.compare(ps_net, pi_net)


class CtrlWhile2ElifBInIfElif(Cell):
    def __init__(self):
        super().__init__()
        self.cast = P.Cast()

    def construct(self, x):
        while self.cast(x, dtype.bool_):
            x -= 1
            if x < -1:
                return x
            elif x < 3:
                return x
            elif x < 9:
                x -= 1
            else:
                x -= 2
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_2elif_return_in_ifelif():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with if-2elif-else in while, return in if elif
        2. parameter as condition
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as torch
    '''
    x = 12
    fact = CtrlFactory(x)
    ps_net = CtrlWhile2ElifBInIfElif()
    pi_net = CtrlWhile2ElifBInIfElif()
    fact.compare(ps_net, pi_net)


class CtrlWhile2ElifReturnIfElif(Cell):
    def __init__(self):
        super().__init__()
        self.sqrt = F.sqrt
        self.square = F.square

    def construct(self, x):
        while x < 20:
            if self.sqrt(x) > 4:
                x = x + 1
                return x
            elif x > 10:
                x = x + 4
                return x
            elif self.square(x) > 4:
                x += 3
            else:
                x += 2
            x += 1
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_2elif_return_in_if_elif_usef():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with if-2elif-else in while, return in if elif
        2. use F
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as torch
    '''
    x = 1
    fact = CtrlFactory(x)
    ps_net = CtrlWhile2ElifReturnIfElif()
    pi_net = CtrlWhile2ElifReturnIfElif()
    fact.compare(ps_net, pi_net)


class CtrlWhile2ElifReturnInIfElse(Cell):
    def __init__(self, t):
        super().__init__()
        self.assign = P.Assign()
        self.weight = Parameter(Tensor(t, dtype.float32), name="w")

    def construct(self, x):
        while x < 2:
            x += 1
            if x < -4:
                return x
            elif x < -3:
                self.assign(self.weight, x)
            elif x < 0:
                x += 2
            else:
                return x
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_2elif_return_in_if_else():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with if-2elif-else in while, return in if else
        2. assign parameter
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as torch
    '''
    t = 4
    x = -4
    fact = CtrlFactory(x)
    ps_net = CtrlWhile2ElifReturnInIfElse(t)
    pi_net = CtrlWhile2ElifReturnInIfElse(t)
    fact.compare(ps_net, pi_net)


class CtrlWhile2ElifReturnInElifElse(Cell):
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
                return x
            else:
                return x
            x += 1
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_2elif_return_in_elif_else():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net, with if-2elif-else in while, return in elif else
        2. use print
    Expectation:
        1. the network run ok
        2. the network forward and backward result is the same as torch
    '''
    x = 3
    fact = CtrlFactory(x)
    ps_net = CtrlWhile2ElifReturnInElifElse()
    pi_net = CtrlWhile2ElifReturnInElifElse()
    fact.compare(ps_net, pi_net)
