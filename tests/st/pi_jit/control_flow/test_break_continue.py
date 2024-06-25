import numpy as np
from mindspore.nn import Cell
from mindspore.common import dtype as ms
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore import context, jit
from mindspore.common.parameter import Parameter
from ..share.utils import match_array
from tests.mark_utils import arg_mark

grad_all = C.GradOperation(get_all=True)

class Grad(nn.Cell):
    def __init__(self, net):
        super(Grad, self).__init__(auto_prefix=False)
        self.forward_net = net
        self.grad = C.GradOperation(get_all=True)
    @jit(mode="PIJit")
    def construct(self, *inputs):
        grads = self.grad(self.forward_net)(*inputs)
        return grads


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_while_true_break():
    """
    Feature: PIJit
    Description: Test while true with control flow.
    Expectation: No exception.
    """
    class WhileTrueBreakNet(nn.Cell):
        def __init__(self, t):
            super(WhileTrueBreakNet, self).__init__()
            self.add = P.Add()
            self.mul = P.Mul()
            self.para = Parameter(Tensor(t, ms.int32), name="a")

        @jit(mode="PIJit")
        def construct(self, x, y):
            out = self.mul(y, self.para)
            while True:
                if x == 5:
                    x = x - 3
                    continue
                if x == 2:
                    break
                out = self.add(out, out)
            return out

    context.set_context(mode=context.PYNATIVE_MODE)
    t = np.array([1]).astype(np.int32)
    y = Tensor([1], ms.int32)
    x = Tensor([5], ms.int32)
    net = WhileTrueBreakNet(t)
    grad_net = Grad(net)
    grad_out = grad_net(x, y)
    expect = (Tensor([0], ms.int32), Tensor([1], ms.int32))
    assert expect == grad_out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_while_concatenation_10_layer():
    """
    TEST_SUMMARY:
    Description: create a net, with ten serial while loop
    Expectation: result match
    """
    class Net2(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.add = P.Add()

        def construct(self, x, y, z):
            out = z
            for _ in range(10):
                while x < y:
                    out = self.add(out, out)
                    x = x + 1
                x = x - 2
            out = self.relu(out)
            return out
    x = Tensor([2], ms.float32)
    y = Tensor([4], ms.float32)
    z = Tensor(np.random.randn(4, 4, 4), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = Net2()
    jit(fn=Net2.construct, mode="PSJit")(ps_net, x, y, z)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = Net2()
    jit(fn=Net2.construct, mode="PIJit")(pi_net, x, y, z)
    match_array(ps_net(x, y, z), pi_net(x, y, z))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_while_break():
    """
    TEST_SUMMARY:
    Description: create a net, with ten serial while loop
    Expectation: result match
    """
    class Net3(Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()

        def construct(self, x, y, z):
            out = z
            while x < y:
                out = self.add(z, z)
                x = x + 1
                if x == y:
                    break
            return out
    x = Tensor([2], ms.float32)
    y = Tensor([4], ms.float32)
    z = Tensor(np.random.randn(4, 4, 4), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = Net3()
    jit(fn=Net3.construct, mode="PSJit")(ps_net, x, y, z)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = Net3()
    jit(fn=Net3.construct, mode="PIJit")(pi_net, x, y, z)
    match_array(ps_net(x, y, z), pi_net(x, y, z))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_while_nested_break():
    """
    TEST_SUMMARY:
    Description: create a net, with break in while in while
    Expectation: result match
    """
    class Net4(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.add = P.Add()

        def construct(self, x, y, z):
            out = z
            while x < y:
                while x + 1 < y:
                    out = self.add(z, z)
                    x = x + 1
                    if x == y - 1:
                        break
                x = x + 1
            out = self.relu(out)
            return out
    x = Tensor([2], ms.float32)
    y = Tensor([8], ms.float32)
    z = Tensor(np.random.randn(4, 4, 4), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = Net4()
    jit(fn=Net4.construct, mode="PSJit")(ps_net, x, y, z)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = Net4()
    jit(fn=Net4.construct, mode="PIJit")(pi_net, x, y, z)
    match_array(ps_net(x, y, z), pi_net(x, y, z))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_while_alone():
    """
    TEST_SUMMARY:
    Description: create a net, with while independent of output
    Expectation: result match
    """
    class Net5(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.add = P.Add()

        def construct(self, x, y, z):
            out = z
            a = z
            while x < y:
                a = self.add(a, a)
                x = x + 1
            out = self.relu(out)
            return out
    x = Tensor([2], ms.float32)
    y = Tensor([4], ms.float32)
    z = Tensor(np.random.randn(4, 4, 4), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = Net5()
    jit(fn=Net5.construct, mode="PSJit")(ps_net, x, y, z)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = Net5()
    jit(fn=Net5.construct, mode="PIJit")(pi_net, x, y, z)
    match_array(ps_net(x, y, z), pi_net(x, y, z))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_while_if_single_break_in_true():
    """
    TEST_SUMMARY:
    Description: create a net, with break in if(True) in while
    Expectation: result match
    """
    class Net6(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.add = P.Add()

        def construct(self, x, y, z):
            out = z
            while x < y:
                x = x + 1
                if x == y:
                    out = self.add(out, out)
                    break
            out = self.relu(out)
            return out
    x = Tensor([2], ms.float32)
    y = Tensor([4], ms.float32)
    z = Tensor(np.random.randn(4, 4, 4), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = Net6()
    jit(fn=Net6.construct, mode="PSJit")(ps_net, x, y, z)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = Net6()
    jit(fn=Net6.construct, mode="PIJit")(pi_net, x, y, z)
    match_array(ps_net(x, y, z), pi_net(x, y, z))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_while_if_single_break_in_false():
    """
    TEST_SUMMARY:
    Description: create a net, with break in if(True) in while
    Expectation: result match
    """
    class Net7(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.add = P.Add()

        def construct(self, x, y, z):
            out = z
            while x < y:
                x = x + 1
                if x < y:
                    pass
                else:
                    out = self.add(out, out)
                    if 2 * x == y:
                        break
            out = self.relu(out)
            return out
    x = Tensor([2], ms.float32)
    y = Tensor([4], ms.float32)
    z = Tensor(np.random.randn(4, 4, 4), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = Net7()
    jit(fn=Net7.construct, mode="PSJit")(ps_net, x, y, z)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = Net7()
    jit(fn=Net7.construct, mode="PIJit")(pi_net, x, y, z)
    match_array(ps_net(x, y, z), pi_net(x, y, z))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_while_multi_if_break_nested_if_001():
    """
    TEST_SUMMARY:
    Description: create a net, with break in if(True) in while
    Expectation: result match
    """
    class Net8(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.add = P.Add()

        def construct(self, x, y, z):
            out = z
            while x < y:
                x = x + 1
                if x < y:
                    if x + 2 < y:
                        x = x + 2
                        break
                    else:
                        pass
                if y > 2 * x:
                    if y > 2 * x + 1:
                        if y > 3 * x:
                            out = self.add(out, out)
                            break

            out = self.relu(out)
            return out
    x = Tensor([2], ms.float32)
    y = Tensor([8], ms.float32)
    z = Tensor(np.random.randn(4, 4, 4), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = Net8()
    jit(fn=Net8.construct, mode="PSJit")(ps_net, x, y, z)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = Net8()
    jit(fn=Net8.construct, mode="PIJit")(pi_net, x, y, z)
    match_array(ps_net(x, y, z), pi_net(x, y, z))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_while_multi_if_break_nested_if_002():
    """
    TEST_SUMMARY:
    Description: create a net, with break in if in if in while
    Expectation: result match
    """
    class CtrlWhileMultiIf(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.add = P.Add()

        def construct(self, x, y, z):
            out = z
            while x < y:
                x = x + 1
                if x < y:
                    x = x + 2
                    out = self.add(out, out)
                    if x + 2 < y:
                        x = x + 1
                    else:
                        pass
                    if x == y - 2:
                        break

                if y > 2 * x:
                    if y > 2 * x + 1:
                        out = self.add(out, out)
                        if y > 3 * x:
                            y = y - 1
                        if 3 * x == y:
                            break

            out = self.relu(out)
            return out
    x = Tensor([2], ms.float32)
    y = Tensor([20], ms.float32)
    z = Tensor(np.random.randn(4, 4, 4), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlWhileMultiIf()
    jit(fn=CtrlWhileMultiIf.construct, mode="PSJit")(ps_net, x, y, z)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlWhileMultiIf()
    jit(fn=CtrlWhileMultiIf.construct, mode="PIJit")(pi_net, x, y, z)
    match_array(ps_net(x, y, z), pi_net(x, y, z))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_while_multi_if_break_concatenation_if():
    """
    TEST_SUMMARY:
    Description: create a net, with break in 3 if in while
    Expectation: result match
    """
    class Net10(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.add = P.Add()

        def construct(self, x, y, z):
            out = z
            while x < y:
                x = x + 1
                out = self.relu(out)
                if x + 2 == y:
                    x = x + 2
                    out = self.add(out, out)
                    break

                if x + 4 == y:
                    y = y - 2
                    out = self.relu(out)
                    break

                if x == y:
                    out = self.relu(out)
                    break

            out = self.relu(out)
            return out
    x = Tensor([2], ms.float32)
    y = Tensor([10], ms.float32)
    z = Tensor(np.random.randn(4, 4, 4), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = Net10()
    jit(fn=Net10.construct, mode="PSJit")(ps_net, x, y, z)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = Net10()
    jit(fn=Net10.construct, mode="PIJit")(pi_net, x, y, z)
    match_array(ps_net(x, y, z), pi_net(x, y, z))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_multi_while_nested_if_break_001():
    """
    TEST_SUMMARY:
    Description: create a net, with break in if in while in while
    Expectation: result match
    """
    class Net11(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.add = P.Add()

        def construct(self, x, y, z):
            out = z
            while x < y:
                while 3 * x < y:
                    if 2 * x == y:
                        out = self.add(out, out)
                        break
                    out = self.relu(out)
                    y = y - 1
                x = x + 1
            out = self.relu(out)
            return out
    x = Tensor([2], ms.float32)
    y = Tensor([10], ms.float32)
    z = Tensor(np.random.randn(4, 4, 4), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = Net11()
    jit(fn=Net11.construct, mode="PSJit")(ps_net, x, y, z)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = Net11()
    jit(fn=Net11.construct, mode="PIJit")(pi_net, x, y, z)
    match_array(ps_net(x, y, z), pi_net(x, y, z))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_multi_while_nested_if_break_002():
    """
    TEST_SUMMARY:
    Description: create a net, with break in second if in while in while
    Expectation: result match
    """
    class Net12(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.add = P.Add()

        def construct(self, x, y, z):
            out = z
            while x < y:
                while 3 * x < y:
                    out = self.relu(out)
                    if 2 * x == y:
                        out = self.add(out, out)
                    if x + 6 == y:
                        break
                    y = y - 1
                x = x + 1
            out = self.relu(out)
            return out
    x = Tensor([2], ms.float32)
    y = Tensor([10], ms.float32)
    z = Tensor(np.random.randn(4, 4, 4), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = Net12()
    jit(fn=Net12.construct, mode="PSJit")(ps_net, x, y, z)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = Net12()
    jit(fn=Net12.construct, mode="PIJit")(pi_net, x, y, z)
    match_array(ps_net(x, y, z), pi_net(x, y, z))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_multi_while_nested_if_break_003():
    """
    TEST_SUMMARY:
    Description: create a net, with break in both if in while in while
    Expectation: result match
    """
    class Net13(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.add = P.Add()

        def construct(self, x, y, z):
            out = z
            while x < y:
                while 3 * x < y:
                    if 2 * x == y:
                        out = self.add(out, out)
                        break
                    x = x + 1
                    if x + 6 == y:
                        break
                x = x + 1
            out = self.relu(out)
            return out
    x = Tensor([2], ms.float32)
    y = Tensor([10], ms.float32)
    z = Tensor(np.random.randn(4, 4, 4), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = Net13()
    jit(fn=Net13.construct, mode="PSJit")(ps_net, x, y, z)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = Net13()
    jit(fn=Net13.construct, mode="PIJit")(pi_net, x, y, z)
    match_array(ps_net(x, y, z), pi_net(x, y, z))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_multi_while_concatenation_if_break():
    """
    TEST_SUMMARY:
    Description: create a net, with break in all 3 if in while
    Expectation: result match
    """
    class Net14(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.add = P.Add()

        def construct(self, x, y, z):
            out = z
            while x < y:
                if 2 * x < y:
                    out = self.add(out, out)
                    break

                if 3 * x < y:
                    out = self.relu(out)
                    break

                if x == y:
                    out = self.relu(out)
                    break
                x = x + 1

            out = self.relu(out)
            return out
    x = Tensor([2], ms.float32)
    y = Tensor([10], ms.float32)
    z = Tensor(np.random.randn(4, 4, 4), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = Net14()
    jit(fn=Net14.construct, mode="PSJit")(ps_net, x, y, z)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = Net14()
    jit(fn=Net14.construct, mode="PIJit")(pi_net, x, y, z)
    match_array(ps_net(x, y, z), pi_net(x, y, z))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_for_if_break_in_true():
    """
    TEST_SUMMARY:
    Description: create a net, with break in if(True) in for
    Expectation: result match
    """
    class Net15(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.add = P.Add()

        def construct(self, x, y, z):
            out = z
            for _ in range(5):
                if 2 * x < y:
                    out = self.add(out, out)
                    if x + 6 == y:
                        break
                else:
                    out = self.relu(out)
                x = x + 1
            out = self.relu(out)
            return out
    x = Tensor([2], ms.float32)
    y = Tensor([8], ms.float32)
    z = Tensor(np.random.randn(4, 4, 4), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = Net15()
    jit(fn=Net15.construct, mode="PSJit")(ps_net, x, y, z)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = Net15()
    jit(fn=Net15.construct, mode="PIJit")(pi_net, x, y, z)
    match_array(ps_net(x, y, z), pi_net(x, y, z))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_for_if_break_in_false():
    """
    TEST_SUMMARY:
    Description: create a net, with break in if(False) in for
    Expectation: result match
    """
    class Net16(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.add = P.Add()

        def construct(self, x, y, z):
            out = z
            for _ in range(5):
                if 3 * x < y:
                    out = self.add(out, out)
                else:
                    out = self.relu(out)
                    if x + 6 == y:
                        break
            out = self.relu(out)
            return out
    x = Tensor([2], ms.float32)
    y = Tensor([8], ms.float32)
    z = Tensor(np.random.randn(4, 4, 4), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = Net16()
    jit(fn=Net16.construct, mode="PSJit")(ps_net, x, y, z)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = Net16()
    jit(fn=Net16.construct, mode="PIJit")(pi_net, x, y, z)
    match_array(ps_net(x, y, z), pi_net(x, y, z))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_for_multi_if_break_nested_001():
    """
    TEST_SUMMARY:
    Description: create a net, with break in if(third) in while
    Expectation: result match
    """
    class Net17(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.add = P.Add()

        def construct(self, x, y, z):
            out = z
            for _ in range(5):
                if 2 * x < y:
                    out = self.relu(out)
                    if 3 * x < y:
                        out = self.add(out, out)
                        if 3 * x + 1 == y:
                            break
                    x = x + 1
            out = self.relu(out)
            return out
    x = Tensor([2], ms.float32)
    y = Tensor([10], ms.float32)
    z = Tensor(np.random.randn(4, 4, 4), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = Net17()
    jit(fn=Net17.construct, mode="PSJit")(ps_net, x, y, z)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = Net17()
    jit(fn=Net17.construct, mode="PIJit")(pi_net, x, y, z)
    match_array(ps_net(x, y, z), pi_net(x, y, z))
