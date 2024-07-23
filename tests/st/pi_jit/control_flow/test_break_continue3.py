from mindspore import context, jit
from mindspore.nn import Cell
import numpy as np
import pytest
from mindspore.common import Tensor
from mindspore.common import dtype as ms
from mindspore.common import Parameter
import mindspore.ops.operations as P
from ..share.utils import match_array
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_for_multi_if_break_nested_002():
    """
    TEST_SUMMARY:
    Description: create a net, with break in 3 nested if in for
    Expectation: result match
    """
    class Net18(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.add = P.Add()

        def construct(self, x, y, z):
            out = z
            for _ in range(5):
                if 2 * x < y:
                    if 3 * x < y:
                        out = self.add(out, out)
                        x = x + 1
                    out = self.relu(out)
                if x + 6 == y:
                    break
            out = self.relu(out)
            return out
    x = Tensor([2], ms.float32)
    y = Tensor([10], ms.float32)
    z = Tensor(np.random.randn(4, 4, 4), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = Net18()
    jit(fn=Net18.construct, mode="PSJit")(ps_net, x, y, z)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = Net18()
    jit(fn=Net18.construct, mode="PIJit")(pi_net, x, y, z)
    match_array(ps_net(x, y, z), pi_net(x, y, z))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_for_multi_if_break_nested_003():
    """
    TEST_SUMMARY:
    Description: create a net, with break in 3 nested if in for
    Expectation: result match
    """
    class Net19(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.add = P.Add()

        def construct(self, x, y, z):
            out = z
            for _ in range(5):
                if 2 * x < y:
                    if 3 * x < y:
                        out = self.add(out, out)
                        x = x + 1
                        if 2 * x + 1 == y:
                            break
                    out = self.relu(out)
                    if x + 6 == y:
                        break
            out = self.relu(out)
            return out
    x = Tensor([2], ms.float32)
    y = Tensor([10], ms.float32)
    z = Tensor(np.random.randn(4, 4, 4), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = Net19()
    jit(fn=Net19.construct, mode="PSJit")(ps_net, x, y, z)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = Net19()
    jit(fn=Net19.construct, mode="PIJit")(pi_net, x, y, z)
    match_array(ps_net(x, y, z), pi_net(x, y, z))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_for_multi_if_break_concatenation():
    """
    TEST_SUMMARY:
    Description: create a net, with break in 3 nested if in for
    Expectation: result match
    """
    class Net20(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.add = P.Add()

        def construct(self, x, y, z):
            out = z
            for _ in range(2):
                for _ in range(3):
                    if 2 * x < y:
                        out = self.add(out, out)
                        x = x + 1
                        if x + 6 == y:
                            break

                for _ in range(2):
                    if 2 * x < y:
                        out = self.relu(out)
                        y = y - 1
                        if x + 5 == y:
                            break

            out = self.relu(out)
            return out
    x = Tensor([2], ms.float32)
    y = Tensor([10], ms.float32)
    z = Tensor(np.random.randn(4, 4, 4), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = Net20()
    jit(fn=Net20.construct, mode="PSJit")(ps_net, x, y, z)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = Net20()
    jit(fn=Net20.construct, mode="PIJit")(pi_net, x, y, z)
    match_array(ps_net(x, y, z), pi_net(x, y, z))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_for_multi_if_continue_concatenation():
    """
    TEST_SUMMARY:
    Description: create a net, with break in 3 nested if in for
    Expectation: result match
    """
    class Net21(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.add = P.Add()

        def construct(self, x, y, z):
            out = z
            for _ in range(2):
                for _ in range(3):
                    if 2 * x < y:
                        out = self.add(out, out)
                        x = x + 1
                    else:
                        continue

                for _ in range(2):
                    if 3 * x < y:
                        out = self.relu(out)
                        y = y - 1
                    else:
                        continue

            out = self.relu(out)
            return out
    x = Tensor([2], ms.float32)
    y = Tensor([10], ms.float32)
    z = Tensor(np.random.randn(4, 4, 4), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = Net21()
    jit(fn=Net21.construct, mode="PSJit")(ps_net, x, y, z)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = Net21()
    jit(fn=Net21.construct, mode="PIJit")(pi_net, x, y, z)
    match_array(ps_net(x, y, z), pi_net(x, y, z))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_while_for_if_combine_break_continue_001():
    """
    TEST_SUMMARY:
    Description: create a net, with break in 3 nested if in for
    Expectation: result match
    """
    class Net22(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.add = P.Add()

        def construct(self, x, y, z):
            out = z
            for _ in range(2):
                while 2 * x < y:
                    if 2 * x < y:
                        out = self.add(out, out)

                    if 3 * x < y:
                        x = x + 2
                    else:
                        break
                    x = x + 1

                for _ in range(2):
                    if x + 5 < y:
                        out = self.relu(out)
                    else:
                        continue

            out = self.relu(out)
            return out
    x = Tensor([2], ms.float32)
    y = Tensor([16], ms.float32)
    z = Tensor(np.random.randn(4, 4, 4), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = Net22()
    jit(fn=Net22.construct, mode="PSJit")(ps_net, x, y, z)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = Net22()
    jit(fn=Net22.construct, mode="PIJit")(pi_net, x, y, z)
    match_array(ps_net(x, y, z), pi_net(x, y, z))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_while_for_if_combine_break_continue_002():
    """
    TEST_SUMMARY:
    Description: create a net, with break in 3 nested if in for
    Expectation: result match
    """
    class Net23(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.add = P.Add()

        def construct(self, x, y, z):
            out = z
            for _ in range(2):
                for _ in range(2):
                    if 4 * x < y:
                        out = self.relu(out)
                    else:
                        continue

                while x < y:
                    if 2 * x < y:
                        out = self.add(out, out)
                        x = x + 2
                    if 3 * x < y:
                        x = x + 1
                    else:
                        break
                    x = x + 2

            out = self.relu(out)
            return out
    x = Tensor([2], ms.float32)
    y = Tensor([20], ms.float32)
    z = Tensor(np.random.randn(4, 4, 4), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = Net23()
    jit(fn=Net23.construct, mode="PSJit")(ps_net, x, y, z)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = Net23()
    jit(fn=Net23.construct, mode="PIJit")(pi_net, x, y, z)
    match_array(ps_net(x, y, z), pi_net(x, y, z))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_while_for_if_combine_break_continue_003():
    """
    TEST_SUMMARY:
    Description: create a net, with break in 3 nested if in for
    Expectation: result match
    """
    class Net24(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.add = P.Add()

        def construct(self, x, y, z):
            out = z
            for _ in range(2):
                for _ in range(2):
                    if 3 * x < y:
                        break
                    else:
                        y = y - 1

                while x < y:
                    if 2 * x < y:
                        out = self.add(out, out)
                        x = x + 2
                    if 3 * x < y:
                        x = x + 1
                    else:
                        break
                    x = x + 2

            out = self.relu(out)
            return out
    x = Tensor([2], ms.float32)
    y = Tensor([10], ms.float32)
    z = Tensor(np.random.randn(4, 4, 4), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = Net24()
    jit(fn=Net24.construct, mode="PSJit")(ps_net, x, y, z)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = Net24()
    jit(fn=Net24.construct, mode="PIJit")(pi_net, x, y, z)
    match_array(ps_net(x, y, z), pi_net(x, y, z))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_while_for_if_combine_break_continue_004():
    """
    TEST_SUMMARY:
    Description: create a net, with break in 3 nested if in for
    Expectation: result match
    """
    class Net25(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.add = P.Add()

        def construct(self, x, y, z):
            out = z
            if x < y:
                while 2 * x < y:
                    for _ in range(2):
                        if 3 * x < y:
                            out = self.add(out, out)
                        else:
                            continue
                    x = x + 2
                    if 2 * x == y:
                        break

                while x + 2 < y:
                    if x + 5 < y:
                        out = self.relu(out)
                        x = x + 1
                    x = x + 1
                    if x + 2 == y:
                        break

            out = self.relu(out)
            return out
    x = Tensor([2], ms.float32)
    y = Tensor([20], ms.float32)
    z = Tensor(np.random.randn(4, 4, 4), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = Net25()
    jit(fn=Net25.construct, mode="PSJit")(ps_net, x, y, z)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = Net25()
    jit(fn=Net25.construct, mode="PIJit")(pi_net, x, y, z)
    match_array(ps_net(x, y, z), pi_net(x, y, z))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_while_for_if_combine_break_continue_005():
    """
    TEST_SUMMARY:
    Description: create a net, with break in 3 nested if in for
    Expectation: result match
    """
    class Net26(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.add = P.Add()

        def construct(self, x, y, z):
            out = z
            while x < y:
                for _ in range(2):
                    if 2 * x < y:
                        out = self.add(out, out)
                        if 2 * x + 10 == y:
                            break

                if 3 * x < y:
                    for _ in range(2):
                        if 2 * x < y:
                            out = self.relu(out)
                        else:
                            continue
                else:
                    while 2 * x < y:
                        for _ in range(2):
                            out = self.relu(out)
                            if x + 9 == y:
                                break
                        y = y - 1
                        continue
                x = x + 2
            out = self.relu(out)
            return out
    x = Tensor([2], ms.float32)
    y = Tensor([20], ms.float32)
    z = Tensor(np.random.randn(4, 4, 4), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = Net26()
    jit(fn=Net26.construct, mode="PSJit")(ps_net, x, y, z)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = Net26()
    jit(fn=Net26.construct, mode="PIJit")(pi_net, x, y, z)
    match_array(ps_net(x, y, z), pi_net(x, y, z))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_while_for_if_break_return_001():
    """
    TEST_SUMMARY:
    Description: create a net, with break in 3 nested if in for
    Expectation: result match
    """
    class Net27(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.add = P.Add()

        def construct(self, x, y, z):
            out = z
            while x < y:
                if 2 * x < y:
                    out = self.relu(out)
                    x = x + 1
                elif 3 * x < y:
                    out = self.add(out, out)
                    x = x - 1
                else:
                    out = self.relu(out)
                if 2 * x == y:
                    break
            out = self.relu(out)
            return out
    x = Tensor([2], ms.float32)
    y = Tensor([20], ms.float32)
    z = Tensor(np.random.randn(4, 4, 4), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = Net27()
    jit(fn=Net27.construct, mode="PSJit")(ps_net, x, y, z)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = Net27()
    jit(fn=Net27.construct, mode="PIJit")(pi_net, x, y, z)
    match_array(ps_net(x, y, z), pi_net(x, y, z))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_while_for_if_break_return_002():
    """
    TEST_SUMMARY:
    Description: create a net, with break in 3 nested if in for
    Expectation: result match
    """
    class Net28(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.add = P.Add()

        def construct(self, x, y, z):
            out = z
            while x < y:
                if 2 * x == y:
                    continue
                elif 3 * x < y:
                    out = self.add(out, out)
                    x = x + 1
                else:
                    out = self.relu(out)
                    x = x - 1
                if 3 * x - 1 == y:
                    break

            out = self.relu(out)
            return out
    x = Tensor([2], ms.float32)
    y = Tensor([20], ms.float32)
    z = Tensor(np.random.randn(4, 4, 4), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = Net28()
    jit(fn=Net28.construct, mode="PSJit")(ps_net, x, y, z)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = Net28()
    jit(fn=Net28.construct, mode="PIJit")(pi_net, x, y, z)
    match_array(ps_net(x, y, z), pi_net(x, y, z))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_while_for_if_condition_define_in_init():
    """
    TEST_SUMMARY:
    Description: create a net, with break in 3 nested if in for
    Expectation: result match
    """
    class Net29(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.add = P.Add()
            self.x = 2
            self.y = 20

        def construct(self, z):
            out = z
            while self.x < self.y:
                if 2 * self.x < self.y:
                    out = self.add(out, out)
                if self.x + 18 == self.y:
                    break
            out = self.relu(out)
            return out
    z = Tensor(np.random.randn(4, 4, 4), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = Net29()
    jit(fn=Net29.construct, mode="PSJit")(ps_net, z)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = Net29()
    jit(fn=Net29.construct, mode="PIJit")(pi_net, z)
    match_array(ps_net(z), pi_net(z))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_while_for_if_break_parameter():
    """
    TEST_SUMMARY:
    Description: create a net, with break in 3 nested if in for
    Expectation: result match
    """
    class Net30(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.add = P.Add()
            add_np = np.full((4, 4, 4), 0.5, dtype=np.float32)
            self.add_weight = Parameter(Tensor(add_np), name="add_weight")

        def construct(self, x, y, z):
            out = z
            while x < y:
                if 2 * x < y:
                    out = self.add(out, self.add_weight)
                elif 3 * x < y:
                    out = self.relu(out)
                    x = x + 1
                else:
                    break
                x = x + 1

            out = self.relu(out)
            return out
    x = Tensor([2], ms.float32)
    y = Tensor([20], ms.float32)
    z = Tensor(np.random.randn(4, 4, 4), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = Net30()
    jit(fn=Net30.construct, mode="PSJit")(ps_net, x, y, z)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = Net30()
    jit(fn=Net30.construct, mode="PIJit")(pi_net, x, y, z)
    match_array(ps_net(x, y, z), pi_net(x, y, z))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_while_for_if_break_plus_continue():
    """
    TEST_SUMMARY:
    Description: create a net, with break in 3 nested if in for
    Expectation: result match
    """
    class Net31(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.add = P.Add()

        def construct(self, x, y, z):
            out = z
            while x < y:
                if 3 * x < y:
                    out = self.add(out, out)
                    if 3 * x == y:
                        break
                    if x + 20 == y:
                        continue
                elif 2 * x < y:
                    out = self.relu(out)
                    x = x + 1
                else:
                    break
                x = x + 1

            out = self.relu(out)
            return out
    x = Tensor([2], ms.float32)
    y = Tensor([10], ms.float32)
    z = Tensor(np.random.randn(4, 4, 4), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = Net31()
    jit(fn=Net31.construct, mode="PSJit")(ps_net, x, y, z)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = Net31()
    jit(fn=Net31.construct, mode="PIJit")(pi_net, x, y, z)
    match_array(ps_net(x, y, z), pi_net(x, y, z))
