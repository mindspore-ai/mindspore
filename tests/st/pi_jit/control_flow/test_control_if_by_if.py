import numpy as np
from mindspore.nn import Cell
from mindspore import Tensor
from mindspore.common.parameter import Parameter
import mindspore.ops.operations as op
from ..parse.parser_factory import ParserFactory
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ctrl_if_by_if_basic():
    """
    Feature: PIJit
    Description: create a net, with if by if
    Expectation: No exception.
    """
    class Net41(Cell):
        def __init__(self):
            super().__init__()
            self.relu = op.ReLU()
            self.sigmoid = op.Sigmoid()
            self.tanh = op.Tanh()
            self.add = op.TensorAdd()
            a = np.full((1,), 5, dtype=np.float32)
            self.a = Parameter(Tensor(a), name="a")
            b = np.full((1,), 4, dtype=np.float32)
            self.b = Parameter(Tensor(b), name="b")
            c = np.full((1,), 7, dtype=np.float32)
            self.c = Parameter(Tensor(c), name="c")

        def construct(self, x):
            if self.a > self.b:
                if self.a < self.c:
                    out = self.relu(x)
                else:
                    out = x + 1
            else:
                out = x + 2

            if self.b > self.c:
                out = x + 3
            else:
                pass
            return out

    input_np_a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    ps_net = Net41()
    pi_net = Net41()
    fact = ParserFactory(ps_net, pi_net, input_np_a)
    fact.forward_cmp()
    fact.backward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ctrl_if_by_if_with_for():
    """
    Feature: PIJit
    Description: create a net, with for in if
    Expectation: No exception.
    """
    class Net42(Cell):
        def __init__(self):
            super().__init__()
            self.relu = op.ReLU()
            self.sigmoid = op.Sigmoid()
            self.tanh = op.Tanh()
            self.add = op.TensorAdd()
            a = np.full((1,), 5, dtype=np.float32)
            self.a = Parameter(Tensor(a), name="a")
            b = np.full((1,), 4, dtype=np.float32)
            self.b = Parameter(Tensor(b), name="b")
            c = np.full((1,), 7, dtype=np.float32)
            self.c = Parameter(Tensor(c), name="c")

        def construct(self, x):
            if self.a > self.b:
                for _ in range(0, 2):
                    x = self.relu(x)
                out = x
            else:
                out = x + 2

            if self.b > self.c:
                out = x + 3
            else:
                pass
            return out

    input_np_a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    ps_net = Net42()
    pi_net = Net42()
    fact = ParserFactory(ps_net, pi_net, input_np_a)
    fact.forward_cmp()
    fact.backward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ctrl_if_by_if_second_if_match_the_false_branch_of_first_if():
    """
    Feature: PIJit
    Description: create a net, with if by if
    Expectation: No exception.
    """
    class Net44(Cell):
        def __init__(self):
            super().__init__()
            self.relu = op.ReLU()
            self.sigmoid = op.Sigmoid()
            self.tanh = op.Tanh()
            self.add = op.TensorAdd()
            a = np.full((1,), 5, dtype=np.float32)
            self.a = Parameter(Tensor(a), name="a")
            b = np.full((1,), 4, dtype=np.float32)
            self.b = Parameter(Tensor(b), name="b")

        def construct(self, x):
            if self.a > self.b:
                x = self.relu(x)
            if self.a <= self.b:
                x = self.tanh(x)
            return x

    input_np_a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    ps_net = Net44()
    pi_net = Net44()
    fact = ParserFactory(ps_net, pi_net, input_np_a)
    fact.forward_cmp()
    fact.backward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ctrl_if_by_if_combine_with_elif_else():
    """
    Feature: PIJit
    Description: create a net, with if by if and elif
    Expectation: No exception.
    """
    class Net45(Cell):
        def __init__(self):
            super().__init__()
            self.relu = op.ReLU()
            self.sigmoid = op.Sigmoid()
            self.tanh = op.Tanh()
            self.add = op.TensorAdd()
            a = np.full((1,), 5, dtype=np.float32)
            self.a = Parameter(Tensor(a), name="a")
            b = np.full((1,), 4, dtype=np.float32)
            self.b = Parameter(Tensor(b), name="b")
            c = np.full((1,), 7, dtype=np.float32)
            self.c = Parameter(Tensor(c), name="c")

        def construct(self, x):
            out = x
            if self.a > self.b:
                if self.a < self.c:
                    out = self.relu(x)
            elif self.b == self.c:
                out = self.tanh(x)
            else:
                out = self.sigmoid(x)

            if self.c <= self.b:
                out = self.add(out, out)

            return out

    input_np_a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    ps_net = Net45()
    pi_net = Net45()
    fact = ParserFactory(ps_net, pi_net, input_np_a)
    fact.forward_cmp()
    fact.backward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ctrl_if_by_if_call_func():
    """
    Feature: PIJit
    Description: create a net, with if by if
    Expectation: No exception.
    """
    class Net49(Cell):
        def __init__(self):
            super().__init__()
            self.relu = op.ReLU()
            self.sigmoid = op.Sigmoid()
            self.tanh = op.Tanh()
            self.add = op.TensorAdd()
            a = np.full((1,), 5, dtype=np.float32)
            self.a = Parameter(Tensor(a), name="a")
            b = np.full((1,), 4, dtype=np.float32)
            self.b = Parameter(Tensor(b), name="b")
            c = np.full((1,), 7, dtype=np.float32)
            self.c = Parameter(Tensor(c), name="c")

        def func1(self, x):
            x = self.relu(x)
            return x

        def func2(self, x):
            x = self.add(x, x)
            return x

        def construct(self, x):
            if self.a > self.b:
                if self.a < self.c:
                    out = self.func1(x)
                else:
                    out = self.func2(x)
            else:
                out = x + 2
            if self.b > self.c:
                out = x + 3
            else:
                pass
            return out

    input_np_a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    ps_net = Net49()
    pi_net = Net49()
    fact = ParserFactory(ps_net, pi_net, input_np_a)
    fact.forward_cmp()
    fact.backward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ctrl_if_by_if_call_func_which_include_ctrl_flow():
    """
    Feature: PIJit
    Description: create a net, with for in if
    Expectation: No exception.
    """
    class Net50(Cell):
        def __init__(self):
            super().__init__()
            self.relu = op.ReLU()
            self.sigmoid = op.Sigmoid()
            self.tanh = op.Tanh()
            self.add = op.TensorAdd()
            a = np.full((1,), 5, dtype=np.float32)
            self.a = Parameter(Tensor(a), name="a")
            b = np.full((1,), 4, dtype=np.float32)
            self.b = Parameter(Tensor(b), name="b")
            c = np.full((1,), 7, dtype=np.float32)
            self.c = Parameter(Tensor(c), name="c")

        def func1(self, x):
            if self.a > self.b:
                x = self.relu(x)
            else:
                x = x * 2
            return x

        def func2(self, x):
            while self.c < 10:
                if self.a > 3:
                    x = self.relu(x)
                    self.a -= 1
                self.c += 1
            return x

        def construct(self, x):
            if self.a > self.b:
                if self.a < self.c:
                    out = self.func1(x)
                else:
                    out = self.func2(x)
            else:
                out = x + 2
            if self.b > self.c:
                out = x + 3
            else:
                pass
            return out

    input_np_a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    ps_net = Net50()
    pi_net = Net50()
    fact = ParserFactory(ps_net, pi_net, input_np_a)
    fact.forward_cmp()
    fact.backward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ctrl_if_by_if_call_subnet():
    """
    Feature: PIJit
    Description: create a net, with for in if
    Expectation: No exception.
    """
    class SubNet(Cell):
        def __init__(self):
            super().__init__()
            self.relu = op.ReLU()

        def construct(self, x):
            x = self.relu(x)
            return x

    class Net51(Cell):
        def __init__(self):
            super().__init__()
            self.net_inside = SubNet()
            self.relu = op.ReLU()
            self.sigmoid = op.Sigmoid()
            self.add = op.TensorAdd()
            a = np.full((1,), 5, dtype=np.float32)
            self.a = Parameter(Tensor(a), name="a")
            b = np.full((1,), 4, dtype=np.float32)
            self.b = Parameter(Tensor(b), name="b")
            c = np.full((1,), 7, dtype=np.float32)
            self.c = Parameter(Tensor(c), name="c")

        def construct(self, x):
            if self.a > self.b:
                x = self.net_inside(x)
            else:
                x = self.sigmoid(x)

            if self.a < self.c:
                x = self.add(x, 0)

            return x

    input_np_a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    ps_net = Net51()
    pi_net = Net51()
    fact = ParserFactory(ps_net, pi_net, input_np_a)
    fact.forward_cmp()
    fact.backward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ctrl_if_by_if_call_subnet_which_include_ctrl_flow():
    """
    Feature: PIJit
    Description: create a net, with for in if
    Expectation: No exception.
    """
    class SubNet(Cell):
        def __init__(self):
            super().__init__()
            self.relu = op.ReLU()
            self.add = op.TensorAdd()
            a = np.full((1,), 5, dtype=np.float32)
            self.a = Parameter(Tensor(a), name="a")
            b = np.full((1,), 4, dtype=np.float32)
            self.b = Parameter(Tensor(b), name="b")

        def construct(self, x):
            if self.a > self.b:
                x = self.relu(x)
                while self.b < 6:
                    x = self.add(x, 0)
                    self.b += 1
            return x

    class Net52(Cell):
        def __init__(self):
            super().__init__()
            self.net_inside = SubNet()
            self.relu = op.ReLU()
            self.sigmoid = op.Sigmoid()
            self.add = op.TensorAdd()
            a = np.full((1,), 5, dtype=np.float32)
            self.a = Parameter(Tensor(a), name="a")
            b = np.full((1,), 4, dtype=np.float32)
            self.b = Parameter(Tensor(b), name="b")
            c = np.full((1,), 7, dtype=np.float32)
            self.c = Parameter(Tensor(c), name="c")

        def construct(self, x):
            if self.a > self.b:
                x = self.net_inside(x)
            else:
                x = self.sigmoid(x)

            if self.a > self.c:
                x = self.add(x, 0)
            else:
                x = self.relu(x)
            return x

    input_np_a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    ps_net = Net52()
    pi_net = Net52()
    fact = ParserFactory(ps_net, pi_net, input_np_a)
    fact.forward_cmp()
    fact.backward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ctrl_if_by_if_combine_with_not_or_and():
    """
    Feature: PIJit
    Description: create a net, with for in if
    Expectation: No exception.
    """
    class Net53(Cell):
        def __init__(self):
            super().__init__()
            self.relu = op.ReLU()
            self.sigmoid = op.Sigmoid()
            self.add = op.TensorAdd()
            a = np.full((1,), 5, dtype=np.float32)
            self.a = Parameter(Tensor(a), name="a")
            b = np.full((1,), 4, dtype=np.float32)
            self.b = Parameter(Tensor(b), name="b")
            c = np.full((1,), 7, dtype=np.float32)
            self.c = Parameter(Tensor(c), name="c")

        def construct(self, x):
            if self.a > self.b and self.a < self.c:
                x = self.relu(x)
            if self.b > self.c or self.a < self.b:
                x = self.add(x, x)
            if not self.a < self.c:
                x = self.sigmoid(x)
            return x

    input_np_a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    ps_net = Net53()
    pi_net = Net53()
    fact = ParserFactory(ps_net, pi_net, input_np_a)
    fact.forward_cmp()
    fact.backward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ctrl_if_by_if_combine_with_dynamic_shape():
    """
    Feature: PIJit
    Description: create a net, with for in if
    Expectation: No exception.
    """
    class Net54(Cell):
        def __init__(self):
            super().__init__()
            self.relu = op.ReLU()
            self.sigmoid = op.Sigmoid()
            self.add = op.TensorAdd()
            self.expanddims1 = op.ExpandDims()
            self.expanddims2 = op.ExpandDims()
            a = np.full((1,), 5, dtype=np.float32)
            self.a = Parameter(Tensor(a), name="a")
            b = np.full((1,), 4, dtype=np.float32)
            self.b = Parameter(Tensor(b), name="b")
            c = np.full((1,), 7, dtype=np.float32)
            self.c = Parameter(Tensor(c), name="c")

        def construct(self, x):
            if self.a > self.b:
                out = 1
            else:
                out = 2
            if self.b < self.c:
                out = self.expanddims1(x, out)
            else:
                out = self.expanddims2(x, out)
            return out

    input_np_a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    ps_net = Net54()
    pi_net = Net54()
    fact = ParserFactory(ps_net, pi_net, input_np_a)
    fact.forward_cmp()
    fact.backward_cmp()
