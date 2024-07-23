from mindspore import context, jit
from mindspore.nn import Cell
from mindspore.common import Tensor
import numpy as np
from ..share.utils import match_array
from ..share.grad import GradOfFirstInput
import mindspore.ops.operations as op
from tests.mark_utils import arg_mark


class ControlOneForAddn(Cell):
    def __init__(self, start, stop, step):
        super().__init__()
        self.addn = op.AddN()
        self.start = start
        self.stop = stop
        self.step = step

    def construct(self, input_x):
        out = input_x
        for _ in range(self.start, self.stop, self.step):
            out = self.addn([out, input_x, input_x])
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_for_range_addn():
    """
    TEST_SUMMARY:
    Description: create a net, with break continue in while
    Expectation: result match
    """
    input_shape = (214, 214, 7, 7)
    start, stop, step = 10, 25, 3
    input_np = np.random.randn(*input_shape).astype(np.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = ControlOneForAddn(start, stop, step)
    jit(fn=ControlOneForAddn.construct, mode="PSJit")(ps_net, Tensor(input_np))
    out_ps = ps_net(Tensor(input_np))
    grad_net = GradOfFirstInput(ps_net, sens_param=False)
    ps_grad = grad_net(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = ControlOneForAddn(start, stop, step)
    jit(fn=ControlOneForAddn.construct, mode="PIJit")(pi_net, Tensor(input_np))
    out_pi = pi_net(Tensor(input_np))
    grad_net = GradOfFirstInput(pi_net, sens_param=False)
    pi_grad = grad_net(Tensor(input_np))
    match_array(out_ps, out_pi, error=4)
    match_array(ps_grad, pi_grad, error=4)


class ControlOneForSplit(Cell):
    def __init__(self):
        super().__init__()
        self.split = op.Split(1, 4)
        self.addn = op.AddN()

    def construct(self, input_x):
        x = self.addn([input_x, input_x])
        sub_tensors = self.split(x)
        out = sub_tensors[0]
        for s in sub_tensors:
            out = self.addn([out, s])
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_for_split():
    """
    TEST_SUMMARY:
    Description: create a net, with break continue in while
    Expectation: result match
    """
    input_shape = (4, 4)
    input_np = np.random.randn(*input_shape).astype(np.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = ControlOneForSplit()
    jit(fn=ControlOneForSplit.construct, mode="PSJit")(ps_net, Tensor(input_np))
    out_ps = ps_net(Tensor(input_np))
    grad_net = GradOfFirstInput(ps_net, sens_param=False)
    ps_grad = grad_net(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = ControlOneForSplit()
    jit(fn=ControlOneForSplit.construct, mode="PIJit")(pi_net, Tensor(input_np))
    out_pi = pi_net(Tensor(input_np))
    grad_net = GradOfFirstInput(pi_net, sens_param=False)
    pi_grad = grad_net(Tensor(input_np))
    match_array(out_ps, out_pi, error=4)
    match_array(ps_grad, pi_grad, error=4)


class ControlOneForOneIf(Cell):
    def __init__(self):
        super().__init__()
        self.addn = op.AddN()

    def construct(self, input_x, x, y, z):
        out = input_x
        for i in [x, y]:
            if i > z:
                out = self.addn([out, out])
            else:
                out = self.addn([out, input_x])
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_for_if():
    """
    TEST_SUMMARY:
    Description: create a net, with for in list of input
    Expectation: result match
    """
    input_shape = (4, 3, 4)
    x = np.array(1, np.float32)
    y = np.array(-1, np.float32)
    z = np.array(0, np.float32)
    input_np = np.random.randn(*input_shape).astype(np.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = ControlOneForOneIf()
    jit(fn=ControlOneForOneIf.construct, mode="PSJit")(ps_net, Tensor(input_np), Tensor(x), Tensor(y), Tensor(z))
    out_ps = ps_net(Tensor(input_np), Tensor(x), Tensor(y), Tensor(z))
    grad_net = GradOfFirstInput(ps_net, sens_param=False)
    ps_grad = grad_net(Tensor(input_np), Tensor(x), Tensor(y), Tensor(z))
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = ControlOneForOneIf()
    jit(fn=ControlOneForOneIf.construct, mode="PIJit")(pi_net, Tensor(input_np), Tensor(x), Tensor(y), Tensor(z))
    out_pi = pi_net(Tensor(input_np), Tensor(x), Tensor(y), Tensor(z))
    grad_net = GradOfFirstInput(pi_net, sens_param=False)
    pi_grad = grad_net(Tensor(input_np), Tensor(x), Tensor(y), Tensor(z))
    match_array(out_ps, out_pi, error=4)
    match_array(ps_grad, pi_grad, error=4)


class ControlOneForOneFor(Cell):
    def __init__(self):
        super().__init__()
        self.addn = op.AddN()

    def construct(self, input_x):
        out = input_x
        for _ in range(5):
            for _ in range(4):
                out = self.addn([out, input_x])
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_for_in_for():
    """
    TEST_SUMMARY:
    Description: create a net, with for in for
    Expectation: result match
    """
    input_shape = (4, 3, 4)
    input_np = np.random.randn(*input_shape).astype(np.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = ControlOneForOneFor()
    jit(fn=ControlOneForOneFor.construct, mode="PSJit")(ps_net, Tensor(input_np))
    out_ps = ps_net(Tensor(input_np))
    grad_net = GradOfFirstInput(ps_net, sens_param=False)
    ps_grad = grad_net(Tensor(input_np))
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = ControlOneForOneFor()
    jit(fn=ControlOneForOneFor.construct, mode="PIJit")(pi_net, Tensor(input_np))
    out_pi = pi_net(Tensor(input_np))
    grad_net = GradOfFirstInput(pi_net, sens_param=False)
    pi_grad = grad_net(Tensor(input_np))
    match_array(out_ps, out_pi, error=4)
    match_array(ps_grad, pi_grad, error=4)


class ControlOneWhileInFor(Cell):
    def __init__(self):
        super().__init__()
        self.addn = op.AddN()

    def construct(self, input_x, x, y):
        out = input_x
        for _ in range(3):
            y = y + 1
            while x < y:
                out = self.addn([out, input_x])
                x = x + 1
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_while_in_for():
    """
    TEST_SUMMARY:
    Description: create a net, with while in for
    Expectation: result match
    """
    input_shape = (4, 3, 4)
    x = np.array(1, np.float32)
    y = np.array(4, np.float32)
    input_np = np.random.randn(*input_shape).astype(np.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = ControlOneWhileInFor()
    jit(fn=ControlOneWhileInFor.construct, mode="PSJit")(ps_net, Tensor(input_np), Tensor(x), Tensor(y))
    out_ps = ps_net(Tensor(input_np), Tensor(x), Tensor(y))
    grad_net = GradOfFirstInput(ps_net, sens_param=False)
    ps_grad = grad_net(Tensor(input_np), Tensor(x), Tensor(y))
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = ControlOneWhileInFor()
    jit(fn=ControlOneWhileInFor.construct, mode="PIJit")(pi_net, Tensor(input_np), Tensor(x), Tensor(y))
    out_pi = pi_net(Tensor(input_np), Tensor(x), Tensor(y))
    grad_net = GradOfFirstInput(pi_net, sens_param=False)
    pi_grad = grad_net(Tensor(input_np), Tensor(x), Tensor(y))
    match_array(out_ps, out_pi, error=4)
    match_array(ps_grad, pi_grad, error=4)
