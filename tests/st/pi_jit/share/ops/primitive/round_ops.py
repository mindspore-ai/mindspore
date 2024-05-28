import numpy as np
import mindspore.ops.operations as op
from mindspore import Tensor, jit, context
from mindspore.nn import Cell
from ...utils import allclose_nparray
from ...grad import GradOfFirstInput


class Round(Cell):
    def __init__(self):
        super().__init__()
        self.op = op.Round()

    def construct(self, input_data):
        return self.op(input_data)


class RoundFactory():
    def __init__(self, input_shape, dtype=np.float16):
        self.input_np = np.random.randn(*input_shape).astype(dtype)
        self.out_grad_np = np.random.randn(*input_shape).astype(dtype)
        self.dtype = dtype
        if self.dtype == np.float16:
            self.loss = 1e-3
        elif self.dtype in (np.float32, np.complex64):
            self.loss = 1e-4
        elif self.dtype in (np.float64, np.complex128):
            self.loss = 1e-5
        else:
            self.loss = 0
        self.input_data = Tensor(self.input_np)

    def forward_mindspore_impl(self, net):
        out = net(self.input_data)
        return out.asnumpy()

    def grad_mindspore_impl(self, net):
        out_grad = Tensor(self.out_grad_np)
        grad_net = GradOfFirstInput(net)
        grad_net.set_train()
        input_grad = grad_net(self.input_data, out_grad)
        return input_grad.asnumpy()

    def forward_cmp(self):
        ps_net = Round()
        jit(ps_net.construct, mode="PSJit")(self.input_data)
        context.set_context(mode=context.GRAPH_MODE)
        out_psjit = self.forward_mindspore_impl(ps_net)
        pi_net = Round()
        jit(pi_net.construct, mode="PIJit")(self.input_data)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pijit = self.forward_mindspore_impl(pi_net)
        allclose_nparray(out_pijit, out_psjit, self.loss, self.loss)

    def grad_cmp(self):
        ps_net = Round()
        jit(ps_net.construct, mode="PSJit")(self.input_data)
        context.set_context(mode=context.GRAPH_MODE)
        input_grad_psjit = self.grad_mindspore_impl(ps_net)
        pi_net = Round()
        jit(pi_net.construct, mode="PIJit")(self.input_data)
        context.set_context(mode=context.PYNATIVE_MODE)
        input_grad_pijit = self.grad_mindspore_impl(pi_net)
        allclose_nparray(input_grad_pijit, input_grad_psjit, self.loss, self.loss)
