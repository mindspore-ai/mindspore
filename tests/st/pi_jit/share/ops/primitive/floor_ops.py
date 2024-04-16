import numpy as np
from mindspore import Tensor, jit, context
import mindspore.ops.operations as op
from mindspore.nn import Cell
from ...utils import allclose_nparray
from ...grad import GradOfFirstInput


class Floor(Cell):
    def __init__(self):
        super().__init__()
        self.floor = op.Floor()

    def construct(self, input_x):
        return self.floor(input_x)


class FloorFactory():
    def __init__(self, input_shape, dtype=np.float16):
        self.dtype = dtype
        self.input_np = np.random.randn(*input_shape).astype(dtype)
        self.output_grad_np = np.random.randn(*input_shape).astype(dtype)
        if self.dtype == np.float16:
            self.loss = 1e-3
        elif self.dtype in (np.float32, np.complex64):
            self.loss = 1e-4
        elif self.dtype in (np.float64, np.complex128):
            self.loss = 1e-5
        else:
            self.loss = 0
        self.input_tensor = Tensor(self.input_np)

    def forward_mindspore_impl(self, net):
        out = net(self.input_tensor)
        return out.asnumpy()

    def grad_mindspore_impl(self, net):
        ms_input = Tensor(self.input_np)
        output_grad = Tensor(self.output_grad_np)
        grad_net = GradOfFirstInput(net)
        grad_net.set_train()
        input_grad = grad_net(ms_input, output_grad)
        return input_grad.asnumpy()

    def forward_cmp(self):
        ps_net = Floor()
        jit(ps_net.construct, mode="PSJit")(self.input_tensor)
        context.set_context(mode=context.GRAPH_MODE)
        out_psjit = self.forward_mindspore_impl(ps_net)
        pi_net = Floor()
        jit(pi_net.construct, mode="PIJit")(self.input_tensor)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pijit = self.forward_mindspore_impl(pi_net)
        allclose_nparray(out_pijit, out_psjit, self.loss, self.loss)

    def grad_cmp(self):
        ps_net = Floor()
        jit(ps_net.construct, mode="PSJit")(self.input_tensor)
        context.set_context(mode=context.GRAPH_MODE)
        input_grad_psjit = self.grad_mindspore_impl(ps_net)
        pi_net = Floor()
        jit(pi_net.construct, mode="PIJit")(self.input_tensor)
        context.set_context(mode=context.PYNATIVE_MODE)
        input_grad_pijit = self.grad_mindspore_impl(pi_net)
        allclose_nparray(input_grad_pijit, input_grad_psjit, self.loss, self.loss)
