import numpy as np
from mindspore.common.tensor import Tensor
from mindspore import ops
from mindspore import context, jit
from mindspore.ops import operations as op
from mindspore.nn import Cell
from ...utils import allclose_nparray, is_empty
from ...grad import GradOfFirstInput



class Pow(Cell):
    def __init__(self, exp):
        super().__init__()
        self.pow = op.Pow()
        self.exp = exp

    def construct(self, input_ms):
        return self.pow(input_ms, self.exp)


class PowVmap(Cell):
    def __init__(self):
        super().__init__()
        self.pow = op.Pow()

    def construct(self, base, exp):
        return self.pow(base, exp)


class PowFactory():
    def __init__(self, input_shape, exp, dtype=np.float32):
        self.dtype = dtype
        self.input_shape = input_shape
        if is_empty(input_shape):
            self.input_np = np.abs(np.random.randn(*input_shape))
        else:
            self.input_np = np.abs(np.random.randn(*input_shape).astype(self.dtype))
        if self.dtype == np.float16:
            self.loss = 1e-3
        elif self.dtype in (np.float32, np.complex64):
            self.loss = 1e-4
        elif self.dtype in (np.float64, np.complex128):
            self.loss = 1e-5
        else:
            self.loss = 0
        self.exp = exp
        self.output_grad_np = None
        self.input_ms = Tensor(self.input_np)

    def forward_mindspore_impl(self, net):
        out = net(self.input_ms)
        return out.asnumpy()


    def grad_mindspore_impl(self, net, output_grad_np):
        input_ms = Tensor(self.input_np)
        output_grad = Tensor(output_grad_np)
        grad_net = GradOfFirstInput(net)
        grad_net.set_train()
        input_grad = grad_net(self.input_ms, output_grad)
        return input_grad.asnumpy()


    def forward_cmp(self):
        ps_net = Pow(self.exp)
        jit(ps_net.construct, mode="PSJit")(self.input_ms)
        context.set_context(mode=context.GRAPH_MODE)
        out_psjit = self.forward_mindspore_impl(ps_net)
        pi_net = Pow(self.exp)
        jit(pi_net.construct, mode="PIJit")(self.input_ms)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pijit = self.forward_mindspore_impl(pi_net)
        allclose_nparray(out_pijit, out_psjit, self.loss, self.loss)

    def grad_cmp(self):
        self.output_grad_np = np.random.randn(*self.input_shape).astype(self.dtype)
        ps_net = Pow(self.exp)
        jit(ps_net.construct, mode="PSJit")(self.input_ms)
        context.set_context(mode=context.GRAPH_MODE)
        input_grad_psjit = self.grad_mindspore_impl(ps_net, self.output_grad_np)
        pi_net = Pow(self.exp)
        jit(pi_net.construct, mode="PIJit")(self.input_ms)
        context.set_context(mode=context.PYNATIVE_MODE)
        input_grad_pijit = self.grad_mindspore_impl(pi_net, self.output_grad_np)
        allclose_nparray(input_grad_pijit, input_grad_psjit, self.loss, self.loss)


    @jit(mode="PIJit")
    def forward_mindspore_func(self):
        out = ops.pow(self.input_ms, self.exp)
        return out.asnumpy()

    def forward_mindspore_tensor(self):
        out = self.input_ms.pow(self.exp)
        return out.asnumpy()
