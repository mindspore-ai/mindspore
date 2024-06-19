import numpy as np
from mindspore.common.tensor import Tensor
from mindspore.nn import Cell
import mindspore.ops.operations as op
from ...utils import allclose_nparray
from ...grad import GradOfAllInputs
from mindspore import jit, context
import mindspore.ops as op


class Div(Cell):
    def __init__(self):
        super().__init__()
        self.div = op.Div()

    def construct(self, inputx, inputy):
        x = self.div(inputx, inputy)
        return x


class DivFactory():
    def __init__(self, inputx_shape, inputy_shape, dtype=np.float32):
        if dtype == np.uint64:
            self.inputx = np.random.uniform(1, 100, size=inputx_shape).astype(dtype)
        else:
            self.inputx = np.random.uniform(-100, 100, size=inputx_shape).astype(dtype)
        self.inputy = np.random.uniform(1, 10, size=inputy_shape).astype(dtype)
        self.dtype = dtype
        self.output_grad_np = None
        self.inputx_ms = Tensor(self.inputx)
        self.inputy_ms = Tensor(self.inputy)
        if self.dtype == np.float16:
            self.loss = 1e-3
        elif self.dtype in (np.float32, np.complex64):
            self.loss = 1e-4
        elif self.dtype in (np.float64, np.complex128):
            self.loss = 1e-5
        else:
            self.loss = 0

    def forward_mindspore_impl(self, net):
        out = net(self.inputx_ms, self.inputy_ms)
        return out.asnumpy()

    def grad_mindspore_impl(self, net):
        inputx_ms = Tensor(self.inputx)
        inputy_ms = Tensor(self.inputy)
        net_me = GradOfAllInputs(net)
        net_me.set_train()
        if self.output_grad_np is None:
            out = self.forward_mindspore_impl(net)
            sens = np.random.randn(*list(out.shape))
            self.output_grad_np = np.array(sens, dtype=out.dtype)
        output = net_me(self.inputx_ms, self.inputy_ms, Tensor(self.output_grad_np))
        return output[0].asnumpy(), output[1].asnumpy()

    def forward_cmp(self):
        ps_net = Div()
        jit(ps_net.construct, mode="PSJit")(self.inputx_ms, self.inputy_ms)
        context.set_context(mode=context.GRAPH_MODE)
        out_psjit = self.forward_mindspore_impl(ps_net)
        pi_net = Div()
        jit(pi_net.construct, mode="PIJit")(self.inputx_ms, self.inputy_ms)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pijit = self.forward_mindspore_impl(pi_net)
        allclose_nparray(out_pijit, out_psjit, self.loss, self.loss)

    def grad_cmp(self):
        ps_net = Div()
        jit(ps_net.construct, mode="PSJit")(self.inputx_ms, self.inputy_ms)
        context.set_context(mode=context.GRAPH_MODE)
        input_grad_psjit_a, input_grad_psjit_b = self.grad_mindspore_impl(ps_net)
        pi_net = Div()
        jit(pi_net.construct, mode="PIJit")(self.inputx_ms, self.inputy_ms)
        context.set_context(mode=context.PYNATIVE_MODE)
        input_grad_pijit_a, input_grad_pijit_b = self.grad_mindspore_impl(ps_net)
        allclose_nparray(input_grad_pijit_a, input_grad_psjit_a, self.loss, self.loss)
        allclose_nparray(input_grad_pijit_b, input_grad_psjit_b, self.loss, self.loss)
