import numpy as np
from mindspore import Tensor, jit, context
from mindspore.nn import Cell
from mindspore.ops import operations as op
import mindspore.ops as ops
from ...utils import allclose_nparray


class Invert(Cell):
    def __init__(self):
        super().__init__()
        self.invert = op.Invert()

    def construct(self, input_x):
        return self.invert(input_x)


class InvertFunction(Cell):
    def construct(self, input_x):
        return ops.invert(input_x)


class InvertTensor(Cell):
    def construct(self, input_x):
        return input_x.invert()


class InvertFactory():
    def __init__(self, input_shape, dtype=np.float32):
        self.input_np = np.random.randint(-100, 100, input_shape).astype(dtype)
        self.dtype = dtype
        if self.dtype == np.float16:
            self.loss = 1e-3
        elif self.dtype in (np.float32, np.complex64):
            self.loss = 1e-4
        elif self.dtype in (np.float64, np.complex128):
            self.loss = 1e-5
        else:
            self.loss = 0
        self.input_me = Tensor(self.input_np)

    def forward_mindspore_impl(self, net):
        out = net(self.input_me)
        return out.asnumpy()

    def forward_cmp(self):
        ps_net = Invert()
        jit(ps_net.construct, mode="PSJit")(self.input_me)
        context.set_context(mode=context.GRAPH_MODE)
        out_psjit = self.forward_mindspore_impl(ps_net)
        pi_net = Invert()
        jit(pi_net.construct, mode="PIJit")(self.input_me)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pijit = self.forward_mindspore_impl(pi_net)

        allclose_nparray(out_pijit, out_psjit, self.loss, self.loss)

    def forward_function_cmp(self):
        ps_net = InvertFunction()
        jit(ps_net.construct, mode="PSJit")(self.input_me)
        context.set_context(mode=context.GRAPH_MODE)
        out_psjit = ps_net(self.input_me)
        pi_net = InvertFunction()
        jit(pi_net.construct, mode="PIJit")(self.input_me)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pijit = pi_net(self.input_me)
        allclose_nparray(out_pijit, out_psjit, self.loss, self.loss)

    def forward_tensor_cmp(self):
        ps_net = InvertFunction()
        jit(ps_net.construct, mode="PSJit")(self.input_me)
        context.set_context(mode=context.GRAPH_MODE)
        out_psjit = ps_net(self.input_me)
        pi_net = InvertFunction()
        jit(pi_net.construct, mode="PIJit")(self.input_me)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pijit = pi_net(self.input_me)
        allclose_nparray(out_pijit, out_psjit, self.loss, self.loss)
