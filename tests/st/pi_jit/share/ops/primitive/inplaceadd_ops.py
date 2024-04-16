import numpy as np
from mindspore import Tensor, jit, context, ops
from mindspore.nn import Cell
from ...utils import allclose_nparray


class InplaceAdd(Cell):
    def __init__(self, indices):
        super().__init__()
        self.op = ops.operations.InplaceAdd(indices)

    def construct(self, input_x, input_v):
        return self.op(input_x, input_v)


class InplaceAddFactory():
    def __init__(self, inputx_shape, inputv_shape, indices, dtype1=np.float32, dtype2=np.float32):
        self.input_x_np = np.random.randn(*inputx_shape).astype(dtype1)
        self.input_v_np = np.random.randn(*inputv_shape).astype(dtype2)
        self.indices = indices
        self.dtype1 = dtype1
        self.out_grad_np = np.random.randn(*inputx_shape).astype(dtype=dtype1)
        if self.dtype1 == np.float16:
            self.loss = 1e-3
        elif self.dtype1 in (np.float32, np.complex64):
            self.loss = 1e-4
        elif self.dtype1 in (np.float64, np.complex128):
            self.loss = 1e-5
        else:
            self.loss = 0
        self.input_x_me = Tensor(self.input_x_np)
        self.input_v_me = Tensor(self.input_v_np)

    def forward_mindspore_impl(self, net):
        out = net(self.input_x_me, self.input_v_me)
        return out.asnumpy()

    def forward_cmp(self):
        ps_net = InplaceAdd(self.indices)
        jit(ps_net.construct, mode="PSJit")(self.input_x_me, self.input_v_me)
        context.set_context(mode=context.GRAPH_MODE)
        out_psjit = self.forward_mindspore_impl(ps_net)
        pi_net = InplaceAdd(self.indices)
        jit(pi_net.construct, mode="PIJit")(self.input_x_me, self.input_v_me)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pijit = self.forward_mindspore_impl(pi_net)

        allclose_nparray(out_pijit, out_psjit, self.loss, self.loss)
