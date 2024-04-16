from ...utils import allclose_nparray
from mindspore import Tensor, jit, context
from mindspore import ops
from mindspore.nn import Cell
import mindspore.ops.operations as op
import numpy as np


class InplaceSub(Cell):
    def __init__(self, indices):
        super().__init__()
        self.op = op.InplaceSub(indices)

    def construct(self, input_x, input_v):
        return self.op(input_x, input_v)


class InplaceSubFactory():
    def __init__(self, input_shape, target_shape, indices, dtype=np.float32):
        self.input_x_np = np.random.randn(*input_shape).astype(dtype)
        self.input_v_np = np.random.randn(*target_shape).astype(dtype)
        self.indices = indices
        self.dtype = dtype
        self.out_grad_np = np.random.randn(*input_shape).astype(dtype=dtype)
        if self.dtype == np.float16:
            self.loss = 1e-3
        elif self.dtype in (np.float32, np.complex64):
            self.loss = 1e-4
        elif self.dtype in (np.float64, np.complex128):
            self.loss = 1e-5
        else:
            self.loss = 0
        self.input_x_me = Tensor(self.input_x_np)
        self.input_v_me = Tensor(self.input_v_np)

    def forward_mindspore_impl(self, net):
        out = net(self.input_x_me, self.input_v_me)
        return out.asnumpy()

    def forward_mindspore_func_impl(self):
        out = ops.inplace_sub(self.input_x_me, self.input_v_me, self.indices)
        return out.asnumpy()

    def forward_cmp(self):
        ps_net = InplaceSub(self.indices)
        jit(ps_net.construct, mode="PSJit")(self.input_x_me, self.input_v_me)
        context.set_context(mode=context.GRAPH_MODE)
        out_psjit = self.forward_mindspore_impl(ps_net)
        pi_net = InplaceSub(self.indices)
        jit(pi_net.construct, mode="PIJit")(self.input_x_me, self.input_v_me)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pijit = self.forward_mindspore_impl(pi_net)

        allclose_nparray(out_pijit, out_psjit, self.loss, self.loss)

    def forward_func_cmp(self):
        out_me = self.forward_mindspore_func_impl()

        allclose_nparray(out_tf, out_me, self.loss, self.loss)

    def forward_mindspore_dynamic_shape_impl(self, net):
        input_x_dyn = Tensor(shape=[None for _ in self.input_x_np.shape], dtype=Tensor(self.input_x_np).dtype)
        input_v_dyn = Tensor(shape=[None for _ in self.input_v_np.shape], dtype=Tensor(self.input_v_np).dtype)
        net.set_inputs(input_x_dyn, input_v_dyn)
        out = net(self.input_x_me, self.input_v_me)
        return out.asnumpy()

    def forward_dynamic_shape_cmp(self):
        ps_net = InplaceSub(self.indices)
        jit(ps_net.construct, mode="PSJit")(self.input_x_me, self.input_v_me)
        context.set_context(mode=context.GRAPH_MODE)
        out_psjit = self.forward_mindspore_dynamic_shape_impl(ps_net)
        pi_net = InplaceSub(self.indices)
        jit(pi_net.construct, mode="PIJit")(self.input_x_me, self.input_v_me)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pijit = self.forward_mindspore_dynamic_shape_impl(pi_net)

        allclose_nparray(out_pijit, out_psjit, self.loss, self.loss)
