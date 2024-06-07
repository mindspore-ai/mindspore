from mindspore.nn import Cell
from mindspore.ops import operations as op
from mindspore import Tensor, jit, context
from mindspore import dtype as mstype
import numpy as np
from ..share.utils import allclose_nparray
from ..share.grad import GradOfAllInputs


class CosDynamicShapeNetMS(Cell):
    def __init__(self):
        super().__init__()
        self.reducesum = op.ReduceSum(keep_dims=False)
        self.relu = op.ReLU()
        self.cos = op.Cos()

    def construct(self, x, indices):
        unique_indices = self.relu(indices)
        x = self.reducesum(x, unique_indices)
        return self.cos(x)


class CosDynamicShapeFactory():
    def __init__(self, inputs, dtype=mstype.float32):
        self.input_x = inputs[0]
        self.input_x_np = inputs[0].asnumpy()
        self.indices = inputs[1]
        self.indices_np = inputs[1].asnumpy()
        self.out_grad_np = None
        self.dtype = dtype
        if self.dtype == np.float16:
            self.loss = 1e-3
        elif self.dtype in (np.float32, np.complex64):
            self.loss = 1e-4
        elif self.dtype in (np.float64, np.complex128):
            self.loss = 1e-5
        else:
            self.loss = 0

    def forward_mindspore_impl(self, ms_net):
        input_x_dyn = Tensor(shape=[None for _ in self.input_x.shape], dtype=self.input_x.dtype)
        indices_dyn = Tensor(shape=[None], dtype=self.indices.dtype)
        ms_net.set_inputs(input_x_dyn, indices_dyn)
        out_ms = ms_net(self.input_x, self.indices)
        return out_ms.asnumpy()

    def forward_cmp(self):
        ps_net = CosDynamicShapeNetMS()
        jit(ps_net.construct, mode="PSJit")(self.input_x, self.indices)
        context.set_context(mode=context.GRAPH_MODE)
        out_ps = self.forward_mindspore_impl(ps_net)
        pi_net = CosDynamicShapeNetMS()
        jit(pi_net.construct, mode="PIJit")(self.input_x, self.indices)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pi = self.forward_mindspore_impl(pi_net)
        allclose_nparray(out_pi, out_ps, self.loss, self.loss)

    def grad_mindspore_impl(self, net):
        input_x = self.input_x
        out_grad_ms = Tensor(self.out_grad_np)
        grad_net = GradOfAllInputs(net)
        grad_net.set_train()
        input_x_dyn = Tensor(shape=[None for _ in input_x.shape], dtype=input_x.dtype)
        output_grad_dyn = Tensor(shape=[None for _ in out_grad_ms.shape], dtype=out_grad_ms.dtype)
        indices_dyn = Tensor(shape=[None], dtype=self.indices.dtype)
        grad_net.set_inputs(input_x_dyn, indices_dyn, output_grad_dyn)
        input_grad = grad_net(input_x, self.indices, out_grad_ms)
        return input_grad[0].asnumpy().astype(self.dtype)

    def grad_cmp(self):
        ps_net = CosDynamicShapeNetMS()
        jit(ps_net.construct, mode="PSJit")(self.input_x, self.indices)
        context.set_context(mode=context.GRAPH_MODE)
        out_ps = self.grad_mindspore_impl(ps_net)
        pi_net = CosDynamicShapeNetMS()
        jit(pi_net.construct, mode="PSJit")(self.input_x, self.indices)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pi = self.grad_mindspore_impl(pi_net)
        allclose_nparray(out_pi.asnumpy(), out_ps.asnumpy(), self.loss, self.loss)
