import numpy as np
from mindspore import dtype
from mindspore import Tensor, jit, context
from mindspore.nn import Cell
from mindspore.ops import operations as op
from ..share.utils import allclose_nparray, is_empty
from mindspore.common import dtype as mstype


class FloorDivDynamicShapeFactory():
    def __init__(self, inputs, grads=None):
        self.input = inputs
        self.input_x = inputs[0]
        self.input_y = inputs[1]
        self.input_x_np = inputs[0].asnumpy()
        self.input_y_np = inputs[1].asnumpy()
        self.indices = inputs[2]
        self.indices_np = inputs[2].asnumpy()
        if is_empty(grads):
            self.out_grad_np = None
        else:
            self.out_grad_np = grads.asnumpy()
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
        input_y_dyn = Tensor(shape=[None for _ in self.input_y.shape], dtype=self.input_y.dtype)
        indices_dyn = Tensor(shape=[None], dtype=self.indices.dtype)
        ms_net.set_inputs(input_x_dyn, input_y_dyn, indices_dyn)
        out_ms = ms_net(self.input_x, self.input_y, self.indices)
        return out_ms.asnumpy()


    def forward_cmp(self):
        ps_net = FloorDivDynamicShapeNetMS()
        jit(ps_net.construct, mode="PSJit")(self.input_x, self.input_y, self.indices)
        context.set_context(mode=context.GRAPH_MODE)
        out_ps = self.forward_mindspore_impl(ps_net)
        pi_net = FloorDivDynamicShapeNetMS()
        jit(pi_net.construct, mode="PIJit")(self.input_x, self.input_y, self.indices)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pi = self.forward_mindspore_impl(pi_net)
        allclose_nparray(out_pi, out_ps, self.loss, self.loss)


    def grad_cmp(self):
        ps_net = FloorDivDynamicShapeNetMS()
        jit(ps_net.construct, mode="PSJit")(self.input_x, self.input_y, self.indices)
        context.set_context(mode=context.GRAPH_MODE)
        input_grad_psjit = self.grad_mindspore_impl(ps_net)
        pi_net = CosDynamicShapeNetMS()
        jit(pi_net.construct, mode="PIJit")(self.input_x, self.input_y, self.indices)
        context.set_context(mode=context.PYNATIVE_MODE)
        input_grad_pijit = self.grad_mindspore_impl(pi_net)
        allclose_nparray(input_grad_pijit[0], input_grad_psjit[0], self.loss, self.loss)
        allclose_nparray(input_grad_pijit[1], input_grad_psjit[1], self.loss, self.loss)


class FloorDivDynamicShapeNetMS(Cell):
    def __init__(self):
        super().__init__()
        self.reducesum = op.ReduceSum(keep_dims=False)
        self.relu = op.ReLU()
        self.floordiv = op.FloorDiv()
        self.cast = op.Cast()

    def construct(self, x, y, indices):
        unique_indices = self.relu(indices)
        x = self.cast(self.reducesum(self.cast(x, mstype.float32), unique_indices), x.dtype)
        y = self.cast(self.reducesum(self.cast(y, mstype.float32), unique_indices), y.dtype)
        return self.floordiv(x, y)
