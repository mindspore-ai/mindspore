from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore import Tensor, jit, context
import numpy as np
from ..share.utils import allclose_nparray
from ..share.grad import GradOfAllInputs


class DynamicShapeAbs(Cell):
    def __init__(self):
        super().__init__()
        self.reducemean = P.ReduceSum(keep_dims=False)
        self.abs = P.Abs()
        self.relu = P.ReLU()

    def construct(self, x, indices):
        x = self.abs(x)
        unique_indices = self.relu(indices)
        x = self.reducemean(x, unique_indices)
        return self.abs(x)


class AbsDynamicShapeMock():
    def __init__(self, input_np, indices_np, dtype=np.float32):
        self.input_np = input_np
        self.indices_np = indices_np
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
        self.input_ms = Tensor(self.input_np)
        self.indices_ms = Tensor(self.indices_np)
        self.input_dyn = Tensor(shape=[None for _ in self.input_ms.shape], dtype=self.input_ms.dtype)
        self.indices_dyn = Tensor(shape=[None], dtype=self.indices_ms.dtype)

    def forward_mindspore_impl(self, ms_net):
        ms_net.set_inputs(self.input_dyn, self.indices_dyn)
        out_ms = ms_net(self.input_ms, self.indices_ms)
        return out_ms


    def grad_mindspore_impl(self, net):
        if self.out_grad_np is None:
            print(*self.input_np.shape, "mindspore shape")
            self.out_grad_np = np.random.randn(*self.input_np.shape).astype(self.dtype)
        grad_net = GradOfAllInputs(net)
        grad_net.set_train()
        out_grad_np = Tensor(self.out_grad_np)
        out_grad_dyn = Tensor(shape=[None for _ in out_grad_np.shape], dtype=out_grad_np.dtype)
        grad_net.set_inputs(self.input_dyn, self.indices_dyn, out_grad_dyn)
        input_grad = grad_net(self.input_ms, self.indices_ms, out_grad_np)
        return input_grad[0]


    def forward_cmp(self):
        ps_net = DynamicShapeAbs()
        jit(ps_net.construct, mode="PSJit")(self.input_ms, self.indices_ms)
        context.set_context(mode=context.GRAPH_MODE)
        out_ps = self.forward_mindspore_impl(ps_net)
        pi_net = DynamicShapeAbs()
        jit(pi_net.construct, mode="PIJit")(self.input_ms, self.indices_ms)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pi = self.forward_mindspore_impl(pi_net)
        allclose_nparray(out_pi.asnumpy(), out_ps.asnumpy(), self.loss, self.loss)

    def grad_cmp(self):
        ps_net = DynamicShapeAbs()
        jit(ps_net.construct, mode="PSJit")(self.input_ms, self.indices_ms)
        context.set_context(mode=context.GRAPH_MODE)
        out_ps = self.grad_mindspore_impl(ps_net)
        pi_net = DynamicShapeAbs()
        jit(pi_net.construct, mode="PSJit")(self.input_ms, self.indices_ms)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pi = self.grad_mindspore_impl(pi_net)
        allclose_nparray(out_pi.asnumpy(), out_ps.asnumpy(), self.loss, self.loss)
