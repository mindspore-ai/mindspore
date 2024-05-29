from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore import Tensor, jit, context
import numpy as np
from ..share.grad import GradOfFirstInput
from ..share.utils import allclose_nparray, is_empty


class DynamicShapeSliceNet(Cell):
    def __init__(self):
        super().__init__()
        self.slice = P.Slice()
        self.reduce = P.ReduceSum(False)

    def construct(self, x, begin, size, axis):
        x_re = self.reduce(x, axis)
        return self.slice(x_re, begin, size)


class DynamicShapeSliceFactory():
    def __init__(self, input_shape, begin, size, axis, dtype=np.float32):
        if dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
            if is_empty(input_shape):
                self.input_np = np.random.uniform(low=0, high=200000, size=input_shape)
            else:
                self.input_np = np.random.uniform(low=0, high=200000, size=input_shape).astype(
                    dtype)
        else:
            if is_empty(input_shape):
                self.input_np = np.random.randn(*input_shape)
            else:
                self.input_np = np.random.randn(*input_shape).astype(dtype)
        self.dtype = dtype
        if self.dtype == np.float16:
            self.loss = 1e-3
        elif self.dtype in (np.float32, np.complex64):
            self.loss = 1e-4
        elif self.dtype in (np.float64, np.complex128):
            self.loss = 1e-5
        else:
            self.loss = 0
        self.dtype = dtype
        self.begin = begin
        self.size = size
        self.axis = axis
        self.out_grad_np = None
        self.input_ms = Tensor(self.input_np)
        self.begin_ms = Tensor(self.begin)
        self.size_ms = Tensor(self.size)
        self.axis_ms = Tensor(self.axis)
        self.input_dyn = Tensor(shape=[None for _ in self.input_ms.shape], dtype=self.input_ms.dtype)
        self.begin_dyn = Tensor(shape=[None for _ in self.begin_ms.shape], dtype=self.begin_ms.dtype)
        self.size_dyn = Tensor(shape=[None for _ in self.size_ms.shape], dtype=self.size_ms.dtype)
        self.axis_dyn = Tensor(shape=[None], dtype=self.axis_ms.dtype)

    def forward_mindspore_impl(self, ms_net):
        ms_net.set_inputs(self.input_dyn, self.begin_dyn, self.size_dyn, self.axis_dyn)
        out_ms = ms_net(self.input_ms, self.begin_ms, self.size_ms, self.axis_ms)
        return out_ms.asnumpy()


    def forward_cmp(self):
        ps_net = DynamicShapeSliceNet()
        jit(ps_net.construct, mode="PSJit")(self.input_ms, self.begin_ms, self.size_ms, self.axis_ms)
        context.set_context(mode=context.GRAPH_MODE)
        out_psjit = self.forward_mindspore_impl(ps_net)
        pi_net = DynamicShapeSliceNet()
        jit(pi_net.construct, mode="PIJit")(self.input_ms, self.begin_ms, self.size_ms, self.axis_ms)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pijit = self.forward_mindspore_impl(pi_net)
        allclose_nparray(out_pijit, out_psjit, self.loss, self.loss)

    def grad_mindspore_impl(self, net):
        ms_net = DynamicShapeSliceNet()
        if self.out_grad_np is None:
            self.out_grad_np = np.random.randn(*self.forward_mindspore_impl(ms_net).shape).astype(self.dtype)
        grad_net = GradOfFirstInput(net)
        grad_net.set_train()

        output_grad_ms = Tensor(self.out_grad_np)
        out_grad_dyn = Tensor(shape=[None for _ in output_grad_ms.shape],
                              dtype=output_grad_ms.dtype)
        grad_net.set_inputs(self.input_dyn, self.begin_dyn, self.size_dyn, self.axis_dyn, self.out_grad_dyn)
        out_grad = grad_net(self.input_ms, self.begin_ms, self.size_ms, self.axis_ms, self.output_grad_ms)
        return out_grad.asnumpy()


    def grad_cmp(self):
        ps_net = DynamicShapeSliceNet()
        jit(ps_net.construct, mode="PSJit")(self.input_ms, self.begin_ms, self.size_ms, self.axis_ms)
        context.set_context(mode=context.GRAPH_MODE)
        out_psjit = self.grad_mindspore_impl(ps_net)
        pi_net = DynamicShapeSliceNet()
        jit(pi_net.construct, mode="PIJit")(self.input_ms, self.begin_ms, self.size_ms, self.axis_ms)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pijit = self.grad_mindspore_impl(pi_net)
        allclose_nparray(out_psjit, out_pijit, self.loss, self.loss)
