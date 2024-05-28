import numpy as np
import mindspore.ops.operations as op
from mindspore.common import dtype_to_nptype
from mindspore.nn import Cell
from ...utils import allclose_nparray, is_empty
from mindspore import jit, context, Tensor


class Slice(Cell):
    def __init__(self, begin, size):
        super().__init__()
        self.slice = op.Slice()
        self.begin = begin
        self.size = size

    def construct(self, input_shape):
        x = self.slice(input_shape, self.begin, self.size)
        return x


class SliceFactory():
    def __init__(self, input_shape, begin, size, dtype=np.float32):
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
        self.output_grad_np = None
        self.input_ms = Tensor(self.input_np)

    def forward_mindspore_impl(self, net):
        out = net(self.input_ms)
        return out.asnumpy()

    def forward_mindspore_dynamic_shape_impl(self, ms_net):
        input_dyn = Tensor(shape=[None for _ in self.input_ms.shape], dtype=self.input_ms.dtype)
        ms_net.set_inputs(input_dyn)
        out_ms = ms_net(self.input_ms)

        return out_ms.asnumpy()

    def forward_cmp(self):
        ps_net = Slice(self.begin, self.size)
        jit(ps_net.construct, mode="PSJit")(self.input_ms)
        context.set_context(mode=context.GRAPH_MODE)
        out_psjit = self.forward_mindspore_impl(ps_net)
        pi_net = Slice(self.begin, self.size)
        jit(pi_net.construct, mode="PIJit")(self.input_ms)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pijit = self.forward_mindspore_impl(pi_net)
        allclose_nparray(out_pijit, out_psjit, self.loss, self.loss)

    def forward_dynamic_shape_cmp(self):
        ps_net = Slice(self.begin, self.size)
        jit(ps_net.construct, mode="PSJit")(self.input_ms)
        context.set_context(mode=context.GRAPH_MODE)
        out_psjit = self.forward_mindspore_dynamic_shape_impl(ps_net)
        pi_net = Slice(self.begin, self.size)
        jit(pi_net.construct, mode="PIJit")(self.input_ms)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pijit = self.forward_mindspore_dynamic_shape_impl(pi_net)
        allclose_nparray(out_pijit, out_psjit, self.loss, self.loss)


class SliceMock(SliceFactory):
    def __init__(self, inputs=None, grads=None):
        input_x = inputs[0]
        self.begin = inputs[1]
        self.size = inputs[2]
        input_shape = input_x.shape
        self.ms_type = input_x.dtype

        super().__init__(input_shape=input_shape, begin=self.begin, size=self.size,
                         dtype=dtype_to_nptype(self.ms_type))
        self.input_np = input_x.asnumpy()

        if is_empty(grads):
            self.output_grad_np = None
        else:
            self.output_grad_np = grads[0].asnumpy()
