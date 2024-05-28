import numpy as np
from mindspore.nn import Cell
from mindspore.ops.operations.math_ops import Median as MedianOp
from mindspore import jit, Tensor, context
from ...utils import allclose_nparray
from ...grad import GradOfFirstInput
import mindspore.ops.operations._grad_ops as G


class Median(Cell):
    def __init__(self, global_median, axis, keep_dims):
        super().__init__()
        self.global_median = global_median
        self.axis = axis
        self.keep_dims = keep_dims
        self.median = MedianOp(self.global_median, self.axis, self.keep_dims)

    def construct(self, x):
        return self.median(x)


class MedianGrad(Cell):
    def __init__(self, global_median, axis, keep_dims):
        super().__init__()
        self.global_median = global_median
        self.axis = axis
        self.keep_dims = keep_dims
        self.median_grad = G.MedianGrad(self.global_median, self.axis, self.keep_dims)

    def construct(self, dy, x, y, indices):
        return self.median_grad(dy, x, y, indices)


class MedianFactory():
    def __init__(self, input_shape, global_median, axis=0, keep_dims=False, dtype=np.float32):
        self.dtype = dtype
        if dtype == np.int16 or dtype == np.int32 or dtype == np.int64:
            self.input = np.random.choice(100000, size=input_shape,
                                          replace=False).astype(self.dtype)
        else:
            self.input = np.random.randn(*input_shape).astype(dtype)
        self.global_median = global_median
        self.axis = axis
        self.keep_dims = keep_dims
        if self.dtype == np.float16:
            self.loss = 1e-3
        elif self.dtype in (np.float32, np.complex64):
            self.loss = 1e-4
        elif self.dtype in (np.float64, np.complex128):
            self.loss = 1e-5
        else:
            self.loss = 0
        self.input_ms = Tensor(self.input)

    def forward_mindspore_impl(self, net):
        y, indices = net(self.input_ms)
        return y.asnumpy(), indices.asnumpy()

    def grad_mindspore_impl(self, net):
        grad_net = GradOfFirstInput(net, sens_param=False)
        res = grad_net(self.input_ms)
        return res.asnumpy()


    def forward_cmp(self):
        ps_net = Median(self.global_median, self.axis, self.keep_dims)
        jit(ps_net.construct, mode="PSJit")(self.input_ms)
        context.set_context(mode=context.GRAPH_MODE)
        y_psjit, indices_psjit = self.forward_mindspore_impl(ps_net)
        pi_net = Median(self.global_median, self.axis, self.keep_dims)
        jit(pi_net.construct, mode="PIJit")(self.input_ms)
        context.set_context(mode=context.PYNATIVE_MODE)
        y_pijit, indices_pijit = self.forward_mindspore_impl(pi_net)
        if not self.global_median:
            return allclose_nparray(y_pijit, y_psjit, self.loss, self.loss) and \
                   allclose_nparray(indices_pijit, indices_psjit, self.loss, self.loss)
        return allclose_nparray(y_pijit, y_psjit, self.loss, self.loss)

    def grad_cmp(self):
        ps_net = Median(self.global_median, self.axis, self.keep_dims)
        jit(ps_net.construct, mode="PSJit")(self.input_ms)
        context.set_context(mode=context.GRAPH_MODE)
        grad_psjit = self.grad_mindspore_impl(ps_net)
        pi_net = Median(self.global_median, self.axis, self.keep_dims)
        jit(pi_net.construct, mode="PIJit")(self.input_ms)
        context.set_context(mode=context.PYNATIVE_MODE)
        grad_pijit = self.grad_mindspore_impl(pi_net)
        self.loss = 0.01
        assert np.allclose(grad_psjit, grad_pijit, self.loss, self.loss)
