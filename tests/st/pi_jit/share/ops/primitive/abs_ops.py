from mindspore.nn import Cell
from mindspore import jit, context
import mindspore.ops.operations as op
from ...grad import GradOfFirstInput
from ...utils import allclose_nparray, is_empty
from mindspore import Tensor
import numpy as np


class Abs(Cell):
    def __init__(self):
        super().__init__()
        self.abs = op.Abs()

    def construct(self, inputs):
        return self.abs(inputs)


class AbsFactory():
    def __init__(self, input_shape, dtype=np.float16):
        self.input_np = np.random.randn(*input_shape).astype(dtype)
        self.input = Tensor(self.input_np)
        self.dtype = dtype
        self.shape = None
        self.out_grad_np = None
        if self.dtype == np.float16:
            self.loss = 1e-3
        elif self.dtype in (np.float32, np.complex64):
            self.loss = 1e-4
        elif self.dtype in (np.float64, np.complex128):
            self.loss = 1e-5
        else:
            self.loss = 0

    def forward_mindspore_impl(self):
        pi_net = Abs()
        jit(pi_net.construct, mode="PIJit")(self.input)
        context.set_context(mode=context.PYNATIVE_MODE)
        pi_out = pi_net(self.input)
        return pi_out.asnumpy()

    def forward_mindspore_dynamic_shape_impl(self):
        input_ms = Tensor(self.input_np)
        pi_net = Abs()
        jit(pi_net.construct, mode="PIJit")(self.input)
        context.set_context(mode=context.PYNATIVE_MODE)
        input_dyn = Tensor(shape=[None for _ in input_ms.shape], dtype=input_ms.dtype)
        pi_net.set_inputs(input_dyn)
        pi_out = pi_net(input_ms)
        return pi_out.asnumpy()

    def forward_numpy_impl(self):
        out = np.abs(self.input_np)
        return out

    def grad_mindspore_impl(self, net):
        if is_empty(self.shape):
            grad_net = GradOfFirstInput(net, sens_param=False)
            input_grad = grad_net(self.input)
        else:
            out_grad = Tensor(self.out_grad_np)
            grad_net = GradOfFirstInput(net)
            input_grad = grad_net(self.input, out_grad)
        return input_grad.asnumpy()

    def grad_mindspore_dynamic_shape_impl(self, net):
        input_ms = Tensor(self.input_np)
        output_grad = Tensor(self.out_grad_np)
        grad_net = GradOfFirstInput(net)
        grad_net.set_train()
        input_dyn = Tensor(shape=[None for _ in input_ms.shape], dtype=input_ms.dtype)
        output_grad_dyn = Tensor(shape=[None for _ in output_grad.shape], dtype=output_grad.dtype)
        grad_net.set_inputs(input_dyn, output_grad_dyn)
        input_grad = grad_net(input_ms, output_grad)
        return input_grad.asnumpy()

    def forward_cmp(self):
        out_numpy = self.forward_numpy_impl()
        out_mindspore = self.forward_mindspore_impl()
        allclose_nparray(out_numpy, out_mindspore, self.loss, self.loss)

    def forward_dynamic_shape_cmp(self):
        out_numpy = self.forward_numpy_impl()
        out_mindspore = self.forward_mindspore_dynamic_shape_impl()
        allclose_nparray(out_numpy, out_mindspore, self.loss, self.loss)

    def grad_cmp(self):
        ps_net = Abs()
        jit(ps_net.construct, mode="PSJit")(self.input)
        context.set_context(mode=context.GRAPH_MODE)
        input_grad_psjit = self.grad_mindspore_impl(ps_net)
        pi_net = Abs()
        jit(pi_net.construct, mode="PIJit")(self.input)
        context.set_context(mode=context.PYNATIVE_MODE)
        input_grad_pijit = self.grad_mindspore_impl(pi_net)
        allclose_nparray(input_grad_pijit, input_grad_psjit, self.loss, self.loss)

    def grad_dynamic_shape_cmp(self):
        ps_net = Abs()
        jit(ps_net.construct, mode="PSJit")(self.input)
        context.set_context(mode=context.GRAPH_MODE)
        input_grad_psjit = self.grad_mindspore_dynamic_shape_impl(ps_net)
        pi_net = Abs()
        jit(pi_net.construct, mode="PIJit")(self.input)
        context.set_context(mode=context.PYNATIVE_MODE)
        input_grad_pijit = self.grad_mindspore_dynamic_shape_impl(pi_net)
        allclose_nparray(input_grad_pijit, input_grad_psjit, self.loss, self.loss)

    def get_vmap_forward_output(self):
        output = Tensor(self.forward_mindspore_impl())
        return output

    def grad_vmap_cmp(self, in_axes=-1, run_time=10, improve_times=1.5):
        out_grad = Tensor(self.get_vmap_forward_output())
        net = Abs()
        inputs = [self.input, out_grad]
        grad_net = GradOfFirstInput(net)
        self.vmap_cmp(grad_net, in_axes, run_time, improve_times, *inputs)
