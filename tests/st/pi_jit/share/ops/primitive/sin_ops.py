import numpy as np
from mindspore.ops import operations as op
from mindspore.nn import Cell
from mindspore.common.tensor import Tensor
from mindspore.common import dtype_to_nptype
from ...utils import allclose_nparray, is_empty
from ...grad import GradOfFirstInput
from mindspore import jit, context


class Sin(Cell):
    def __init__(self):
        super().__init__()
        self.sin = op.Sin()

    def construct(self, x):
        return self.sin(x)


class SinTensorNet(Cell):

    def construct(self, x):
        return x.sin()


class SinMock():
    def __init__(self, inputs=None, grads=None):
        self.ms_type = inputs[0].dtype
        self.dtype = dtype_to_nptype(self.ms_type)
        self.input_x = inputs[0]
        self.input_x_np = inputs[0].asnumpy()
        if self.dtype == np.float16:
            self.loss = 1e-3
        elif self.dtype in (np.float32, np.complex64):
            self.loss = 1e-4
        elif self.dtype in (np.float64, np.complex128):
            self.loss = 1e-5
        else:
            self.loss = 0

        if is_empty(grads):
            self.output_grad_np = None
        else:
            self.output_grad_np = grads[0].asnumpy()

    def forward_mindspore_impl(self, net):
        out = net(self.input_x)
        return out.asnumpy()


    def grad_mindspore_impl(self, sin_net):
        if self.output_grad_np is None:
            if self.dtype == np.complex64 or self.dtype == np.complex128:
                self.output_grad_np = np.ones_like(self.forward_mindspore_impl(sin_net), dtype=self.dtype)
            else:
                self.output_grad_np = self.forward_mindspore_impl(sin_net)
        out_grad_np = Tensor(self.output_grad_np.astype(self.dtype))

        if self.dtype == np.complex64 or self.dtype == np.complex128:
            sin_grad = GradOfFirstInput(sin_net)
            sin_grad.set_train()
            out_grad = sin_grad(self.input_x, out_grad_np)
            return np.conj(out_grad.asnumpy())
        sin_grad = GradOfFirstInput(sin_net)
        sin_grad.set_train()
        out_grad = sin_grad(self.input_x, out_grad_np)
        return out_grad.asnumpy()


    def forward_cmp(self):
        ps_net = Sin()
        jit(ps_net.construct, mode="PSJit")(self.input_x)
        context.set_context(mode=context.GRAPH_MODE)
        out_psjit = self.forward_mindspore_impl(ps_net)
        pi_net = Sin()
        jit(pi_net.construct, mode="PIJit")(self.input_x)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pijit = self.forward_mindspore_impl(pi_net)
        if self.dtype == np.complex64 or np.complex128:
            ps_real = np.real(out_psjit)
            pi_real = np.real(out_pijit)
            ps_imag = np.imag(out_psjit)
            pi_imag = np.imag(out_pijit)
            allclose_nparray(pi_real, ps_real, self.loss, self.loss)
            allclose_nparray(pi_imag, ps_imag, self.loss, self.loss)
        else:
            allclose_nparray(out_pijit, out_psjit, self.loss, self.loss)

    def forward_mindspore_dynamic_shape_impl(self, net):
        input_x = Tensor(self.input_x_np)
        input_dyn = Tensor(shape=[None for _ in input_x.shape], dtype=input_x.dtype)
        net.set_inputs(input_dyn)
        out = net(input_x)
        return out.asnumpy()

    def forward_dynamic_shape_cmp(self):
        ps_net = Sin()
        jit(ps_net.construct, mode="PSJit")(self.input_x)
        context.set_context(mode=context.GRAPH_MODE)
        out_psjit = self.forward_mindspore_dynamic_shape_impl(ps_net)
        pi_net = Sin()
        jit(pi_net.construct, mode="PIJit")(self.input_x)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pijit = self.forward_mindspore_dynamic_shape_impl(pi_net)

        if self.dtype == np.complex64 or np.complex128:
            ps_real = np.real(out_psjit)
            pi_real = np.real(out_pijit)
            ps_imag = np.imag(out_psjit)
            pi_imag = np.imag(out_pijit)
            allclose_nparray(pi_real, ps_real, self.loss, self.loss)
            allclose_nparray(pi_imag, ps_imag, self.loss, self.loss)
        else:
            allclose_nparray(out_pijit, out_psjit, self.loss, self.loss)

    def grad_cmp(self):
        ps_net = Sin()
        jit(ps_net.construct, mode="PSJit")(self.input_x)
        context.set_context(mode=context.GRAPH_MODE)
        input_grad_psjit = self.grad_mindspore_impl(ps_net)
        pi_net = Sin()
        jit(pi_net.construct, mode="PIJit")(self.input_x)
        context.set_context(mode=context.PYNATIVE_MODE)
        input_grad_pijit = self.grad_mindspore_impl(pi_net)

        if self.dtype == np.complex64 or np.complex128:
            ps_real = np.real(input_grad_psjit)
            pi_real = np.real(input_grad_pijit)
            ps_imag = np.imag(input_grad_psjit)
            pi_imag = np.imag(input_grad_pijit)
            allclose_nparray(pi_real, ps_real, self.loss, self.loss)
            allclose_nparray(pi_imag, ps_imag, self.loss, self.loss)
        else:
            allclose_nparray(input_grad_pijit, input_grad_psjit, self.loss, self.loss)

    def grad_mindspore_dynamic_shape_impl(self, net):
        sin_net = Sin()
        if self.output_grad_np is None:
            self.output_grad_np = np.random.randn(*self.forward_mindspore_impl(sin_net).shape).astype(self.dtype)

        output_grad = Tensor(self.output_grad_np)
        input_x = Tensor(self.input_x_np)
        net = Sin()
        grad_net = GradOfFirstInput(net)
        input_dyn = Tensor(shape=[None for _ in input_x.shape], dtype=input_x.dtype)
        output_grad_dyn = Tensor(shape=[None for _ in output_grad.shape], dtype=output_grad.dtype)
        grad_net.set_inputs(input_dyn, output_grad_dyn)
        grad_net.set_train()
        out_grad = grad_net(input_x, output_grad)
        return out_grad.asnumpy()

    def grad_dynamic_shape_cmp(self):
        ps_net = Sin()
        jit(ps_net.construct, mode="PSJit")(self.input_x)
        context.set_context(mode=context.GRAPH_MODE)
        input_grad_psjit = self.grad_mindspore_dynamic_shape_impl(ps_net)
        pi_net = Sin()
        jit(pi_net.construct, mode="PIJit")(self.input_x)
        context.set_context(mode=context.PYNATIVE_MODE)
        input_grad_pijit = self.grad_mindspore_dynamic_shape_impl(pi_net)

        if self.dtype == np.complex64 or np.complex128:
            ps_real = np.real(input_grad_psjit)
            pi_real = np.real(input_grad_pijit)
            ps_imag = np.imag(input_grad_psjit)
            pi_imag = np.imag(input_grad_pijit)
            allclose_nparray(pi_real, ps_real, self.loss, self.loss)
            allclose_nparray(pi_imag, ps_imag, self.loss, self.loss)
        else:
            allclose_nparray(input_grad_pijit, input_grad_psjit, self.loss, self.loss)

    def forward_mindspore_tensor_impl(self, net):
        output = net(self.input_x)
        return output.asnumpy()

    def forward_tensor_cmp(self):
        ps_net = SinTensorNet()
        jit(ps_net.construct, mode="PSJit")(self.input_x)
        context.set_context(mode=context.GRAPH_MODE)
        out_psjit = self.forward_mindspore_tensor_impl(ps_net)
        pi_net = SinTensorNet()
        jit(pi_net.construct, mode="PIJit")(self.input_x)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pijit = self.forward_mindspore_tensor_impl(pi_net)
        allclose_nparray(out_pijit, out_psjit, self.loss, self.loss)
