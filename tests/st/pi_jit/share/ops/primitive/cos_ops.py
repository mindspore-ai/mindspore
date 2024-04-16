from mindspore.nn import Cell
from mindspore import jit, context
import mindspore.ops.operations as op
from ...grad import GradOfFirstInput
from ...utils import allclose_nparray, is_empty
from mindspore import Tensor
import numpy as np


class Cos(Cell):
    def __init__(self):
        super().__init__()
        self.cos = op.Cos()

    def construct(self, input_x):
        return self.cos(input_x)

class CosMock():
    def __init__(self, inputs=None, grads=None):
        self.dtype = inputs[0].dtype

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

    def cmp_complex(self, ps_res, pi_res):
        if self.dtype == np.complex64 or np.complex128:
            ps_real = np.real(ps_res)
            pi_real = np.real(pi_res)
            ps_imag = np.imag(ps_res)
            pi_imag = np.imag(pi_res)
            allclose_nparray(pi_real, ps_real, self.loss, self.loss)
            allclose_nparray(pi_imag, ps_imag, self.loss, self.loss)
        else:
            allclose_nparray(pi_res, ps_res, self.loss, self.loss)

    def forward_mindspore_impl(self, net):
        out = net(self.input_x)
        return out.asnumpy()


    def forward_cmp(self):
        ps_net = Cos()
        jit(ps_net.construct, mode="PSJit")(self.input_x)
        context.set_context(mode=context.GRAPH_MODE)
        out_psjit = self.forward_mindspore_impl(ps_net)

        pi_net = Cos()
        jit(pi_net.construct, mode="PIJit")(self.input_x)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pijit = self.forward_mindspore_impl(pi_net)

        self.cmp_complex(out_psjit, out_pijit)

    def forward_mindspore_dynamic_shape_impl(self, net):
        input_x = Tensor(self.input_x_np)
        input_dyn = Tensor(shape=[None for _ in input_x.shape], dtype=input_x.dtype)
        net.set_inputs(input_dyn)
        out = net(input_x)
        return out.asnumpy()

    def forward_dynamic_shape_cmp(self):
        ps_net = Cos()
        jit(ps_net.construct, mode="PSJit")(self.input_x)
        context.set_context(mode=context.GRAPH_MODE)
        out_psjit = self.forward_mindspore_dynamic_shape_impl(ps_net)

        pi_net = Cos()
        jit(pi_net.construct, mode="PIJit")(self.input_x)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pijit = self.forward_mindspore_dynamic_shape_impl(pi_net)

        self.cmp_complex(out_psjit, out_pijit)


    def grad_mindspore_dynamic_shape_impl(self, net):
        input_x = Tensor(self.input_x_np)
        output_grad = Tensor(self.output_grad_np)
        grad_net = GradOfFirstInput(net)
        input_dyn = Tensor(shape=[None for _ in input_x.shape], dtype=input_x.dtype)
        output_grad_dyn = Tensor(shape=[None for _ in output_grad.shape], dtype=output_grad.dtype)
        grad_net.set_inputs(input_dyn, output_grad_dyn)
        grad_net.set_train()
        out_grad = grad_net(input_x, output_grad)
        return out_grad.asnumpy()

    def grad_dynamic_shape_cmp(self):
        ps_net = Cos()
        jit(ps_net.construct, mode="PSJit")(self.input_x)
        context.set_context(mode=context.GRAPH_MODE)
        input_grad_psjit = self.grad_mindspore_dynamic_shape_impl(ps_net)

        pi_net = Cos()
        jit(pi_net.construct, mode="PIJit")(self.input_x)
        context.set_context(mode=context.PYNATIVE_MODE)
        input_grad_pijit = self.grad_mindspore_dynamic_shape_impl(pi_net)

        self.cmp_complex(input_grad_psjit, input_grad_pijit)
