import numpy as np
from mindspore.nn import Cell
import mindspore.ops.operations as op
from mindspore import Tensor, jit, context
from ...utils import allclose_nparray
from ...utils import tensor_to_numpy, nptype_to_mstype
from ...grad import GradOfAllInputs


class Minimum(Cell):
    def __init__(self):
        super().__init__()
        self.min = op.Minimum()

    def construct(self, left_input, right_input):
        return self.min(left_input, right_input)


class MinimumFactory():
    def __init__(self, left_input, right_input, dtype=np.float32, grad=None, ms_type=None):
        self.left_input = left_input
        self.right_input = right_input
        self.out_grad_np = grad
        self.ms_type = ms_type
        self.dtype = dtype
        self.ms_dtype = self.dtype
        if self.dtype == np.float16:
            self.loss = 1e-3
        elif self.dtype in (np.float32, np.complex64):
            self.loss = 1e-4
        elif self.dtype in (np.float64, np.complex128):
            self.loss = 1e-5
        else:
            self.loss = 0
        if isinstance(self.left_input, float):
            self.left_input_ms = Tensor(self.left_input, dtype=self.ms_dtype)
        else:
            self.left_input_ms = Tensor(self.left_input)
        if isinstance(self.right_input, float):
            self.right_input_ms = Tensor(self.right_input, dtype=self.ms_dtype)
        else:
            self.right_input_ms = Tensor(self.right_input)

    def forward_mindspore_impl(self, net):
        out = net(self.left_input_ms, self.right_input_ms)
        return out.asnumpy()

    def grad_mindspore_impl(self, net):
        out_grad_me = Tensor(self.out_grad_np, dtype=nptype_to_mstype(self.ms_dtype))
        net_me = GradOfAllInputs(net)
        net_me.set_train()
        input_grad = net_me(self.left_input_ms, self.right_input_ms, out_grad_me)
        return input_grad[0].asnumpy(), input_grad[1].asnumpy()

    def forward_cmp(self):
        ps_net = Minimum()
        jit(ps_net.construct, mode="PSJit")(self.left_input_ms, self.right_input_ms)
        context.set_context(mode=context.GRAPH_MODE)
        out_psjit = self.forward_mindspore_impl(ps_net)
        pi_net = Minimum()
        jit(pi_net.construct, mode="PIJit")(self.left_input_ms, self.right_input_ms)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pijit = self.forward_mindspore_impl(pi_net)
        allclose_nparray(out_pijit, out_psjit, self.loss, self.loss)

    def grad_cmp(self):
        ps_net = Minimum()
        jit(ps_net.construct, mode="PSJit")(self.left_input_ms, self.right_input_ms)
        context.set_context(mode=context.GRAPH_MODE)
        output_grad_psjit = self.grad_mindspore_impl(ps_net)
        pi_net = Minimum()
        jit(pi_net.construct, mode="PIJit")(self.left_input_ms, self.right_input_ms)
        context.set_context(mode=context.PYNATIVE_MODE)
        output_grad_pijit = self.grad_mindspore_impl(pi_net)

        allclose_nparray(output_grad_pijit[0], output_grad_psjit[0], self.loss, self.loss)
        allclose_nparray(output_grad_pijit[1], output_grad_psjit[1], self.loss, self.loss)


class MinimumMock(MinimumFactory):
    def __init__(self, inputs=None, grads=None):
        if grads is not None:
            grad = grads[0]
            grad_np = tensor_to_numpy(grad)
        ms_type = None

        left_input = tensor_to_numpy(inputs[0])
        right_input = tensor_to_numpy(inputs[1])
        dtype = left_input.dtype
        MinimumFactory.__init__(self, left_input, right_input, dtype, grad_np, ms_type=ms_type)
