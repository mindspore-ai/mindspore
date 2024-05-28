import numpy as np
from mindspore import jit, context
from mindspore.common.tensor import Tensor
from mindspore.nn import Cell
import mindspore.ops.operations as op
from ...utils import allclose_nparray, is_empty
from ...utils import tensor_to_numpy, nptype_to_mstype
from ...grad import GradOfAllInputs


class Maximum(Cell):
    def __init__(self):
        super().__init__()
        self.max = op.Maximum()

    def construct(self, left_input, right_input):
        return self.max(left_input, right_input)


class MaximumFactory():
    def __init__(self, left_input, right_input, dtype=np.float32, ms_type=None):
        self.left_input = left_input
        self.right_input = right_input
        self.out_grad_np = None
        self.dtype = dtype
        self.ms_dtype = self.dtype
        self.ms_type = ms_type
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
        ps_net = Maximum()
        jit(ps_net.construct, mode="PSJit")(self.left_input_ms, self.right_input_ms)
        context.set_context(mode=context.GRAPH_MODE)
        out_psjit = self.forward_mindspore_impl(ps_net)
        pi_net = Maximum()
        jit(pi_net.construct, mode="PIJit")(self.left_input_ms, self.right_input_ms)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pijit = self.forward_mindspore_impl(pi_net)
        allclose_nparray(out_pijit, out_psjit, self.loss, self.loss)

    def grad_cmp(self):
        ps_net = Maximum()
        jit(ps_net.construct, mode="PSJit")(self.left_input_ms, self.right_input_ms)
        context.set_context(mode=context.GRAPH_MODE)
        output_grad_psjit = self.grad_mindspore_impl(ps_net)
        pi_net = Maximum()
        jit(pi_net.construct, mode="PIJit")(self.left_input_ms, self.right_input_ms)
        context.set_context(mode=context.PYNATIVE_MODE)
        output_grad_pijit = self.grad_mindspore_impl(pi_net)
        allclose_nparray(output_grad_pijit[0], output_grad_psjit[0], self.loss, self.loss)
        allclose_nparray(output_grad_pijit[1], output_grad_psjit[1], self.loss, self.loss)


class MaximumMock(MaximumFactory):
    def __init__(self, inputs=None, grads=None):
        input_x = inputs[0]
        input_y = inputs[1]
        ms_type = None
        left_input = tensor_to_numpy(input_x)
        right_input = tensor_to_numpy(input_y)
        dtype = left_input.dtype
        MaximumFactory.__init__(self, left_input=left_input, right_input=right_input, dtype=dtype, ms_type=ms_type)
        self.left_input = left_input
        self.right_input = right_input
        if is_empty(grads):
            self.out_grad_np = None
        else:
            self.out_grad_np = grads[0].asnumpy()
