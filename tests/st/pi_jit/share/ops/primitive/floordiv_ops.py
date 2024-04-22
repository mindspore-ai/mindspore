import numpy as np
from mindspore.common.tensor import Tensor
from mindspore.nn import Cell
from mindspore import jit, context
import mindspore.ops.operations as op
from ...utils import allclose_nparray, is_empty
from ...grad import GradOfAllInputs


class FloorDiv(Cell):
    def __init__(self):
        super().__init__()
        self.op = op.FloorDiv()

    def construct(self, left_input, right_input):
        return self.op(left_input, right_input)


class FloorDivMock():
    def __init__(self, inputs=None, grads=None):
        self.ms_type = inputs[0].dtype
        self.input_x1 = inputs[0]
        self.input_x2 = inputs[1]
        self.left_input_np = self.input_x1.asnumpy()
        self.right_input_np = self.input_x2.asnumpy()
        if self.ms_type == np.float16:
            self.loss = 1e-3
        elif self.ms_type in (np.float32, np.complex64):
            self.loss = 1e-4
        elif self.ms_type in (np.float64, np.complex128):
            self.loss = 1e-5
        else:
            self.loss = 0
        if is_empty(grads):
            self.out_grad_np = \
                np.random.randn(*self.input_x1.shape).astype(self.left_input_np.dtype)
        else:
            self.out_grad_np = grads.asnumpy()

    def forward_mindspore_impl(self, net):
        left_input = self.input_x1
        right_input = self.input_x2
        out = net(left_input, right_input)
        return out.asnumpy()


    def grad_mindspore_impl(self, net):
        left_input = Tensor(self.left_input_np)
        right_input = Tensor(self.right_input_np)
        out_grad = Tensor(self.out_grad_np)
        grad_net = GradOfAllInputs(net)
        grad_net.set_train()
        input_grad = grad_net(left_input, right_input, out_grad)
        return input_grad[0].asnumpy(), input_grad[1].asnumpy()


    def forward_cmp(self):
        left_input = self.input_x1
        right_input = self.input_x2
        ps_net = FloorDiv()
        jit(ps_net.construct, mode="PSJit")(left_input, right_input)
        context.set_context(mode=context.GRAPH_MODE)
        out_psjit = self.forward_mindspore_impl(ps_net)
        pi_net = FloorDiv()
        jit(pi_net.construct, mode="PIJit")(left_input, right_input)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pijit = self.forward_mindspore_impl(pi_net)
        allclose_nparray(out_pijit, out_psjit, self.loss, self.loss)

    def grad_cmp(self):
        left_input = Tensor(self.left_input_np)
        right_input = Tensor(self.right_input_np)
        ps_net = FloorDiv()
        jit(ps_net.construct, mode="PSJit")(left_input, right_input)
        context.set_context(mode=context.GRAPH_MODE)
        input_grad_psjit = self.grad_mindspore_impl(ps_net)
        pi_net = FloorDiv()
        jit(pi_net.construct, mode="PIJit")(left_input, right_input)
        context.set_context(mode=context.PYNATIVE_MODE)
        input_grad_pijit = self.grad_mindspore_impl(pi_net)
        allclose_nparray(input_grad_pijit[0], input_grad_psjit[0], self.loss, self.loss)
        allclose_nparray(input_grad_pijit[1], input_grad_psjit[1], self.loss, self.loss)

    def forward_mindspore_dynamic_shape_impl(self, net):
        input_x1_dyn = Tensor(shape=[None for _ in self.input_x1.shape], dtype=self.input_x1.dtype)
        input_x2_dyn = Tensor(shape=[None for _ in self.input_x2.shape], dtype=self.input_x2.dtype)
        net.set_inputs(input_x1_dyn, input_x2_dyn)
        out = net(self.input_x1, self.input_x2)
        return out.asnumpy()

    def grad_mindspore_dynamic_shape_impl(self):
        out_grad = Tensor(self.out_grad_np)
        net = FloorDiv()
        grad_net = GradOfAllInputs(net)
        grad_net.set_train()
        input_x1_dyn = Tensor(shape=[None for _ in self.input_x1.shape], dtype=self.input_x1.dtype)
        input_x2_dyn = Tensor(shape=[None for _ in self.input_x2.shape], dtype=self.input_x2.dtype)
        grad_net.set_inputs(input_x1_dyn, input_x2_dyn, out_grad)
        input_grad = grad_net(self.input_x1, self.input_x2, out_grad)
        return input_grad[0].asnumpy(), input_grad[1].asnumpy()

    def forward_dynamic_shape_cmp(self):
        left_input = self.input_x1
        right_input = self.input_x2
        ps_net = FloorDiv()
        jit(ps_net.construct, mode="PSJit")(left_input, right_input)
        context.set_context(mode=context.GRAPH_MODE)
        out_psjit = self.forward_mindspore_dynamic_shape_impl(ps_net)
        pi_net = FloorDiv()
        jit(pi_net.construct, mode="PIJit")(left_input, right_input)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pijit = self.forward_mindspore_dynamic_shape_impl(pi_net)
        allclose_nparray(out_pijit, out_psjit, self.loss, self.loss)

    def grad_dynamic_shape_cmp(self):
        left_input = Tensor(self.left_input_np)
        right_input = Tensor(self.right_input_np)
        ps_net = FloorDiv()
        jit(ps_net.construct, mode="PSJit")(left_input, right_input)
        context.set_context(mode=context.GRAPH_MODE)
        input_grad_psjit = self.grad_mindspore_impl(ps_net)
        pi_net = FloorDiv()
        jit(pi_net.construct, mode="PIJit")(left_input, right_input)
        context.set_context(mode=context.PYNATIVE_MODE)
        input_grad_pijit = self.grad_mindspore_impl(pi_net)
        allclose_nparray(input_grad_pijit[0], input_grad_psjit[0], self.loss, self.loss)
        allclose_nparray(input_grad_pijit[1], input_grad_psjit[1], self.loss, self.loss)


class FloorDivFactory(FloorDivMock):
    def __init__(self, input_shape, dtype=np.float16):
        self.dtype = dtype
        self.left_input_np = np.random.randn(*input_shape).astype(dtype)
        self.right_input_np = np.random.uniform(1, 100, input_shape).astype(dtype)
        self.out_grad_np = np.random.randn(*input_shape).astype(dtype)
        FloorDivMock.__init__(self, inputs=[Tensor(self.left_input_np), Tensor(self.right_input_np)],
                              grads=Tensor(self.out_grad_np))
