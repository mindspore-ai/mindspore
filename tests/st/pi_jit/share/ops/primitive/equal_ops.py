from mindspore import Tensor, jit, context
from mindspore.nn import Cell
import mindspore.ops.operations as op
from ...utils import tensor_to_numpy
from ...utils import allclose_nparray
from ...grad import GradOfAllInputs
import numpy as np


class Equal(Cell):
    def __init__(self):
        super().__init__()
        self.equal = op.Equal()

    def construct(self, left_input, right_input):
        return self.equal(left_input, right_input)


class EqualFactory():
    def __init__(self, input_shape, dtype=np.float16):
        if dtype == np.uint32:
            self.left_input_np = np.random.randint(1, 100, size=(input_shape)).astype(dtype)
            self.right_input_np = np.random.randint(1, 100, size=(input_shape)).astype(dtype)
            self.out_grad_np = np.random.randint(1, 100, size=(input_shape)).astype(dtype)
        else:
            self.left_input_np = np.random.randn(*input_shape).astype(dtype)
            self.right_input_np = np.random.randn(*input_shape).astype(dtype)
            self.out_grad_np = np.random.randn(*input_shape).astype(dtype)

        self.dtype = dtype
        if self.dtype == np.float16:
            self.loss = 1e-3
        elif self.dtype in (np.float32, np.complex64):
            self.loss = 1e-4
        elif self.dtype in (np.float64, np.complex128):
            self.loss = 1e-5
        else:
            self.loss = 0
        self.left_input = Tensor(self.left_input_np)
        self.right_input = Tensor(self.right_input_np)

    def forward_mindspore_impl(self, net):
        out = net(self.left_input, self.right_input)
        return out.asnumpy()


    def forward_numpy_impl(self):
        out = np.equal(self.left_input_np, self.right_input_np, dtype=self.dtype)
        return out

    def grad_mindspore_impl(self, net):
        out_grad = Tensor(self.out_grad_np)
        grad_net = GradOfAllInputs(net)
        grad_net.set_train()
        input_grad = grad_net(self.left_input, self.right_input, out_grad)
        return input_grad[0].asnumpy(), input_grad[1].asnumpy()


    def forward_cmp(self):
        ps_net = Equal()
        jit(ps_net.construct, mode="PSJit")(self.left_input, self.right_input)
        context.set_context(mode=context.GRAPH_MODE)
        out_psjit = self.forward_mindspore_impl(ps_net)
        pi_net = Equal()
        jit(pi_net.construct, mode="PIJit")(self.left_input, self.right_input)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pijit = self.forward_mindspore_impl(ps_net)
        allclose_nparray(out_pijit, out_psjit, self.loss, self.loss)

    def grad_cmp(self):
        ps_net = Equal()
        jit(ps_net.construct, mode="PSJit")(self.left_input, self.right_input)
        context.set_context(mode=context.GRAPH_MODE)
        out_psjit = self.grad_mindspore_impl(ps_net)
        pi_net = Equal()
        jit(pi_net.construct, mode="PIJit")(self.left_input, self.right_input)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pijit = self.grad_mindspore_impl(pi_net)
        allclose_nparray(out_pijit, out_psjit, self.loss, self.loss)


class EqualMock(EqualFactory):
    def __init__(self, inputs=None, grads=None):
        input_x = inputs[0]
        input_y = inputs[1]
        input_shape = tuple()
        if isinstance(input_x, Tensor):
            input_x = tensor_to_numpy(input_x)
            input_shape = input_x.shape
        if isinstance(input_y, Tensor):
            input_y = tensor_to_numpy(input_y)
        EqualFactory.__init__(self, input_shape=input_shape)
        self.left_input_np = input_x
        self.right_input_np = input_y
        if grads is not None:
            self.out_grad_np = grads[0]
