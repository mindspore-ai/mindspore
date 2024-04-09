import numpy as np
from mindspore.common.tensor import Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as op
import mindspore
from ...utils import tensor_to_numpy, is_empty
from ...grad import GradOfFirstInput


class DType(Cell):
    def __init__(self):
        super().__init__()
        self.dtype = op.DType()

    def construct(self, x):
        return self.dtype(x)


class DTypeFactory():
    def __init__(self, input_shape, dtype=np.float16, input_x=None):
        if not is_empty(input_shape):
            self.input_np = np.random.randn(*input_shape).astype(dtype=dtype)
        else:
            self.input_np = input_x
        self.dtype = dtype

    def forward_mindspore_impl(self, net):
        input_ms = Tensor(self.input_np)
        out = net(input_ms)
        return out

    def grad_mindspore_impl(self, net):
        input_ms = Tensor(self.input_np)
        grad_net = GradOfFirstInput(net, sens_param=False)
        grad_net.set_train()
        input_grad = grad_net(input_ms)
        return input_grad.asnumpy()

    def forward_cmp(self, net):
        out_mindspore = mindspore.dtype_to_nptype(self.forward_mindspore_impl(net))
        assert out_mindspore == self.dtype

    def grad_cmp(self, net):
        input_grad_mindspore = self.grad_mindspore_impl(net)
        input_grad_np = np.zeros_like(self.input_np)
        assert input_grad_mindspore.all() == input_grad_np.all()


class DTypeMock(DTypeFactory):
    def __init__(self, inputs=None, grads=None):
        input_x = inputs[0]
        input_shape = input_x.shape
        input_np = tensor_to_numpy(input_x)
        dtype = input_np.dtype
        DTypeFactory.__init__(self, input_shape=input_shape, dtype=dtype, input_x=input_np)
        self.input_np = input_np
        if is_empty(grads):
            self.output_grad_np = None
        else:
            self.output_grad_np = grads[0].asnumpy()
