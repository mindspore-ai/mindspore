import numpy as np
import mindspore.ops.operations as ops
from mindspore import ms_function
from mindspore.common.tensor import Tensor
from mindspore.nn import Cell
from mindspore.common import dtype_to_nptype
from ...utils import allclose_nparray, is_empty
from ...grad import HighGrad
from ...grad import GradOfFirstInput
from ...grad import GradOfAllInputs
from mindspore import jit, context


class MaxPool(Cell):
    def __init__(self, pad_mode, kernel_size, strides):
        super().__init__()
        self.maxpool = ops.MaxPool(pad_mode=pad_mode,
                                   kernel_size=kernel_size,
                                   strides=strides)

    def construct(self, x):
        return self.maxpool(x)


class MaxPoolMock():
    def __init__(self, attributes=None, inputs=None, grads=None):
        self.ms_type = inputs[0][0].dtype
        self.dtype = dtype_to_nptype(self.ms_type)
        self.pad_mode = attributes.get('pad_mode')
        self.kernel_size = attributes.get('kernel_size')
        self.strides = attributes.get('strides')
        self.data_format = attributes.get('data_format', "NCHW")
        self.input_ms = []
        self.input_x_np = []
        self.input_x = inputs[0]
        self.input_x_np.append(inputs[0][0].asnumpy().transpose(0, 2, 3, 1))
        if self.dtype == np.float16:
            self.loss = 1e-3
        elif self.dtype in (np.float32, np.complex64):
            self.loss = 1e-4
        elif self.dtype in (np.float64, np.complex128):
            self.loss = 1e-5
        else:
            self.loss = 0

        if is_empty(grads):
            self.out_grad_np = None
        else:
            self.out_grad_np = grads[0].asnumpy()

        if isinstance(self.kernel_size, tuple):
            if len(self.kernel_size) == 4:
                self.tf_kernel_size = [1, self.kernel_size[1], self.kernel_size[3], 1]
            if len(self.kernel_size) == 2:
                self.tf_kernel_size = [1, self.kernel_size[0], self.kernel_size[1], 1]
            else:
                self.tf_kernel_size = list(self.kernel_size)
        else:
            self.tf_kernel_size = [1, self.kernel_size, self.kernel_size, 1]
        if isinstance(self.strides, tuple):
            if len(self.strides) == 4:
                self.tf_strides = [1, self.strides[1], self.strides[3], 1]
            if len(self.strides) == 2:
                self.tf_strides = [1, self.strides[0], self.strides[1], 1]
            else:
                self.tf_strides = list(self.strides)
        else:
            self.tf_strides = [1, self.strides, self.strides, 1]


    def grad_mindspore_impl(self, net):
        return net(self.input_x[0]).asnumpy()


    def forward_cmp(self):
        ps_net = MaxPool(pad_mode=self.pad_mode, kernel_size=self.kernel_size, strides=self.strides)
        jit(ps_net.construct, mode="PSJit")
        context.set_context(mode=context.GRAPH_MODE)
        out_psjit = ps_net(self.input_x[0]).asnumpy()
        pi_net = MaxPool(pad_mode=self.pad_mode, kernel_size=self.kernel_size, strides=self.strides)
        jit(pi_net.construct, mode="PIJit")
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pijit = ps_net(self.input_x[0]).asnumpy()
        allclose_nparray(out_pijit, out_psjit, self.loss, self.loss)

    def grad_cmp(self):
        ps_net = MaxPool(pad_mode=self.pad_mode, kernel_size=self.kernel_size, strides=self.strides)
        jit(ps_net.construct, mode="PSJit")
        context.set_context(mode=context.GRAPH_MODE)
        input_grad_psjit = self.grad_mindspore_impl(ps_net)
        pi_net = MaxPool(pad_mode=self.pad_mode, kernel_size=self.kernel_size, strides=self.strides)
        jit(pi_net.construct, mode="PIJit")
        context.set_context(mode=context.PYNATIVE_MODE)
        input_grad_pijit = self.grad_mindspore_impl(pi_net)
        for index in range(0, len(input_grad_pijit)):
            allclose_nparray(input_grad_pijit[index],
                             input_grad_psjit[index],
                             self.loss, self.loss)

    def highgrad_mindspore_impl(self):
        net = MaxPool(pad_mode=self.pad_mode, kernel_size=self.kernel_size, strides=self.strides)
        grad_net = HighGrad(net, grad_list=[GradOfFirstInput, GradOfFirstInput])
        out = grad_net(self.input_x[0])
        return out.asnumpy()


    def highgrad_cmp(self):
        ps_net = MaxPool(pad_mode=self.pad_mode, kernel_size=self.kernel_size, strides=self.strides)
        jit(ps_net.construct, mode="PSJit")
        context.set_context(mode=context.GRAPH_MODE)
        grad_psnet = HighGrad(ps_net, grad_list=[GradOfFirstInput, GradOfFirstInput])
        out_psjit = grad_psnet(self.input_x[0]).asnumpy()

        pi_net = MaxPool(pad_mode=self.pad_mode, kernel_size=self.kernel_size, strides=self.strides)
        jit(ps_net.construct, mode="PIJit")
        context.set_context(mode=context.PYNATIVE_MODE)
        grad_pinet = HighGrad(pi_net, grad_list=[GradOfFirstInput, GradOfFirstInput])
        out_pijit = grad_pinet(self.input_x[0]).asnumpy()

        allclose_nparray(out_pijit, out_psjit, self.loss, self.loss)

    @ms_function
    def foreach_vmap(self, *inputs):
        out_vmap = []
        nest_axis_size = inputs[0].shape[-1]
        for i in range(nest_axis_size):
            inner_args = []
            for arg in inputs:
                inner_args.append(arg[..., i])
            out = self.pure_net_vmap(*inner_args)
            out_vmap.append(out[0])
        return [ops.Stack()(out_vmap)]

    def grad_vmap_cmp(self, batch_size, in_axes=-1, run_time=100, improve_times=2.0):
        if self.out_grad_np is None:
            self.out_grad_np = self.forward_mindspore_impl()
        net = MaxPool(pad_mode=self.pad_mode, kernel_size=self.kernel_size, strides=self.strides)
        grad_net = GradOfAllInputs(net)
        grad_net.set_train()
        stack = ops.Stack(axis=-1)
        stack_input = stack([stack([self.input_x[0]] * batch_size)] * batch_size)
        stack_grad = ops.Stack(axis=in_axes)([ops.Stack(axis=in_axes)([Tensor(self.out_grad_np).astype(self.ms_type)] *
                                                                      batch_size)] * batch_size)
        inputs = [stack_input, stack_grad]
        self.vmap_cmp(grad_net, in_axes, run_time, improve_times, *inputs)
