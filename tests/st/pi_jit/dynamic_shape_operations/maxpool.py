import numpy as np
import mindspore
from mindspore.nn import Cell
from mindspore import Tensor, jit, context
from mindspore.ops import operations as P
from ..share.utils import allclose_nparray
from ..share.grad import GradOfFirstInput



class DynamicRankMaxPoolNet(Cell):
    def __init__(self, pad_mode="SAME", kernel_size=3, strides=2, data_format='NCHW'):
        super(DynamicRankMaxPoolNet, self).__init__()
        self.unique = P.Unique()
        self.reducesum = P.ReduceSum(keep_dims=False)
        self.maxpool = P.MaxPool(pad_mode=pad_mode, kernel_size=kernel_size,
                                 strides=strides, data_format=data_format)

    def construct(self, x, indices):
        unique_indices, _ = self.unique(indices)
        x = self.reducesum(x, unique_indices)
        output = self.maxpool(x)
        return output


class DynamicRankMaxPoolMock():
    def __init__(self, input_x, indices, dtype=np.float32, kernel_size=3, stride=2,
                 data_format='NCHW', pad_mode='SAME', loss=0.001):
        self.input_x = (input_x).astype(dtype)
        self.indices = indices
        self.dtype = dtype
        self.ksize = kernel_size
        self.data_format = data_format
        self.stride = stride
        self.pad_mode = pad_mode
        self.loss = loss
        self.out_grad_np = None
        self.input_me_x = Tensor(self.input_x)
        self.input_me_indices = Tensor(self.indices, mindspore.int64)

    def forward_mindspore_impl(self, net):
        tmp_input = Tensor(shape=[None for _ in self.input_me_x.shape], dtype=self.input_me_x.dtype)
        net.set_inputs(tmp_input, self.input_me_indices)
        out = net(self.input_me_x, self.input_me_indices)
        return out.asnumpy()


    def forward_cmp(self):
        ps_net = DynamicRankMaxPoolNet(self.pad_mode, self.ksize, self.stride, self.data_format)
        jit(ps_net.construct, mode="PSJit")(self.input_me_x, self.input_me_indices)
        context.set_context(mode=context.GRAPH_MODE)
        out_psjit = self.forward_mindspore_impl(ps_net)
        pi_net = DynamicRankMaxPoolNet(self.pad_mode, self.ksize, self.stride, self.data_format)
        jit(pi_net.construct, mode="PIJit")(self.input_me_x, self.input_me_indices)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pijit = self.forward_mindspore_impl(pi_net)
        allclose_nparray(out_pijit, out_psjit, self.loss, self.loss)

    def grad_mindspore_impl(self, net):
        if self.data_format == 'NCHW':
            self.out_grad_np = self.out_grad_np.transpose(0, 3, 1, 2)
        grad_net = GradOfFirstInput(net)
        grad_net.set_train()
        grad_input0 = Tensor(shape=[None for _ in self.input_me_x.shape], dtype=self.input_me_x.dtype)
        grad_input1 = Tensor(shape=[None for _ in Tensor(self.out_grad_np).shape],
                             dtype=input_me_x.dtype)
        grad_net.set_inputs(grad_input0, self.input_me_indices, grad_input1)
        input_grad = grad_net(self.input_me_x, self.input_me_indices, Tensor(self.out_grad_np))
        return input_grad.asnumpy()


    def grad_cmp(self):
        ps_net = DynamicRankMaxPoolNet(self.pad_mode, self.ksize, self.stride, self.data_format)
        jit(ps_net.construct, mode="PSJit")(self.input_me_x, self.input_me_indices)
        context.set_context(mode=context.GRAPH_MODE)
        out_psjit = self.grad_mindspore_impl(ps_net)
        pi_net = DynamicRankMaxPoolNet(self.pad_mode, self.ksize, self.stride, self.data_format)
        jit(pi_net.construct, mode="PIJit")(self.input_me_x, self.input_me_indices)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pijit = self.grad_mindspore_impl(pi_net)

        allclose_nparray(out_pijit, out_psjit, self.loss, self.loss)
