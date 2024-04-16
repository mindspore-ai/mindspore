import numpy as np
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.ops.operations.math_ops import Median
from mindspore import Tensor, jit, context
import mindspore.numpy as ms_np
from ..share.utils import allclose_nparray


class MedianDynamicRankNet(Cell):
    def __init__(self, global_median=False, axis=0, keep_dims=False):
        super().__init__()
        self.global_median = global_median
        self.axis = axis
        self.keep_dims = keep_dims
        self.median = Median(self.global_median, self.axis, self.keep_dims)
        self.reducesum = P.ReduceSum(keep_dims=False)

    def construct(self, input_x):
        n = 8
        axis = ms_np.randint(1, n, (n - 1,))
        random_axis = ms_np.unique(axis)
        x = self.reducesum(input_x, random_axis)
        ms_y, ms_indices = self.median(x)
        return ms_y, ms_indices

class MedianDynamicRankFactory():
    def __init__(self, input_x_shape, global_median, axis, keep_dims, dtype, loss=0.0001):
        if dtype in [np.int16, np.int32, np.int64]:
            self.input_x_np = np.random.choice(100000, size=input_x_shape,
                                               replace=False).astype(dtype=dtype)
        else:
            self.input_x_np = np.random.randn(*input_x_shape).astype(dtype=dtype)
        self.global_median = global_median
        self.axis = axis
        self.keep_dims = keep_dims
        self.dtype = dtype
        self.loss = loss
        self.output_grad_np = None
        self.input_x_ms = Tensor(self.input_x_np)

    def forward_mindspore_impl(self, net):
        ms_y, ms_indices = net(self.input_x_ms)
        return ms_y.asnumpy(), ms_indices.asnumpy()


    def forward_cmp(self):
        """正向比较"""
        ps_net = MedianDynamicRankNet(global_median=self.global_median, axis=self.axis,
                                      keep_dims=self.keep_dims)
        jit(ps_net.construct, mode="PSJit")(self.input_x_ms)
        context.set_context(mode=context.GRAPH_MODE)
        psjit_y, _ = self.forward_mindspore_impl(ps_net)
        pi_net = MedianDynamicRankNet(global_median=self.global_median, axis=self.axis,
                                      keep_dims=self.keep_dims)
        jit(pi_net.construct, mode="PIJit")(self.input_x_ms)
        context.set_context(mode=context.PYNATIVE_MODE)
        pijit_y, _ = self.forward_mindspore_impl(pi_net)

        allclose_nparray(psjit_y, pijit_y, self.loss, self.loss)
