from ...utils import allclose_nparray
from mindspore import jit, context, Tensor, ops, common
from mindspore.nn import Cell
import numpy as np
from ...grad import GradOfAllInputs


class Range0(Cell):
    def __init__(self, maxlen=1000000):
        super().__init__()
        self.range = ops.operations.Range(maxlen)

    def construct(self, start, limit, delta):
        return self.range(start, limit, delta)


class OpsRangeFactory():
    def __init__(self, start, limit, delta, maxlen=1000000, dtype=np.int32):
        self.maxlen = maxlen
        self.start = start
        self.limit = limit
        self.delta = delta
        self.input_dtype = dtype
        self.input_dtype = common.dtype.pytype_to_dtype(dtype)
        self.out_grad_np = None
        if self.input_dtype == np.float16:
            self.loss = 1e-3
        elif self.input_dtype in (np.float32, np.complex64):
            self.loss = 1e-4
        elif self.input_dtype in (np.float64, np.complex128):
            self.loss = 1e-5
        else:
            self.loss = 0
        self.start_ms = Tensor(self.start, self.input_dtype)
        self.limit_ms = Tensor(self.limit, self.input_dtype)
        self.delta_ms = Tensor(self.delta, self.input_dtype)

    def forward_mindspore_impl(self, net):
        out = net(self.start_ms, self.limit_ms, self.delta_ms)
        return out.asnumpy()


    def forward_cmp(self):
        ps_net = Range0(self.maxlen)
        jit(ps_net.construct, mode="PSJit")(self.start_ms, self.limit_ms, self.delta_ms)
        context.set_context(mode=context.GRAPH_MODE)
        out_psjit = self.forward_mindspore_impl(ps_net)
        pi_net = Range0(self.maxlen)
        jit(pi_net.construct, mode="PIJit")(self.start_ms, self.limit_ms, self.delta_ms)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pijit = self.forward_mindspore_impl(pi_net)
        allclose_nparray(out_pijit, out_psjit, self.loss, self.loss)

    def grad_mindspore_impl(self):
        net = Range0(self.maxlen)
        if self.out_grad_np is None:
            out = self.forward_mindspore_impl(net)
            self.out_grad_np = np.random.randn(*list(out.shape)).astype(out.dtype)
        net = Range0(self.maxlen)
        grad_net = GradOfAllInputs(net)
        grad_net.set_train()
        input_grad = grad_net(self.start_ms, self.limit_ms, self.delta_ms, Tensor(self.out_grad_np))
        return input_grad[0].asnumpy()


    def grad_cmp(self):
        input_grad_mindspore = self.grad_mindspore_impl()
        # 标杆无反向，且mindspore返回为0.0 2022年11月24日
        assert input_grad_mindspore == 0.0
