from mindspore.nn import Cell
from mindspore import jit, context, Tensor, ops
from ...grad import GradOfAllInputsAndParams
from ...utils import allclose_nparray
from mindspore.common import dtype_to_nptype
import numpy as np


class Dense(Cell):
    def __init__(self):
        super().__init__()
        self.dense = ops.Dense()

    def construct(self, x, w, b):
        x = self.dense(x, w, b)
        return x


class DenseFactory():
    def __init__(self, input_shape, in_channel, out_channel, dtype=np.float16):
        self.dtype = dtype
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.x_np = np.random.randn(*input_shape).astype(dtype)
        if len(input_shape) > 1:
            self.w_np = np.random.randn(self.out_channel, self.in_channel).astype(dtype)
            self.b_np = np.random.randn(self.out_channel).astype(dtype)
        else:
            self.w_np = np.random.randn(self.x_np.shape[0]).astype(dtype)
            self.b_np = np.array(np.random.randn()).astype(dtype)
        self.x_ms = Tensor(self.x_np)
        self.w_ms = Tensor(self.w_np)
        self.b_ms = Tensor(self.b_np) if self.b_np is not None else None
        self.output_grad_np = None
        if self.dtype == np.float16:
            self.loss = 1e-3
        elif self.dtype in (np.float32, np.complex64):
            self.loss = 1e-4
        elif self.dtype in (np.float64, np.complex128):
            self.loss = 1e-5
        else:
            self.loss = 0

    def forward_mindspore_impl(self, net):
        out = net(self.x_ms, self.w_ms, self.b_ms)
        return out.asnumpy()

    def grad_mindspore_impl(self, net):
        if self.output_grad_np is None:
            out = self.forward_mindspore_impl(net)
            sens = np.random.randn(*list(out.shape))
            self.output_grad_np = np.array(sens, dtype=out.dtype)
        output_grad = Tensor(self.output_grad_np.astype(dtype=self.dtype))
        grad_net = GradOfAllInputsAndParams(net)
        grad_net.set_train()
        input_grad = grad_net(self.x_ms, self.w_ms, self.b_ms, output_grad)

        if self.b_ms is not None:
            return input_grad[0][0].asnumpy(), input_grad[0][1].asnumpy(), input_grad[0][2].asnumpy()
        return input_grad[0][0].asnumpy(), input_grad[0][1].asnumpy()

    def forward_cmp(self):
        if self.dtype == np.float16:
            self.loss *= 100  # 累加次数达到200000+，ccb结论放宽精度标准
        elif self.dtype == np.float32:
            self.loss *= 10  # 累加次数达到200000+，ccb结论放宽精度标准
        ps_net = Dense()
        jit(ps_net.construct, mode="PSJit")(self.x_ms, self.w_ms, self.b_ms)
        context.set_context(mode=context.GRAPH_MODE)
        out_psjit = self.forward_mindspore_impl(ps_net)
        pi_net = Dense()
        jit(pi_net.construct, mode="PIJit")(self.x_ms, self.w_ms, self.b_ms)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pijit = self.forward_mindspore_impl(pi_net)
        allclose_nparray(out_pijit, out_psjit, self.loss, self.loss)

    def grad_cmp(self):
        ps_net = Dense()
        jit(ps_net.construct, mode="PSJit")(self.x_ms, self.w_ms, self.b_ms)
        context.set_context(mode=context.GRAPH_MODE)
        input_grad_psjit = self.grad_mindspore_impl(ps_net)
        pi_net = Dense()
        jit(pi_net.construct, mode="PIJit")(self.x_ms, self.w_ms, self.b_ms)
        context.set_context(mode=context.PYNATIVE_MODE)
        input_grad_pijit = self.grad_mindspore_impl(pi_net)
        allclose_nparray(input_grad_pijit[0], input_grad_psjit[0], self.loss, self.loss)
        allclose_nparray(input_grad_pijit[1], input_grad_psjit[1], self.loss, self.loss)

        if self.b_np is not None:
            allclose_nparray(input_grad_pijit[2], input_grad_psjit[2], self.loss, self.loss)

    def mindspore_profile(self, net, run_time, op_names, *inputs):
        """
        profiler forward and grad for mindspore
        :param net: the mindspore net
        :param run_time: the times for running
        :param op_names: the list of the primitive op name in API
        :param inputs: the inputs
        :return: the profiler data
        """
        profiler_ms = Profiler()
        for _ in range(run_time):
            net(*inputs)
        profiler_ms.analyse()
        profile_ms = 0.0
        profiler_table = profiler_ms.op_analyse(op_names)
        profiler_table = json.loads(profiler_table)
        for op_name in op_names:
            profile_ms += float(profiler_table[op_name][0]["op_avg_time(us)"])
        return profile_ms

class DenseMock(DenseFactory):
    def __init__(self, inputs=None):
        x = inputs[0]
        w = inputs[1]
        in_c = w.shape[-1]
        out_c = w.shape[0]
        d = x.dtype
        super().__init__(x.shape, in_c, out_c, dtype_to_nptype(d))
