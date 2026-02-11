from mindspore import jit, context
from mindspore.common import dtype
from mindspore.common import Tensor
from mindspore.nn import ForwardValueAndGrad
from ..share.utils import allclose_nparray


class CtrlFactory():
    def __init__(self, *inputs):
        super().__init__()
        self.ms_input = [Tensor(x, dtype.float32) for x in inputs]

        self.count = 0
        self.dyn = []
        for x in self.ms_input:
            xshp = x.shape
            if xshp:
                dshp = [None for _ in x.shape]
                dynt = Tensor(shape=dshp, dtype=x.dtype)
                self.dyn.append(dynt)
            else:
                self.dyn.append(x)

    def compare(self, ps_net, pi_net, dyn=False):
        self.count += 1
        if self.count == 2:
            for x in self.tc_input:
                if x.grad is not None:
                    x.grad.data.zero_()
        if dyn:
            ps_net.set_inputs(*self.dyn)
            pi_net.set_inputs(*self.dyn)
        context.set_context(mode=context.GRAPH_MODE)
        jit(function=ps_net.construct, capture_mode="ast")
        grad_net = ForwardValueAndGrad(ps_net, get_all=True)
        ps_out, ps_grad = grad_net(*self.ms_input)
        context.set_context(mode=context.PYNATIVE_MODE)
        jit(function=pi_net.construct, capture_mode="bytecode")
        grad_net = ForwardValueAndGrad(pi_net, get_all=True)
        pi_out, pi_grad = grad_net(*self.ms_input)

        allclose_nparray(pi_out.asnumpy(), ps_out.asnumpy(), 0.001, 0.001)
        for s, i in zip(ps_grad, pi_grad):
            allclose_nparray(s.asnumpy(), i.asnumpy(), 0.001, 0.001)
