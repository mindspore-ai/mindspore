from mindspore import context, jit
from mindspore.common.tensor import Tensor
from ..share.grad import GradOfFirstInput
from ..share.grad import GradOfAllInputs
from ..share.utils import allclose_nparray
import numpy as np
import copy
from tests.mark_utils import arg_mark


class ParserFactory():
    def __init__(self, ps_net, pi_net, *inputs):
        self._input_num = len(inputs)
        self.loss = 1e-3
        self.net_ps = ps_net
        self.net_pi = pi_net
        self.input_me_list = []
        for item in inputs:
            self._input_me = Tensor(item)
            self.input_me_list.append(self._input_me)

        self.out_np_shape = self.forward_mindspore_impl(self.net_ps).shape
        if not self.out_np_shape:
            self.out_np = np.array(1).astype(np.float32)
        else:
            self.out_np = np.random.randn(*self.out_np_shape).astype(np.float32)
        self.output_grad = Tensor(self.out_np)

    def forward_mindspore_impl(self, net_me):
        net_me.set_grad()
        input_me_use_list = copy.deepcopy(self.input_me_list)
        output_me = net_me(*input_me_use_list)
        return output_me

    def grad_mindspore_impl(self, net_me):
        grad_func = GradOfFirstInput if self._input_num == 1 else GradOfAllInputs
        grad_net = grad_func(net_me)
        grad_net.set_train()
        input_me_use_list = copy.deepcopy(self.input_me_list)
        grad_ms = grad_net(*input_me_use_list, self.output_grad)
        return grad_ms

    def forward_cmp(self):
        input_me_use_list = copy.deepcopy(self.input_me_list)
        context.set_context(mode=context.GRAPH_MODE)
        jit(fn=self.net_ps.construct, mode="PSJit")(*input_me_use_list)
        out_ps = self.forward_mindspore_impl(self.net_ps).asnumpy()
        context.set_context(mode=context.PYNATIVE_MODE)
        jit(fn=self.net_pi.construct, mode="PIJit")(*input_me_use_list)
        out_pi = self.forward_mindspore_impl(self.net_pi).asnumpy()
        allclose_nparray(out_pi, out_ps, self.loss, self.loss)

    def backward_cmp(self):
        context.set_context(mode=context.GRAPH_MODE)
        grad_ps = self.grad_mindspore_impl(self.net_ps)
        context.set_context(mode=context.PYNATIVE_MODE)
        grad_pi = self.grad_mindspore_impl(self.net_pi)
        for i in range(self._input_num):
            _grad_ps = grad_ps if self._input_num == 1 else grad_ps[i]
            _grad_pi = grad_pi if self._input_num == 1 else grad_pi[i]
            input_grad_ps = _grad_ps.asnumpy()
            input_grad_pi = _grad_pi.asnumpy()
            allclose_nparray(input_grad_pi, input_grad_ps, self.loss, self.loss)
