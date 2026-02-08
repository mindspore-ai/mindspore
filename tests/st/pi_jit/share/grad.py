import time
import stat
import os
from mindspore.nn import Cell
from mindspore import Tensor
from mindspore.ops.composite import GradOperation
from mindspore.common import ParameterTuple


class _Grad(Cell):
    def __init__(self, grad, network, wrt_params=False, real_inputs_count=None):
        super().__init__()
        self.network = network
        self.grad = grad
        self.sens_param = self.grad.sens_param
        self.wrt_params = wrt_params
        self.real_inputs_count = real_inputs_count
        if self.wrt_params:
            self.params = ParameterTuple(self.network.trainable_params())

    def __call__(self, *inputs):
        if self.sens_param and self._dynamic_shape_inputs is not None:
            # not support dynamic shape sens
            if self.real_inputs_count is None:
                dyn_inputs = self._dynamic_shape_inputs[:-1]
                real_sens = inputs[-1:]
            else:
                idx = self.real_inputs_count
                dyn_inputs = self._dynamic_shape_inputs[:idx]
                real_sens = inputs[idx:]
            static_sens = list(dyn_inputs) + list(real_sens)
            super().set_inputs(*static_sens)

        a = time.perf_counter()
        out = super().__call__(*inputs)
        b = time.perf_counter()
        if os.environ.get("perf") == '1':
            phase = os.environ.get("PHASE")
            flags = os.O_WRONLY | os.O_CREAT
            modes = stat.S_IWUSR | stat.S_IRUSR
            with os.fdopen(os.open(phase, flags, modes), 'w') as f:
                f.write(str(b - a))
        return out

    def construct(self, *inputs):
        if self.wrt_params:
            if self.real_inputs_count is None or self.sens_param is False:
                return self.grad(self.network, self.params)(*inputs)
            real_inputs = inputs[:self.real_inputs_count]
            sense_param_inputs = inputs[self.real_inputs_count:]
            return self.grad(self.network, self.params)(*real_inputs, sense_param_inputs)
        if self.real_inputs_count is None or self.sens_param is False:
            return self.grad(self.network)(*inputs)
        real_inputs = inputs[:self.real_inputs_count]
        sense_param_inputs = inputs[self.real_inputs_count:]
        return self.grad(self.network)(*real_inputs, sense_param_inputs)


class GradOfFirstInput(_Grad):
    """
    get grad of first input
    """
    def __init__(self, network, sens_param=True, real_inputs_count=None):
        super().__init__(grad=GradOperation(sens_param=sens_param),
                         network=network, real_inputs_count=real_inputs_count)


class GradOfAllInputs(_Grad):
    """
    get grads of all inputs
    """
    def __init__(self, network, sens_param=True, real_inputs_count=None):
        super().__init__(grad=GradOperation(get_all=True, sens_param=sens_param),
                         network=network, real_inputs_count=real_inputs_count)


class GradOfAllParams(_Grad):
    """
    get grads of all params
    """
    def __init__(self, network, sens_param=True, real_inputs_count=None):
        super().__init__(grad=GradOperation(get_by_list=True, sens_param=sens_param),
                         network=network, wrt_params=True, real_inputs_count=real_inputs_count)


class GradOfAllInputsAndParams(_Grad):
    """
    get grads of all inputs and params
    """
    def __init__(self, network, sens_param=True, real_inputs_count=None):
        super().__init__(grad=GradOperation(get_all=True, get_by_list=True,
                                            sens_param=sens_param),
                         network=network, wrt_params=True, real_inputs_count=real_inputs_count)


class HighGrad(Cell):
    """
    get any order of grad
    """
    def __init__(self, network, grad_list, sens_param=False, real_inputs_count=None):
        super().__init__()
        self.grads = [network,]
        for i in range(len(grad_list)-1):
            _grad = grad_list[i](self.grads[i], sens_param=False)
            self.grads.append(_grad)
        self.final_grad = grad_list[-1](self.grads[-1], sens_param=sens_param, real_inputs_count=real_inputs_count)

    def construct(self, *inputs):
        return self.final_grad(*inputs)


def compute_grad_of_net_inputs(net: Cell, *inputs, sens: Tensor):
    grad_op = GradOfFirstInput if len(inputs) == 1 else GradOfAllInputs
    grad_net = grad_op(net, sens_param=True)
    grad_net.set_train()
    return grad_net(*inputs, sens)
