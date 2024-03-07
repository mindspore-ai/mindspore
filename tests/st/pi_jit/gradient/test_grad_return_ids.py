import numpy as onp
import pytest

from mindspore import jit, context, ops
from mindspore.common import Parameter, Tensor, dtype
from mindspore.nn import Cell
from ..share.utils import match_array


class GradFactory:
    def __init__(self, shape, pos, weight):
        npw = onp.random.randn(*shape)
        npb = onp.random.randn(*shape)
        self.net = Net(npw, npb)
        self.npx = onp.random.randn(*shape)
        self.npy = onp.random.randn(*shape)
        self.nps = onp.ones(shape)
        self.pos = pos
        self.weight = weight
        self.loss = 0.0001

    def get_mindspore_grad(self):
        get = []
        if self.pos is None:
            pass
        elif isinstance(self.pos, int):
            get.append(self.pos)
        else:
            get.extend(self.pos)
        real_weights = []
        if self.weight is not None:
            if 'w' in self.weight:
                real_weights.append(self.net.w)
                get.append(self.net.w)
            if 'b' in self.weight:
                real_weights.append(self.net.b)
                get.append(self.net.b)
            self.weight = real_weights

        grad_net = Grad(self.net,
                        self.pos, self.weight, get)
        x = Tensor(self.npx, dtype.float32)
        y = Tensor(self.npy, dtype.float32)
        grads = grad_net(x, y)
        return grads


# condition1, network construct
class Net(Cell):
    def __init__(self, w, b):
        super().__init__()
        mw = Tensor(w, dtype.float32)
        mb = Tensor(b, dtype.float32)
        self.w = Parameter(mw, name='w')
        self.b = Parameter(mb, name='b')

    def construct(self, x, y):
        out = self.w * x + self.b + y
        return out


# condition2, grad class construct
class Grad(Cell):
    def __init__(self, net, pos, param, get):
        super().__init__()
        self.net = net
        self.pos = pos
        self.param = param
        self.get = get

    def construct(self, x, y):
        grad_net = ops.grad(self.net, self.pos, self.param, return_ids=True)
        grads = grad_net(x, y)
        out = []
        for i in self.get:
            grad = ops.get_grad(grads, i)
            out.append(grad)
        return out


def grad_return_ids_pos0(class_name):
    fact = GradFactory(shape=(3, 4), pos=0, weight=None)
    jit(class_name.construct, mode="PSJit")
    context.set_context(mode=context.PYNATIVE_MODE)
    jit_grad = fact.get_mindspore_grad()
    jit(class_name.construct, mode="PIJit")
    context.set_context(mode=context.PYNATIVE_MODE)
    pijit_grad = fact.get_mindspore_grad()
    return jit_grad, pijit_grad


def grad_return_ids_pos01(class_name):
    fact = GradFactory(shape=(5, 4), pos=(0, 1), weight=None)
    jit(class_name.construct, mode="PSJit")
    context.set_context(mode=context.PYNATIVE_MODE)
    jit_grad = fact.get_mindspore_grad()
    jit(class_name.construct, mode="PIJit")
    context.set_context(mode=context.PYNATIVE_MODE)
    pijit_grad = fact.get_mindspore_grad()
    return jit_grad, pijit_grad


def grad_return_ids_weight_w(class_name):
    fact = GradFactory(shape=(2, 4), pos=None, weight=['w'])
    jit(class_name.construct, mode="PSJit")
    context.set_context(mode=context.PYNATIVE_MODE)
    jit_grad = fact.get_mindspore_grad()
    jit(class_name.construct, mode="PIJit")
    context.set_context(mode=context.PYNATIVE_MODE)
    pijit_grad = fact.get_mindspore_grad()
    return jit_grad, pijit_grad


def grad_return_ids_weight_wb(class_name):
    fact = GradFactory(shape=(2, 5, 3), pos=None, weight=['w', 'b'])
    jit(class_name.construct, mode="PSJit")
    context.set_context(mode=context.PYNATIVE_MODE)
    jit_grad = fact.get_mindspore_grad()
    jit(class_name.construct, mode="PIJit")
    context.set_context(mode=context.PYNATIVE_MODE)
    pijit_grad = fact.get_mindspore_grad()
    return jit_grad, pijit_grad


def grad_return_ids_pos_weight(class_name):
    fact = GradFactory(shape=(2, 3, 4), pos=None, weight=('w', 'b'))
    fact_pijit = GradFactory(shape=(2, 3, 4), pos=None, weight=('w', 'b'))
    jit(class_name.construct, mode="PSJit")
    context.set_context(mode=context.PYNATIVE_MODE)
    jit_grad = fact.get_mindspore_grad()
    jit(class_name.construct, mode="PIJit")
    context.set_context(mode=context.PYNATIVE_MODE)
    pijit_grad = fact_pijit.get_mindspore_grad()
    return jit_grad, pijit_grad


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.skip
@pytest.mark.parametrize('func', [grad_return_ids_pos0])
def test_grad_return_ids_pos0_pynative(func):
    """
    Feature:
        Validate the computation of gradients at position 0 in Pynative mode.

    Description:
        1. Initialize a neural network with the equation: w * x + y + b.
        2. Compute the gradient at position 0.
        3. Retrieve the calculated gradient at position 0.

    Expectation:
        The neural network should execute without errors.
        The computed gradient should match the reference gradient from PyTorch.
    """

    # condition1
    jit_grad, pijit_grad = func(Net)
    match_array(jit_grad, pijit_grad)
    # condition2
    jit_grad2, pijit_grad2 = func(Grad)
    match_array(jit_grad2, pijit_grad2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.skip
@pytest.mark.parametrize('func', [grad_return_ids_pos01])
def test_grad_return_ids_pos01_pynative(func):
    """
    Feature:
        Validate the computation of gradients at positions 0 and 1 in Pynative mode.

    Description:
        1. Initialize a neural network with the equation: w * x + y + b.
        2. Compute the gradient at positions 0 and 1.
        3. Retrieve the calculated gradients at positions 0 and 1.

    Expectation:
        The neural network should execute without errors.
        The computed gradients should match the reference gradients from PyTorch.
    """

    # condition1
    jit_grad, pijit_grad = func(Net)
    match_array(jit_grad, pijit_grad)
    # condition2
    jit_grad2, pijit_grad2 = func(Grad)
    match_array(jit_grad2, pijit_grad2)

@pytest.mark.skip
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [grad_return_ids_weight_w])
def test_grad_return_ids_weight_w_pynative(func):
    """
    Feature:
        Validate the computation of gradients for parameter w in Pynative mode.

    Description:
        1. Initialize a neural network with the equation: w * x + y + b.
        2. Compute the gradient for weights w.
        3. Retrieve the calculated gradients for parameters w and b.

    Expectation:
        The neural network should execute without errors.
        The computed gradient should match the reference gradient from PyTorch.
    """

    # condition1
    jit_grad, pijit_grad = func(Net)
    match_array(jit_grad, pijit_grad)
    # condition2
    jit_grad2, pijit_grad2 = func(Grad)
    match_array(jit_grad2, pijit_grad2)

@pytest.mark.skip
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [grad_return_ids_weight_wb])
def test_grad_return_ids_weight_wb_pynative(func):
    """
    Feature:
        Validate gradient computation for parameter w in Pynative mode.

    Description:
        1. Create a network: w * x + y + b.
        2. Compute the gradient for weights w.
        3. Retrieve the calculated gradient of parameter w.

    Expectation:
        The network should execute without errors.
        The computed gradient should match the reference gradient from PyTorch.
    """

    # condition1
    jit_grad, pijit_grad = func(Net)
    match_array(jit_grad, pijit_grad)
    # condition2
    jit_grad2, pijit_grad2 = func(Grad)
    match_array(jit_grad2, pijit_grad2)

@pytest.mark.skip
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [grad_return_ids_pos_weight])
def test_grad_return_ids_pos_weight_pynative(func):
    """
    Feature:
        Validate gradient computation for all inputs and weights in Pynative mode.

    Description:
        1. Create a network: w * x + y + b.
        2. Compute the gradient for all inputs and weights.
        3. Retrieve the calculated gradients of all parameters.

    Expectation:
        The network should execute without errors.
        The computed gradients should match the reference gradients from PyTorch.
    """

    # condition1
    jit_grad, pijit_grad = func(Net)
    match_array(jit_grad, pijit_grad)
    # condition2
    jit_grad2, pijit_grad2 = func(Grad)
    match_array(jit_grad2, pijit_grad2)
