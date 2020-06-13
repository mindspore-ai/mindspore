import numpy as np
import mindspore.context as context
from mindspore import Tensor, Parameter
from mindspore.nn import Cell, Composite
from mindspore.ops import operations as P
import mindspore.ops.composite as C
import logging

log = logging.getLogger("ME")
log.setLevel(level=logging.DEBUG)
context.set_context(mode=context.GRAPH_MODE, save_graphs=True, device_target="Ascend")

class Sigmoid(Composite):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.neg = P.Neg()
        self.exp = P.Exp()
        self.div = P.Div()

    def construct(self, x):
        neg_val = self.neg(x)
        exp_val = self.exp(neg_val)
        sigmoid = 1.0 / (1.0 + exp_val)
        return sigmoid


class Net(Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.sigmoid = Sigmoid()
        self.exp = P.Exp()

    def construct(self, x):
        return self.sigmoid(x) * self.exp(x)

class NetComposite(Composite):
    def __init__(self):
        super(NetComposite, self).__init__()
        self.sigmoid = Sigmoid()
        self.exp = P.Exp()

    def construct(self, x):
        return self.sigmoid(x) * self.exp(x)

class Net1(Cell):
    def __init__(self):
        super(Net1, self).__init__()
        self.exp = P.Exp()

    def construct(self, x):
        return self.exp(x)

class NetComposite1(Composite):
    def __init__(self):
        super(NetComposite1, self).__init__()
        self.net = Net1()
        self.exp = P.Exp()

    def construct(self, x):
        return self.exp(x) * self.net(x)

class Net_grad(Cell):
    def __init__(self):
        super(Net_grad, self).__init__()
        self.sigmoid = Sigmoid()
        self.exp = P.Exp()

    def construct(self, x, dout):
        dout = C.grad_with_sens(self.sigmoid)(x, dout)
        #out = self.sigmoid(x)
        return dout


def vm_impl(x):
    return (1.0 / (1.0 + np.exp(-x))) * np.exp(x)

def vm_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# composite not inline funcGraph
def test_composite_sigmoid1():
    x = np.random.normal(0, 1, [2, 3]).astype(np.float32)
    net = Net()
    result = net(Tensor(x))
    vm_result = vm_impl(x)
    print("=======================================")
    print("x: {}".format(x))
    print("result: {}".format(result))
    print("vm_result: {}".format(vm_result))
    print("=======================================")

# composite inline composite
def test_composite_sigmoid2():
    x = Tensor(np.random.normal(0, 1, [2, 3]).astype(np.float32))
    net = NetComposite()
    result = net(x)
    print("=======================================")
    print(result)
    print("=======================================")

# composite inline func
def test_composite_sigmoid3():
    x = Tensor(np.random.normal(0, 1, [2, 3]).astype(np.float32))
    net = NetComposite1()
    result = net(x)
    print("=======================================")
    print(result)
    print("=======================================")


def test_composite_sigmoid_grad():
    x = np.random.normal(0, 1, [2, 3]).astype(np.float32)
    dout = np.random.normal(0, 1, [2, 3]).astype(np.float32)
    net = Net_grad()
    result = net(Tensor(x), Tensor(dout))
    print("=======================================")
    print("x: {}".format(x))
    print("result: {}".format(result))
    print("=======================================")

test_composite_sigmoid1()

test_composite_sigmoid_grad()
