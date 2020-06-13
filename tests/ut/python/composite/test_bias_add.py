import logging
import numpy as np
import mindspore.context as context
import mindspore.ops.composite as C
from mindspore import Tensor, Parameter
from mindspore.nn import Cell
from mindspore.ops.operations import _grad_ops as G
import mindspore.ops.operations as P
from mindspore.nn.composite_ops import BiasAdd, BiasAddGrad

log = logging.getLogger("ME")
log.setLevel(level=logging.DEBUG)
context.set_context(mode=context.GRAPH_MODE, save_graphs=True, device_target="Ascend")

class Net(Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.bias_add = BiasAdd()

    def construct(self, x, bias):
        return self.bias_add(x, bias)

class Net_grad(Cell):
    def __init__(self):
        super(Net_grad, self).__init__()
        self.bias_add = BiasAdd()

    def construct(self, x, bias, dout):
        return C.grad_all_with_sens(self.bias_add)(x, bias, dout)

class Net1(Cell):
    def __init__(self):
        super(Net1, self).__init__()
        self.bias_add = P.BiasAdd()

    def construct(self, x, bias):
        return self.bias_add(x, bias)

class Net_grad1(Cell):
    def __init__(self):
        super(Net_grad1, self).__init__()
        self.bias_add_grad = G.BiasAddGrad()

    def construct(self, dout):
        return self.bias_add_grad(dout)

class Net_grad2(Cell):
    def __init__(self):
        super(Net_grad2, self).__init__()
        self.bias_add_grad = BiasAddGrad()

    def construct(self, dout):
        return self.bias_add_grad(dout)

# composite not inline funcGraph
def test_composite_bias_add():
    x = np.random.normal(0, 1, [2, 3, 2, 3]).astype(np.float32)
    bias = np.random.normal(0, 1, [3]).astype(np.float32)
    net = Net()
    net1 = Net1()

    result = net(Tensor(x), Tensor(bias))
    result1 = net1(Tensor(x), Tensor(bias))
    print("=======================================")
    print("x:\n{}".format(x))
    print("bias:\n{}".format(bias))
    print("result:\n{}".format(result))
    print("result1:\n{}".format(result1))
    print("=======================================")

def test_composite_bias_add_grad():
    x = np.random.normal(0, 1, [2, 3, 2, 3]).astype(np.float32)
    bias = np.random.normal(0, 1, [3]).astype(np.float32)
    dout = np.random.normal(0, 1, [2, 3, 2, 3]).astype(np.float32)
    # dout = np.ones([2, 3, 4, 4]).astype(np.float32)
    net1 = Net_grad1()
    net = Net_grad()

    result1 = net1(Tensor(dout))
    result_1 = net(Tensor(x), Tensor(bias), Tensor(dout))
    print("=======================================")
    print("x:\n{}".format(x))
    print("bias:\n{}".format(bias))
    print("dout:\n{}".format(dout))
    print("result_1:\n{}".format(result_1))
    print("=======================================")
    print("result1:\n{}".format(result1))
    print("=======================================")

def test_composite_bias_add_grad1():
    x = np.random.normal(0, 1, [2, 3, 2, 3]).astype(np.float32)
    bias = np.random.normal(0, 1, [3]).astype(np.float32)
    dout = np.random.normal(0, 1, [2, 3, 2, 3]).astype(np.float32)
    # dout = np.ones([2, 3, 4, 4]).astype(np.float32)
    net1 = Net_grad1()
    net = Net_grad2()

    result1 = net1(Tensor(dout))
    result_1 = net(Tensor(dout))
    print("=======================================")
    print("x:\n{}".format(x))
    print("bias:\n{}".format(bias))
    print("dout:\n{}".format(dout))
    print("result_1:\n{}".format(result_1))
    print("=======================================")
    print("result1:\n{}".format(result1))
    print("=======================================")

#test_composite_bias_add()
test_composite_bias_add_grad1()
