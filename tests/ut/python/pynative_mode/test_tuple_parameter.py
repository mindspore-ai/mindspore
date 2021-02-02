import numpy as np

import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.ops import operations as P
from mindspore.ops import composite as C



def setup_module(module):
    context.set_context(mode=context.PYNATIVE_MODE)


class Block1(nn.Cell):
    """ Define Cell with tuple input as paramter."""

    def __init__(self):
        super(Block1, self).__init__()
        self.mul = P.Mul()

    def construct(self, tuple_xy):
        x, y = tuple_xy
        z = self.mul(x, y)
        return z

class Block2(nn.Cell):
    """ definition with tuple in tuple output in Cell."""

    def __init__(self):
        super(Block2, self).__init__()
        self.mul = P.Mul()
        self.add = P.Add()

    def construct(self, x, y):
        z1 = self.mul(x, y)
        z2 = self.add(z1, x)
        z3 = self.add(z1, y)
        return (z1, (z2, z3))

class Net1(nn.Cell):
    def __init__(self):
        super(Net1, self).__init__()
        self.block = Block1()

    def construct(self, x, y):
        res = self.block((x, y))
        return res


class Net2(nn.Cell):
    def __init__(self):
        super(Net2, self).__init__()
        self.add = P.Add()
        self.block = Block2()

    def construct(self, x, y):
        z1, (z2, z3) = self.block(x, y)
        res = self.add(z1, z2)
        res = self.add(res, z3)
        return res

def test_net():
    context.set_context(save_graphs=True)
    x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32) * 2)
    y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32) * 3)
    net1 = Net1()
    grad_op = C.GradOperation(get_all=True)
    output = grad_op(net1)(x, y)
    assert np.all(output[0].asnumpy() == y.asnumpy())
    assert np.all(output[1].asnumpy() == x.asnumpy())

    net2 = Net2()
    output = grad_op(net2)(x, y)
    expect_x = np.ones([1, 1, 3, 3]).astype(np.float32) * 10
    expect_y = np.ones([1, 1, 3, 3]).astype(np.float32) * 7
    assert np.all(output[0].asnumpy() == expect_x)
    assert np.all(output[1].asnumpy() == expect_y)
