# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" test_stop_gradient """
import numpy as np
import pytest

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Parameter, ParameterTuple
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import jit
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.ops.functional import stop_gradient
from mindspore.ops.primitive import prim_attr_register, PrimitiveWithInfer
from tests.security_utils import security_off_wrap
from ..ut_filter import non_graph_engine
from ....mindspore_test_framework.utils.bprop_util import bprop


grad_by_list = C.GradOperation(get_by_list=True)
grad_all = C.GradOperation(get_all=True)


def setup_module(module):
    context.set_context(mode=context.PYNATIVE_MODE)


def stop_func(x, y):
    """ stop_func"""
    c = x * y
    c_s = x + y
    return c_s, c


def stop_test1(x, y):
    """ stop_test1 """
    c = x * y
    c_s = stop_gradient(c)
    return c_s


def stop_test2(x, y):
    """ stop_test2 """
    c = x * y
    c_s = stop_gradient(c)
    d = c_s + x * y
    return d * y


def stop_test3(x, y):
    """ stop_test3 """
    x = x * y
    z = stop_test1(x, y)
    k = z * y
    return k


def stop_test5(x, y):
    """ stop_test3 """
    x = x + y
    o1, o2 = stop_func(x, y)
    c = stop_gradient(o1)
    c = o2 + c
    return c


def stop_test4(x, y):
    """ stop_test4 """
    c = x + y
    c_s = stop_gradient(c)
    e = c + c_s
    return e


@jit
def grad_stop_test(x, y):
    """ grad_stop_test """
    return grad_all(stop_test2)(x, y)


@jit
def grad_stop_test1(x, y):
    """ grad_stop_test1 """
    return grad_all(stop_test3)(x, y)


@jit
def grad_stop_test5(x, y):
    """ grad_stop_test5 """
    return grad_all(stop_test5)(x, y)


def test_stop():
    """ test_stop """
    print("test_stop:", grad_stop_test(1, 1))


def test_stop1():
    """ test_stop1 """
    print("test_stop1:", grad_stop_test1(2, 3))


def test_stop5():
    """ test_stop1 """
    print("test_stop5:", grad_stop_test5(2, 3))


class GradWrap(nn.Cell):
    """ GradWrap definition """

    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network
        self.weights = ParameterTuple(network.get_parameters())

    @jit
    def construct(self, x, label):
        weights = self.weights
        return grad_by_list(self.network, weights)(x, label)


@non_graph_engine
def test_softmaxloss_grad():
    """ test_softmaxloss_grad """

    class NetWithLossClass(nn.Cell):
        """ NetWithLossClass definition """

        def __init__(self, network):
            super(NetWithLossClass, self).__init__()
            self.loss = nn.SoftmaxCrossEntropyWithLogits()
            self.network = network

        @jit
        def construct(self, x, label):
            predict = self.network(x)
            return self.loss(predict, label)

    class Net(nn.Cell):
        """ Net definition """

        def __init__(self):
            super(Net, self).__init__()
            self.weight = Parameter(Tensor(np.ones([64, 10]).astype(np.float32)), name="weight")
            self.bias = Parameter(Tensor(np.ones([10]).astype(np.float32)), name="bias")
            self.fc = P.MatMul()
            self.fc2 = nn.Dense(10, 10)
            self.biasAdd = P.BiasAdd()
            self.relu = nn.ReLU()
            self.cast = P.Cast()

        @jit
        def construct(self, x):
            x = self.fc(x, self.weight)
            x = self.cast(x, mstype.float32)
            x = self.relu(self.fc2(x))
            x = self.fc2(x)
            x = stop_gradient(x)
            x = self.biasAdd(x, self.bias)
            return x

    net = GradWrap(NetWithLossClass(Net()))

    predict = Tensor(np.ones([1, 64]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    print("pynative run")
    out = net(predict, label)
    print("out:", out)


def test_stop_gradient_1():
    class Mul(nn.Cell):
        def __init__(self):
            super(Mul, self).__init__()

        @jit
        def construct(self, x, y):
            ret = x * y
            ret = stop_gradient(ret)
            return ret

    dx, dy = bprop(Mul(), Tensor(np.ones([2, 2]).astype(np.float32)),
                   Tensor(np.ones([2, 2]).astype(np.float32)), wrt=['inputs'])
    expect = np.zeros([2, 2])
    assert (dx.asnumpy() == expect).all()
    assert (dy.asnumpy() == expect).all()


def test_stop_gradient_2():
    class Mul(nn.Cell):
        def __init__(self):
            super(Mul, self).__init__()

        @jit
        def construct(self, x, y):
            c = x * y
            z = x * y
            return c, z

    class MulAdd(nn.Cell):
        def __init__(self):
            super(MulAdd, self).__init__()
            self.mul = Mul()

        @jit
        def construct(self, x, y):
            u = x + y
            v = x - y
            c, z = self.mul(u, v)
            c = stop_gradient(c)
            ret1 = c + x + y
            ret2 = z + y + y
            return ret1, ret2

    dx = bprop(MulAdd(), Tensor(np.ones([2, 2]).astype(np.float32)),
               Tensor(np.ones([2, 2]).astype(np.float32)))
    expect = np.array([[3.0, 3.0], [3.0, 3.0]])
    assert (dx.asnumpy() == expect).all()


def test_stop_gradient_3():
    class TupleGetItem(nn.Cell):
        def __init__(self):
            super(TupleGetItem, self).__init__()

        @jit
        def construct(self, x1, x2, x3, x4, x5):
            z1 = x1 + x1
            z2 = x1 * x2
            t = (z1, z2, x3, x4, x5)
            z2 = t[1]
            z2 = stop_gradient(z2)
            return z1, z2, x3, x4, x5

    dx = bprop(TupleGetItem(),
               Tensor(np.ones([2]).astype(np.float32)),
               Tensor(np.ones([2]).astype(np.float32)),
               Tensor(np.ones([2]).astype(np.float32)),
               Tensor(np.ones([2]).astype(np.float32)),
               Tensor(np.ones([2]).astype(np.float32)))
    expect = np.array([[2.0, 2.0], [2.0, 2.0]])
    assert (dx.asnumpy() == expect).all()


def test_stop_gradient_4():
    def stop_test(x):
        return stop_gradient(x)

    assert grad_all(stop_test)(Tensor(1, dtype=ms.int32)) == (0,)


def test_stop_gradient_5():
    def stop_test(x):
        y = x + x
        y = stop_gradient(y)
        ret = x + y
        return ret

    assert grad_all(stop_test)(Tensor(1, dtype=ms.int32)) == (1,)


def test_stop_gradient_6():
    def stop_test(x, y):
        ret = x * y
        ret = stop_gradient(ret)
        return ret

    assert grad_all(stop_test)(Tensor(1, dtype=ms.int32), Tensor(3, dtype=ms.int32)) == (0, 0)


class PrimWithMultiOutputs(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self):
        """init"""

    def __call__(self, x, y):
        """Implement by vm mode."""
        return x, y

    def infer_shape(self, x_shape, y_shape):
        return x_shape, y_shape

    def infer_dtype(self, x_type, y_type):
        return x_type, y_type

    def get_bprop(self):
        def bprop(x, y, out, dout):
            return (dout[0], dout[1])

        return bprop


def test_stop_gradient_7():
    class PrimWithMultiOutputs_(nn.Cell):
        def __init__(self):
            super(PrimWithMultiOutputs_, self).__init__()
            self.prim_with_multi_outputs = PrimWithMultiOutputs()

        @jit
        def construct(self, x1, x2):
            x1, x2 = self.prim_with_multi_outputs(x1, x2)
            x1 = stop_gradient(x1)
            return x1, x2

    dx, dy = bprop(PrimWithMultiOutputs_(), Tensor(np.ones([2]).astype(np.float32)),
                   Tensor(np.ones([2]).astype(np.float32)), wrt=['inputs'])
    expect_dx = np.zeros([2])
    expect_dy = np.ones([2])
    assert (dx.asnumpy() == expect_dx).all()
    assert (dy.asnumpy() == expect_dy).all()


def test_stop_gradient_8():
    class PrimWithMultiOutputs_(nn.Cell):
        def __init__(self):
            super(PrimWithMultiOutputs_, self).__init__()
            self.prim_with_multi_output = PrimWithMultiOutputs()

        @jit
        def construct(self, x1, x2):
            x1, x2 = stop_gradient(self.prim_with_multi_output(x1, x2))
            return x1, x2

    dx, dy = bprop(PrimWithMultiOutputs_(), Tensor(np.ones([2]).astype(np.float32)),
                   Tensor(np.ones([2]).astype(np.float32)), wrt=['inputs'])
    expect_dx = np.zeros([2])
    expect_dy = np.zeros([2])
    assert (dx.asnumpy() == expect_dx).all()
    assert (dy.asnumpy() == expect_dy).all()


def test_stop_gradient_9():
    class Mul(nn.Cell):
        def __init__(self):
            super(Mul, self).__init__()

        @jit
        def construct(self, x, y):
            c = x * y
            z = x * y
            return c, z

    class MulAdd(nn.Cell):
        def __init__(self):
            super(MulAdd, self).__init__()
            self.mul = Mul()

        @jit
        def construct(self, x, y):
            u = x + y
            v = x - y
            c, z = self.mul(u, v)
            c1 = stop_gradient(c)
            c2 = c
            ret1 = c1 + x + y + c2
            ret2 = z + y + y
            return ret1, ret2

    dx = bprop(MulAdd(), Tensor(np.ones([2, 2]).astype(np.float32)),
               Tensor(np.ones([2, 2]).astype(np.float32)))
    expect = np.array([[5.0, 5.0], [5.0, 5.0]])
    assert (dx.asnumpy() == expect).all()


class PrimWithNoBprop(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self):
        """init"""

    def __call__(self, x, y):
        """Implement by vm mode."""
        return x, y

    def infer_shape(self, x_shape, y_shape):
        return x_shape, y_shape

    def infer_dtype(self, x_type, y_type):
        return x_type, y_type


def test_stop_gradient_10():
    class PrimWithNoBprop_(nn.Cell):
        def __init__(self):
            super(PrimWithNoBprop_, self).__init__()
            self.prim_with_no_bprop = PrimWithNoBprop()

        @jit
        def construct(self, x, y):
            x = x * y
            x, y = self.prim_with_no_bprop(x, y)
            x = stop_gradient(x)
            y = stop_gradient(y)
            return x, y

    dx = bprop(PrimWithNoBprop_(), Tensor(np.ones([2]).astype(np.float32)),
               Tensor(np.ones([2]).astype(np.float32)))
    expect_dx = np.zeros([2])
    assert (dx.asnumpy() == expect_dx).all()


def test_stop_gradient_11():
    class PrimWithNoBprop_(nn.Cell):
        def __init__(self):
            super(PrimWithNoBprop_, self).__init__()
            self.prim_with_no_bprop = PrimWithNoBprop()

        @jit
        def construct(self, x, y):
            x, y = self.prim_with_no_bprop(x, y)
            x = stop_gradient(x)
            return x, y

    with pytest.raises(RuntimeError):
        bprop(PrimWithNoBprop_(), Tensor(np.ones([2]).astype(np.float32)),
              Tensor(np.ones([2]).astype(np.float32)))


@security_off_wrap
def test_stop_print():
    class StopPrint(nn.Cell):
        def __init__(self):
            super(StopPrint, self).__init__()
            self.printm = P.Print()

        def construct(self, x, y):
            self.printm("StopPrint", x)
            self.printm(y)
            return x, y

    grad_all(StopPrint())(Tensor(np.ones([2]).astype(np.float32)),
                          Tensor(np.ones([2]).astype(np.float32)))
