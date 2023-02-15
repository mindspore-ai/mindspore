# Copyright 2022 Huawei Technologies Co., Ltd
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
""" test graph fallback """
import math
import pytest
import numpy as np
import mindspore.nn as nn
from mindspore import context, Tensor, jit
from mindspore.ops import Primitive

context.set_context(mode=context.GRAPH_MODE)


def test_parse_return():
    """
    Feature: JIT Fallback
    Description: Test Interpret node in return statement in graph mode.
    Expectation: No exception.
    """

    class Network(nn.Cell):
        def __init__(self):
            super(Network, self).__init__()
            self.x = np.array([1, 2, 3])

        def construct(self):
            return Tensor(self.x)

    net = Network()
    out = net()
    assert (out.asnumpy() == [1, 2, 3]).all()


def test_parse_def_function():
    """
    Feature: JIT Fallback
    Description: Test args default value is Interpret node in graph mode.
    Expectation: No exception.
    """

    class Network(nn.Cell):
        def __init__(self):
            super(Network, self).__init__()
            self.x = np.array([1, 2, 3])

        def construct(self, y=np.array([2, 3, 4])):
            return Tensor(self.x + y)

    net = Network()
    out = net()
    assert (out.asnumpy() == [3, 5, 7]).all()


def test_parse_lambda():
    """
    Feature: JIT Fallback
    Description: Test Interpret node in lambda in graph mode.
    Expectation: No exception.
    """

    class Network(nn.Cell):
        def __init__(self):
            super(Network, self).__init__()
            self.x = np.array([1, 2, 3])

        def construct(self):
            new_x = lambda x: 2 * x + self.x
            y = new_x(1)
            return Tensor(y)

    net = Network()
    out = net()
    assert (out.asnumpy() == [3, 4, 5]).all()


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_parse_lambda_2():
    """
    Feature: JIT Fallback
    Description: Test Interpret node in lambda in graph mode.
    Expectation: No exception.
    """

    class Network(nn.Cell):
        def __init__(self):
            super(Network, self).__init__()
            self.x = np.array([1, 2, 3])

        def construct(self):
            new_x = lambda x: 2 * x + self.x
            return Tensor(new_x(1))

    net = Network()
    out = net()
    assert (out.asnumpy() == [3, 4, 5]).all()


def test_parse_bool_op():
    """
    Feature: JIT Fallback
    Description: Test Interpret node in bool op in graph mode.
    Expectation: No exception.
    """

    class Network(nn.Cell):
        def __init__(self):
            super(Network, self).__init__()
            self.x = np.array([1, 2, 3])

        def construct(self):
            if Tensor(2) and self.x.all():
                return Tensor(self.x + 1)
            return Tensor(self.x)

    net = Network()
    out = net()
    assert (out.asnumpy() == [2, 3, 4]).all()


def test_parse_tuple():
    """
    Feature: JIT Fallback
    Description: Test Interpret node in tuple in graph mode.
    Expectation: No exception.
    """

    class Network(nn.Cell):
        def construct(self):
            x = Tensor([1])
            tuple_num = [x, x + 1, Tensor([3])]
            return tuple_num

    net = Network()
    out = net()
    assert out[0].asnumpy() == 1 and out[1].asnumpy() == 2 and out[2].asnumpy() == 3


def test_parse_slice():
    """
    Feature: JIT Fallback
    Description: Test Interpret node in slice in graph mode.
    Expectation: No exception.
    """

    class Network(nn.Cell):
        def construct(self):
            x = [Tensor([11]), Tensor([22]), Tensor([33])]
            y = x[Tensor([0]): Tensor([2])]
            return y

    net = Network()
    out = net()
    assert out[0].asnumpy() == 11 and out[1].asnumpy() == 22


def test_list_count():
    """
    Feature: Fallback feature
    Description: support attr/method of builtin type.
    Expectation: No exception.
    """
    @jit
    def list_count():
        x = list([1, 2, 3])
        res = x.count(1)
        return res
    assert list_count() == 1


def test_list_append():
    """
    Feature: Fallback feature
    Description: support attr/method of builtin type.
    Expectation: No exception.
    """
    @jit
    def list_append():
        x = list([1, 2, 3])
        x.append(4)
        return Tensor(x)
    assert np.all(list_append().asnumpy() == np.array([1, 2, 3, 4]))


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_parse_subscript():
    """
    Feature: JIT Fallback
    Description: Test Interpret node in subscript in graph mode.
    Expectation: No exception.
    """

    class Network(nn.Cell):
        def construct(self):
            x = [Tensor([11]), Tensor([22]), Tensor([33])]
            y = x[Tensor([0])] + x[Tensor([1])] + x[Tensor([2])]
            return y

    net = Network()
    out = net()
    assert out.asnumpy() == 66


def test_parse_subscript_2():
    """
    Feature: JIT Fallback
    Description: Test Interpret node in subscript in graph mode.
    Expectation: No exception.
    """

    class Network(nn.Cell):
        def construct(self):
            x = [Tensor([11]), Tensor([22]), Tensor([33])]
            y = x[np.array(0)]
            return y

    net = Network()
    out = net()
    assert out.asnumpy() == 11


def test_parse_unary_op():
    """
    Feature: JIT Fallback
    Description: Test Interpret node in unary op in graph mode.
    Expectation: No exception.
    """

    class Network(nn.Cell):
        def construct(self):
            x = np.array([1, 2, 3])
            y = -x
            return Tensor(y)

    net = Network()
    out = net()
    assert (out.asnumpy() == [-1, -2, -3]).all()


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_parse_dict():
    """
    Feature: JIT Fallback
    Description: Test Interpret node in dict in graph mode.
    Expectation: No exception.
    """

    class Network(nn.Cell):
        def construct(self):
            x = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])}
            key = "b"
            try:
                value = x[key]
            except KeyError:
                print("The key is not exist in x.")
            return Tensor(value)

    net = Network()
    out = net()
    assert (out.asnumpy() == [4, 5, 6]).all()


def test_parse_ifexpr():
    """
    Feature: JIT Fallback
    Description: Test Interpret node in ifexpr in graph mode.
    Expectation: No exception.
    """

    class Network(nn.Cell):
        def construct(self):
            y = Tensor([0]) if np.array([1]) else Tensor([1])
            return y

    net = Network()
    out = net()
    assert out == 0


def test_fallback_tensor_array_astype():
    """
    Feature: JIT Fallback
    Description: Test Tensor(array) with astype() in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        me_x = Tensor([1.1, -2.1]).astype("float32")
        return me_x
    print(foo())


def test_fallback_tuple_with_mindspore_function():
    """
    Feature: JIT Fallback
    Description: Test fallback when local input has tuple with mindspore function type, such as Cell, Primitive.
    Expectation: No exception.
    """
    def test_isinstance(a, base_type):
        mro = type(a).mro()
        for i in base_type:
            if i in mro:
                return True
        return False

    @jit
    def foo():
        return test_isinstance(np.array(1), (np.ndarray, nn.Cell, Primitive))

    assert foo()


def test_fallback_math_method():
    """
    Feature: JIT Fallback
    Description: Test math.floor()
    Expectation: No exception
    """
    @jit
    def foo():
        return math.floor(1.5)

    assert foo() == 1
