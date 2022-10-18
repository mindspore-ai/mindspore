# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
import pytest
import numpy as np
from mindspore import Tensor, jit, context
import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.common.initializer import One


context.set_context(mode=context.GRAPH_MODE)


def test_fallback_tensor():
    """
    Feature: JIT Fallback
    Description: Test Tensor() in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        me_x = Tensor(1)
        return me_x
    print(foo())


def test_fallback_tensor_bool():
    """
    Feature: JIT Fallback
    Description: Test Tensor(bool) in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        me_x = Tensor([True, True, False])
        return me_x
    print(foo())


def test_fallback_tensor_array():
    """
    Feature: JIT Fallback
    Description: Test Tensor(array) in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        me_x = Tensor([1])
        return me_x
    print(foo())


def test_fallback_tensor_with_mstype():
    """
    Feature: JIT Fallback
    Description: Test Tensor() with mstype in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        me_x = Tensor(1, mstype.int32)
        return me_x
    print(foo())


def test_fallback_tensor_array_with_mstype():
    """
    Feature: JIT Fallback
    Description: Test Tensor(array) with mstype in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        me_x = Tensor([1], mstype.int32)
        return me_x
    print(foo())


def test_fallback_tensor_with_numpy():
    """
    Feature: JIT Fallback
    Description: Test Tensor() with numpy in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        me_x = Tensor(np.zeros([1, 2, 3]), mstype.float32)
        return me_x
    print(foo())


def test_fallback_tensor_with_init():
    """
    Feature: JIT Fallback
    Description: Test Tensor() with init in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        me_x = Tensor(shape=(1, 3), dtype=mstype.float32, init=One())
        return me_x
    print(foo())


def test_fallback_tensor_reshape():
    """
    Feature: JIT Fallback
    Description: Test Tensor() with reshape() in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        me_x = Tensor(np.arange(10, 16).reshape(2, 3).astype("float32"))
        return me_x
    print(foo())


def test_fallback_tensor_abs():
    """
    Feature: JIT Fallback
    Description: Test Tensor.abs() in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        a = Tensor([1.1, -2.1])
        out = a.abs()
        return out
    print(foo())


def test_fallback_tensor_all():
    """
    Feature: JIT Fallback
    Description: Test Tensor.all() in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        a = Tensor([True, True, False])
        out = a.all()
        return out
    print(foo())


def test_fallback_tensor_any():
    """
    Feature: JIT Fallback
    Description: Test Tensor.any() in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        a = Tensor([True, True, False])
        out = a.any()
        return out
    print(foo())


def test_fallback_tensor_argmax():
    """
    Feature: JIT Fallback
    Description: Test Tensor.argmax() in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        a = Tensor(np.arange(10, 16).reshape(2, 3).astype("float32"))
        out = a.argmax()
        return out
    print(foo())


def test_fallback_tensor_argmin():
    """
    Feature: JIT Fallback
    Description: Test Tensor.argmin() in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        a = Tensor(np.arange(10, 16).reshape(2, 3).astype("float32"))
        out = a.argmin()
        return out
    print(foo())


def test_fallback_tensor_astype():
    """
    Feature: JIT Fallback
    Description: Test Tensor.astype() in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        a = Tensor(np.ones((1, 2, 2, 1), dtype=np.float32))
        out = a.astype("int32")
        return out
    print(foo())


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_fallback_tensor_asnumpy():
    """
    Feature: JIT Fallback
    Description: Test Tensor.asnumpy() in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        me_x = Tensor(np.arange(0, 6).reshape(2, 3))
        np_x = me_x.asnumpy()
        return Tensor(np_x)
    print(foo())


def test_fallback_tensor_from_numpy():
    """
    Feature: JIT Fallback
    Description: Test Tensor.from_numpy() in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        np_x = np.array([1, 2])
        me_x = Tensor.from_numpy(np_x)
        return me_x
    print(foo())


def test_fallback_tensor_binop():
    """
    Feature: Fallback feature
    Description: support binop's interpreted nodes.
    Expectation: No exception.
    """
    class BinOpNet(nn.Cell):
        def construct(self):
            np_array = np.array(9)
            res = Tensor(np_array) + Tensor(np_array)
            return res

    net = BinOpNet()
    print(net())


def test_fallback_tensor_compare():
    """
    Feature: Fallback feature
    Description: support compare op's interpreted nodes.
    Expectation: No exception.
    """
    class CompareNet(nn.Cell):
        def construct(self):
            np_array_1 = np.array(1)
            np_array_2 = np.array(2)
            res = Tensor(np_array_1) < Tensor(np_array_2)
            return res

    compare_net = CompareNet()
    print(compare_net())


def test_fallback_tensor_not():
    """
    Feature: Fallback feature
    Description: support bool op's interpreted nodes.
    Expectation: No exception.
    """
    class NotNet(nn.Cell):
        def construct(self):
            np_array_1 = np.array(True, dtype=np.bool_)
            res = not Tensor(np_array_1)
            return res

    net = NotNet()
    res = net()
    print("res:", res)


def test_fallback_tensor_and():
    """
    Feature: Fallback feature
    Description: support bool op's interpreted nodes.
    Expectation: No exception.
    """
    class AndNet(nn.Cell):
        def construct(self):
            np_array_1 = np.array(True, dtype=np.bool_)
            np_array_2 = np.array(False, dtype=np.bool_)
            res = Tensor(np_array_1) and Tensor(np_array_2)
            return res

    net = AndNet()
    res = net()
    print("res:", res)


def test_fallback_tensor_or():
    """
    Feature: Fallback feature
    Description: support bool op's interpreted nodes.
    Expectation: No exception.
    """
    class OrNet(nn.Cell):
        def construct(self):
            np_array_1 = np.array(True, dtype=np.bool_)
            np_array_2 = np.array(False, dtype=np.bool_)
            res = Tensor(np_array_1) or Tensor(np_array_2)
            return res

    net = OrNet()
    res = net()
    print("res:", res)


def test_fallback_tensor_augassign():
    """
    Feature: Fallback feature
    Description: support interpreted nodes in augassign.
    Expectation: No exception.
    """
    class OrNet(nn.Cell):
        def construct(self):
            np_array_1 = np.array(1)
            np_array_2 = np.array(2)
            res = Tensor(np_array_1)
            res += Tensor(np_array_2)
            return res

    net = OrNet()
    res = net()
    print("res:", res)


def test_fallback_tensor_subscript():
    """
    Feature: Fallback feature
    Description: support interpreted nodes in subscript.
    Expectation: No exception.
    """
    class SubScriptNet(nn.Cell):
        def construct(self):
            np_array_1 = np.array([1, 2, 3, 4, 5])
            np_array_2 = np.array(2)
            res = Tensor(np_array_1)[Tensor(np_array_2)]
            return res

    net = SubScriptNet()
    res = net()
    print("res:", res)


def test_fallback_tensor_if():
    """
    Feature: Fallback feature
    Description: support interpreted nodes in if statement.
    Expectation: No exception.
    """
    class IfNet(nn.Cell):
        def construct(self):
            np_array_1 = np.array(1)
            if Tensor(np_array_1):
                return 1
            return 0

    net = IfNet()
    res = net()
    print("res:", res)


def test_fallback_tensor_slice():
    """
    Feature: JIT Fallback
    Description: support interpreted nodes in slice.
    Expectation: No exception.
    """
    @jit
    def foo():
        array = np.arange(10)
        out = Tensor(array)[1:5]
        return out
    print(foo())


def test_fallback_ms_tensor():
    """
    Feature: JIT Fallback
    Description: Test ms.Tensor() in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        me_x = ms.Tensor([1])
        return me_x
    res = foo()
    assert (res.asnumpy() == [1]).all()


def test_fallback_ms_tensor_numpy():
    """
    Feature: JIT Fallback
    Description: Test ms.Tensor() in graph mode.
    Expectation: No exception.
    """
    @jit
    def foo():
        me_x = ms.Tensor(np.array([1, 2], dtype=np.float32))
        return me_x
    res = foo()
    assert (res.asnumpy() == [1, 2]).all()


def test_fallback_ms_tensor_class():
    """
    Feature: Fallback feature
    Description: Test ms.Tensor() in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self):
            np_array = np.array(9)
            x = ms.Tensor(np_array)
            res = x + ms.Tensor(np_array)
            return res

    net = Net()
    res = net()
    assert res == 18


def test_fallback_ms_tensor_user():
    """
    Feature: Fallback feature
    Description: Test ms.Tensor() and Tensor created by user in graph mode.
    Expectation: No exception.
    """
    class InnerNet(nn.Cell):
        def Tensor(self):
            return 10

        def construct(self):
            return self.Tensor()

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.inner_net = InnerNet()

        def construct(self):
            np_array = np.array(8)
            x = ms.Tensor(np_array)
            y = self.inner_net.Tensor()
            return x + y

    net = Net()
    res = net()
    assert res == 18
