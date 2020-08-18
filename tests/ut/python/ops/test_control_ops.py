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
""" test control ops """
import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor
from mindspore import context
from mindspore import nn
from mindspore.common import dtype as mstype
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.common import ms_function

context.set_context(mode=context.GRAPH_MODE)


def cond_data_test(x_init, y_init):
    class Net(nn.Cell):
        def __init__(self):
            """"""
            super(Net, self).__init__()
            self.square = P.Square()
            self.add = P.TensorAdd()
            self.value = Tensor(3, dtype=ms.float32)
            self.switch = P.GeSwitch()
            self.merge = P.Merge()
            self.less = P.Less()

        def construct(self, x, y):
            cond = self.less(x, y)
            st1, _ = self.switch(x, cond)
            st2, _ = self.switch(y, cond)
            add_ret = self.add(st1, st2)
            _, sf3 = self.switch(self.value, cond)
            sq_ret = self.square(sf3)
            ret = self.merge((add_ret, sq_ret))
            return ret[0]

    x = Tensor(x_init, dtype=ms.float32)
    y = Tensor(y_init, dtype=ms.float32)
    net = Net()
    output = net(x, y)
    return output


def test_cond_data_true():
    output = cond_data_test(3, 8)
    print("test_cond_data_true:", output)


def test_cond_data_false():
    output = cond_data_test(8, 3)
    print("test_cond_data_false:", output)


def if_compile_test(x_init, y_init):
    class Net(nn.Cell):
        def __init__(self):
            """"""
            super(Net, self).__init__()
            self.square = P.Square()
            self.add = P.TensorAdd()
            self.value = Tensor(3, dtype=ms.float32)
            self.switch = P.GeSwitch()
            self.merge = P.Merge()
            self.less = P.Less()

        def construct(self, x, y):
            cond = self.less(x, y)
            ret = self.value
            if cond:
                ret = self.add(x, ret)
                ret = self.add(y, ret)
            else:
                ret = self.square(self.value)
            return ret

    x = Tensor(x_init, dtype=ms.float32)
    y = Tensor(y_init, dtype=ms.float32)
    net = Net()
    output = net(x, y)
    return output


def test_if_none():
    class Net(nn.Cell):
        def __init__(self, z: None):
            """"""
            super(Net, self).__init__()
            self.z = z

        def construct(self, x, y):
            if self.z:
                ret = x
            else:
                ret = y
            return ret

    x = Tensor(np.ones([6, 8, 10], np.int32))
    y = Tensor(np.zeros([3, 4, 5], np.int32))
    z = None
    net = Net(z)
    assert np.all(net(x, y).asnumpy() == y.asnumpy())

def test_if_str_is_not_none_right():
    class Net(nn.Cell):
        def __init__(self, z: str):
            """"""
            super(Net, self).__init__()
            self.z = z

        def construct(self, x, y):
            if self.z is None:
                ret = x
            else:
                ret = y
            return ret

    x = Tensor(np.ones([6, 8, 10], np.int32))
    y = Tensor(np.zeros([3, 4, 5], np.int32))
    z = "ok"
    net = Net(z)
    assert np.all(net(x, y).asnumpy() == y.asnumpy())


def test_if_str_is_not_none_left():
    class Net(nn.Cell):
        def __init__(self, z: str):
            """"""
            super(Net, self).__init__()
            self.z = z

        def construct(self, x, y):
            if self.z is None:
                ret = x
            else:
                ret = y
            return ret

    x = Tensor(np.ones([6, 8, 10], np.int32))
    y = Tensor(np.zeros([3, 4, 5], np.int32))
    z = "ok"
    net = Net(z)
    assert np.all(net(x, y).asnumpy() == y.asnumpy())


def test_if_none_equal_none():
    class Net(nn.Cell):
        def __init__(self, z: None):
            """"""
            super(Net, self).__init__()
            self.z = z

        def construct(self, x, y):
            if self.z is None:
                ret = x
            else:
                ret = y
            return ret

    x = Tensor(np.ones([6, 8, 10], np.int32))
    y = Tensor(np.zeros([3, 4, 5], np.int32))
    z = None
    net = Net(z)
    assert np.all(net(x, y).asnumpy() == x.asnumpy())


def test_if_str_is_null():
    class Net(nn.Cell):
        def __init__(self, z: str):
            """"""
            super(Net, self).__init__()
            self.z = z

        def construct(self, x, y):
            if self.z:
                ret = x
            else:
                ret = y
            return ret

    x = Tensor(np.ones([6, 8, 10], np.int32))
    y = Tensor(np.zeros([3, 4, 5], np.int32))
    z = ""
    net = Net(z)
    assert np.all(net(x, y).asnumpy() == y.asnumpy())


def test_if_str_is_true():
    class Net(nn.Cell):
        def __init__(self, z: str):
            """"""
            super(Net, self).__init__()
            self.z = z

        def construct(self, x, y):
            if self.z:
                ret = x
            else:
                ret = y
            return ret

    x = Tensor(np.ones([6, 9, 10], np.int32))
    y = Tensor(np.zeros([3, 4, 5], np.int32))
    z = "ok"
    net = Net(z)
    assert np.all(net(x, y).asnumpy() == x.asnumpy())


def test_if_str_equal():
    class Net(nn.Cell):
        def __init__(self, z: str):
            """"""
            super(Net, self).__init__()
            self.z = z

        def construct(self, x, y):
            if self.z == "ok":
                ret = x
            else:
                ret = y
            return ret

    x = Tensor(np.ones([6, 8, 10], np.int32))
    y = Tensor(np.zeros([3, 4, 5], np.int32))
    z = "ok"
    net = Net(z)
    assert np.all(net(x, y).asnumpy() == x.asnumpy())


def test_if_tuple_is_null():
    class Net(nn.Cell):
        def __init__(self, z: tuple):
            """"""
            super(Net, self).__init__()
            self.z = z

        def construct(self, x, y):
            if self.z:
                ret = x
            else:
                ret = y
            return ret

    x = Tensor(np.ones([6, 8, 10], np.int32))
    y = Tensor(np.zeros([3, 4, 5], np.int32))
    z = ()
    net = Net(z)
    assert np.all(net(x, y).asnumpy() == y.asnumpy())


def test_if_tuple_is_not_null():
    class Net(nn.Cell):
        def __init__(self, z: tuple):
            """"""
            super(Net, self).__init__()
            self.z = z

        def construct(self, x, y):
            if self.z:
                ret = x
            else:
                ret = y
            return ret

    x = Tensor(np.ones([6, 8, 10], np.int32))
    y = Tensor(np.zeros([3, 4, 5], np.int32))
    z = (1, 2, 3)
    net = Net(z)
    assert np.all(net(x, y).asnumpy() == x.asnumpy())


def test_if_dict_is_null():
    class Net(nn.Cell):
        def __init__(self, z: dict):
            """"""
            super(Net, self).__init__()
            self.z = z

        def construct(self, x, y):
            if self.z:
                ret = x
            else:
                ret = y
            return ret

    x = Tensor(np.ones([6, 8, 10], np.int32))
    y = Tensor(np.zeros([3, 4, 5], np.int32))
    z = {}
    net = Net(z)
    assert np.all(net(x, y).asnumpy() == y.asnumpy())


def test_if_dict_is_not_null():
    class Net(nn.Cell):
        def __init__(self, z: dict):
            """"""
            super(Net, self).__init__()
            self.z = z

        def construct(self, x, y):
            if self.z:
                ret = x
            else:
                ret = y
            return ret

    x = Tensor(np.ones([6, 8, 10], np.int32))
    y = Tensor(np.zeros([3, 4, 5], np.int32))
    z = {"one": 1, "two": 2}
    net = Net(z)
    assert np.all(net(x, y).asnumpy() == x.asnumpy())


def test_if_else_assign():
    class Net(nn.Cell):
        def __init__(self, m: list):
            """"""
            super(Net, self).__init__()
            self.m = m
            self.n = [4, 5, 6]

        def construct(self, x, y):
            exp_1 = self.m if self.m else self.n
            exp_2 = self.m if exp_1 == self.n else self.n
            if exp_2 == self.m:
                if self.m:
                    ret = x
                else:
                    ret = y
            else:
                if self.m:
                    ret = x
                else:
                    ret = y
            return ret

    x = Tensor(np.ones([6, 8, 10], np.int32))
    y = Tensor(np.zeros([3, 4, 5], np.int32))
    z = [1, 2]
    net = Net(z)
    assert np.all(net(x, y).asnumpy() == x.asnumpy())


def test_if_compile_true():
    output = if_compile_test(3, 8)
    print("test_if_compile_true:", output)


def test_if_compile_false():
    output = if_compile_test(8, 3)
    print("test_if_compile_false:", output)


def test_switch_layer():
    class Layer1(nn.Cell):
        def __init__(self):
            super(Layer1, self).__init__()
            self.z1 = Parameter(
                Tensor(np.full([128, 96], 0.6, dtype=np.float32)), name='z1')

        def construct(self, x):
            return x * self.z1

    class Layer2(nn.Cell):
        def __init__(self):
            super(Layer2, self).__init__()
            self.z2 = Parameter(
                Tensor(np.full([128, 96], 0.6, dtype=np.float32)), name='z2')

        def construct(self, x):
            return x * self.z2

    class SwitchLayerCell(nn.Cell):
        def __init__(self):
            super(SwitchLayerCell, self).__init__()
            self.layers = (Layer1(), Layer2())
            self.z3 = Parameter(
                Tensor(np.full([128, 96], 0.6, dtype=np.float32)), name='z3')

        def construct(self, index, x):
            ret = F.switch_layer(index, self.layers)(x) * self.z3
            return ret

    index = Tensor(0, dtype=mstype.int32)
    net = SwitchLayerCell()
    net(index, Tensor(np.full([128, 96], 0.6, dtype=np.float32)))
    C.grad_by_list(net, ParameterTuple(net.trainable_params()))(index,
                                                                Tensor(np.full([128, 96], 0.6, dtype=np.float32)))
    C.grad_all(net)(index, Tensor(np.full([128, 96], 0.6, dtype=np.float32)))


def test_index_to_switch_layer():
    class Layer1(nn.Cell):
        def __init__(self):
            super(Layer1, self).__init__()
            self.z1 = Parameter(
                Tensor(np.full([128, 96], 0.6, dtype=np.float32)), name='z1')

        def construct(self, x):
            return x * self.z1

    class Layer2(nn.Cell):
        def __init__(self):
            super(Layer2, self).__init__()
            self.z2 = Parameter(
                Tensor(np.full([128, 96], 0.6, dtype=np.float32)), name='z2')

        def construct(self, x):
            return x * self.z2

    class SwitchLayerCell(nn.Cell):
        def __init__(self):
            super(SwitchLayerCell, self).__init__()
            self.layers = (Layer1(), Layer2())
            self.z3 = Parameter(
                Tensor(np.full([128, 96], 0.6, dtype=np.float32)), name='z3')

        def construct(self, index, x):
            ret = self.layers[index](x) * self.z3
            return ret

    index = Tensor(0, dtype=mstype.int32)
    net = SwitchLayerCell()
    net(index, Tensor(np.full([128, 96], 0.6, dtype=np.float32)))
    C.grad_by_list(net, ParameterTuple(net.trainable_params()))(index,
                                                                Tensor(np.full([128, 96], 0.6, dtype=np.float32)))
    C.grad_all(net)(index, Tensor(np.full([128, 96], 0.6, dtype=np.float32)))


def test_switch_layer_with_single_prim():
    class SwitchLayerCell(nn.Cell):
        def __init__(self):
            super(SwitchLayerCell, self).__init__()
            self.layers = (nn.ReLU(), nn.ReLU())
            self.z3 = Parameter(
                Tensor(np.full([128, 96], 0.6, dtype=np.float32)), name='z3')

        def construct(self, index, x):
            ret = self.layers[index](x) * self.z3
            return ret

    index = Tensor(0, dtype=mstype.int32)
    net = SwitchLayerCell()
    net(index, Tensor(np.full([128, 96], 0.6, dtype=np.float32)))
    C.grad_by_list(net, ParameterTuple(net.trainable_params()))(index,
                                                                Tensor(np.full([128, 96], 0.6, dtype=np.float32)))
    C.grad_all(net)(index, Tensor(np.full([128, 96], 0.6, dtype=np.float32)))


def test_control_depend_check():
    with pytest.raises(TypeError) as e:
        P.ControlDepend(0.0)
        print(e)
    with pytest.raises(ValueError) as e:
        P.ControlDepend(2)
        print(e)
    with pytest.raises(TypeError) as e:
        P.ControlDepend((2,))
        print(e)


def test_if_nested_compile():
    class Net(nn.Cell):
        def __init__(self, auto_prefix=True):
            super().__init__(auto_prefix=auto_prefix)
            self.squre = P.Square()
            self.value = Tensor(3, dtype=ms.float32)

        def construct(self, x, y):
            res = self.value
            if x <= y:
                res = x + res
                res = y + res
            else:
                if x == y:
                    res = self.squre(self.value * y)
                else:
                    res = self.squre(self.value)
            return res
    x = Tensor(1.0, dtype=ms.float32)
    y = Tensor(2.0, dtype=ms.float32)
    net = Net()
    net(x, y)


def test_if_inside_for():
    class Net(nn.Cell):
        def __init__(self, auto_prefix=True):
            super().__init__(auto_prefix=auto_prefix)
            self.squre = P.Square()
            self.value = Tensor(3, dtype=ms.float32)
            self.count = 4

        def construct(self, x, y):
            res = 0
            for i in range(self.count):
                if i == x:
                    res = res + x
                else:
                    res = res - y
            return res
    c1 = Tensor(1, dtype=ms.int32)
    c2 = Tensor(1, dtype=ms.int32)
    net = Net()
    net(c1, c2)


def test_while_in_while():
    c1 = Tensor(1, dtype=ms.int32)
    c2 = Tensor(2, dtype=ms.int32)
    c3 = Tensor(3, dtype=ms.int32)
    c4 = Tensor(4, dtype=ms.int32)
    @ms_function
    def while_in_while(x, y, z, u):
        out = c4
        while x < y:
            z = c4 + c4
            while z < y:
                z = z + 1
                out = out + 1
            x = x + 1

        out = out + 3
        return out
    while_in_while(c1, c2, c3, c4)


def test_tensor_cond():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.t = Tensor(np.array(0, np.bool))
            self.t1 = Tensor(np.array([True], np.bool))
        def construct(self, x, y):
            t = 0
            if self.t:
                t = t - x * y
            else:
                t = t - x / y
            if self.t1:
                t = t + x / y
            else:
                t = t + x * y
            return t
            
            
    x = Tensor(np.ones([6, 8, 10], np.int32))
    y = Tensor(np.ones([6, 8, 10], np.int32))
    net = Net()
    out = net(x, y)

def test_tensor_cond_exception():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.t = Tensor(np.array([True, False], np.bool))
        def construct(self, x, y):
            t = 0
            if self.t:
                t = t - x * y
            else:
                t = t - x / y
            return t
            
            
    x = Tensor(np.ones([6, 8, 10], np.int32))
    y = Tensor(np.ones([6, 8, 10], np.int32))
    net = Net()
    with pytest.raises(ValueError):
        out = net(x, y)

def test_while_scalar():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.x = 10
        def construct(self, x, y):
            i = 0
            t = 0
            while (i < 10):
                t = t + x + y
                i = i + 1
            return t
    net = Net()
    x = Tensor(np.ones([6, 8, 10], np.int32))
    y = Tensor(np.ones([6, 8, 10], np.int32))
    out = net(x, y)

def test_while_tensor():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.t = Tensor(np.ones([6, 8, 10], np.int32))
            self.count = Tensor(np.array([10], np.int32))
        def construct(self, x, y):
            i = 0
            t = self.t
            while (i < self.count):
                t = t + x + y
                i = i + 1
            return t
    net = Net()
    x = Tensor(np.ones([6, 8, 10], np.int32))
    y = Tensor(np.ones([6, 8, 10], np.int32))
    out = net(x, y)


def test_large_for_loop():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.flatten = P.ReLU() #nn.Flatten()

        def construct(self, x):
            for elem in range(1, 19000):
                x = self.flatten(x + elem)
            return x

    t = Tensor(np.ones([2, 3], dtype=np.float32))
    net = Net()
    net(t)


def test_large_for_loop_with_continue_break():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.flatten = P.ReLU() #nn.Flatten()

        def construct(self, x):
            idx = 0
            for elem1 in range(200):
                idx = idx + 1
                if idx < 10:
                    x = x + 0.5
                    continue
                if idx > 500:
                    break
                x = self.flatten(x + elem1)
            return x

    t = Tensor(np.ones([2, 3], dtype=np.float32))
    net = Net()
    net(t)


def test_mixed_precision_cast():
    x = Tensor(np.ones([2, 3], dtype=np.float32))
    z = F.mixed_precision_cast(mstype.float16, x)
    assert z.dtype == mstype.float16


def test_while_concat():
    class Net(nn.Cell):
        def __init__(self, data):
            super(Net, self).__init__()
            self.start = Tensor(0, dtype=mstype.int32)
            self.end = Tensor(2, dtype=mstype.int32)
            self.out = Tensor(np.zeros([2, 3], dtype=np.float32))
            self.concat = P.Concat()

        def construct(self, inputs):
            idx = self.start
            end = self.end
            out = self.out
            while idx < end:
                xi = inputs[idx, :, :]
                out = self.concat((out, xi))
                idx = idx + 1
            return out

    x = Tensor(np.arange(10 * 2 * 3).reshape(10, 2, 3).astype(np.float32))
    net = Net(x)
    net(x)


def test_tensor_all_construct_lack_branch():
    class NetConditionLackBranch(nn.Cell):
        def __init__(self):
            super(NetConditionLackBranch, self).__init__()
            self.logicaland = P.LogicalAnd()
            self.logicalor = P.LogicalOr()

        def construct(self, input1, input2):
            if input1.all():
                return self.logicaland(input1, input2)
            while input1.any():
                return  self.logicalor(input1, input2)
            # NOTICE: here missing return statement, default return None

    input_np_1 = np.random.choice([True], size=(2, 3, 4, 5))
    input_tensor_1 = Tensor(input_np_1)
    input_np_2 = np.random.choice([True, False], size=(2, 3, 4, 5))
    input_tensor_2 = Tensor(input_np_2)
    net = NetConditionLackBranch()
    with pytest.raises(Exception):
        net(input_tensor_1, input_tensor_2)
