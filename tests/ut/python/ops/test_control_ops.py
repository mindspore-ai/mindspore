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
import os
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
from mindspore.common import jit

context.set_context(mode=context.GRAPH_MODE)

grad_by_list = C.GradOperation(get_by_list=True)
grad_all = C.GradOperation(get_all=True)
grad_all_with_sens = C.GradOperation(get_all=True, sens_param=True)


def cond_data_test(x_init, y_init):
    class Net(nn.Cell):
        def __init__(self):
            """"""
            super(Net, self).__init__()
            self.square = P.Square()
            self.add = P.Add()
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
            self.add = P.Add()
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
    grad_by_list(net, ParameterTuple(net.trainable_params()))(index,
                                                              Tensor(np.full([128, 96], 0.6, dtype=np.float32)))
    grad_all(net)(index, Tensor(np.full([128, 96], 0.6, dtype=np.float32)))


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
    grad_by_list(net, ParameterTuple(net.trainable_params()))(index,
                                                              Tensor(np.full([128, 96], 0.6, dtype=np.float32)))
    grad_all(net)(index, Tensor(np.full([128, 96], 0.6, dtype=np.float32)))


def test_parser_switch_layer_switch_in_bprop():
    class OneInputBprop(nn.Cell):
        def __init__(self, funcs):
            super(OneInputBprop, self).__init__()
            self.op = P.ReLU()
            self.funcs = funcs

        def construct(self, i, x):
            return self.op(x)

        def bprop(self, i, x, out, dout):
            return i, self.funcs[i](x, dout)

    class Add(nn.Cell):
        def __init__(self):
            super().__init__()
            self.op = P.Add()

        def construct(self, x, y):
            return self.op(x, y)

    class Mul(nn.Cell):
        def __init__(self):
            super().__init__()
            self.op = P.Mul()

        def construct(self, x, y):
            return self.op(x, y)

    func1 = Add()
    func2 = Mul()
    funcs = (func1, func2)
    net = OneInputBprop(funcs)
    input1 = Tensor(np.ones([2, 2]).astype(np.float32))
    grad = Tensor(np.random.randn(2, 2).astype(np.float32))
    i = Tensor(1, mstype.int32)
    grad_net = grad_all_with_sens(net)
    grad_net(i, input1, grad)


def test_parser_switch_layer_inputs_tuple():
    class TwoInputTupleFinalNet(nn.Cell):
        def __init__(self, funcs):
            super().__init__()
            self.funcs = funcs

        def construct(self, i, inputa, inputb):
            inputs = (inputa, inputb)
            x = self.funcs[i](inputs)
            return x

    class Add(nn.Cell):
        def __init__(self):
            super().__init__()
            self.op = P.Add()

        def construct(self, x):
            y = self.op(x[0], x[1])
            return self.op(x[0], y)

    class Mul(nn.Cell):
        def __init__(self):
            super().__init__()
            self.op = P.Mul()

        def construct(self, x):
            y = self.op(x[0], x[1])
            return self.op(x[0], y)

    func1 = Add()
    func2 = Mul()

    funcs = (func1, func2)
    net = TwoInputTupleFinalNet(funcs)

    input1 = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    input2 = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    i = Tensor(1, mstype.int32)
    grad = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    back_net = grad_all_with_sens(net)
    back_out = back_net(i, input1, input2, grad)


def test_switch_layer_env_eliminate():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(1, 1, 3, pad_mode='same')
            self.conv2 = nn.Conv2d(1, 1, 5, pad_mode='same')
            self.funs = (self.conv, self.conv2)

        def construct(self, x, index):
            x = self.funs[index](x)
            return x

    class NetGrad(nn.Cell):
        def __init__(self, net):
            super(NetGrad, self).__init__()
            self.grad_op = C.GradOperation(get_by_list=True, sens_param=False)
            self.net = net
            self.weights = ParameterTuple(self.net.trainable_params())

        def construct(self, x, index):
            weights = self.weights
            grad = self.grad_op(self.net, weights)(x, index)
            return grad

    net = Net()
    net2 = NetGrad(net)
    x = Tensor(np.ones((3, 1, 12, 12)), ms.float32)
    i = Tensor(1, ms.int32)
    net2(x, i)


def test_switch_layer_single_layer():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(1, 1, 3, pad_mode='same')
            self.funs = (self.conv,)

        def construct(self, x, index):
            x = self.funs[index](x)
            return x

    class NetGrad(nn.Cell):
        def __init__(self, net):
            super(NetGrad, self).__init__()
            self.grad_op = C.GradOperation(get_by_list=True, sens_param=False)
            self.net = net
            self.weights = ParameterTuple(self.net.trainable_params())

        def construct(self, x, index):
            weights = self.weights
            grad = self.grad_op(self.net, weights)(x, index)
            return grad

    net = Net()
    net2 = NetGrad(net)
    x = Tensor(np.ones((3, 1, 12, 12)), ms.float32)
    i = Tensor(1, ms.int32)
    net2(x, i)


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

    @jit
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


def test_while_with_weight_in_condition():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.loop = Parameter(Tensor(1, dtype=ms.float32), name="loop")

        def construct(self, x):
            while self.loop < 5:
                self.loop += 1
                x += 1
            return x

    net = Net()
    x = Tensor(-1, dtype=ms.float32)
    grad_all(net)(x)


def test_mixed_precision_cast():
    x = Tensor(np.ones([2, 3], dtype=np.float32))
    z = F.mixed_precision_cast(mstype.float16, x)
    assert z.dtype == mstype.float16


def test_while_add():
    class Net(nn.Cell):
        def __init__(self, data):
            super(Net, self).__init__()
            self.start = Tensor(0, dtype=mstype.int32)
            self.end = Tensor(2, dtype=mstype.int32)
            self.out = Tensor(np.zeros([2, 3], dtype=np.float32))
            self.add = P.Add()

        def construct(self, inputs):
            idx = self.start
            end = self.end
            out = self.out
            while idx < end:
                xi = inputs[idx, :, :]
                out = self.add(out, xi)
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
                return self.logicalor(input1, input2)
            # NOTICE: here missing return statement, default return None

    input_np_1 = np.random.choice([True], size=(2, 3, 4, 5))
    input_tensor_1 = Tensor(input_np_1)
    input_np_2 = np.random.choice([True, False], size=(2, 3, 4, 5))
    input_tensor_2 = Tensor(input_np_2)
    net = NetConditionLackBranch()
    with pytest.raises(Exception):
        net(input_tensor_1, input_tensor_2)


def test_parser_switch_layer_func_primitive():
    class FinalNet(nn.Cell):
        def __init__(self, funcs):
            super().__init__()
            self.funcs = funcs

        def construct(self, i, input1):
            x = self.funcs[i](input1)
            return x

    func1 = P.ReLU()
    func2 = P.Softmax()
    funcs = (func1, func2)
    net = FinalNet(funcs)

    input1 = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    i = Tensor(1, mstype.int32)

    with pytest.raises(ValueError):
        net(i, input1)


def test_switch_layer_shape_join_failed():
    class AddFuncNet(nn.Cell):
        def __init__(self, funcs, new_func):
            super(AddFuncNet, self).__init__()
            self.funcs = funcs
            self.new_func = new_func

        def construct(self, i, inputs):
            final_funcs = self.funcs + (self.new_func,)
            x = final_funcs[i](inputs)
            return x

    class ReLUTuple(nn.Cell):
        def __init__(self):
            super(ReLUTuple, self).__init__()
            self.op = nn.ReLU()

        def construct(self, x):
            return self.op(x[0])

    func1 = nn.Softmax()
    func2 = nn.ReLU()
    func3 = ReLUTuple()

    funcs = (func1, func2)

    net = AddFuncNet(funcs, func3)

    inp = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    i = Tensor(1, mstype.int32)
    net(i, inp)


def test_switch_layer_dtype_join_failed():
    class Cast(nn.Cell):
        def __init__(self, dtype):
            super(Cast, self).__init__()
            self.op = P.Cast()
            self.dtype = dtype

        def construct(self, x):
            y = self.op(x, self.dtype)
            return y + y

    class SwitchNegNet(nn.Cell):
        def __init__(self, funcs):
            super(SwitchNegNet, self).__init__()
            self.funcs = funcs
            self.op = P.Neg()

        def construct(self, i, inputs):
            x = self.funcs[i](inputs)
            x = self.op(x)
            return x

    func1 = nn.ReLU()
    func2 = Cast(mstype.int32)
    funcs = (func1, func2)
    net = SwitchNegNet(funcs)

    inp = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    i = Tensor(0, mstype.int32)

    with pytest.raises(TypeError) as err:
        net(i, inp)


def test_large_for_loop():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.flatten = P.ReLU()  # nn.Flatten()

        def construct(self, x):
            for elem in range(1, 1900):
                x = self.flatten(x + elem)
            return x

    t = Tensor(np.ones([2, 3], dtype=np.float32))
    net = Net()
    os.environ['MS_DEV_RECURSIVE_EVAL'] = '1'
    old_max_call_depth = context.get_context('max_call_depth')
    context.set_context(max_call_depth=60)
    with pytest.raises(RuntimeError) as err:
        net(t)
    context.set_context(max_call_depth=old_max_call_depth)
    os.environ['MS_DEV_RECURSIVE_EVAL'] = '0'
    assert 'Exceed function call depth limit 60' in str(err.value)


def test_large_for_loop_case2():
    class Menet(nn.Cell):
        def __init__(self, axis, flag_boottom, flag_top):
            super(Menet, self).__init__()
            self.squeeze = P.Squeeze(axis)
            self.expanddims = P.ExpandDims()
            self.flatten = nn.Flatten()
            self.neg = P.Neg()
            self.axis = axis
            self.flag_boottom = flag_boottom
            self.flag_top = flag_top

        def construct(self, x):
            if self.flag_boottom:
                x = self.neg(x)
            for i in range(0, 1500):
                x = self.expanddims(x, self.axis)
                x = self.squeeze(x)
                x = self.flatten(x)
            if self.flag_top:
                x = self.neg(x)
            return x

    x = Tensor(np.ones([2, 3], dtype=np.float32))
    net = Menet(axis=0, flag_boottom=True, flag_top=True)
    os.environ['MS_DEV_RECURSIVE_EVAL'] = '1'
    old_max_call_depth = context.get_context('max_call_depth')
    context.set_context(max_call_depth=80)
    with pytest.raises(RuntimeError) as err:
        net(x)
    os.environ['MS_DEV_RECURSIVE_EVAL'] = '0'
    context.set_context(max_call_depth=old_max_call_depth)
    assert 'Exceed function call depth limit 80' in str(err.value)


def test_large_for_loop_with_continue_break():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.flatten = P.ReLU()  # nn.Flatten()

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

    os.environ['MS_DEV_RECURSIVE_EVAL'] = '1'
    old_max_call_depth = context.get_context('max_call_depth')
    context.set_context(max_call_depth=2000)
    t = Tensor(np.ones([2, 3], dtype=np.float32))
    net = Net()
    net(t)
    os.environ['MS_DEV_RECURSIVE_EVAL'] = '0'
    context.set_context(max_call_depth=old_max_call_depth)


def test_recursive_call():
    class Net(nn.Cell):
        """ Net definition """

        def __init__(self):
            super(Net, self).__init__()
            self.fc = nn.Dense(10, 10)  # padding=0

        def construct(self, x):
            net2 = Net2()
            x = net2(x)
            out = self.fc(x)
            return out

    class Net2(nn.Cell):
        def __init__(self):
            super(Net2, self).__init__()
            self.net = Net()
            self.fc = nn.Dense(10, 10)

        def construct(self, x):
            x = self.net(x)
            out = self.fc(x)
            return out

    context.set_context(mode=context.GRAPH_MODE)
    os.environ['MS_DEV_RECURSIVE_EVAL'] = '1'
    old_max_call_depth = context.get_context('max_call_depth')
    context.set_context(max_call_depth=80)
    input_data = Tensor(np.identity(10).astype(np.float32))
    net = Net2()
    with pytest.raises(RuntimeError):
        net(input_data)
    os.environ['MS_DEV_RECURSIVE_EVAL'] = '0'
    context.set_context(max_call_depth=old_max_call_depth)


# grad for Tensor(Bool) input and eliminate AddN(MakeTuple(Xs, zeros_like(Bool)))
def test_grad_tensor_bool():
    class Net(nn.Cell):

        def construct(self, x, y, z):
            out = z
            while x:
                out = out + z
                x = y
            return out

    x = Tensor(np.array(False).astype(np.bool))
    y = Tensor(np.array(False).astype(np.bool))
    z = Tensor(np.ones([2, 3], dtype=np.float32))
    net = grad_all(Net())
    net(x, y, z)
