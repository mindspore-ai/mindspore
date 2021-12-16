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
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter

context.set_context(mode=context.GRAPH_MODE)

grad_by_list = C.GradOperation(get_by_list=True)
grad_all = C.GradOperation(get_all=True)
grad_all_with_sens = C.GradOperation(get_all=True, sens_param=True)


def if_compile_test(x_init, y_init):
    """
    Feature: if compile test.
    Description: if compile test
    Expectation: compile done without error.
    """
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


def test_if_nested_compile():
    """
    Feature: if nested compile test.
    Description: if nested compile test
    Expectation: compile done without error.
    """
    class Net(nn.Cell):
        def __init__(self, auto_prefix=True):
            """"""
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
    """
    Feature: if inside test.
    Description: if inside test
    Expectation: compile done without error.
    """
    class Net(nn.Cell):
        def __init__(self, auto_prefix=True):
            """"""
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


def test_while_with_weight_in_condition():
    """
    Feature: while with weight in condition test.
    Description: while with weight in condition test
    Expectation: compile done without error.
    """
    class Net(nn.Cell):
        def __init__(self):
            """"""
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


def test_while_add():
    """
    Feature: while add test.
    Description: while add test
    Expectation: compile done without error.
    """
    class Net(nn.Cell):
        def __init__(self, data):
            """"""
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
    """
    Feature: tensor all construct lack test.
    Description: tensor all construct lack test
    Expectation: compile done without error.
    """
    class NetConditionLackBranch(nn.Cell):
        def __init__(self):
            """"""
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
    """
    Feature: parser switch layer func primitive test.
    Description: parser switch layer func primitive test
    Expectation: compile done without error.
    """
    class FinalNet(nn.Cell):
        def __init__(self, funcs):
            """"""
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


def test_large_for_loop():
    """
    Feature: large for loop test.
    Description: large for loop test
    Expectation: compile done without error.
    """
    class Net(nn.Cell):
        def __init__(self):
            """"""
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


def test_large_for_loop_with_continue_break():
    """
    Feature: large for loop with continue break test.
    Description: large for loop with continue break test
    Expectation: compile done without error.
    """
    class Net(nn.Cell):
        def __init__(self):
            """"""
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
    """
    Feature: recursive call test.
    Description: recursive call test
    Expectation: compile done without error.
    """
    class Net(nn.Cell):
        """ Net definition """
        def __init__(self):
            """"""
            super(Net, self).__init__()
            self.fc = nn.Dense(10, 10)  # padding=0
            # self.net2 = Net2()

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


def test_pow():
    """
    Feature: pow test.
    Description: pow test
    Expectation: compile done without error.
    """
    input_tensor = Tensor(np.array([[2, 2], [3, 3]]))
    power = Tensor(np.array(3.0, np.int64))
    testpow = P.Pow()
    expect = np.array([[8, 8], [27, 27]])
    result = testpow(input_tensor, power)
    assert np.all(result.asnumpy() == expect)


def test_pow1():
    """
    Feature: pow one test.
    Description: pow one test
    Expectation: compile done without error.
    """
    input_tensor = Tensor(np.array([[2, 2], [2, 2]]))
    power = Tensor(np.array(3.0, np.int64))
    testpow = P.Pow()
    expect = np.array([[8, 8], [8, 8]])
    result = testpow(input_tensor, power)
    assert np.all(result.asnumpy() == expect)


def test_pow2():
    """
    Feature: pow two test.
    Description: pow two test
    Expectation: compile done without error.
    """
    input_tensor = Tensor(np.array([[1, 1], [2, 2]]))
    power = Tensor(np.array(3.0, np.int64))
    testpow = P.Pow()
    expect = np.array([[1, 1], [8, 8]])
    result = testpow(input_tensor, power)
    assert np.all(result.asnumpy() == expect)


def test_pow3():
    """
    Feature: pow three test.
    Description: pow three test
    Expectation: compile done without error.
    """
    input_tensor = Tensor(np.array([[2, 2], [1, 1]]))
    power = Tensor(np.array(3.0, np.int64))
    testpow = P.Pow()
    expect = np.array([[8, 8], [1, 1]])
    result = testpow(input_tensor, power)
    assert np.all(result.asnumpy() == expect)


def test_exp():
    """
    Feature: exp test.
    Description: exp test
    Expectation: compile done without error.
    """
    input_tensor = Tensor(np.array([[2, 2], [3, 3]]))
    testexp = P.Exp()
    result = testexp(input_tensor)
    expect = np.exp(np.array([[2, 2], [3, 3]]))
    assert np.all(result.asnumpy() == expect)


def test_exp1():
    """
    Feature: exp one test.
    Description: exp one test
    Expectation: compile done without error.
    """
    input_tensor = Tensor(np.array([[2, 2], [3, 3]]))
    testexp = P.Exp()
    result = testexp(input_tensor)
    expect = np.exp(np.array([[2, 2], [3, 3]]))
    assert np.all(result.asnumpy() == expect)


def test_realdiv():
    """
    Feature: realdiv test.
    Description: realdiv test
    Expectation: compile done without error.
    """
    x = Tensor(2048.0)
    y = Tensor(128.0)
    div = P.RealDiv()
    result = div(x, y)
    x = x.asnumpy()
    y = y.asnumpy()
    expect = x / y
    assert np.all(result.asnumpy() == expect)


def test_realdiv1():
    """
    Feature: realdiv one test.
    Description: realdiv one test
    Expectation: compile done without error.
    """
    x = Tensor(256.0)
    y = Tensor(128.0)
    div = P.RealDiv()
    result = div(x, y)
    x = x.asnumpy()
    y = y.asnumpy()
    expect = x / y
    assert np.all(result.asnumpy() == expect)


def test_eye():
    """
    Feature: eye test.
    Description: eye test
    Expectation: compile done without error.
    """
    x = np.arange(3)
    expect = np.ones_like(x)
    expect = np.diag(expect)
    eye = P.Eye()
    eye_output = eye(3, 3, ms.float32)
    assert np.all(eye_output.asnumpy() == expect)


def test_sub():
    """
    Feature: sub test.
    Description: sub test
    Expectation: compile done without error.
    """
    input_x = Tensor(np.ones(shape=[3]))
    input_y = Tensor(np.zeros(shape=[3]))

    sub = P.Sub()
    result = sub(input_x, input_y)
    expect = np.ones(shape=[3])
    assert np.all(result.asnumpy() == expect)


def test_square():
    """
    Feature: square test.
    Description: square test
    Expectation: compile done without error.
    """
    input_tensor = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    square = P.Square()
    result = square(input_tensor)
    expect = np.array([[1, 4, 9], [16, 25, 36]])
    assert np.all(result.asnumpy() == expect)


def test_sqrt():
    """
    Feature: sqrt test.
    Description: sqrt test
    Expectation: compile done without error.
    """
    input_tensor = Tensor(np.array([[4, 4], [9, 9]]))

    sqrt = P.Sqrt()
    expect = np.array([[2, 2], [3, 3]])
    result = sqrt(input_tensor)
    assert np.all(result.asnumpy() == expect)
