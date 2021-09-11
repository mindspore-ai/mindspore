# Copyright 2021 Huawei Technologies Co., Ltd
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
""" test math ops """
import numpy as np
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE)


class Add(nn.Cell):
    def __init__(self):
        super(Add, self).__init__()
        self.add = P.Add()

    def construct(self, x, y):
        z = self.add(x, y)
        return z


def test_number_add_number():
    input_x = 0.1
    input_y = -3.2
    result1 = input_x + input_y
    add_net = Add()
    result2 = add_net(input_x, input_y)
    expect = -3.1
    assert result1 == expect
    assert result2 == expect


def test_tensor_add_tensor_int8():
    input_x = Tensor(np.ones(shape=[3])).astype(np.int8)
    input_y = Tensor(np.zeros(shape=[3])).astype(np.int8)
    result1 = input_x + input_y
    add_net = Add()
    result2 = add_net(input_x, input_y)
    expect = np.ones(shape=[3])
    assert np.all(result1.asnumpy() == expect)
    assert np.all(result2.asnumpy() == expect)


def test_tensor_add_tensor_int16():
    input_x = Tensor(np.ones(shape=[3])).astype(np.int16)
    input_y = Tensor(np.zeros(shape=[3])).astype(np.int16)
    result1 = input_x + input_y
    add_net = Add()
    result2 = add_net(input_x, input_y)
    expect = np.ones(shape=[3])
    assert np.all(result1.asnumpy() == expect)
    assert np.all(result2.asnumpy() == expect)


def test_tensor_add_tensor_int32():
    input_x = Tensor(np.ones(shape=[3])).astype(np.int32)
    input_y = Tensor(np.zeros(shape=[3])).astype(np.int32)
    result1 = input_x + input_y
    add_net = Add()
    result2 = add_net(input_x, input_y)
    expect = np.ones(shape=[3])
    assert np.all(result1.asnumpy() == expect)
    assert np.all(result2.asnumpy() == expect)


def test_tensor_add_tensor_int64():
    input_x = Tensor(np.ones(shape=[3])).astype(np.int64)
    input_y = Tensor(np.zeros(shape=[3])).astype(np.int64)
    result1 = input_x + input_y
    add_net = Add()
    result2 = add_net(input_x, input_y)
    expect = np.ones(shape=[3])
    assert np.all(result1.asnumpy() == expect)
    assert np.all(result2.asnumpy() == expect)


def test_tensor_add_tensor_uint8():
    input_x = Tensor(np.ones(shape=[3])).astype(np.uint8)
    input_y = Tensor(np.zeros(shape=[3])).astype(np.uint8)
    result1 = input_x + input_y
    add_net = Add()
    result2 = add_net(input_x, input_y)
    expect = np.ones(shape=[3])
    assert np.all(result1.asnumpy() == expect)
    assert np.all(result2.asnumpy() == expect)


def test_tensor_add_tensor_uint16():
    input_x = Tensor(np.ones(shape=[3])).astype(np.uint16)
    input_y = Tensor(np.zeros(shape=[3])).astype(np.uint16)
    result1 = input_x + input_y
    add_net = Add()
    result2 = add_net(input_x, input_y)
    expect = np.ones(shape=[3])
    assert np.all(result1.asnumpy() == expect)
    assert np.all(result2.asnumpy() == expect)


def test_tensor_add_tensor_uint32():
    input_x = Tensor(np.ones(shape=[3])).astype(np.uint32)
    input_y = Tensor(np.zeros(shape=[3])).astype(np.uint32)
    result1 = input_x + input_y
    add_net = Add()
    result2 = add_net(input_x, input_y)
    expect = np.ones(shape=[3])
    assert np.all(result1.asnumpy() == expect)
    assert np.all(result2.asnumpy() == expect)


def test_tensor_add_tensor_uint64():
    input_x = Tensor(np.ones(shape=[3])).astype(np.uint64)
    input_y = Tensor(np.zeros(shape=[3])).astype(np.uint64)
    result1 = input_x + input_y
    add_net = Add()
    result2 = add_net(input_x, input_y)
    expect = np.ones(shape=[3])
    assert np.all(result1.asnumpy() == expect)
    assert np.all(result2.asnumpy() == expect)


def test_tensor_add_tensor_float16():
    input_x = Tensor(np.ones(shape=[3])).astype(np.float16)
    input_y = Tensor(np.zeros(shape=[3])).astype(np.float16)
    result1 = input_x + input_y
    add_net = Add()
    result2 = add_net(input_x, input_y)
    expect = np.ones(shape=[3])
    assert np.all(result1.asnumpy() == expect)
    assert np.all(result2.asnumpy() == expect)


def test_tensor_add_tensor_float32():
    input_x = Tensor(np.ones(shape=[3])).astype(np.float32)
    input_y = Tensor(np.zeros(shape=[3])).astype(np.float32)
    result1 = input_x + input_y
    add_net = Add()
    result2 = add_net(input_x, input_y)
    expect = np.ones(shape=[3])
    assert np.all(result1.asnumpy() == expect)
    assert np.all(result2.asnumpy() == expect)


def test_tensor_add_tensor_float64():
    input_x = Tensor(np.ones(shape=[3])).astype(np.float64)
    input_y = Tensor(np.zeros(shape=[3])).astype(np.float64)
    result1 = input_x + input_y
    add_net = Add()
    result2 = add_net(input_x, input_y)
    expect = np.ones(shape=[3])
    assert np.all(result1.asnumpy() == expect)
    assert np.all(result2.asnumpy() == expect)


def test_tensor_add_number():
    input_x = Tensor(np.ones(shape=[3])).astype(np.float32)
    input_y = -0.4
    result1 = input_x + input_y
    add_net = Add()
    result2 = add_net(input_x, input_y)
    expect = np.ones(shape=[3]) * 0.6
    assert np.all(result1.asnumpy() == expect.astype(np.float32))
    assert np.all(result2.asnumpy() == expect.astype(np.float32))


def test_tuple_add_tuple():
    input_x = (Tensor(np.ones(shape=[3])).astype(np.float32))
    input_y = (Tensor(np.ones(shape=[3])).astype(np.float32) * 2)
    result1 = input_x + input_y
    add_net = Add()
    result2 = add_net(input_x, input_y)
    expect = (np.ones(shape=[3]) * 3)
    assert np.all(result1.asnumpy() == expect.astype(np.float32))
    assert np.all(result2.asnumpy() == expect.astype(np.float32))


def test_tuple_add_tuple_shape():
    input_x = (Tensor(np.ones(shape=[3])).astype(np.float32))
    input_y = (Tensor(np.ones(shape=[4])).astype(np.float32) * 2)

    result1 = input_x + input_y
    add_net = Add()
    result2 = add_net(input_x, input_y)
    expect = (np.ones(shape=[3]) * 3)
    assert np.all(result1.asnumpy() == expect.astype(np.float32))
    assert np.all(result2.asnumpy() == expect.astype(np.float32))


def test_string_add_string():
    input_x = "string111_"
    input_y = "add_string222"
    result = input_x + input_y
    expect = "string111_add_string222"
    assert result == expect


def test_list_add_list():
    input_x = [1, 3, 5, 7, 9]
    input_y = ["0", "6"]
    result = input_x + input_y
    expect = [1, 3, 5, 7, 9, "0", "6"]
    assert result == expect


class Sub(nn.Cell):
    def __init__(self):
        super(Sub, self).__init__()
        self.sub = P.Sub()

    def construct(self, x, y):
        z = self.sub(x, y)
        return z


def test_number_sub_number():
    input_x = 10.11
    input_y = 902
    result1 = input_x - input_y
    sub_net = Sub()
    result2 = sub_net(input_x, input_y)
    expect = -891.89
    assert np.all(result1 == expect)
    assert np.all(result2 == expect)


def test_tensor_sub_tensor():
    input_x = Tensor(np.array([[2, 2], [3, 3]]))
    input_y = Tensor(np.array([[1, 2], [-3, 3]]))
    result1 = input_x - input_y
    sub_net = Sub()
    result2 = sub_net(input_x, input_y)
    expect = Tensor(np.array([[1, 0], [6, 0]]))
    assert np.all(result1.asnumpy() == expect.asnumpy())
    assert np.all(result2.asnumpy() == expect.asnumpy())


def test_tensor_sub_number():
    input_x = Tensor(np.array([[2, 2], [3, 3]]))
    input_y = -2
    result1 = input_x - input_y
    sub_net = Sub()
    result2 = sub_net(input_x, input_y)
    expect = Tensor(np.array([[4, 4], [5, 5]]))
    assert np.all(result1.asnumpy() == expect.asnumpy())
    assert np.all(result2.asnumpy() == expect.asnumpy())


def test_number_sub_tensor():
    input_x = Tensor(np.array([[2, 2], [3, 3]]))
    input_y = -2
    result1 = input_x - input_y
    sub_net = Sub()
    result2 = sub_net(input_x, input_y)
    expect = Tensor(np.array([[-4, -4], [-5, -5]]))
    assert np.all(result1.asnumpy() == expect.asnumpy())
    assert np.all(result2.asnumpy() == expect.asnumpy())


class Mul(nn.Cell):
    def __init__(self):
        super(Mul, self).__init__()
        self.mul = P.Mul()

    def construct(self, x, y):
        z = self.mul(x, y)
        return z


def test_number_mul_number():
    input_x = 4.91
    input_y = 0.16
    result1 = input_x * input_y
    mul_net = Mul()
    result2 = mul_net(input_x, input_y)
    expect = 0.7856
    diff1 = result1 - expect
    diff2 = result2 - expect
    error = 1.0e-6
    assert np.all(diff1 < error)
    assert np.all(-diff1 < error)
    assert np.all(diff2 < error)
    assert np.all(-diff2 < error)


def test_tensor_mul_tensor():
    input_x = Tensor(np.array([[2, 2], [3, 3]])).astype(np.float32)
    input_y = Tensor(np.array([[1, 2], [3, 1]])).astype(np.float32)
    result1 = input_x * input_y
    mul_net = Mul()
    result2 = mul_net(input_x, input_y)
    expect = Tensor(np.array([[2, 4], [9, 3]]))
    assert np.all(result1.asnumpy() == expect.asnumpy())
    assert np.all(result2.asnumpy() == expect.asnumpy())


def test_tensor_mul_number():
    input_x = Tensor(np.array([[2, 2], [3, 3]])).astype(np.float32)
    input_y = -1
    result1 = input_x * input_y
    mul_net = Mul()
    result2 = mul_net(input_x, input_y)
    expect = Tensor(np.array([[-2, -2], [-3, -3]]))
    assert np.all(result1.asnumpy() == expect.asnumpy())
    assert np.all(result2.asnumpy() == expect.asnumpy())


def test_number_mul_tensor():
    input_x = Tensor(np.array([[2, 2], [3, 3]])).astype(np.float32)
    input_y = -1
    result1 = input_x * input_y
    mul_net = Mul()
    result2 = mul_net(input_x, input_y)
    expect = Tensor(np.array([[-2, -2], [-3, -3]]))
    assert np.all(result1.asnumpy() == expect.asnumpy())
    assert np.all(result2.asnumpy() == expect.asnumpy())


class Div(nn.Cell):
    def __init__(self):
        super(Div, self).__init__()
        self.div = P.Div()

    def construct(self, x, y):
        z = self.div(x, y)
        return z


def test_number_div_number():
    input_x = 4
    input_y = -1
    result1 = input_x / input_y
    div_net = Div()
    result2 = div_net(input_x, input_y)
    expect = -4
    assert np.all(result1 == expect)
    assert np.all(result2 == expect)


def test_tensor_div_tensor():
    input_x = Tensor(np.array([[2, 2], [3, 3]])).astype(np.float32)
    input_y = Tensor(np.array([[1, 2], [3, 1]])).astype(np.float32)
    result1 = input_x / input_y
    div_net = Div()
    result2 = div_net(input_x, input_y)
    expect = Tensor(np.array([[2, 1], [1, 3]]))
    assert np.all(result1.asnumpy() == expect.asnumpy())
    assert np.all(result2.asnumpy() == expect.asnumpy())


def test_tensor_div_number():
    input_x = Tensor(np.array([[2, 2], [3, 3]])).astype(np.float32)
    input_y = 2
    result1 = input_x / input_y
    div_net = Div()
    result2 = div_net(input_x, input_y)
    expect = Tensor(np.array([[1, 1], [1.5, 1.5]]))
    assert np.all(result1.asnumpy() == expect.asnumpy())
    assert np.all(result2.asnumpy() == expect.asnumpy())


def test_number_div_tensor():
    input_x = Tensor(np.array([[2, 2], [4, 4]])).astype(np.float32)
    input_y = 2
    result1 = input_x / input_y
    div_net = Div()
    result2 = div_net(input_x, input_y)
    expect = Tensor(np.array([[1, 1], [0.5, 0.5]]))
    assert np.all(result1.asnumpy() == expect.asnumpy())
    assert np.all(result2.asnumpy() == expect.asnumpy())


class Mod(nn.Cell):
    def __init__(self):
        super(Mod, self).__init__()
        self.mod = P.Mod()

    def construct(self, x, y):
        z = self.mod(x, y)
        return z


def test_number_mod_number():
    input_x = 19
    input_y = 2
    result1 = input_x % input_y
    mod_net = Mod()
    result2 = mod_net(input_x, input_y)
    expect = 1
    assert np.all(result1 == expect)
    assert np.all(result2 == expect)


def test_tensor_mod_tensor():
    input_x = Tensor(np.array([[2, 2], [4, 4]])).astype(np.float32)
    input_y = Tensor(np.array([[2, 2], [4, 4]])).astype(np.float32)
    result1 = input_x % input_y
    mod_net = Mod()
    result2 = mod_net(input_x, input_y)
    expect = Tensor(np.array([[0, 0], [0, 0]])).astype(np.float32)
    assert np.all(result1.asnumpy() == expect.asnumpy())
    assert np.all(result2.asnumpy() == expect.asnumpy())


def test_tensor_mod_number():
    input_x = Tensor(np.array([[2, 2], [4, 4]])).astype(np.float32)
    input_y = -1
    result1 = input_x % input_y
    mod_net = Mod()
    result2 = mod_net(input_x, input_y)
    expect = Tensor(np.array([[0, 0], [0, 0]])).astype(np.float32)
    assert np.all(result1.asnumpy() == expect.asnumpy())
    assert np.all(result2.asnumpy() == expect.asnumpy())


def test_number_mod_tensor():
    input_x = Tensor(np.array([[2, 2], [4, 4]])).astype(np.float32)
    input_y = 5
    result1 = input_x % input_y
    mod_net = Mod()
    result2 = mod_net(input_x, input_y)
    expect = Tensor(np.array([[1, 1], [1, 1]])).astype(np.float32)
    assert np.all(result1.asnumpy() == expect.asnumpy())
    assert np.all(result2.asnumpy() == expect.asnumpy())


class Pow(nn.Cell):
    def __init__(self):
        super(Pow, self).__init__()
        self.pow = P.Pow()

    def construct(self, x, y):
        z = self.pow(x, y)
        return z


def test_number_pow_number():
    input_x = 2
    input_y = 5
    result1 = input_x ** input_y
    pow_net = Pow()
    result2 = pow_net(input_x, input_y)
    expect = 32
    assert np.all(result1 == expect)
    assert np.all(result2 == expect)


def test_tensor_pow_tensor():
    input_x = Tensor(np.array([[2, 2], [4, 4]])).astype(np.float32)
    input_y = Tensor(np.array([[2, 2], [4, 4]])).astype(np.float32)
    result1 = input_x ** input_y
    pow_net = Pow()
    result2 = pow_net(input_x, input_y)
    expect = Tensor(np.array([[4, 4], [256, 256]])).astype(np.float32)
    assert np.all(result1.asnumpy() == expect.asnumpy())
    assert np.all(result2.asnumpy() == expect.asnumpy())


def test_tensor_pow_number():
    input_x = Tensor(np.array([[2, 2], [4, 4]])).astype(np.float32)
    input_y = 3
    result1 = input_x ** input_y
    pow_net = Pow()
    result2 = pow_net(input_x, input_y)
    expect = Tensor(np.array([[8, 8], [64, 64]])).astype(np.float32)
    assert np.all(result1.asnumpy() == expect.asnumpy())
    assert np.all(result2.asnumpy() == expect.asnumpy())


def test_number_pow_tensor():
    input_x = Tensor(np.array([[2, 2], [4, 4]])).astype(np.float32)
    input_y = 3
    result1 = input_x ** input_y
    pow_net = Pow()
    result2 = pow_net(input_x, input_y)
    expect = Tensor(np.array([[9, 9], [81, 81]])).astype(np.float32)
    assert np.all(result1.asnumpy() == expect.asnumpy())
    assert np.all(result2.asnumpy() == expect.asnumpy())


class FloorDiv(nn.Cell):
    def __init__(self):
        super(FloorDiv, self).__init__()
        self.floordiv = P.FloorDiv()

    def construct(self, x, y):
        z = self.floordiv(x, y)
        return z


def test_number_floordiv_number():
    input_x = 2
    input_y = 5
    result1 = input_x // input_y
    floordiv_net = FloorDiv()
    result2 = floordiv_net(input_x, input_y)
    expect = 0
    assert np.all(result1 == expect)
    assert np.all(result2 == expect)


def test_tensor_floordiv_tensor():
    input_x = Tensor(np.array([[2, 2], [4, 4]])).astype(np.float32)
    input_y = Tensor(np.array([[1, 2], [-2, 4]])).astype(np.float32)
    result1 = input_x // input_y
    floordiv_net = FloorDiv()
    result2 = floordiv_net(input_x, input_y)
    expect = Tensor(np.array([[2, 1], [-2, 1]])).astype(np.float32)
    assert np.all(result1.asnumpy() == expect.asnumpy())
    assert np.all(result2.asnumpy() == expect.asnumpy())


def test_tensor_floordiv_number():
    input_x = Tensor(np.array([[2, 2], [4, 4]])).astype(np.float32)
    input_y = 3
    result1 = input_x // input_y
    floordiv_net = FloorDiv()
    result2 = floordiv_net(input_x, input_y)
    expect = Tensor(np.array([[0, 0], [1, 1]])).astype(np.float32)
    assert np.all(result1.asnumpy() == expect.asnumpy())
    assert np.all(result2.asnumpy() == expect.asnumpy())


def test_number_floordiv_tensor():
    input_x = Tensor(np.array([[2, 2], [4, 4]])).astype(np.float32)
    input_y = 3
    result1 = input_x // input_y
    floordiv_net = FloorDiv()
    result2 = floordiv_net(input_x, input_y)
    expect = Tensor(np.array([[1, 1], [0, 0]])).astype(np.float32)
    assert np.all(result1.asnumpy() == expect.asnumpy())
    assert np.all(result2.asnumpy() == expect.asnumpy())


def test_number_floormod_number():
    input_x = 2
    input_y = 5
    result1 = input_x // input_y
    floordiv_net = FloorDiv()
    result2 = floordiv_net(input_x, input_y)
    expect = 2
    assert np.all(result1 == expect)
    assert np.all(result2 == expect)


def test_tensor_floormod_tensor():
    input_x = Tensor(np.array([[2, 2], [4, 4]])).astype(np.float32)
    input_y = Tensor(np.array([[1, 2], [-2, 4]])).astype(np.float32)
    result1 = input_x // input_y
    floordiv_net = FloorDiv()
    result2 = floordiv_net(input_x, input_y)
    expect = Tensor(np.array([[1, 0], [-2, 0]])).astype(np.float32)
    assert np.all(result1.asnumpy() == expect.asnumpy())
    assert np.all(result2.asnumpy() == expect.asnumpy())


def test_tensor_floormod_number():
    input_x = Tensor(np.array([[2, 2], [4, 4]])).astype(np.float32)
    input_y = 3
    result1 = input_x // input_y
    floordiv_net = FloorDiv()
    result2 = floordiv_net(input_x, input_y)
    expect = Tensor(np.array([[2, 2], [1, 1]])).astype(np.float32)
    assert np.all(result1.asnumpy() == expect.asnumpy())
    assert np.all(result2.asnumpy() == expect.asnumpy())


def test_number_floormod_tensor():
    input_x = Tensor(np.array([[2, 2], [4, 4]])).astype(np.float32)
    input_y = 3
    result1 = input_x // input_y
    floordiv_net = FloorDiv()
    result2 = floordiv_net(input_x, input_y)
    expect = Tensor(np.array([[1, 1], [3, 3]])).astype(np.float32)
    assert np.all(result1.asnumpy() == expect.asnumpy())
    assert np.all(result2.asnumpy() == expect.asnumpy())
