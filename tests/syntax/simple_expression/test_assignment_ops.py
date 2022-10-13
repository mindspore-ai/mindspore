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

import numpy as np
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE)


class Assign(nn.Cell):
    def __init__(self, x, y):
        super(Assign, self).__init__()
        self.x = Parameter(initializer(x, x.shape), name="x")
        self.y = Parameter(initializer(y, y.shape), name="y")
        self.assign = P.Assign()

    def construct(self):
        self.assign(self.y, self.x)
        return self.y


def test_assign_bool():
    x = Tensor(np.ones([3, 3]).astype(np.bool_))
    y = Tensor(np.zeros([3, 3]).astype(np.bool_))
    assign = Assign(x, y)
    output = assign()
    output = output.asnumpy()
    output_expect = np.ones([3, 3]).astype(np.bool_)
    print(output)
    assert np.all(output == output_expect)


def test_assign_int8():
    x = Tensor(np.ones([3, 3]).astype(np.int8))
    y = Tensor(np.zeros([3, 3]).astype(np.int8))
    assign = Assign(x, y)
    output = assign()
    output = output.asnumpy()
    output_expect = np.ones([3, 3]).astype(np.int8)
    print(output)
    assert np.all(output == output_expect)


def test_assign_uint8():
    x = Tensor(np.ones([3, 3]).astype(np.uint8))
    y = Tensor(np.zeros([3, 3]).astype(np.uint8))
    assign = Assign(x, y)
    output = assign()
    output = output.asnumpy()
    output_expect = np.ones([3, 3]).astype(np.uint8)
    print(output)
    assert np.all(output == output_expect)


def test_assign_int16():
    x = Tensor(np.ones([3, 3]).astype(np.int16))
    y = Tensor(np.zeros([3, 3]).astype(np.int16))
    assign = Assign(x, y)
    output = assign()
    output = output.asnumpy()
    output_expect = np.ones([3, 3]).astype(np.int16)
    print(output)
    assert np.all(output == output_expect)


def test_assign_uint16():
    x = Tensor(np.ones([3, 3]).astype(np.uint16))
    y = Tensor(np.zeros([3, 3]).astype(np.uint16))
    assign = Assign(x, y)
    output = assign()
    output = output.asnumpy()
    output_expect = np.ones([3, 3]).astype(np.uint16)
    print(output)
    assert np.all(output == output_expect)


def test_assign_int32():
    x = Tensor(np.ones([3, 3]).astype(np.int32))
    y = Tensor(np.zeros([3, 3]).astype(np.int32))
    assign = Assign(x, y)
    output = assign()
    output = output.asnumpy()
    output_expect = np.ones([3, 3]).astype(np.int32)
    print(output)
    assert np.all(output == output_expect)


def test_assign_uint32():
    x = Tensor(np.ones([3, 3]).astype(np.uint32))
    y = Tensor(np.zeros([3, 3]).astype(np.uint32))
    assign = Assign(x, y)
    output = assign()
    output = output.asnumpy()
    output_expect = np.ones([3, 3]).astype(np.uint32)
    print(output)
    assert np.all(output == output_expect)


def test_assign_int64():
    x = Tensor(np.ones([3, 3]).astype(np.int64))
    y = Tensor(np.zeros([3, 3]).astype(np.int64))
    assign = Assign(x, y)
    output = assign()
    output = output.asnumpy()
    output_expect = np.ones([3, 3]).astype(np.int64)
    print(output)
    assert np.all(output == output_expect)


def test_assign_uint64():
    x = Tensor(np.ones([3, 3]).astype(np.uint64))
    y = Tensor(np.zeros([3, 3]).astype(np.uint64))
    assign = Assign(x, y)
    output = assign()
    output = output.asnumpy()
    output_expect = np.ones([3, 3]).astype(np.uint64)
    print(output)
    assert np.all(output == output_expect)


def test_assign_float16():
    x = Tensor(np.array([[0.1, 0.2, 0.3],
                         [0.4, 0.5, 0.5],
                         [0.6, 0.7, 0.8]]).astype(np.float16))
    y = Tensor(np.array([[0.4, 0.5, 0.5],
                         [0.6, 0.7, 0.8],
                         [0.1, 0.2, 0.3]]).astype(np.float16))
    assign = Assign(x, y)
    output = assign()
    output = output.asnumpy()
    output_expect = np.array([[0.1, 0.2, 0.3],
                              [0.4, 0.5, 0.5],
                              [0.6, 0.7, 0.8]]).astype(np.float16)
    print(output)
    assert np.all(output - output_expect < 1e-6)


def test_assign_float32():
    x = Tensor(np.array([[0.1, 0.2, 0.3],
                         [0.4, 0.5, 0.5],
                         [0.6, 0.7, 0.8]]).astype(np.float32))
    y = Tensor(np.array([[0.4, 0.5, 0.5],
                         [0.6, 0.7, 0.8],
                         [0.1, 0.2, 0.3]]).astype(np.float32))
    assign = Assign(x, y)
    output = assign()
    output = output.asnumpy()
    output_expect = np.array([[0.1, 0.2, 0.3],
                              [0.4, 0.5, 0.5],
                              [0.6, 0.7, 0.8]]).astype(np.float32)
    print(output)
    assert np.all(output - output_expect < 1e-6)


def test_assign_float64():
    x = Tensor(np.array([[0.1, 0.2, 0.3],
                         [0.4, 0.5, 0.5],
                         [0.6, 0.7, 0.8]]).astype(np.float64))
    y = Tensor(np.array([[0.4, 0.5, 0.5],
                         [0.6, 0.7, 0.8],
                         [0.1, 0.2, 0.3]]).astype(np.float64))
    assign = Assign(x, y)
    output = assign()
    output = output.asnumpy()
    output_expect = np.array([[0.1, 0.2, 0.3],
                              [0.4, 0.5, 0.5],
                              [0.6, 0.7, 0.8]]).astype(np.float64)
    print(output)
    assert np.all(output - output_expect < 1e-6)


class AssignAdd(nn.Cell):
    def __init__(self, x, y):
        super(AssignAdd, self).__init__()
        self.x = Parameter(initializer(x, x.shape), name="x")
        self.y = Parameter(initializer(y, y.shape), name="y")
        self.assignadd = P.AssignAdd()

    def construct(self):
        self.assignadd(self.y, self.x)
        return self.y


def test_number_assignadd_number():
    input_x = 2
    result1 = 5
    result2 = 5
    result1 += input_x
    assignadd = AssignAdd(result2, input_x)
    result2 = assignadd()
    expect = 7
    assert np.all(result1 == expect)
    assert np.all(result2 == expect)


def test_tensor_assignadd_tensor():
    input_x = Tensor(np.array([[2, 2], [3, 3]]))
    result1 = Tensor(np.array([[4, -2], [2, 17]]))
    result2 = Tensor(np.array([[4, -2], [2, 17]]))
    result1 += input_x
    result2 = AssignAdd(result2, input_x)()
    expect = Tensor(np.array([[6, 0], [5, 20]]))
    assert np.all(result1.asnumpy() == expect)
    assert np.all(result2.asnumpy() == expect)


def test_tensor_assignadd_number():
    input_x = 3
    result1 = Tensor(np.array([[4, -2], [2, 17]])).astype(np.float16)
    result2 = Tensor(np.array([[4, -2], [2, 17]])).astype(np.float16)
    result1 += input_x
    result2 = AssignAdd(result2, input_x)()
    expect = Tensor(np.array([[7, 1], [5, 20]]))
    assert np.all(result1.asnumpy() == expect)
    assert np.all(result2.asnumpy() == expect)


def test_number_assignadd_tensor():
    result1 = 3
    result2 = 3
    input_x = Tensor(np.array([[4, -2], [2, 17]])).astype(np.float16)
    result1 += input_x
    result2 = AssignAdd(result2, input_x)()
    expect = Tensor(np.array([[7, 1], [5, 20]]))
    assert np.all(result1.asnumpy() == expect)
    assert np.all(result2.asnumpy() == expect)


def test_tuple_assignadd_tuple():
    result1 = (1, 2, 3, 4)
    result2 = (1, 2, 3, 4)
    input_x = (2, 3, 4, 5, 6)
    result1 += input_x
    result2 = AssignAdd(result2, input_x)()
    expect = (1, 2, 3, 4, 2, 3, 4, 5, 6)
    assert np.all(result1.asnumpy() == expect)
    assert np.all(result2.asnumpy() == expect)


def test_string_assignadd_string():
    result1 = "string111"
    input2 = "string111"
    input_x = "string222"
    result1 += input_x
    result2 = AssignAdd(input2, input_x)
    expect = "string111string222"
    assert result1 == expect
    assert result2 == expect


class AssignSub(nn.Cell):
    def __init__(self, x, y):
        super(AssignSub, self).__init__()
        self.x = Parameter(initializer(x, x.shape), name="x")
        self.y = Parameter(initializer(y, y.shape), name="y")
        self.assignsub = P.AssignSub()

    def construct(self):
        self.assignsub(self.y, self.x)
        return self.y


def test_number_assignsub_number():
    input_x = 2
    result1 = 5
    input2 = 5
    result1 -= input_x
    result2 = AssignSub(input2, input_x)
    expect = 3
    assert np.all(result1 == expect)
    assert np.all(result2 == expect)


def test_tensor_assignsub_tensor():
    input_x = Tensor(np.array([[2, 2], [3, 3]]))
    result1 = Tensor(np.array([[4, -2], [2, 17]]))
    result2 = Tensor(np.array([[4, -2], [2, 17]]))
    result1 -= input_x
    result2 = AssignSub(result2, input_x)
    expect = Tensor(np.array([[2, -4], [-1, 14]]))
    assert np.all(result1.asnumpy() == expect)
    assert np.all(result2.asnumpy() == expect)


def test_tensor_assignsub_number():
    input_x = 3
    result1 = Tensor(np.array([[4, -2], [2, 17]])).astype(np.float16)
    input2 = Tensor(np.array([[4, -2], [2, 17]])).astype(np.float16)
    result1 -= input_x
    result2 = AssignSub(input2, input_x)
    expect = Tensor(np.array([[1, -5], [-1, 14]]))
    assert np.all(result1.asnumpy() == expect)
    assert np.all(result2.asnumpy() == expect)


def test_number_assignsub_tensor():
    result1 = 3
    input2 = 3
    input_x = Tensor(np.array([[4, -2], [2, 17]])).astype(np.float16)
    result1 -= input_x
    result2 = AssignSub(input2, input_x)
    expect = Tensor(np.array([[-1, 5], [1, -14]]))
    assert np.all(result1.asnumpy() == expect)
    assert np.all(result2.asnumpy() == expect)


def test_number_assignmul_number():
    input_x = 2
    result = 5
    result *= input_x
    expect = 10
    assert np.all(result == expect)


def test_tensor_assignmul_tensor():
    input_x = Tensor(np.array([[2, 2], [3, 3]]))
    result = Tensor(np.array([[4, -2], [2, 17]]))
    result *= input_x
    expect = Tensor(np.array([[8, -4], [6, 51]]))
    assert np.all(result.asnumpy() == expect)


def test_tensor_assignmul_number():
    input_x = 3
    result = Tensor(np.array([[4, -2], [2, 17]])).astype(np.float16)
    result *= input_x
    expect = Tensor(np.array([[12, -6], [6, 51]]))
    assert np.all(result.asnumpy() == expect)


def test_number_assignmul_tensor():
    result = 3
    input_x = Tensor(np.array([[4, -2], [2, 17]])).astype(np.float16)
    result *= input_x
    expect = Tensor(np.array([[12, -6], [6, 51]]))
    assert np.all(result.asnumpy() == expect)


def test_number_assigndiv_number():
    input_x = 2
    result = 5
    result /= input_x
    expect = 2.5
    assert np.all(result == expect)


def test_tensor_assigndiv_tensor():
    input_x = Tensor(np.array([[2, 2], [3, 3]]))
    result = Tensor(np.array([[4, -2], [6, 15]]))
    result /= input_x
    expect = Tensor(np.array([[2, -1], [2, 5]]))
    assert np.all(result.asnumpy() == expect)


def test_tensor_assigndiv_number():
    input_x = 3
    result = Tensor(np.array([[9, -3], [6, 15]])).astype(np.float16)
    result /= input_x
    expect = Tensor(np.array([[3, -1], [2, 5]]))
    assert np.all(result.asnumpy() == expect)


def test_number_assigndiv_tensor():
    result = 3
    input_x = Tensor(np.array([[2, -2], [2, -2]])).astype(np.float16)
    result /= input_x
    expect = Tensor(np.array([[1.5, -1.5], [1.5, -1.5]]))
    assert np.all(result.asnumpy() == expect)


def test_number_assignmod_number():
    input_x = 2
    result = 5
    result %= input_x
    expect = 1
    assert np.all(result == expect)


def test_tensor_assignmod_tensor():
    input_x = Tensor(np.array([[2, 2], [3, 3]]))
    result = Tensor(np.array([[4, -2], [6, 15]]))
    result %= input_x
    expect = Tensor(np.array([[0, 0], [0, 0]]))
    assert np.all(result.asnumpy() == expect)


def test_tensor_assignmod_number():
    input_x = 3
    result = Tensor(np.array([[9, -3], [7, 15]])).astype(np.float16)
    result %= input_x
    expect = Tensor(np.array([[0, 0], [1, 0]]))
    assert np.all(result.asnumpy() == expect)


def test_number_assignmod_tensor():
    result = 3
    input_x = Tensor(np.array([[2, -2], [2, -2]])).astype(np.float16)
    result %= input_x
    expect = Tensor(np.array([[1, -1], [1, -1]]))
    assert np.all(result.asnumpy() == expect)


def test_number_assignmulmul_number():
    input_x = 2
    result = 5
    result **= input_x
    expect = 25
    assert np.all(result == expect)


def test_tensor_assignmulmul_tensor():
    input_x = Tensor(np.array([[2, 2], [3, 3]]))
    result = Tensor(np.array([[4, -2], [6, 5]]))
    result **= input_x
    expect = Tensor(np.array([[16, 4], [216, 125]]))
    assert np.all(result.asnumpy() == expect)


def test_tensor_assignmulmul_number():
    input_x = 3
    result = Tensor(np.array([[9, -3], [7, 5]])).astype(np.float16)
    result **= input_x
    expect = Tensor(np.array([[729, -27], [343, 125]]))
    assert np.all(result.asnumpy() == expect)


def test_number_assignmulmul_tensor():
    result = 3
    input_x = Tensor(np.array([[2, 2], [2, 2]])).astype(np.float16)
    result **= input_x
    expect = Tensor(np.array([[9, 9], [9, 9]]))
    assert np.all(result.asnumpy() == expect)


def test_number_assigndivdiv_number():
    input_x = 2
    result = 5
    result //= input_x
    expect = 2
    assert np.all(result == expect)


def test_tensor_assigndivdiv_tensor():
    input_x = Tensor(np.array([[2, 2], [3, 3]]))
    result = Tensor(np.array([[4, -2], [6, 6]]))
    result //= input_x
    expect = Tensor(np.array([[2, -1], [2, 2]]))
    assert np.all(result.asnumpy() == expect)


def test_tensor_assigndivdiv_number():
    input_x = 3
    result = Tensor(np.array([[9, -3], [15, 9]])).astype(np.float16)
    result //= input_x
    expect = Tensor(np.array([[3, -1], [5, 3]]))
    assert np.all(result.asnumpy() == expect)


def test_number_assigndivdiv_tensor():
    result = 3
    input_x = Tensor(np.array([[1, 2], [2, 2]])).astype(np.float16)
    result //= input_x
    expect = Tensor(np.array([[3, 1], [1, 1]]))
    assert np.all(result.asnumpy() == expect)
