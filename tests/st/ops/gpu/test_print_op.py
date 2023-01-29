# Copyright 2021-2023 Huawei Technologies Co., Ltd
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
import pytest

from mindspore import Tensor, jit
import mindspore.nn as nn
import mindspore.numpy as ms_np
from mindspore.ops import operations as P
import mindspore.context as context
import mindspore as ms
from tests.security_utils import security_off_wrap


class PrintNetOneInput(nn.Cell):
    def __init__(self):
        super(PrintNetOneInput, self).__init__()
        self.op = P.Print()

    def construct(self, x):
        self.op(x)
        return x


class PrintNetTwoInputs(nn.Cell):
    def __init__(self):
        super(PrintNetTwoInputs, self).__init__()
        self.op = P.Print()

    def construct(self, x, y):
        self.op(x, y)
        return x


class PrintNetIndex(nn.Cell):
    def __init__(self):
        super(PrintNetIndex, self).__init__()
        self.op = P.Print()

    def construct(self, x):
        self.op(x[0][0][6][3])
        return x


@security_off_wrap
def print_testcase(nptype):
    # large shape
    x = np.arange(20808).reshape(6, 3, 34, 34).astype(nptype)
    # a value that can be stored as int8_t
    x[0][0][6][3] = 125
    # small shape
    y = np.arange(9).reshape(3, 3).astype(nptype)
    x = Tensor(x)
    y = Tensor(y)

    net_1 = PrintNetOneInput()
    net_2 = PrintNetTwoInputs()
    net_3 = PrintNetIndex()
    net_1(x)
    net_2(x, y)
    net_3(x)


class PrintNetString(nn.Cell):
    def __init__(self):
        super(PrintNetString, self).__init__()
        self.op = P.Print()

    def construct(self, x, y):
        self.op("The first Tensor is", x)
        self.op("The second Tensor is", y)
        self.op("This line only prints string", "Another line")
        self.op("The first Tensor is", x, y, "is the second Tensor")
        return x


@security_off_wrap
def print_testcase_string(nptype):
    x = np.ones(18).astype(nptype)
    y = np.arange(9).reshape(3, 3).astype(nptype)
    x = Tensor(x)
    y = Tensor(y)
    net = PrintNetString()
    net(x, y)


class PrintTypes(nn.Cell):
    def __init__(self):
        super(PrintTypes, self).__init__()
        self.op = P.Print()

    def construct(self, x, y, z):
        self.op("This is a scalar:", 34, "This is int:", x,
                "This is float64:", y, "This is int64:", z)
        return x


@security_off_wrap
@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_print_multiple_types(mode):
    """
    Feature: GPU Print op.
    Description: test print with multiple types.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="GPU")
    x = Tensor(np.array([[1], [3], [4], [6], [3]], dtype=np.int32))
    y = Tensor(np.array([[1], [3], [4], [6], [3]]).astype(np.float64))
    z = Tensor(np.arange(9).reshape(3, 3).astype(np.int64))
    net = PrintTypes()
    net(x, y, z)


@security_off_wrap
@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64,
                                   np.bool, np.float64, np.float32, np.float16, np.complex64, np.complex128])
def test_print_dtype(mode, dtype):
    """
    Feature: GPU Print op.
    Description: test print with the different types.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="GPU")
    print_testcase(dtype)


@security_off_wrap
@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_print_string(mode):
    """
    Feature: GPU Print op.
    Description: test Print with string.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="GPU")
    print_testcase_string(np.float32)


@security_off_wrap
@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_print_dynamic_shape(mode):
    """
    Feature: GPU Print op.
    Description: test Print with dynamic shape.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="GPU")

    net = PrintNetOneInput()
    x = Tensor(np.random.randn(3, 4, 5).astype(np.float32))
    x_dyn = Tensor(shape=[None, None, None], dtype=ms.float32)
    net.set_inputs(x_dyn)
    net(x)


@security_off_wrap
@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_print_op_tuple():
    """
    Feature: cpu Print op.
    Description: test Print with tuple input.
    Expectation: success.
    """
    class PrintTupleNet(nn.Cell):
        def construct(self, x):
            tuple_x = tuple((1, 2, 3, 4, 5))
            print("tuple_x:", tuple_x, x, "print success!")
            return x

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = PrintTupleNet()
    x = Tensor([6, 7, 8, 9, 10])
    net(x)


@security_off_wrap
@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_print_tensor_dtype_in_nested_tuple(mode):
    """
    Feature: Print op.
    Description: test Print with tensor dtype in nested tuple.
    Expectation: success.
    """
    class PrintDtypeNet(nn.Cell):
        def construct(self, x, y):
            dtype_tuple = (x.dtype, y)
            dtype_tuple_tuple = (x, dtype_tuple)
            print("Tensor type in tuple:", dtype_tuple_tuple)
            return x + y

    context.set_context(mode=mode, device_target="GPU")
    x = Tensor([3, 4], dtype=ms.int32)
    y = Tensor([1, 2], dtype=ms.int32)
    net = PrintDtypeNet()
    net(x, y)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_print_abs():
    """
    Feature: Print op.
    Description: Print the result of max.
    Expectation: success.
    """
    @jit
    def function():
        tuple_x = (Tensor(10).astype("float32"), Tensor(30).astype("float32"), Tensor(50).astype("float32"))
        sum_x = Tensor(0).astype("float32")
        max_x = Tensor(0).astype("float32")
        for i in range(3):
            max_x = max(tuple_x)
            sum_x += max_x
            print(max_x)
            print(i)
        for x in zip(tuple_x):
            sum_x = sum(x, sum_x)
            print(sum_x)
        return sum_x

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    out = function()
    print("out:", out)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_print_tensor():
    """
    Feature: Print op.
    Description: Print tensor.
    Expectation: success.
    """
    class ReLUDynamicNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()
            self.reduce = P.ReduceSum(keep_dims=False)
            self.shape = P.TensorShape()

        def construct(self, x, y, z):
            rand_axis = ms_np.randint(1, 3, (3,))
            axis = ms_np.unique(rand_axis)
            print("the input y is:", y)
            print("the input z is:", z)
            print("the input z is:", z, " the shape of x:", self.shape(x))
            print("before reduce the shape of x is: ", self.shape(x))
            x = self.reduce(x, axis)
            x_shape = self.shape(x)
            print("after reduce the shape of x is: ", self.shape(x))
            return self.relu(x), axis, x_shape

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    relu_net = ReLUDynamicNet()
    input_x = Tensor(np.random.randn(8, 3, 5, 8, 5, 6).astype(np.float32))
    input_y = [1, 2, 3, 4]
    input_z = (5, 6, 7, 8, 9)
    _, _, x_shape = relu_net(input_x, input_y, input_z)
    print(x_shape)
    print("test end")
