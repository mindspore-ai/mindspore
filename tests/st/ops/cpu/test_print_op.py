# Copyright 2022-2025 Huawei Technologies Co., Ltd
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
import mindspore.context as context
import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor, ops, Parameter
from mindspore.common import dtype as ms_dtype
from tests.mark_utils import arg_mark


class PrintNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.print = ops.Print()

    def construct(self, x):
        self.print("scalar int:", 2, "scalar float:", 1.2, "scalar bool:", False, "Tensor type:", x)
        return x


class PrintFunc(nn.Cell):
    def construct(self, x):
        ops.print_("scalar int:", 2, "scalar float:", 1.2, "scalar bool:", False, "Tensor :", x)
        return x


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64,
                                   np.bool_, np.float64, np.float32, np.float16, np.complex64, np.complex128])
def test_print_op_dtype(mode, dtype):
    """
    Feature: cpu Print ops.
    Description: test print with the different types.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="CPU")

    net = PrintNet()
    x = Tensor(np.array([[1, 2], [3, 4], [5, 6]]).astype(dtype))
    net(x)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_print_op_dynamic_shape(mode):
    """
    Feature: cpu Print op.
    Description: test Print with dynamic shape.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="CPU")

    net = PrintNet()
    x = Tensor(np.random.randn(3, 4, 5).astype(np.float32))
    x_dyn = Tensor(shape=[None, None, None], dtype=ms.float32)
    net.set_inputs(x_dyn)
    net(x)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_print_op_functional(mode):
    """
    Feature: cpu Print op.
    Description: test Print with functional interface.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="CPU")

    net = PrintFunc()
    x = Tensor(np.random.randn(3, 4, 5).astype(np.float32))
    net(x)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_print_op_tuple():
    """
    Feature: cpu Print op.
    Description: test Print with tuple input.
    Expectation: success.
    """

    class PrintTupleNet(nn.Cell):
        def construct(self, x):
            tuple_x = tuple((1, 2, 3, 4, 5))
            ops.print_("tuple_x:", tuple_x, x, "print success!")
            return x

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    net = PrintTupleNet()
    x = Tensor([6, 7, 8, 9, 10])
    net(x)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_print_op_string_twice():
    """
    Feature: cpu Print op.
    Description: test Print string twice. Avoid memory reuse with dirty data.
    Expectation: success.
    """

    class Network(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor(np.random.randn(5, 3), ms.float32), name='w')
            self.b = Parameter(Tensor(np.random.randn(3), ms.float32), name='b')

        def construct(self, x):
            out = ops.matmul(x, self.w)
            print('matmul: ', out)
            out = out + self.b
            print('add bias: ', out)
            return out

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    x = ops.ones(5, ms.float32)
    model = Network()
    model.compile(x)
    out = model(x)
    print('out: ', out)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_print_check():
    """
    Feature: Print op.
    Description: Check if print is a built-in function.
    Expectation: There is no print implementation under the ops module, an error needs to be reported.
    """
    class Net(nn.Cell):
        def construct(self, x):
            ops.print(x)
            return x

    with pytest.raises(AttributeError) as raise_info:
        context.set_context(mode=context.GRAPH_MODE)
        net = Net()
        x = Tensor([1, 2], ms_dtype.float32)
        net(x)
    assert "Module: 'Namespace:mindspore.ops' object has no attribute 'print'" in str(raise_info.value)
