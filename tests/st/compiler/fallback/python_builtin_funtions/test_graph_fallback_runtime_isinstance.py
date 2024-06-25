# Copyright 2023 Huawei Technologies Co., Ltd
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

import pytest
import numpy as np
from mindspore import Tensor, context, Parameter, jit_class, COOTensor, CSRTensor
from mindspore.common.sparse_tensor import RowTensorInner
import mindspore.nn as nn
import mindspore as ms
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_isinstance():
    """
    Feature: JIT Fallback
    Description: Test isinstance() in fallback runtime
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            x_is_tensor = isinstance(x, Tensor)
            x_is_sequence = isinstance(x, (tuple, list))
            y_is_sequence = isinstance(y, (tuple, list))
            return x_is_tensor, x_is_sequence, y_is_sequence

    x = Tensor([-1, 2, 4])
    y = (1, 2)
    net = Net()
    out = net(x, y)
    assert out[0] == out[2] == True
    assert not out[1]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_isinstance_numpy():
    """
    Feature: JIT Fallback
    Description: Test isinstance() in fallback runtime
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, x):
            return isinstance(x.asnumpy(), np.ndarray)

    x = Tensor(np.array([-1, 2, 4]))
    net = Net()
    out = net(x)
    assert out


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_isinstance_numpy_type():
    """
    Feature: JIT Fallback
    Description: Test isinstance() in fallback runtime
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = np.array([-1, 2, 4])
            return isinstance(x.asnumpy(), type(y))

    x = Tensor(np.array([-1, 2, 4]))
    net = Net()
    out = net(x)
    assert out


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_isinstance_parameter():
    """
    Feature: JIT Fallback
    Description: Test isinstance() in fallback runtime
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.para = Parameter(Tensor(2, dtype=ms.float64), name='para')

        def construct(self, x):
            return isinstance(self.para, type(x)), isinstance(self.para, (list, type(x)))

    x = Tensor([-1, 2, 4])
    net = Net()
    out = net(x)
    assert out[0], out[1]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_isinstance_tuple():
    """
    Feature: JIT Fallback
    Description: Test isinstance() in fallback runtime
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            if isinstance(x.asnumpy(), (int, float, complex, bool)):
                return x
            return x + 1

    x = Tensor([-1])
    print("type:", type(x.asnumpy()))
    net = Net()
    out = net(x)
    assert out == 0


@jit_class
class NetIsinstanceClass:
    def __init__(self):
        self.number = 1

    def add(self):
        out = self.number + self.number
        return out


@jit_class
class NetIsinstanceClass2:
    def __init__(self):
        self.number2 = 1

    def add(self):
        out = self.number2 + self.number2
        return out


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_isinstance_jit_class_type():
    """
    Feature: JIT Fallback
    Description: Test isinstance() in fallback runtime
    Expectation: No exception.
    """

    class IsinstanceNet1(nn.Cell):
        def __init__(self, x):
            super().__init__()
            self.x = x

        def construct(self):
            output1 = isinstance(self.x, NetIsinstanceClass)
            net = NetIsinstanceClass()
            output2 = isinstance(net, NetIsinstanceClass)
            return output1, output2

    input_x_nparray = np.array([[2, 2], [2, 2]])
    net_isinstance = IsinstanceNet1(input_x_nparray)
    res = net_isinstance()
    assert not res[0], res[1]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_isinstance_jit_class_type_tuple():
    """
    Feature: JIT Fallback
    Description: Test isinstance() in fallback runtime
    Expectation: No exception.
    """

    class IsinstanceNet2(nn.Cell):
        def __init__(self, x):
            super().__init__()
            self.x = x
            self.y = NetIsinstanceClass()

        def construct(self):
            return isinstance(self.x, (NetIsinstanceClass, int)), isinstance(self.x, (NetIsinstanceClass2, int))

    input_x_nparray = np.array([[2, 2], [2, 2]])
    net_isinstance = IsinstanceNet2(input_x_nparray)
    res = net_isinstance()
    assert not res[0] and not res[1]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_isinstance_tensor():
    """
    Feature: JIT Fallback
    Description: Test isinstance() in fallback runtime
    Expectation: No exception.
    """

    class IsinstanceNet3(nn.Cell):
        def __init__(self):
            super().__init__()
            indices_cootensor = Tensor([[0, 1], [1, 2]], dtype=ms.int32)
            values_cootensor = Tensor([1, 2], dtype=ms.float32)
            shape_cootensor = (3, 4)
            self.x = COOTensor(indices_cootensor, values_cootensor, shape_cootensor)

        def construct(self):
            return isinstance(self.x, Tensor)

    net_isinstance = IsinstanceNet3()
    res = net_isinstance()
    assert not res


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_isinstance_cootensor():
    """
    Feature: JIT Fallback
    Description: Test isinstance() in fallback runtime
    Expectation: No exception.
    """

    class IsinstanceNet4(nn.Cell):
        def __init__(self, x):
            super().__init__()
            self.x = x
            indices_cootensor = Tensor([[0, 1], [1, 2]], dtype=ms.int32)
            values_cootensor = Tensor([1, 2], dtype=ms.float32)
            shape_cootensor = (3, 4)
            self.y = COOTensor(indices_cootensor, values_cootensor, shape_cootensor)

        def construct(self):
            return isinstance(self.x, COOTensor), isinstance(self.y, COOTensor)

    input_x_nparray = np.array([[2, 2], [2, 2]])
    net_isinstance = IsinstanceNet4(input_x_nparray)
    res = net_isinstance()
    assert not res[0] and res[1]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_isinstance_csrtensor():
    """
    Feature: JIT Fallback
    Description: Test isinstance() in fallback runtime
    Expectation: No exception.
    """

    class IsinstanceNet5(nn.Cell):
        def __init__(self, x):
            super().__init__()
            self.x = x
            indptr = Tensor([0, 1, 2])
            indices = Tensor([0, 1])
            values = Tensor([1, 2], dtype=ms.float32)
            shape = (2, 6)
            self.y = CSRTensor(indptr, indices, values, shape)

        def construct(self):
            return isinstance(self.x, CSRTensor), isinstance(self.y, CSRTensor)

    input_x_nparray = np.array([[2, 2], [2, 2]])
    net_isinstance = IsinstanceNet5(input_x_nparray)
    res = net_isinstance()
    assert not res[0] and res[1]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_isinstance_rowtensor():
    """
    Feature: JIT Fallback
    Description: Test isinstance() in fallback runtime
    Expectation: No exception.
    """

    class IsinstanceNet6(nn.Cell):
        def __init__(self, x):
            super().__init__()
            self.x = x

        def construct(self):
            return isinstance(self.x, RowTensorInner)

    input_x_nparray = np.array([[2, 2], [2, 2]])

    net_isinstance = IsinstanceNet6(input_x_nparray)
    res = net_isinstance()
    assert not res


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_isinstance_rowtensor_cootenosr_csrtensor_np():
    """
    Feature: JIT Fallback
    Description: Test isinstance() in fallback runtime
    Expectation: No exception.
    """

    class IsinstanceNet7(nn.Cell):
        def __init__(self, x):
            super().__init__()
            self.x = x
            indptr = Tensor([0, 1, 2])
            indices = Tensor([0, 1])
            values = Tensor([1, 2], dtype=ms.float32)
            shape = (2, 6)
            self.y = CSRTensor(indptr, indices, values, shape)

        def construct(self):
            out1 = isinstance(self.x, (RowTensorInner, COOTensor, CSRTensor, np.ndarray))
            out2 = isinstance(self.y, (RowTensorInner, COOTensor, CSRTensor, np.ndarray))
            return out1, out2

    input_x_nparray = np.array([[2, 2], [2, 2]])
    net_isinstance = IsinstanceNet7(input_x_nparray)
    res = net_isinstance()
    assert res[0] and res[1]
