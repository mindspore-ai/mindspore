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
"""test graph is_instance"""

import pytest
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor, CSRTensor, COOTensor, RowTensor, jit, jit_class, context
from mindspore.ops import Primitive
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.mutable import mutable

context.set_context(mode=context.GRAPH_MODE)


class Net1(nn.Cell):

    def construct(self, x):
        return x


class Net2(nn.Cell):

    def construct(self, x):
        return x + 1


class NetCombine(Net1, Net2):

    def construct(self, x):
        return x + 2


@jit_class
class MSClass1:
    def __init__(self):
        self.num1 = Tensor(1)


@jit_class
class MSClass2:
    def __init__(self):
        self.num2 = Tensor(2)


@jit_class
class MSCombine(MSClass1, MSClass2):
    def __init__(self):
        super(MSCombine, self).__init__()
        self.num3 = Tensor(3)


def test_isinstance_x_cell_obj_base_type_cell():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance support x to be cell object and base type to be cell.
    Expectation: No exception.
    """
    net = Net1()

    @jit
    def foo():
        return isinstance(net, nn.Cell)

    assert foo()


def test_isinstance_x_cell_obj_base_type_cell_2():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance support x to be cell object and base type to be cell.
    Expectation: No exception.
    """
    net = Net1()

    @jit
    def foo():
        return isinstance(net, Net1)

    assert foo()


def test_isinstance_x_cell_obj_base_type_cell_3():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance support x to be cell object and base type to be cell.
    Expectation: No exception.
    """
    net = Net1()

    @jit
    def foo():
        return isinstance(net, Net2)

    assert not foo()


def test_isinstance_x_cell_obj_base_type_cell_4():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance support x to be cell object and base type to be cell.
    Expectation: No exception.
    """
    net = Net1()

    @jit
    def foo():
        return isinstance(net, NetCombine)

    assert not foo()


def test_isinstance_x_cell_obj_base_type_csr_tensor():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance support x to be cell object and base type to be CSRTensor.
    Expectation: No exception.
    """
    net = Net1()

    @jit
    def foo():
        return isinstance(net, CSRTensor)

    assert not foo()


def test_isinstance_x_cell_obj_base_type_tuple():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance support x to be cell object and base type to be tuple.
    Expectation: No exception.
    """
    net = Net1()

    @jit
    def foo():
        return isinstance(net, (Net1, int, np.ndarray, MSCombine))

    assert foo()


def test_isinstance_x_cell_obj_base_type_tuple_2():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance support x to be cell object and base type to be tuple.
    Expectation: No exception.
    """
    net = Net1()

    @jit
    def foo():
        return isinstance(net, (Net2, int, np.ndarray))

    assert not foo()


def test_isinstance_x_cell_obj_base_type_tuple_3():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance support x to be cell object and base type to be tuple.
    Expectation: No exception.
    """
    net = Net1()

    @jit
    def foo():
        return isinstance(net, (int, float, (CSRTensor, (Net1, nn.Cell))))

    assert foo()


def test_isinstance_x_cell_obj_base_type_tuple_4():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance support x to be cell object and base type to be tuple.
    Expectation: No exception.
    """
    net = Net1()

    @jit
    def foo():
        return isinstance(net, (int, float, (CSRTensor, (Net2, Primitive), MSClass1)))

    assert not foo()


def test_isinstance_x_ms_class_obj():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance support x to be ms_class object.
    Expectation: No exception.
    """
    ms_obj = MSClass1()

    @jit
    def foo():
        return isinstance(ms_obj, (int, float, (MSClass1,)))

    assert foo()


def test_isinstance_x_ms_class_obj_2():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance support x to be ms_class object.
    Expectation: No exception.
    """
    ms_obj = MSClass1()

    @jit
    def foo():
        return isinstance(ms_obj, (int, float, (MSClass2,)))

    assert not foo()


def test_isinstance_x_ms_class_obj_3():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance support x to be ms_class object.
    Expectation: No exception.
    """
    ms_obj = MSCombine()

    @jit
    def foo():
        return isinstance(ms_obj, (int, float, (MSClass1, MSCombine)))

    assert foo()


def test_isinstance_x_ms_class_obj_4():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance support x to be ms_class object.
    Expectation: No exception.
    """
    ms_obj = MSClass2()

    @jit
    def foo():
        return isinstance(ms_obj, (int, float, (MSClass1, MSCombine)))

    assert not foo()


def test_isinstance_x_ms_class_obj_5():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance support x to be ms_class object.
    Expectation: No exception.
    """
    ms_obj = MSClass1()

    @jit
    def foo():
        return isinstance(ms_obj, (int, float, (MSClass2(), MSCombine)))

    with pytest.raises(TypeError) as err:
        foo()
    assert "isinstance() arg 2 must be a type or tuple of types" in str(err.value)


def test_isinstance_x_primitive_obj_base_type_primitive():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance support x to be primitive object and base type to be primitive.
    Expectation: No exception.
    """

    @jit
    def foo():
        return isinstance(P.Add(), Primitive) and isinstance(P.Add(), P.Add)

    assert foo()


def test_isinstance_x_primitive_obj_base_type_primitive_2():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance support x to be primitive object and base type to be primitive.
    Expectation: No exception.
    """

    @jit
    def foo():
        return isinstance(P.Add(), (Primitive, np.ndarray, int)) and isinstance(P.Add(), (P.Add, Net2, float))

    assert foo()


def test_isinstance_x_primitive_obj_base_type_primitive_3():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance support x to be primitive object and base type to be primitive.
    Expectation: No exception.
    """

    @jit
    def foo():
        return isinstance(Primitive("test"), Primitive)

    assert foo()


def test_isinstance_x_primitive_obj_base_type_primitive_4():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance support x to be primitive object and base type to be primitive.
    Expectation: No exception.
    """

    @jit
    def foo():
        return isinstance(Primitive("test"), (Primitive, CSRTensor, MSCombine))

    assert foo()


def test_isinstance_x_primitive_obj_base_type_primitive_5():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance support x to be primitive object and base type to be primitive.
    Expectation: No exception.
    """

    @jit
    def foo():
        return isinstance(Primitive("test"), (int, list, (Primitive,), CSRTensor))

    assert foo()


def test_isinstance_x_external_obj():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance support x to be external object which need to use fallback.
    Expectation: No exception.
    """

    @jit
    def foo():
        return isinstance(np.array(1), np.ndarray)

    assert foo()


def test_isinstance_x_external_obj_2():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance support x to be external object which need to use fallback.
    Expectation: No exception.
    """

    @jit
    def foo():
        return isinstance(np.array(1), list)

    assert not foo()


def test_isinstance_x_external_obj_3():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance support x to be external object which need to use fallback.
    Expectation: No exception.
    """

    @jit
    def foo():
        return isinstance(np.array(1), (list, np.ndarray, Tensor, CSRTensor))

    assert foo()


def test_isinstance_x_external_obj_4():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance support x to be external object which need to use fallback.
    Expectation: No exception.
    """

    @jit
    def foo():
        return isinstance(np.array(1), ((int, list), np.ndarray, Tensor, CSRTensor, MSClass2))

    assert foo()


def test_isinstance_x_tensor():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance support x to be tensor.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        return isinstance(x, (list, np.ndarray, Tensor, CSRTensor))

    x = Tensor(1)
    assert foo(x)


def test_isinstance_x_tensor_2():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance support x to be tensor.
    Expectation: No exception.
    """

    @jit
    def foo():
        return isinstance(Tensor(1), (list, np.ndarray, Tensor, CSRTensor))

    assert foo()


def test_isinstance_x_tensor_3():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance support x to be tensor.
    Expectation: No exception.
    """

    @jit
    def foo():
        return isinstance(Tensor(1), (list, np.ndarray, CSRTensor))

    assert not foo()


def test_isinstance_x_tensor_4():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance support x to be tensor.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = Tensor(np.array([4.0, 5.0, 6.0]).astype(np.float32))
        return isinstance(F.sin(a), (list, np.ndarray, Tensor, CSRTensor, MSClass1))

    assert foo()


def test_isinstance_x_coo_tensor():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance support x to be COOTensor.
    Expectation: No exception.
    """
    @jit
    def foo():
        indices = Tensor([[0, 1], [1, 2]])
        values = Tensor([1, 2])
        shape = (3, 4)
        x = COOTensor(indices, values, shape)
        return isinstance(x, (int, list, (Tensor, COOTensor)))

    assert foo()


def test_isinstance_x_coo_tensor_2():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance support x to be COOTensor.
    Expectation: No exception.
    """
    @jit
    def foo():
        indices = Tensor([[0, 1], [1, 2]])
        values = Tensor([1, 2])
        shape = (3, 4)
        x = COOTensor(indices, values, shape)
        return isinstance(x, (int, list, (Tensor, nn.Cell, CSRTensor)))

    assert not foo()


def test_isinstance_x_csr_tensor():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance support x to be CSRTensor.
    Expectation: No exception.
    """
    @jit
    def foo():
        indptr = Tensor([0, 1, 2])
        indices = Tensor([0, 1])
        values = Tensor([1, 2])
        shape = (2, 4)
        x = CSRTensor(indptr, indices, values, shape)
        return isinstance(x, (int, tuple, (CSRTensor, COOTensor), nn.Cell))

    assert foo()


def test_isinstance_x_csr_tensor_2():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance support x to be CSRTensor.
    Expectation: No exception.
    """
    @jit
    def foo():
        indptr = Tensor([0, 1, 2])
        indices = Tensor([0, 1])
        values = Tensor([1, 2])
        shape = (2, 4)
        x = CSRTensor(indptr, indices, values, shape)
        return isinstance(x, (int, tuple, (Tensor, COOTensor), nn.Cell))

    assert not foo()


def test_isinstance_x_row_tensor():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance support x to be RowTensor.
    Expectation: No exception.
    """
    @jit
    def foo():
        indices = Tensor([0])
        values = Tensor([[1, 2]])
        shape = (3, 2)
        x = RowTensor(indices, values, shape)
        return isinstance(x, (Tensor, COOTensor, (list, tuple), RowTensor, MSClass2))

    assert foo()


def test_isinstance_x_row_tensor_2():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance support x to be RowTensor.
    Expectation: No exception.
    """
    @jit
    def foo():
        indices = Tensor([0])
        values = Tensor([[1, 2]])
        shape = (3, 2)
        x = RowTensor(indices, values, shape)
        return isinstance(x, (Tensor, COOTensor, (list, tuple)))

    assert not foo()


def test_isinstance_wrong_cmp_input():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance when cmp input is wrong.
    Expectation: No exception.
    """
    @jit
    def foo():
        return isinstance(Tensor(1), (np.array([1, 2, 3], Tensor)))

    with pytest.raises(TypeError) as err:
        foo()
    assert "isinstance() arg 2 must be a type or tuple of types" in str(err.value)


def test_isinstance_wrong_cmp_input_2():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance when cmp input is wrong.
    Expectation: No exception.
    """
    @jit
    def foo():
        return isinstance(Tensor(1), Tensor(1))

    with pytest.raises(TypeError) as err:
        foo()
    assert "isinstance() arg 2 must be a type or tuple of types" in str(err.value)


def test_isinstance_wrong_cmp_input_3():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance when cmp input is wrong.
    Expectation: No exception.
    """
    @jit
    def foo():
        return isinstance(Tensor(1), [list, tuple])

    with pytest.raises(TypeError) as err:
        foo()
    assert "isinstance() arg 2 must be a type or tuple of types" in str(err.value)


def test_isinstance_wrong_cmp_input_4():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance when cmp input is wrong.
    Expectation: No exception.
    """
    @jit
    def foo():
        return isinstance(Tensor(1), [1, tuple])

    with pytest.raises(TypeError) as err:
        foo()
    assert "isinstance() arg 2 must be a type or tuple of types" in str(err.value)


@pytest.mark.skip(reason='mutable feature not support scalar input')
def test_isinstance_x_mutable():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance when x is mutable.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        return isinstance(x, (int, tuple))

    x = mutable(2)
    assert foo(x)


@pytest.mark.skip(reason='mutable feature not support scalar input')
def test_isinstance_x_mutable_2():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance when x is mutable.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        return isinstance(x, (int, tuple))

    x = mutable(True)
    assert foo(x)


@pytest.mark.skip(reason='mutable feature not support scalar input')
def test_isinstance_x_mutable_3():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance when x is mutable.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        return isinstance(x, (int, tuple))

    x = mutable([1, 2, 3, 4])
    assert not foo(x)


def test_isinstance_x_mutable_4():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance when x is mutable.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        return isinstance(x, (int, list))

    x = mutable([Tensor(1), Tensor(2), Tensor(3)])
    assert foo(x)


def test_isinstance_x_mutable_5():
    """
    Feature: Graph isinstance.
    Description: Graph isinstance when x is mutable.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        return isinstance(x, (int, tuple))

    x = mutable([Tensor(1), Tensor(2), Tensor(3)])
    assert not foo(x)
