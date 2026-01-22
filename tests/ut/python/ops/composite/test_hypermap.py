# Copyright 2025 Huawei Technologies Co., Ltd
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
# pylint: disable=W0612
# pylint: disable=W0621

import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor, context, nn
from mindspore import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops.composite import HyperMap
from mindspore.graph.api import jit
from mindspore.nn import Cell

context.set_context(mode=context.GRAPH_MODE)

add = P.Add()
hyper_map = C.HyperMap()

double_elements_fg = C.MultitypeFuncGraph("double_elements_fg")
@double_elements_fg.register("Tensor", "Tuple")
def double_elements_fg_for_tensor_tuple(x, y):
    return P.Tile()(x, y)

class HyperMapNet(nn.Cell):
    def __init__(self, fg):
        super(HyperMapNet, self).__init__()
        self.common_map = C.HyperMap()
        self.fg = fg

    def construct(self, nest_tensor_list):
        output = self.common_map(self.fg, *nest_tensor_list)
        return output

@jit
def main_noleaf(x, y):
    return hyper_map(add, x, y)


def test_hypermap_noleaf_tuple_list_mix():
    """
    Feature: Check the types of inputs of HyperMap.
    Description: The types of inputs of HyperMap must be the same.
    Expectation: The types of inputs of HyperMap must be the same.
    """
    tensor1 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
    tensor2 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
    with pytest.raises(Exception, match="the types of arguments in HyperMap must be consistent"):
        main_noleaf((tensor1, 1), [tensor2, 2])


def test_hypermap_noleaf_tuple_length():
    """
    Feature: Check the length of arg of Tuple in HyperMap.
    Description: The length of inputs of HyperMap must be the same.
    Expectation: The length of inputs of HyperMap must be the same.
    """
    tensor1 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
    tensor2 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
    with pytest.raises(Exception, match="The tuples in HyperMap should have the same length."):
        main_noleaf((tensor1, 1), (tensor2, 2, 2))


def test_hypermap_noleaf_list_length():
    """
    Feature: Check the length of arg of List in HyperMap.
    Description: Check the length of arg of List in HyperMap.
    Expectation: Check the length of arg of List in HyperMap.
    """
    tensor1 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
    tensor2 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
    with pytest.raises(Exception, match="The lists in HyperMap should have the same length"):
        main_noleaf([tensor1], [tensor2, tensor2])


def test_hypermap_noleaf_list_tuple():
    """
    Feature: Check the types of inputs of HyperMap.
    Description: The types of inputs of HyperMap must be the same.
    Expectation: The types of inputs of HyperMap must be the same.
    """
    tensor1 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
    tensor2 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
    with pytest.raises(Exception, match="the types of arguments in HyperMap must be consistent"):
        main_noleaf([tensor1], (tensor2, tensor2))


def test_tuple_slice_stop_index():
    """
    Feature: Check the type of stop index of slice.
    Description: The type of stop index of slice must be scalar, None or Tensor.
    Expectation: The type of stop index of slice must be scalar, None or Tensor.
    """

    class TupleSliceNet(Cell):
        def __init__(self):
            super(TupleSliceNet, self).__init__()
            self.addn = P.AddN()
            self.index_0 = Tensor(3)

        def construct(self, tensor_tuple):
            tensor_tuple_slice0 = tensor_tuple[:]
            tensor_tuple_slice1 = tensor_tuple[self.index_0:"str"]  # slice should be Scalar or None, rather than string
            sum0 = self.addn(tensor_tuple_slice0)
            sum1 = self.addn(tensor_tuple_slice1)
            ret = sum0 + sum1
            return ret

    data = (Tensor(np.ones([2, 3, 4], np.int32)),
            Tensor(np.zeros([2, 3, 4], np.int32)),
            Tensor(np.ones([2, 3, 4], np.int32)),
            Tensor(np.ones([2, 3, 4], np.int32)),
            Tensor(np.zeros([2, 3, 4], np.int32)),
            Tensor(np.ones([2, 3, 4], np.int32)))

    net = TupleSliceNet()
    with pytest.raises(Exception, match="Slice indices must be integers or bool."):
        output = net(data)
        print("output:", output)


def test_tuple_slice_start_index():
    """
    Feature: Check the type of start index of slice.
    Description: The type of start index of slice must be scalar, None or Tensor.
    Expectation: The type of start index of slice must be scalar, None or Tensor.
    """

    class TupleSliceNet(Cell):
        def __init__(self):
            super(TupleSliceNet, self).__init__()
            self.addn = P.AddN()
            self.index_0 = Tensor(3)
            self.index_1 = Tensor([5])
            self.index_3 = Tensor([True])

        def construct(self, tensor_tuple):
            tensor_tuple_slice0 = tensor_tuple[:]
            tensor_tuple_slice1 = tensor_tuple["str":self.index_0]
            tensor_tuple_slice2 = tensor_tuple[self.index_3:]
            tensor_tuple_slice3 = tensor_tuple[2:self.index_1:]
            sum0 = self.addn(tensor_tuple_slice0)
            sum1 = self.addn(tensor_tuple_slice1)
            sum2 = self.addn(tensor_tuple_slice2)
            sum3 = self.addn(tensor_tuple_slice3)
            ret = sum0 + sum1 + sum2 + sum3
            return ret

    data = (Tensor(np.ones([2, 3, 4], np.int32)),
            Tensor(np.zeros([2, 3, 4], np.int32)),
            Tensor(np.ones([2, 3, 4], np.int32)),
            Tensor(np.ones([2, 3, 4], np.int32)),
            Tensor(np.zeros([2, 3, 4], np.int32)),
            Tensor(np.ones([2, 3, 4], np.int32)))

    net = TupleSliceNet()
    with pytest.raises(Exception, match="Slice indices must be integers or bool."):
        output = net(data)
        print("output:", output)


def test_tuple_slice_step():
    """
    Feature: Check the type of step of slice.
    Description: The type of step of slice must not be 0.
    Expectation: The type of step of slice must be scalar, None or Tensor.
    """

    class TupleSliceNet(Cell):
        def __init__(self):
            super(TupleSliceNet, self).__init__()
            self.addn = P.AddN()
            self.index_0 = Tensor(3)
            self.index_1 = Tensor([5])
            self.index_3 = Tensor([True])

        def construct(self, tensor_tuple):
            tensor_tuple_slice0 = tensor_tuple[:]
            tensor_tuple_slice1 = tensor_tuple[:self.index_0]
            tensor_tuple_slice2 = tensor_tuple[self.index_3:]
            tensor_tuple_slice3 = tensor_tuple[2:self.index_1:0]
            sum0 = self.addn(tensor_tuple_slice0)
            sum1 = self.addn(tensor_tuple_slice1)
            sum2 = self.addn(tensor_tuple_slice2)
            sum3 = self.addn(tensor_tuple_slice3)
            ret = sum0 + sum1 + sum2 + sum3
            return ret

    data = (Tensor(np.ones([2, 3, 4], np.int32)),
            Tensor(np.zeros([2, 3, 4], np.int32)),
            Tensor(np.ones([2, 3, 4], np.int32)),
            Tensor(np.ones([2, 3, 4], np.int32)),
            Tensor(np.zeros([2, 3, 4], np.int32)),
            Tensor(np.ones([2, 3, 4], np.int32)))

    net = TupleSliceNet()
    with pytest.raises(Exception, match="Slice step cannot be zero."):
        output = net(data)
        print("output:", output)


def test_double_elements_hypermap_inputs_length_mismatch():
    """
    Feature: HyperMap
    Description: When the inputs to hypermap have different length, error will be raised.
    Expectation: error.
    """
    x = (Tensor(np.array([1, 2, 3]), mstype.float32), Tensor(np.array([4, 5, 6]), mstype.float32))
    y = ((1, 2), (2, 1), (5, 6))
    common_map = HyperMapNet(double_elements_fg)
    with pytest.raises(Exception, match="The tuples in HyperMap should have the same length."):
        common_map((x, y))


def test_double_elements_hypermap_inconsistent_inputs():
    """
    Feature: HyperMap
    Description: When the inputs to hypermap is inconsistent, error will be raised.
    Expectation: error.
    """
    x = (Tensor(np.array([1, 2, 3]), mstype.float32), Tensor(np.array([4, 5, 6]), mstype.float32))
    y = [(1, 2), (2, 1)]
    common_map = HyperMapNet(double_elements_fg)
    with pytest.raises(Exception, match="the types of arguments in HyperMap must be consistent"):
        common_map((x, y))


def test_hypermap_ops_instance():
    """
    Feature: HyperMap
    Description: Test the HyperMap with switch_layer.
    Expectation: success.
    """
    add = P.Add()
    with pytest.raises(RuntimeError) as err:
        HyperMap(add)
    assert "Unable to cast Python instance of type " \
           "<class 'mindspore.ops.auto_generate.gen_ops_prim.Add'> to C++ type" in str(err.value)


def test_hypermap_ops_none():
    """
    Feature: HyperMap
    Description: Test the HyperMap with tuple.
    Expectation: success.
    """
    x = Tensor([1, 2])
    with pytest.raises(TypeError):
        HyperMap()((x, x), (x, x))


def test_hypermap_multi_dynamic_input_nested():
    """
    Feature: HyperMap
    Description: Test the HyperMap with multi dynamic input nested.
    Expectation: success.
    """
    def add_func(x, y):
        return x + y

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.hypermap = ms.ops.HyperMap()

        def construct(self, x, y):
            out = self.hypermap(add_func, x, y)
            return out

    input_x = ms.mutable(((Tensor([1]), Tensor([2])),
                          (Tensor([3]), Tensor([4]))), dynamic_len=True)
    input_y = ms.mutable(((Tensor([4]), Tensor([2])),
                          (Tensor([3]), Tensor([1]))), dynamic_len=True)
    net = Net()
    with pytest.raises(TypeError):
        net(input_x, input_y)
